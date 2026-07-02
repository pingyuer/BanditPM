from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def map_to_token(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2).contiguous(), H, W


def token_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    B, N, C = tokens.shape
    if N != height * width:
        raise ValueError(f"Token count {N} does not match map size {height}x{width}.")
    return tokens.transpose(1, 2).contiguous().view(B, C, height, width)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_dim), out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, stride: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_dim), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ShiftMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, shift_size: int = 1) -> None:
        super().__init__()
        self.shift_size = int(shift_size)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = token_to_map(tokens, height, width)
        if self.shift_size:
            chunks = torch.chunk(x, 5, dim=1)
            shifts = (-self.shift_size, 0, self.shift_size, 0, -self.shift_size)
            shifted = []
            for idx, chunk in enumerate(chunks):
                dim = -2 if idx % 2 == 0 else -1
                shifted.append(torch.roll(chunk, shifts=shifts[idx], dims=dim))
            x = torch.cat(shifted, dim=1)
        tokens = map_to_token(x)[0]
        return self.fc2(self.act(self.fc1(tokens)))


class ShiftedMLPBlock(nn.Module):
    def __init__(self, dim: int, *, expansion: float = 4.0, shift_size: int = 1) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.mlp = ShiftMLP(dim, hidden, shift_size=shift_size)
        self.channel_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, H, W = map_to_token(x)
        y = self.mlp(self.norm(tokens), H, W)
        y = y * self.channel_scale.view(1, 1, -1)
        return x + token_to_map(y, H, W)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = ConvBlock(in_dim + skip_dim, out_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, skip], dim=1))


class UNeXtOfficialBackbone(nn.Module):
    """Official-like UNeXt anchor backbone with overlap patch embedding and shifted MLP blocks.

    The public interface intentionally matches the lightweight UNeXtBackbone used by CARDIA:
    encode() returns low/mid/high, and up1/up2/full_res/logits_from_decoder_feature remain
    callable by the temporal module.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_dim: int = 96,
        value_dim: int = 256,
        *,
        mlp_expansion: float = 3.0,
        latent_blocks: int = 2,
        decoder_mlp_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_dim = int(base_dim)
        self.value_dim = int(value_dim)
        dims = [self.base_dim, self.base_dim * 2, self.base_dim * 4]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(dims[0]), dims[0]),
            nn.GELU(),
            ConvBlock(dims[0], dims[0]),
        )
        self.patch_mid = OverlapPatchEmbed(dims[0], dims[1], stride=2, kernel_size=3)
        self.mid_blocks = nn.Sequential(
            ConvBlock(dims[1], dims[1]),
            *[ShiftedMLPBlock(dims[1], expansion=mlp_expansion, shift_size=1) for _ in range(max(0, decoder_mlp_blocks))],
        )
        self.patch_high = OverlapPatchEmbed(dims[1], dims[2], stride=2, kernel_size=3)
        self.high_blocks = nn.Sequential(
            ConvBlock(dims[2], dims[2]),
            *[
                ShiftedMLPBlock(dims[2], expansion=mlp_expansion, shift_size=1 if idx % 2 == 0 else -1)
                for idx in range(max(1, latent_blocks))
            ],
        )
        self.up1 = UpBlock(dims[2], dims[1], dims[1])
        self.up2 = UpBlock(dims[1], dims[0], dims[0])
        self.full_res = ConvBlock(dims[0], dims[0])
        self.logit_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        self.value_proj = nn.Conv2d(dims[1], value_dim, kernel_size=1)
        self.high_proj = nn.Conv2d(dims[2], value_dim, kernel_size=1)
        self.decoder_dim = dims[0]

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        low = self.stem(x)
        mid = self.mid_blocks(self.patch_mid(low))
        high = self.high_blocks(self.patch_high(mid))
        return {"low": low, "mid": mid, "high": high}

    def decode(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
        output_size: tuple[int, int],
        modulation: dict[str, dict[str, torch.Tensor] | torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        low = self._apply_modulation(low, modulation, "low")
        mid = self._apply_modulation(mid, modulation, "mid")
        high = self._apply_modulation(high, modulation, "high")
        dec_mid = self.up1(high, mid)
        dec_low = self.up2(dec_mid, low)
        dec = F.interpolate(dec_low, size=output_size, mode="bilinear", align_corners=False)
        dec = self.full_res(dec)
        dec = self._apply_modulation(dec, modulation, "dec")
        return {
            "logits": self.logit_head(dec),
            "decoder_feature": dec,
            "value": self.value_proj(dec_mid),
            "high_value": self.high_proj(high),
        }

    def _apply_modulation(
        self,
        feat: torch.Tensor,
        modulation: dict[str, dict[str, torch.Tensor] | torch.Tensor] | None,
        level: str,
    ) -> torch.Tensor:
        if not modulation or level not in modulation:
            return feat
        item = modulation[level]
        if torch.is_tensor(item):
            scale = None
            shift = item
        else:
            scale = item.get("scale")
            shift = item.get("shift")
        if scale is not None:
            if scale.shape[-2:] != feat.shape[-2:]:
                scale = F.interpolate(scale, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat * (1.0 + scale)
        if shift is not None:
            if shift.shape[-2:] != feat.shape[-2:]:
                shift = F.interpolate(shift, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat + shift
        return feat

    def logits_from_decoder_feature(self, decoder_feature: torch.Tensor) -> torch.Tensor:
        return self.logit_head(decoder_feature)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["low"], encoded["mid"], encoded["high"], x.shape[-2:])
        return {**decoded, **encoded}
