from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.unext.shifted_mlp import ShiftedMLPBlock


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = ConvBlock(in_dim + skip_dim, out_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, skip], dim=1))


class UNeXtBackbone(nn.Module):
    """UNeXt-style single-frame segmenter with U-Net skips and shifted MLP bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_dim: int = 32,
        value_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_dim = base_dim
        dims = [base_dim, base_dim * 2, base_dim * 4]

        self.input_down = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        self.stem = ConvBlock(dims[0], dims[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(dims[0], dims[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(dims[1], dims[2]))
        self.token_mlp = nn.Sequential(
            ShiftedMLPBlock(dims[2]),
            ShiftedMLPBlock(dims[2], shift=-1),
        )
        self.up1 = UpBlock(dims[2], dims[1], dims[1])
        self.up2 = UpBlock(dims[1], dims[0], dims[0])
        self.full_res = ConvBlock(dims[0], dims[0])
        self.logit_head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        self.value_proj = nn.Conv2d(dims[1], value_dim, kernel_size=1)
        self.high_proj = nn.Conv2d(dims[2], value_dim, kernel_size=1)
        self.decoder_dim = dims[0]
        self.value_dim = value_dim

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        low = self.stem(self.input_down(x))
        mid = self.down1(low)
        high = self.token_mlp(self.down2(mid))
        return {"low": low, "mid": mid, "high": high}

    def decode(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        dec_mid = self.up1(high, mid)
        dec_low = self.up2(dec_mid, low)
        dec = F.interpolate(dec_low, size=output_size, mode="bilinear", align_corners=False)
        dec = self.full_res(dec)
        return {
            "logits": self.logit_head(dec),
            "decoder_feature": dec,
            "value": self.value_proj(dec_mid),
            "high_value": self.high_proj(high),
        }

    def logits_from_decoder_feature(self, decoder_feature: torch.Tensor) -> torch.Tensor:
        return self.logit_head(decoder_feature)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded["low"], encoded["mid"], encoded["high"], x.shape[-2:])
        return {
            **decoded,
            **encoded,
        }
