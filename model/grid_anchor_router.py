from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DepthwiseRelationEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.out_dim = hidden

    def forward(self, current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        relation = torch.cat([current, anchor, current - anchor, current * anchor], dim=1)
        return self.net(relation)


class GridAnchorRouter(nn.Module):
    """Feature-level anchor-guided deformable routing.

    Offsets are normalized backward-sampling displacements added to an identity
    grid using align_corners=False pixel-center semantics.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        max_offset: float = 0.12,
        padding_mode: str = "border",
        align_corners: bool = False,
        update_gate_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.max_offset = float(max_offset)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

        self.relation = DepthwiseRelationEncoder(self.channels, hidden_dim)
        hidden = self.relation.out_dim
        self.offset_head = nn.Conv2d(hidden, self.num_heads * 2, kernel_size=1)
        self.selector = nn.Conv2d(hidden, self.num_heads, kernel_size=1)
        self.trust_gate = nn.Conv2d(hidden, 1, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, 1, kernel_size=1)
        self.delta_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.fusion_gate = nn.Conv2d(self.channels * 3, self.channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.constant_(self.update_gate.bias, float(update_gate_bias))

    def _identity_grid(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        if self.align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(batch, height, width, 2)

    def forward(
        self,
        current: torch.Tensor,
        anchor_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if anchor_prev is None:
            anchor_prev = current.detach()
        if anchor_prev.shape[-2:] != current.shape[-2:]:
            anchor_prev = F.interpolate(anchor_prev, size=current.shape[-2:], mode="bilinear", align_corners=self.align_corners)

        B, C, H, W = current.shape
        relation = self.relation(current, anchor_prev)
        raw_offsets = self.offset_head(relation).view(B, self.num_heads, 2, H, W)
        offsets = torch.tanh(raw_offsets) * self.max_offset

        base_grid = self._identity_grid(B, H, W, current.device, current.dtype)
        grid = base_grid[:, None] + offsets.permute(0, 1, 3, 4, 2)
        sampled = F.grid_sample(
            anchor_prev[:, None].expand(-1, self.num_heads, -1, -1, -1).reshape(B * self.num_heads, C, H, W),
            grid.reshape(B * self.num_heads, H, W, 2),
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).view(B, self.num_heads, C, H, W)

        head_logits = self.selector(relation)
        head_weights = torch.softmax(head_logits, dim=1)
        routed = (sampled * head_weights[:, :, None]).sum(dim=1)
        trust = torch.sigmoid(self.trust_gate(relation))

        delta = self.delta_proj(routed - current)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([current, routed, delta], dim=1)))
        out = current + self.gamma * trust * gate * delta

        update = torch.sigmoid(self.update_gate(relation))
        next_anchor = update * out.detach() + (1.0 - update) * routed.detach()

        weights_flat = head_weights.flatten(2)
        entropy = -(weights_flat * weights_flat.clamp_min(1.0e-6).log()).sum(dim=1).mean(dim=-1)
        entropy = entropy / max(math.log(float(self.num_heads)), 1.0e-6)
        top_usage = F.one_hot(head_weights.argmax(dim=1), num_classes=self.num_heads).permute(0, 3, 1, 2).float()
        offset_abs = offsets.detach().abs()
        aux = {
            "relation": relation,
            "warped_features": sampled,
            "head_logits": head_logits,
            "head_weights": head_weights,
            "trust": trust,
            "update_gate": update,
            "offsets": offsets,
            "gamma": self.gamma.detach().reshape(1),
            "offset_abs_mean": offset_abs.mean(dim=(1, 2, 3, 4)),
            "offset_abs_p95": torch.quantile(offset_abs.flatten(1).float(), 0.95, dim=1).to(offset_abs.dtype),
            "trust_mean": trust.detach().mean(dim=(1, 2, 3)),
            "update_gate_mean": update.detach().mean(dim=(1, 2, 3)),
            "head_entropy": entropy.detach(),
            "head_usage": top_usage.detach().mean(dim=(2, 3)),
            "head_top1_usage": top_usage.detach().mean(dim=(1, 2, 3)),
            "head_max_weight": head_weights.detach().amax(dim=1).mean(dim=(1, 2)),
        }
        return out, next_anchor, aux


class BoundaryAwareFusion(nn.Module):
    def __init__(self, feature_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.boundary = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, feature_channels, kernel_size=1),
        )
        self.delta_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.gate = nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor, high_res_skip: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary = self.boundary(high_res_skip)
        if boundary.shape[-2:] != feature.shape[-2:]:
            boundary = F.interpolate(boundary, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.delta_proj(boundary - feature)
        gate = torch.sigmoid(self.gate(torch.cat([feature, boundary, delta], dim=1)))
        out = feature + self.gamma * gate * delta
        aux = {
            "boundary_gamma": self.gamma.detach().reshape(1),
            "boundary_gate_mean": gate.detach().mean(dim=(1, 2, 3)),
            "boundary_delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
        }
        return out, aux
