from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _group_count(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return math.log(math.exp(value) - 1.0)


class DepthwiseRelationEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int | None = None, norm: str = "group") -> None:
        super().__init__()
        hidden = int(hidden_dim or channels)
        norm = str(norm or "none").lower()

        def make_norm(width: int) -> nn.Module:
            if norm in {"group", "gn", "groupnorm"}:
                return nn.GroupNorm(_group_count(width), width)
            if norm in {"batch", "bn", "batchnorm"}:
                return nn.BatchNorm2d(width)
            return nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1),
            make_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            make_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            make_norm(hidden),
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
        max_offset_px: float = 2.0,
        padding_mode: str = "border",
        align_corners: bool = False,
        write_gate_bias: float = -0.5,
        gamma_init: float = 0.05,
        relation_norm: str = "group",
        selector_logit_scale_init: float = 2.0,
        selector_logit_scale_max: float = 8.0,
        use_global_selector: bool = True,
        enable_decay_gate: bool = False,
        decay_gate_bias: float = 1.5,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.max_offset_px = float(max_offset_px)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)
        self.selector_logit_scale_max = float(selector_logit_scale_max)
        self.use_global_selector = bool(use_global_selector and self.num_heads > 1)
        self.enable_decay_gate = bool(enable_decay_gate)

        self.relation = DepthwiseRelationEncoder(self.channels, hidden_dim, norm=relation_norm)
        hidden = self.relation.out_dim
        self.offset_head = nn.Conv2d(hidden, self.num_heads * 2, kernel_size=1)
        self.spatial_selector = nn.Conv2d(hidden, self.num_heads, kernel_size=1)
        self.global_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_heads, kernel_size=1),
        ) if self.use_global_selector else None
        self.raw_selector_logit_scale = nn.Parameter(torch.tensor(math.log(max(float(selector_logit_scale_init), 1.0e-3))))
        self.write_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.decay_head = nn.Conv2d(hidden, 1, kernel_size=1) if self.enable_decay_gate else None
        self.delta_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.fusion_gate = nn.Conv2d(self.channels * 3, self.channels, kernel_size=1)
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))

        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.constant_(self.write_head.bias, float(write_gate_bias))
        if self.decay_head is not None:
            nn.init.constant_(self.decay_head.bias, float(decay_gate_bias))

    def _selector_scale(self) -> torch.Tensor:
        return self.raw_selector_logit_scale.exp().clamp(0.05, self.selector_logit_scale_max)

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
        offset_px = torch.tanh(raw_offsets) * self.max_offset_px
        offset_x = offset_px[:, :, 0] * (2.0 / float(W))
        offset_y = offset_px[:, :, 1] * (2.0 / float(H))
        offsets = torch.stack([offset_x, offset_y], dim=2)

        base_grid = self._identity_grid(B, H, W, current.device, current.dtype)
        grid = base_grid[:, None] + offsets.permute(0, 1, 3, 4, 2)
        sampled = F.grid_sample(
            anchor_prev[:, None].expand(-1, self.num_heads, -1, -1, -1).reshape(B * self.num_heads, C, H, W),
            grid.reshape(B * self.num_heads, H, W, 2),
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).view(B, self.num_heads, C, H, W)

        spatial_logits = self.spatial_selector(relation)
        global_logits = self.global_selector(relation) if self.global_selector is not None else torch.zeros_like(spatial_logits[:, :, :1, :1])
        selector_scale = self._selector_scale()
        head_logits = (spatial_logits + global_logits) * selector_scale
        head_weights = torch.softmax(head_logits, dim=1)
        frame_selector_logits = head_logits.mean(dim=(2, 3))
        routed = (sampled * head_weights[:, :, None]).sum(dim=1)
        write = torch.sigmoid(self.write_head(relation))

        delta = self.delta_proj(routed - current)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([current, routed, delta], dim=1)))
        gamma = F.softplus(self.raw_gamma)
        out = current + gamma * write * gate * delta

        write_detached = write.detach()
        if self.decay_head is not None:
            decay = torch.sigmoid(self.decay_head(relation))
            next_anchor = decay.detach() * anchor_prev.detach() + (1.0 - decay.detach()) * (
                write_detached * out.detach() + (1.0 - write_detached) * routed.detach()
            )
        else:
            decay = None
            next_anchor = write_detached * out.detach() + (1.0 - write_detached) * routed.detach()

        weights_flat = head_weights.flatten(2)
        entropy = -(weights_flat * weights_flat.clamp_min(1.0e-6).log()).sum(dim=1).mean(dim=-1)
        entropy = entropy / max(math.log(float(self.num_heads)), 1.0)
        top_usage = F.one_hot(head_weights.argmax(dim=1), num_classes=self.num_heads).permute(0, 3, 1, 2).float()
        head_usage = top_usage.detach().mean(dim=(2, 3))
        usage_entropy = -(head_usage * head_usage.clamp_min(1.0e-6).log()).sum(dim=1)
        usage_entropy = usage_entropy / max(math.log(float(self.num_heads)), 1.0)
        global_weights = torch.softmax((global_logits.flatten(1) * selector_scale), dim=1)
        global_entropy = -(global_weights * global_weights.clamp_min(1.0e-6).log()).sum(dim=1)
        global_entropy = global_entropy / max(math.log(float(self.num_heads)), 1.0)
        offset_px_abs = offset_px.detach().abs()
        flow_dx = offset_px[:, :, :, :, 1:] - offset_px[:, :, :, :, :-1]
        flow_dy = offset_px[:, :, :, 1:, :] - offset_px[:, :, :, :-1, :]
        flow_smooth = flow_dx.abs().mean(dim=(1, 2, 3, 4)) + flow_dy.abs().mean(dim=(1, 2, 3, 4))
        write_flat = write.detach().flatten(1).float()
        aux = {
            "relation": relation,
            "warped_features": sampled,
            "head_logits": head_logits,
            "spatial_selector_logits": spatial_logits,
            "global_selector_logits": global_logits.flatten(1),
            "head_weights": head_weights,
            "selector_logits": frame_selector_logits,
            "selector_logit_scale": selector_scale.detach().reshape(1),
            "write": write,
            "offsets": offsets,
            "offset_px": offset_px,
            "gamma": gamma.detach().reshape(1),
            "offset_px_mean": offset_px_abs.mean(dim=(1, 2, 3, 4)),
            "offset_px_p95": torch.quantile(offset_px_abs.flatten(1).float(), 0.95, dim=1).to(offset_px_abs.dtype),
            "flow_smooth": flow_smooth,
            "write_mean": write.detach().mean(dim=(1, 2, 3)),
            "write_p05": torch.quantile(write_flat, 0.05, dim=1).to(write.dtype),
            "write_p95": torch.quantile(write_flat, 0.95, dim=1).to(write.dtype),
            "decay_mean": decay.detach().mean(dim=(1, 2, 3)) if decay is not None else torch.zeros(B, device=current.device, dtype=current.dtype),
            "head_entropy": entropy.detach(),
            "global_selector_entropy": global_entropy.detach(),
            "head_usage": head_usage,
            "head_usage_entropy": usage_entropy.detach(),
            "head_usage_max": head_usage.max(dim=1).values,
            "head_usage_min": head_usage.min(dim=1).values,
            "head_max_weight": head_weights.detach().amax(dim=1).mean(dim=(1, 2)),
        }
        return out, next_anchor, aux


class BoundaryAwareFusion(nn.Module):
    def __init__(self, feature_channels: int, skip_channels: int, gamma_init: float = 0.03) -> None:
        super().__init__()
        self.boundary = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GroupNorm(_group_count(skip_channels), skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(_group_count(feature_channels), feature_channels),
            nn.GELU(),
        )
        self.delta_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.edge_gate = nn.Conv2d(feature_channels * 3, 1, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.boundary_aux_head = nn.Conv2d(feature_channels * 3, 1, kernel_size=1)
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.constant_(self.edge_gate.bias, -2.0)

    def forward(self, feature: torch.Tensor, high_res_skip: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary = self.boundary(high_res_skip)
        if boundary.shape[-2:] != feature.shape[-2:]:
            boundary = F.interpolate(boundary, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        raw_delta = self.delta_proj(boundary - feature)
        delta = torch.tanh(raw_delta)
        gate_input = torch.cat([feature, boundary, delta], dim=1)
        edge_gate = torch.sigmoid(self.edge_gate(gate_input))
        channel_gate = self.channel_gate(gate_input)
        gamma = F.softplus(self.raw_gamma)
        out = feature + gamma * edge_gate * channel_gate * delta
        edge_flat = edge_gate.detach().flatten(1).float()
        aux = {
            "boundary_logits": self.boundary_aux_head(gate_input),
            "boundary_edge_gate": edge_gate,
            "boundary_gamma": gamma.detach().reshape(1),
            "boundary_gate_mean": edge_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_gate_mean": edge_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_gate_p05": torch.quantile(edge_flat, 0.05, dim=1).to(edge_gate.dtype),
            "boundary_edge_gate_p95": torch.quantile(edge_flat, 0.95, dim=1).to(edge_gate.dtype),
            "boundary_channel_gate_mean": channel_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
            "boundary_raw_delta_abs_mean": raw_delta.detach().abs().mean(dim=(1, 2, 3)),
        }
        return out, aux
