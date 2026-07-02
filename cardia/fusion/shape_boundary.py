from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..memory.helpers import _get_activation, _group_count, _softplus_inverse


class ShapeBoundaryFusion(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        skip_channels: int,
        context_channels: int,
        gamma_init: float = 0.03,
        edge_gate_floor: float = 0.05,
        edge_gate_bias: float = -1.0,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.edge_gate_floor = float(edge_gate_floor)
        act_cls = _get_activation(activation).__class__
        self.boundary = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GroupNorm(_group_count(skip_channels), skip_channels),
            act_cls(),
            nn.Conv2d(skip_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(_group_count(feature_channels), feature_channels),
            act_cls(),
        )
        self.context_proj = nn.Conv2d(context_channels, feature_channels, kernel_size=1)
        self.delta_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.edge_gate_head = nn.Conv2d(feature_channels * 4, 1, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.constant_(self.edge_gate_head.bias, float(edge_gate_bias))

    def forward(
        self,
        decoder_feature_t: torch.Tensor,
        high_res_anchor_t: torch.Tensor,
        runtime_context_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary = self.boundary(high_res_anchor_t)
        context = self.context_proj(runtime_context_t)
        if boundary.shape[-2:] != decoder_feature_t.shape[-2:]:
            boundary = F.interpolate(boundary, size=decoder_feature_t.shape[-2:], mode="bilinear", align_corners=False)
        if context.shape[-2:] != decoder_feature_t.shape[-2:]:
            context = F.interpolate(context, size=decoder_feature_t.shape[-2:], mode="bilinear", align_corners=False)
        raw_delta = self.delta_proj(boundary + context - decoder_feature_t)
        delta = 0.5 * torch.tanh(raw_delta)
        gate_input = torch.cat([decoder_feature_t, boundary, context, delta], dim=1)
        edge_logit = self.edge_gate_head(gate_input)
        edge_gate = torch.sigmoid(edge_logit)
        edge_effective = self.edge_gate_floor + (1.0 - self.edge_gate_floor) * edge_gate
        channel_gate = self.channel_gate(gate_input)
        gamma = F.softplus(self.raw_gamma)
        out = decoder_feature_t + gamma * edge_effective * channel_gate * delta
        edge_flat = edge_gate.detach().flatten(1).float()
        return out, {
            "boundary_logits": edge_logit,
            "boundary_edge_gate": edge_gate,
            "boundary_edge_effective": edge_effective,
            "boundary_delta_map": delta.detach(),
            "boundary_gamma": gamma.detach().reshape(1),
            "boundary_edge_gate_mean": edge_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_effective_mean": edge_effective.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_gate_p05": torch.quantile(edge_flat, 0.05, dim=1).to(edge_gate.dtype),
            "boundary_edge_gate_p95": torch.quantile(edge_flat, 0.95, dim=1).to(edge_gate.dtype),
            "boundary_channel_gate_mean": channel_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
        }
