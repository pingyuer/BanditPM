from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..memory.helpers import _softplus_inverse


class DynamicAnchorFusion(nn.Module):
    def __init__(self, channels: int, gamma_init: float = 0.05) -> None:
        super().__init__()
        self.delta_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=3.0e-3)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        dynamic_anchor_t: torch.Tensor,
        runtime_state_t: torch.Tensor,
        trust_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = self.delta_proj(dynamic_anchor_t - anchor_feat_t)
        gate = torch.sigmoid(self.gate(torch.cat([anchor_feat_t, dynamic_anchor_t, runtime_state_t], dim=1)))
        if trust_t is None:
            trust = torch.ones(anchor_feat_t.shape[0], 1, 1, 1, device=anchor_feat_t.device, dtype=anchor_feat_t.dtype)
        else:
            trust = trust_t.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype).view(anchor_feat_t.shape[0], 1, 1, 1)
        gamma = F.softplus(self.raw_gamma)
        final_feature_t = anchor_feat_t + gamma * trust * gate * delta
        gate_flat = gate.detach().flatten(1).float()
        return final_feature_t, {
            "gamma": gamma.detach().reshape(1),
            "dynamic_trust_mean": trust.detach().flatten(1).mean(dim=1),
            "fusion_gate_mean": gate.detach().mean(dim=(1, 2, 3)),
            "fusion_gate_p05": torch.quantile(gate_flat, 0.05, dim=1).to(gate.dtype),
            "fusion_gate_p95": torch.quantile(gate_flat, 0.95, dim=1).to(gate.dtype),
            "delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
            "dynamic_anchor_minus_anchor_abs_mean": (dynamic_anchor_t - anchor_feat_t).detach().abs().mean(dim=(1, 2, 3)),
            "fused_minus_anchor_abs_mean": (final_feature_t - anchor_feat_t).detach().abs().mean(dim=(1, 2, 3)),
        }
