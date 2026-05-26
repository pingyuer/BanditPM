from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


class TrustGate(nn.Module):
    """Trust-gated safety scale for residual injection.

    trust: how much to allow FAF correction (higher = more FAF)
    gate: spatial gate for residual magnitude
    """

    def __init__(self, dec_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # Input: dec + base + proposal + proposal_base + uncertainty + boundary = dec_dim + 5
        self.gate_net = nn.Sequential(
            nn.Conv2d(dec_dim + 5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=1.0e-3)
        with torch.no_grad():
            self.gate_net[-1].bias[0] = _logit(0.15)
            self.gate_net[-1].bias[1] = _logit(0.50)

    def forward(
        self,
        residual_in: torch.Tensor,
        trust_max: torch.Tensor,
        trust_floor: torch.Tensor,
        *,
        disable: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            residual_in: [B*N, C+5, H, W] — same concat as ResidualRefiner input
            trust_max: scalar tensor (scheduled)
            trust_floor: scalar tensor (curriculum)
            disable: if True, return ones (ablation)

        Returns:
            trust: [B, N, H, W]
            gate: [B, N, H, W]
        """
        BN = residual_in.shape[0]
        if disable:
            H, W = residual_in.shape[-2:]
            ones = torch.ones(BN, 1, H, W, device=residual_in.device, dtype=residual_in.dtype)
            return ones, ones

        out = self.gate_net(residual_in)  # [BN, 2, H, W]
        trust_raw = torch.sigmoid(out[:, 0:1])  # [BN, 1, H, W]
        trust = trust_floor + (trust_max - trust_floor).clamp_min(0.0) * trust_raw
        gate = torch.sigmoid(out[:, 1:2])  # [BN, 1, H, W]
        return trust, gate
