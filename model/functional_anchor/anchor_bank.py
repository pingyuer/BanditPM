from __future__ import annotations

import math

import torch
import torch.nn as nn


SLOT_NAMES = ("ed_large", "early_systole", "es_small", "early_diastole", "uncertain")


class FunctionalAnchorBank(nn.Module):
    """Semantic phase slots for ED/ES/cycle anchors."""

    def __init__(self, num_slots: int, state_dim: int, phase_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if int(num_slots) < 5:
            raise ValueError("functional_anchor.num_slots must be >= 5 for semantic phase slots")
        self.num_slots = int(num_slots)
        self.selector = nn.Sequential(
            nn.Linear(state_dim + phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_slots),
        )
        centers = torch.linspace(0.0, 1.0, self.num_slots + 1)[:-1]
        centers[:5] = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.5])
        self.register_buffer("phase_centers", centers, persistent=False)
        self.area_bias = nn.Parameter(torch.linspace(0.85, 0.25, self.num_slots))

    def forward(self, z: torch.Tensor, phase_embed: torch.Tensor, norm_time: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.selector(torch.cat([z, phase_embed], dim=-1))
        dist = torch.minimum((norm_time.unsqueeze(-1) - self.phase_centers).abs(), 1.0 - (norm_time.unsqueeze(-1) - self.phase_centers).abs())
        phase_logits = -4.0 * dist
        weights = torch.softmax(logits + phase_logits, dim=-1)
        entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(max(self.num_slots, 2))
        slot_area = torch.sigmoid(self.area_bias)
        expected_area = (weights * slot_area).sum(dim=-1)
        order_violation = torch.relu(slot_area[1:] - slot_area[:-1]).mean()
        return weights, {
            "slot_entropy": entropy,
            "slot_area": slot_area,
            "expected_anchor_area": expected_area,
            "slot_area_order_violation": order_violation,
            "ed_slot_usage": weights[..., 0],
            "es_slot_usage": weights[..., 2],
        }
