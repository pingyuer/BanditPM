from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhaseEncoder(nn.Module):
    """Cardiac phase descriptor used by state update, slot choice, and fusion."""

    def __init__(self, num_slots: int, phase_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.input_dim = 8 + self.num_slots
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, phase_dim),
            nn.LayerNorm(phase_dim),
        )

    def forward(
        self,
        norm_time: torch.Tensor,
        *,
        prev_area: torch.Tensor,
        area_velocity: torch.Tensor,
        slot_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_time = norm_time.clamp(0.0, 1.0)
        sin_phase = torch.sin(2.0 * math.pi * norm_time)
        cos_phase = torch.cos(2.0 * math.pi * norm_time)
        ed_flag = (norm_time <= 1.0e-6).to(norm_time.dtype)
        es_flag = (norm_time - 0.5).abs().le(0.125).to(norm_time.dtype)
        diastole_flag = (norm_time > 0.5).to(norm_time.dtype)
        trend = torch.sign(area_velocity)
        descriptor = torch.cat(
            [
                norm_time.unsqueeze(-1),
                sin_phase.unsqueeze(-1),
                cos_phase.unsqueeze(-1),
                prev_area.unsqueeze(-1),
                area_velocity.unsqueeze(-1),
                trend.unsqueeze(-1),
                ed_flag.unsqueeze(-1),
                es_flag.unsqueeze(-1),
                slot_history,
            ],
            dim=-1,
        )
        return self.net(descriptor), descriptor
