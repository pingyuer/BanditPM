from __future__ import annotations

import torch
import torch.nn as nn


class TemporalStateODE(nn.Module):
    """Euler-style update for video-level cardiac state z_t."""

    def __init__(self, evidence_dim: int, phase_dim: int, state_dim: int, hidden_dim: int, dt: float = 0.5) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.dt = float(dt)
        self.init = nn.Sequential(
            nn.Linear(evidence_dim + phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.vector_field = nn.Sequential(
            nn.Linear(state_dim + evidence_dim + phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.vector_field[-1].bias)

    def forward(
        self,
        prev_z: torch.Tensor | None,
        evidence: torch.Tensor,
        phase_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_z is None:
            prev_z = self.init(torch.cat([evidence, phase_embed], dim=-1))
        dz = self.vector_field(torch.cat([prev_z, evidence, phase_embed], dim=-1))
        update = self.dt * torch.tanh(dz)
        z = prev_z + update
        return z, dz, update
