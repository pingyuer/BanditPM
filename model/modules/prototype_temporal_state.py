from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PrototypeTemporalState(nn.Module):
    """
    Temporal state update for prototype assignments.

    Shapes:
        p_t: [B, K]
        prev_state: [B, K] or None
        T_t: [B, K]
    """

    def __init__(
        self,
        num_proto: int,
        momentum: float = 0.9,
        learnable_momentum: bool = False,
        detach_prev_state: bool = False,
    ) -> None:
        super().__init__()
        self.num_proto = num_proto
        self.detach_prev_state = detach_prev_state

        momentum_tensor = torch.tensor(float(momentum), dtype=torch.float32)
        if learnable_momentum:
            self.momentum = nn.Parameter(momentum_tensor)
        else:
            self.register_buffer("momentum", momentum_tensor, persistent=False)

    def reset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_proto, device=device)

    def forward(self, p_t: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prev_state is None:
            prev_state = torch.zeros_like(p_t)
        elif self.detach_prev_state:
            prev_state = prev_state.detach()

        momentum = self.momentum
        if momentum.ndim == 0:
            momentum = momentum.clamp(0.0, 0.9999)
        T_t = momentum * prev_state + (1.0 - momentum) * p_t
        return T_t
