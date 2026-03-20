"""Original GDR temporal memory core extracted from the main model."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDRCore(nn.Module):
    """Original gated delta-rule memory update.

    Input shapes:
        value_BNCHW: [B, N, C, H, W]
        key_BCHW: [B, K, H, W]

    Internal state:
        state_BNCC: [B, N, K, C]
    """

    def __init__(self, value_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.b_proj = nn.Linear(value_dim, num_heads, bias=False)
        self.a_proj = nn.Linear(value_dim, num_heads, bias=False)

        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.state_BNCC: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: Optional[int] = None,
        num_objects: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        del batch_size, num_objects, device
        self.state_BNCC = None

    def freeze_parameters(self) -> None:
        for param in [
            self.A_log,
            self.dt_bias,
            self.b_proj.weight,
            self.a_proj.weight,
        ]:
            param.requires_grad = False

    def _normalize_key(self, key_BCHW: torch.Tensor) -> torch.Tensor:
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        return (key_BCHW - key_max_B1HW).softmax(dim=1)

    def forward(
        self,
        value_BNCHW: torch.Tensor,
        key_BCHW: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        key_BCHW = self._normalize_key(key_BCHW)

        if self.state_BNCC is None:
            self.state_BNCC = torch.einsum("bkhw,bnvhw->bnkv", key_BCHW, value_BNCHW)
        else:
            state_t_1_BNCC = self.state_BNCC.clone()
            v_old = torch.einsum("bkhw,bnkv->bnvhw", key_BCHW, state_t_1_BNCC).contiguous()
            v_k = torch.einsum("bkhw,bnvhw->bnkv", key_BCHW, v_old)

            beta_t = self.b_proj(state_t_1_BNCC).sigmoid()
            beta_t = beta_t.repeat_interleave(self.value_dim // self.num_heads, dim=3)
            eraser = torch.einsum("bnkv,bnkv->bnkv", beta_t, v_k).contiguous()

            vk_t = torch.einsum("bkhw,bnvhw->bnkv", key_BCHW, value_BNCHW).contiguous()
            new = torch.einsum("bnkv,bnkv->bnkv", beta_t, vk_t).contiguous()

            old = state_t_1_BNCC - eraser
            alpha = -self.A_log.float().exp() * F.softplus(
                self.a_proj(state_t_1_BNCC).float() + self.dt_bias
            )
            alpha = alpha.repeat_interleave(self.value_dim // self.num_heads, dim=3)
            old = torch.einsum("bnkv,bnkv->bnkv", alpha, old).contiguous()

            self.state_BNCC = old + new

        readout_BNCHW = torch.einsum(
            "bkhw,bnkv->bnvhw", key_BCHW, self.state_BNCC
        ).contiguous()
        return readout_BNCHW, {}
