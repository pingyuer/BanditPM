"""Placeholder memory core reserved for future BanditPM integration."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class BanditPMCore(nn.Module):
    """Placeholder core that currently acts as an identity readout."""

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        del batch_size, num_objects, device

    def forward(
        self,
        value_BNCHW: torch.Tensor,
        key_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        mask_BNHW: torch.Tensor | None = None,
        policy_meta: Dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        del key_BCHW, pixfeat_BCHW, mask_BNHW, policy_meta
        return value_BNCHW, {"memory_type": "banditpm_placeholder"}
