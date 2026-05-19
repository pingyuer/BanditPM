from __future__ import annotations

import torch
import torch.nn as nn


class ModelEMA:
    """Small state-dict EMA helper that works with plain modules and DDP modules."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            value = value.detach()
            if key not in self.state:
                self.state[key] = value.clone()
                continue
            if torch.is_floating_point(value):
                self.state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.state[key].copy_(value)

    def state_dict(self) -> dict:
        return {key: value.detach().clone() for key, value in self.state.items()}

    def load_state_dict(self, state: dict) -> None:
        self.state = {key: value.detach().clone() for key, value in state.items()}

