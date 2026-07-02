from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_section(cfg, key: str):
    value = _cfg_get(cfg, key, {})
    return {} if value in (None, "null") else value


def _group_count(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return math.log(math.exp(value) - 1.0)


def _get_activation(name: str) -> nn.Module:
    name = str(name).upper()
    if name == "SILU" or name == "SWISH":
        return nn.SiLU()
    if name == "MISH":
        return nn.Mish()
    if name == "RELU":
        return nn.ReLU()
    if name == "LEAKYRELU":
        return nn.LeakyReLU()
    return nn.GELU()


class RelationEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int | None = None, activation: str = "GELU") -> None:
        super().__init__()
        hidden = int(hidden_dim or channels)
        act = _get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            act.__class__(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(_group_count(hidden), hidden),
            act.__class__(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            act.__class__(),
        )
        self.out_dim = hidden

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([lhs, rhs, lhs - rhs, lhs * rhs], dim=1))
