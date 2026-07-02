from __future__ import annotations

import torch
import torch.nn as nn


class ShiftedMLPBlock(nn.Module):
    """Small spatial-token MLP block used by the lightweight UNeXt prototype."""

    def __init__(self, dim: int, expansion: int = 4, shift: int = 1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.shift = int(shift)
        self.norm = nn.BatchNorm2d(dim)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )
        self.spatial_mlp_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim)
        self.spatial_mlp_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        if self.shift > 0:
            y_h = torch.roll(y, shifts=self.shift, dims=-1)
            y_v = torch.roll(y, shifts=self.shift, dims=-2)
        else:
            y_h = y_v = y
        y = self.spatial_mlp_h(y_h) + self.spatial_mlp_v(y_v)
        return x + self.channel_mlp(y)
