from __future__ import annotations

import torch
import torch.nn as nn


class MultiLevelInjector(nn.Module):
    """Role-separated high/mid/low/decoder anchor injection."""

    def __init__(self, feature_dims: dict[str, int]) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({level: nn.Conv2d(dim, dim, kernel_size=1) for level, dim in feature_dims.items()})

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        anchor_features: dict[str, torch.Tensor],
        gates: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        injected = dict(feats)
        gate_map = {
            "high": gates.get("gate_high"),
            "mid": gates.get("gate_mid"),
            "low": gates.get("gate_low"),
            "dec": gates.get("anchor_trust"),
        }
        for level, gate in gate_map.items():
            if level not in feats or level not in anchor_features or gate is None:
                continue
            anchor = anchor_features[level].mean(dim=1)
            if gate.shape[-2:] != feats[level].shape[-2:]:
                gate = torch.nn.functional.interpolate(gate, size=feats[level].shape[-2:], mode="bilinear", align_corners=False)
            injected[level] = feats[level] + gate * self.proj[level](anchor)
        return injected
