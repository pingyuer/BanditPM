from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebel.ode_field import _groups


class DisagreementCorrectionHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 2, init_scale: float = 0.25, max_scale: float = 1.0) -> None:
        super().__init__()
        self.max_scale = float(max_scale)
        init_ratio = min(max(init_scale / max(self.max_scale, 1.0e-6), 1.0e-4), 1.0 - 1.0e-4)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + num_classes + 3, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.GroupNorm(_groups(feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, num_classes, 1),
        )

    def correction_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.scale_logit) * self.max_scale

    def forward(
        self,
        feature: torch.Tensor,
        aux_logits: torch.Tensor,
        belief_logits: torch.Tensor,
        disagreement: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = feature.shape[-2:]
        aux_logits = F.interpolate(aux_logits, size=size, mode="bilinear", align_corners=False)
        belief_fg = torch.softmax(
            F.interpolate(belief_logits, size=size, mode="bilinear", align_corners=False),
            dim=1,
        )[:, 1:2]
        disagreement = F.interpolate(disagreement, size=size, mode="bilinear", align_corners=False)
        reliability = F.interpolate(reliability, size=size, mode="bilinear", align_corners=False)
        gated_feature = feature * (1.0 + disagreement)
        delta = self.net(torch.cat([gated_feature, aux_logits, belief_fg, disagreement, reliability], dim=1))
        return delta * disagreement.clamp_min(0.05), self.correction_scale()
