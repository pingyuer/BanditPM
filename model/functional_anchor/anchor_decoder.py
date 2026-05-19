from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorDecoder(nn.Module):
    """Decode z_t + phase_t + semantic slots into anchor logits and features."""

    def __init__(
        self,
        num_slots: int,
        state_dim: int,
        phase_dim: int,
        feature_dims: dict[str, int],
        anchor_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.anchor_size = int(anchor_size)
        self.slot_logits = nn.Parameter(torch.randn(num_slots, 1, anchor_size, anchor_size) * 0.02)
        self.slot_features = nn.ParameterDict(
            {
                level: nn.Parameter(torch.randn(num_slots, dim, anchor_size, anchor_size) * 0.02)
                for level, dim in feature_dims.items()
            }
        )
        self.condition = nn.Sequential(
            nn.Linear(state_dim + phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit_mod = nn.Linear(hidden_dim, anchor_size * anchor_size)
        self.feature_mods = nn.ModuleDict({level: nn.Linear(hidden_dim, dim) for level, dim in feature_dims.items()})

    def forward(
        self,
        z: torch.Tensor,
        phase_embed: torch.Tensor,
        slot_weights: torch.Tensor,
        target_sizes: dict[str, tuple[int, int]],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N = z.shape[:2]
        cond = self.condition(torch.cat([z, phase_embed], dim=-1))
        logits = torch.einsum("bnk,kchw->bnchw", slot_weights, self.slot_logits).squeeze(2)
        logits = logits + 0.05 * self.logit_mod(cond).view(B, N, self.anchor_size, self.anchor_size)
        anchor_logits = F.interpolate(logits.flatten(0, 1).unsqueeze(1), size=output_size, mode="bilinear", align_corners=False)
        anchor_logits = anchor_logits.view(B, N, *output_size)

        features: dict[str, torch.Tensor] = {}
        for level, size in target_sizes.items():
            feat = torch.einsum("bnk,kchw->bnchw", slot_weights, self.slot_features[level])
            mod = self.feature_mods[level](cond).view(B, N, -1, 1, 1)
            feat = feat + 0.05 * mod
            feat = F.interpolate(feat.flatten(0, 1), size=size, mode="bilinear", align_corners=False)
            features[level] = feat.view(B, N, -1, *size)
        return anchor_logits, features
