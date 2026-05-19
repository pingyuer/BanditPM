from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHeads(nn.Module):
    """Separate low-frequency shape, boundary, and confidence residuals."""

    def __init__(self, feature_dims: dict[str, int], hidden_dim: int, residual_clip: float) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        self.shape_head = nn.Sequential(
            nn.Conv2d(feature_dims["high"] + feature_dims["mid"], hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(feature_dims["low"] + feature_dims["dec"], hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_dims["dec"] + 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, 1),
        )
        for module in (self.shape_head[-1], self.boundary_head[-1], self.confidence_head[-1]):
            nn.init.zeros_(module.bias)
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        anchor_logits: torch.Tensor,
        base_logits: torch.Tensor,
        anchor_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B, N, H, W = anchor_logits.shape
        high = F.interpolate(feats["high"], size=feats["mid"].shape[-2:], mode="bilinear", align_corners=False)
        shape_seed = torch.cat([high, feats["mid"]], dim=1)
        shape = self.shape_head(shape_seed)
        shape = F.interpolate(shape, size=(H, W), mode="bilinear", align_corners=False).expand(-1, N, -1, -1)

        boundary_seed = torch.cat([feats["low"], F.interpolate(feats["dec"], size=feats["low"].shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        boundary = self.boundary_head(boundary_seed)
        boundary = F.interpolate(boundary, size=(H, W), mode="bilinear", align_corners=False).expand(-1, N, -1, -1)

        conf_seed = torch.cat(
            [
                feats["dec"],
                torch.sigmoid(anchor_logits).mean(dim=1, keepdim=True),
                torch.sigmoid(base_logits).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        conf = torch.sigmoid(self.confidence_head(conf_seed))
        shape = shape.clamp(-self.residual_clip, self.residual_clip)
        boundary = boundary.clamp(-self.residual_clip, self.residual_clip)
        return {
            "shape_residual_logits": shape,
            "boundary_residual_logits": boundary,
            "confidence": conf,
            "gate_low": conf[:, 0:1],
            "gate_mid": conf[:, 1:2],
            "gate_high": conf[:, 2:3],
            "anchor_trust": conf[:, 3:4],
        }
