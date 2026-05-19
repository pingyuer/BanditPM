from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHeads(nn.Module):
    """Separate low-frequency shape, boundary, and confidence residuals."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int,
        residual_clip: float,
        *,
        use_anchor_features: bool = True,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        self.use_anchor_features = bool(use_anchor_features)
        self.anchor_adapters = nn.ModuleDict(
            {level: nn.Conv2d(dim, dim, kernel_size=1) for level, dim in feature_dims.items()}
        )
        self.shape_head = nn.Sequential(
            nn.Conv2d(feature_dims["high"] * 2 + feature_dims["mid"] * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(feature_dims["low"] * 2 + feature_dims["dec"] * 2 + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_dims["dec"] + 5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, 1),
        )
        for module in (self.shape_head[-1], self.boundary_head[-1]):
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.weight)
        nn.init.constant_(self.confidence_head[-1].bias, float(gate_init_bias))
        nn.init.normal_(self.confidence_head[-1].weight, mean=0.0, std=1.0e-3)
        for module in self.anchor_adapters.values():
            nn.init.zeros_(module.bias)

    def _anchor_level(self, anchor_features: dict[str, torch.Tensor], level: str, size: tuple[int, int]) -> torch.Tensor:
        anchor = anchor_features[level].mean(dim=1)
        if not self.use_anchor_features:
            anchor = torch.zeros_like(anchor)
        anchor = self.anchor_adapters[level](anchor)
        if anchor.shape[-2:] != size:
            anchor = F.interpolate(anchor, size=size, mode="bilinear", align_corners=False)
        return anchor

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        anchor_logits: torch.Tensor,
        base_logits: torch.Tensor,
        anchor_features: dict[str, torch.Tensor],
        phase_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N, H, W = anchor_logits.shape
        high = F.interpolate(feats["high"], size=feats["mid"].shape[-2:], mode="bilinear", align_corners=False)
        anchor_high = self._anchor_level(anchor_features, "high", feats["mid"].shape[-2:])
        anchor_mid = self._anchor_level(anchor_features, "mid", feats["mid"].shape[-2:])
        shape_seed = torch.cat([high, feats["mid"], anchor_high, anchor_mid], dim=1)
        shape = self.shape_head(shape_seed)
        shape = F.interpolate(shape, size=(H, W), mode="bilinear", align_corners=False).expand(-1, N, -1, -1)

        dec_low = F.interpolate(feats["dec"], size=feats["low"].shape[-2:], mode="bilinear", align_corners=False)
        anchor_low = self._anchor_level(anchor_features, "low", feats["low"].shape[-2:])
        anchor_dec = self._anchor_level(anchor_features, "dec", feats["low"].shape[-2:])
        anchor_prob_low = F.interpolate(
            torch.sigmoid(anchor_logits).mean(dim=1, keepdim=True),
            size=feats["low"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_seed = torch.cat([feats["low"], dec_low, anchor_low, anchor_dec, anchor_prob_low], dim=1)
        boundary = self.boundary_head(boundary_seed)
        boundary = F.interpolate(boundary, size=(H, W), mode="bilinear", align_corners=False).expand(-1, N, -1, -1)

        base_prob = torch.sigmoid(base_logits).mean(dim=1, keepdim=True)
        anchor_prob = torch.sigmoid(anchor_logits).mean(dim=1, keepdim=True)
        diff_prob = (base_prob - anchor_prob).abs()
        if phase_confidence is None:
            phase_map = torch.ones_like(base_prob)
        else:
            phase_map = phase_confidence
            if phase_map.dim() == 2:
                phase_map = phase_map.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
            phase_map = phase_map.to(device=base_prob.device, dtype=base_prob.dtype).expand(B, 1, H, W)
        conf_seed = torch.cat(
            [
                feats["dec"],
                base_prob,
                anchor_prob,
                diff_prob,
                phase_map,
                (base_prob + anchor_prob) * 0.5,
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
