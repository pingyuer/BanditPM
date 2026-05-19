from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceFusion(nn.Module):
    """Prediction-mode switch for anchor-primary, base-primary, and residual-only."""

    def __init__(self, prediction_mode: str, residual_clip: float) -> None:
        super().__init__()
        self.prediction_mode = prediction_mode.lower()
        self.residual_clip = float(residual_clip)
        if self.prediction_mode not in {"anchor_primary", "base_primary", "residual_only"}:
            raise ValueError(f"Unsupported functional_anchor prediction_mode: {prediction_mode}")

    def forward(
        self,
        *,
        anchor_logits: torch.Tensor,
        base_logits: torch.Tensor,
        shape_residual: torch.Tensor,
        boundary_residual: torch.Tensor,
        anchor_trust: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = (shape_residual + boundary_residual).clamp(-self.residual_clip, self.residual_clip)
        trust = anchor_trust
        if trust.shape[-2:] != residual.shape[-2:]:
            trust = torch.nn.functional.interpolate(trust, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        trust = trust.expand(-1, residual.shape[1], -1, -1)
        if self.prediction_mode == "anchor_primary":
            final = anchor_logits + residual
        elif self.prediction_mode == "base_primary":
            proposal = anchor_logits + residual
            final = base_logits + trust * (proposal - base_logits)
        else:
            final = anchor_logits + 0.5 * residual
        return final, {
            "residual_logits": residual,
            "anchor_trust_ratio": trust.detach().mean(),
            "image_trust_ratio": (1.0 - trust.detach()).mean(),
        }
