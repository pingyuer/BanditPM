from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceFusion(nn.Module):
    """Prediction-mode switch for UNeXt-primary and anchor ablations."""

    def __init__(self, prediction_mode: str, residual_clip: float) -> None:
        super().__init__()
        self.prediction_mode = prediction_mode.lower()
        self.residual_clip = float(residual_clip)
        if self.prediction_mode not in {"anchor_primary", "base_primary", "learned_blend", "residual_only"}:
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
        proposal = anchor_logits + residual
        delta = proposal - base_logits
        if self.prediction_mode == "anchor_primary":
            final = proposal
        elif self.prediction_mode in {"base_primary", "learned_blend"}:
            final = base_logits + trust * delta
        else:
            final = anchor_logits + 0.5 * residual
        return final, {
            "residual_logits": residual,
            "proposal_logits": proposal,
            "delta_logits": delta,
            "trust_mean": trust.detach().mean(),
            "trust_std": trust.detach().std(unbiased=False),
            "residual_abs_mean": residual.detach().abs().mean(),
            "residual_abs_max": residual.detach().abs().amax(),
            "delta_abs_mean": delta.detach().abs().mean(),
            "anchor_trust_ratio": trust.detach().mean(),
            "image_trust_ratio": (1.0 - trust.detach()).mean(),
        }
