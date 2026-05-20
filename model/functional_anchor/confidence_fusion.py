from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceFusion(nn.Module):
    """Prediction-mode switch for UNeXt-primary and anchor ablations."""

    def __init__(
        self,
        prediction_mode: str,
        residual_clip: float,
        *,
        trust_max: float = 1.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.prediction_mode = prediction_mode.lower()
        self.residual_clip = float(residual_clip)
        self.trust_max = float(trust_max)
        self.residual_scale = float(residual_scale)
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
        residual_scale: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale = self.residual_scale if residual_scale is None else residual_scale
        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, device=shape_residual.device, dtype=shape_residual.dtype)
        scale = scale.to(device=shape_residual.device, dtype=shape_residual.dtype)
        raw_residual = shape_residual + boundary_residual
        residual = scale * torch.tanh(raw_residual)
        trust = anchor_trust.clamp(0.0, self.trust_max)
        if trust.shape[-2:] != residual.shape[-2:]:
            trust = torch.nn.functional.interpolate(trust, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        trust = trust.expand(-1, residual.shape[1], -1, -1)
        proposal = anchor_logits + residual
        delta = proposal - base_logits
        if self.prediction_mode == "anchor_primary":
            final = proposal
        elif self.prediction_mode == "base_primary":
            final = base_logits + trust * delta
        elif self.prediction_mode == "learned_blend":
            # Algebraically equivalent to base + trust * delta, kept as an
            # explicit ablation path so tests/configs can exercise it.
            final = (1.0 - trust) * base_logits + trust * proposal
        else:
            final = anchor_logits + 0.5 * residual
        return final, {
            "residual_logits": residual,
            "proposal_logits": proposal,
            "delta_logits": delta,
            "raw_residual_logits": raw_residual,
            "residual_scale": scale.detach().reshape(()),
            "trust_mean": trust.detach().mean(),
            "trust_std": trust.detach().std(unbiased=False),
            "residual_abs_mean": residual.detach().abs().mean(),
            "residual_abs_max": residual.detach().abs().amax(),
            "residual_clip_hit_ratio": (residual.detach().abs() >= (scale.detach().abs() * 0.99).clamp_min(1.0e-8)).float().mean(),
            "delta_abs_mean": delta.detach().abs().mean(),
            "anchor_trust_ratio": trust.detach().mean(),
            "image_trust_ratio": (1.0 - trust.detach()).mean(),
        }
