from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


class ConfidenceFusion(nn.Module):
    def __init__(self, dec_dim: int, hidden_dim: int, *, confidence_init: float, residual_clip: float) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        self.head = nn.Sequential(
            nn.Conv2d(dec_dim + 5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1.0e-3)
        with torch.no_grad():
            self.head[-1].bias[0] = _logit(confidence_init)
            self.head[-1].bias[1] = 0.0
            self.head[-1].weight[1].zero_()

    def forward(
        self,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
        mixture_logits: torch.Tensor,
        uncertainty_map: torch.Tensor,
        boundary_map: torch.Tensor,
        *,
        confidence_max: torch.Tensor,
        residual_scale: torch.Tensor,
        disable_confidence: bool = False,
        disable_residual: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N, H, W = base_logits.shape
        diff = mixture_logits - base_logits
        fusion_in = torch.cat(
            [
                decoder_feature.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1),
                base_logits.flatten(0, 1).unsqueeze(1),
                mixture_logits.flatten(0, 1).unsqueeze(1),
                diff.flatten(0, 1).unsqueeze(1),
                uncertainty_map.flatten(0, 1).unsqueeze(1),
                boundary_map.flatten(0, 1).unsqueeze(1),
            ],
            dim=1,
        )
        raw = self.head(fusion_in).view(B, N, 2, H, W)
        if disable_confidence:
            confidence = torch.ones(B, N, H, W, device=base_logits.device, dtype=base_logits.dtype)
        else:
            confidence = torch.sigmoid(raw[:, :, 0]) * confidence_max
        if disable_residual:
            residual = torch.zeros_like(base_logits)
        else:
            residual = (torch.tanh(raw[:, :, 1]) * residual_scale).clamp(
                min=-self.residual_clip,
                max=self.residual_clip,
            )
            residual = residual * boundary_map.detach().clamp(0.0, 1.0)
        final = base_logits + confidence * diff + residual
        easy = uncertainty_map.detach() < 0.35
        hard = ~easy
        return final, {
            "confidence_gate": confidence,
            "trust": confidence,
            "residual_logits": residual,
            "safety_residual_logits": confidence * diff + residual,
            "confidence_mean": confidence.detach().mean(),
            "confidence_easy_mean": confidence.detach().masked_select(easy).mean() if easy.any() else confidence.detach().mean(),
            "confidence_hard_mean": confidence.detach().masked_select(hard).mean() if hard.any() else confidence.detach().mean(),
            "residual_l1": residual.detach().abs().mean(),
            "residual_l2": residual.detach().float().pow(2).mean().sqrt().to(dtype=residual.dtype),
            "residual_abs_max": residual.detach().abs().amax(),
            "residual_clip_hit_ratio": (residual.detach().abs() >= self.residual_clip * 0.99).float().mean(),
        }
