from __future__ import annotations

import torch
import torch.nn.functional as F


class AnchorProvider:
    """Expose UNeXt logits as the FAF anchor field."""

    def __call__(self, feats: dict[str, torch.Tensor], base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        prob = torch.sigmoid(base_logits)
        area = prob.mean(dim=(-2, -1))
        uncertainty = (1.0 - (prob - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        uncertainty_mean = uncertainty.mean(dim=(-2, -1))
        uncertainty_max = uncertainty.flatten(-2).amax(dim=-1)
        grad_y = F.pad((prob[..., 1:, :] - prob[..., :-1, :]).abs(), (0, 0, 0, 1))
        grad_x = F.pad((prob[..., :, 1:] - prob[..., :, :-1]).abs(), (0, 1, 0, 0))
        boundary_map = (grad_x + grad_y).clamp(0.0, 1.0)
        boundary = 0.5 * (grad_y.mean(dim=(-2, -1)) + grad_x.mean(dim=(-2, -1)))
        max_prob = prob.flatten(-2).amax(dim=-1)
        min_prob = prob.flatten(-2).amin(dim=-1)
        stats = torch.stack([area, uncertainty_mean, uncertainty_max, boundary, max_prob, min_prob], dim=-1)
        pooled = torch.cat([feats[level].mean(dim=(-2, -1)) for level in ("low", "mid", "high", "dec")], dim=1)
        pooled = pooled.unsqueeze(1).expand(-1, base_logits.shape[1], -1)
        return {
            "anchor_logits": base_logits,
            "anchor_prob": prob,
            "anchor_stats": stats,
            "pooled_features": pooled,
            "base_uncertainty_map": uncertainty,
            "base_boundary_map": boundary_map,
            "query_area": area.detach(),
            "query_uncertainty": uncertainty_mean.detach(),
            "query_boundary_strength": boundary.detach(),
        }
