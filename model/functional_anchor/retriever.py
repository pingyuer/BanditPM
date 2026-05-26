from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Retriever(nn.Module):
    """Soft retrieval: query-key matching with quality bias.

    No hard top-k, no fixed phase assignment.
    """

    def __init__(self, query_dim: int, hidden_dim: int, pooled_dim: int) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.query_net = nn.Sequential(
            nn.Linear(pooled_dim + 6 + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        anchor_keys: torch.Tensor,
        quality: torch.Tensor,
        prev_query: torch.Tensor | None,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            feats: multi-scale features
            base_logits: [B, N, H, W]
            anchor_keys: [A, C] (normalized)
            quality: [B, N, A]
            prev_query: [B, N, C] or None
            temperature: scalar tensor

        Returns:
            weights: [B, N, A]
            aux: dict with query, scores, entropy, effective, etc.
        """
        B, N = base_logits.shape[:2]
        device = base_logits.device
        dtype = base_logits.dtype

        pooled = torch.cat([feats[level].mean(dim=(-2, -1)) for level in ("low", "mid", "high", "dec")], dim=1)
        pooled = pooled.unsqueeze(1).expand(-1, N, -1)

        prob = torch.sigmoid(base_logits)
        area = prob.mean(dim=(-2, -1)).unsqueeze(-1)
        uncertainty = (1.0 - (prob - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        uncertainty_mean = uncertainty.mean(dim=(-2, -1)).unsqueeze(-1)
        uncertainty_max = uncertainty.flatten(-2).amax(dim=-1).unsqueeze(-1)
        grad_y = F.pad((prob[..., 1:, :] - prob[..., :-1, :]).abs(), (0, 0, 0, 1))
        grad_x = F.pad((prob[..., :, 1:] - prob[..., :, :-1]).abs(), (0, 1, 0, 0))
        boundary = 0.5 * (grad_y.mean(dim=(-2, -1)) + grad_x.mean(dim=(-2, -1))).unsqueeze(-1)
        stats = torch.cat([area, uncertainty_mean, uncertainty_max, boundary,
                           prob.flatten(-2).amax(dim=-1).unsqueeze(-1),
                           prob.flatten(-2).amin(dim=-1).unsqueeze(-1)], dim=-1)

        if prev_query is None:
            prev_query = torch.zeros(B, N, self.query_dim, device=device, dtype=dtype)
        query = F.normalize(self.query_net(torch.cat([pooled, stats, prev_query], dim=-1)), dim=-1)

        scores = torch.einsum("bnc,ac->bna", query, F.normalize(anchor_keys, dim=-1))
        scores = scores + 0.5 * (quality - 0.5)
        weights = torch.softmax(scores / temperature.clamp_min(1.0e-4), dim=-1)

        entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        effective = entropy.exp()
        top_values = torch.topk(weights, k=min(3, weights.shape[-1]), dim=-1).values

        return weights, {
            "query": query,
            "scores": scores,
            "query_area": area.detach().squeeze(-1),
            "query_uncertainty": uncertainty_mean.detach().squeeze(-1),
            "query_boundary_strength": boundary.detach().squeeze(-1),
            "base_uncertainty_map": uncertainty,
            "base_boundary_map": (grad_x + grad_y).clamp(0.0, 1.0),
            "active_anchor_entropy": entropy,
            "active_anchor_entropy_norm": entropy / math.log(max(weights.shape[-1], 2)),
            "effective_anchor_number": effective,
            "top1_anchor_weight": top_values[..., 0].detach(),
            "top3_anchor_weight_sum": top_values.sum(dim=-1).detach(),
        }
