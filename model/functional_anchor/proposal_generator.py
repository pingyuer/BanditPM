from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalGenerator(nn.Module):
    """Function-code SDF → affine deformation → proposal.

    canonical_sdf + affine_state + affine_delta → per-anchor proposals.
    """

    def __init__(self, query_dim: int, hidden_dim: int, num_anchors: int) -> None:
        super().__init__()
        self.affine_delta_head = nn.Sequential(
            nn.Linear(query_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.affine_delta_head[-1].weight)
        nn.init.zeros_(self.affine_delta_head[-1].bias)
        self._limits = None

    def _get_limits(self, device, dtype):
        if self._limits is None or self._limits.device != device:
            self._limits = torch.tensor(
                [0.08, 0.08, 0.05, 0.05, math.radians(8.0), 0.05],
                device=device, dtype=dtype,
            )
        return self._limits

    def _affine_matrix(self, affine: torch.Tensor) -> torch.Tensor:
        tx = affine[..., 0].clamp(-0.35, 0.35)
        ty = affine[..., 1].clamp(-0.35, 0.35)
        sx = affine[..., 2].clamp(-0.35, 0.35).exp()
        sy = affine[..., 3].clamp(-0.35, 0.35).exp()
        rot = affine[..., 4].clamp(-0.75, 0.75)
        shear = affine[..., 5].clamp(-0.25, 0.25)
        cos = rot.cos()
        sin = rot.sin()
        row0 = torch.stack([cos / sx, (-sin + shear) / sy, -tx], dim=-1)
        row1 = torch.stack([sin / sx, cos / sy, -ty], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def forward(
        self,
        query: torch.Tensor,
        anchor_keys: torch.Tensor,
        canonical_sdf: torch.Tensor,
        affine_state: torch.Tensor,
        weights: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            query: [B, N, C]
            anchor_keys: [A, C]
            canonical_sdf: [A, 1, S, S] (cached)
            affine_state: [B, N, A, 6]
            weights: [B, N, A]
            output_size: (H, W)

        Returns:
            anchor_proposals: [B, N, A, H, W]
            proposal_logits: [B, N, H, W]
            aux: dict
        """
        B, N, A, _ = affine_state.shape
        device = affine_state.device
        dtype = affine_state.dtype
        H, W = output_size

        anchor_query = anchor_keys.view(1, 1, A, -1).expand(B, N, -1, -1)
        delta_in = torch.cat([
            query.unsqueeze(2).expand(-1, -1, A, -1),
            anchor_query,
            affine_state,
        ], dim=-1)
        raw_delta = self.affine_delta_head(delta_in)
        limits = self._get_limits(device, dtype)
        affine_delta = torch.tanh(raw_delta) * limits

        theta = self._affine_matrix(affine_state + affine_delta).flatten(0, 2)
        grid = F.affine_grid(theta, size=(B * N * A, 1, H, W), align_corners=False)
        anchors = canonical_sdf.to(device=device, dtype=dtype)
        anchors = anchors.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1, -1).flatten(0, 2)
        proposals = F.grid_sample(anchors, grid, mode="bilinear", padding_mode="border", align_corners=False)
        proposals = proposals.view(B, N, A, H, W)

        proposal_logits = (weights.unsqueeze(-1).unsqueeze(-1) * proposals).sum(dim=2)
        anchor_area = torch.sigmoid(proposals).mean(dim=(-2, -1))

        return proposals, proposal_logits, {
            "affine_delta": affine_delta,
            "anchor_area": anchor_area,
            "proposal_area_std": anchor_area.std(dim=-1, unbiased=False),
            "proposal_area_range": anchor_area.amax(dim=-1) - anchor_area.amin(dim=-1),
        }
