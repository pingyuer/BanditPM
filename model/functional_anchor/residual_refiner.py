from __future__ import annotations

import torch
import torch.nn as nn


class ResidualRefiner(nn.Module):
    """Proposal-conditioned residual correction.

    Input channels: dec_feature + base_logits + proposal_logits +
                    (proposal - base) + uncertainty_map + boundary_map
    """

    def __init__(self, dec_dim: int, hidden_dim: int, residual_clip: float) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        # 6 input channels: base, proposal, proposal-base, uncertainty, boundary, dec
        self.head = nn.Sequential(
            nn.Conv2d(dec_dim + 5, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
        proposal_logits: torch.Tensor,
        uncertainty_map: torch.Tensor,
        boundary_map: torch.Tensor,
        *,
        disable_proposal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_feature: [B, C, H, W]
            base_logits: [B, N, H, W]
            proposal_logits: [B, N, H, W]
            uncertainty_map: [B, N, H, W]
            boundary_map: [B, N, H, W]
            disable_proposal: if True, zero out proposal channels (ablation)

        Returns:
            raw_residual: [B, N, H, W] (before scale)
            bounded_residual: [B, N, H, W] (after tanh + clip)
            proposal_minus_base: [B, N, H, W]
        """
        B, N, H, W = base_logits.shape

        if disable_proposal:
            proposal = torch.zeros_like(proposal_logits)
        else:
            proposal = proposal_logits
        proposal_minus_base = proposal - base_logits

        residual_in = torch.cat([
            decoder_feature.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1),
            base_logits.flatten(0, 1).unsqueeze(1),
            proposal.flatten(0, 1).unsqueeze(1),
            proposal_minus_base.flatten(0, 1).unsqueeze(1),
            uncertainty_map.flatten(0, 1).unsqueeze(1),
            boundary_map.flatten(0, 1).unsqueeze(1),
        ], dim=1)

        raw = self.head(residual_in).view(B, N, H, W).clamp(-self.residual_clip, self.residual_clip)
        bounded = torch.tanh(raw)
        return raw, bounded, proposal_minus_base
