from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..memory.helpers import _group_count, _softplus_inverse


class Stage3Stage2CrossAttention(nn.Module):
    """Stage3→Stage2 cross-attention as a parallel residual contribution.

    Returns only the attention residual `gamma * attn_out` (not `base + gamma*attn_out`),
    so the call site can add this on top of the V2 linear-interpolation path without
    destroying the validated stage3→stage2 signal at init time.

    Init: small `gamma_init` (softplus-parametrized) keeps the attention contribution
    small at start of training but never identically zero, so gradients can flow.
    """

    def __init__(self, channels: int, num_heads: int = 4, gamma_init: float = 0.1, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.GroupNorm(_group_count(channels), channels)
        self.norm_kv = nn.GroupNorm(_group_count(channels), channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))

    def forward(
        self, base_feat: torch.Tensor, stage3_feat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, C, H, W = base_feat.shape
        q = self.norm_q(base_feat).flatten(2).transpose(1, 2)
        kv = self.norm_kv(stage3_feat).flatten(2).transpose(1, 2)
        attn_out, attn_weights = self.cross_attn(q, kv, kv, need_weights=True)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        gamma = F.softplus(self.gamma)
        residual = gamma * attn_out
        attn_w_flat = attn_weights.reshape(-1, attn_weights.shape[-1])
        entropy = -(attn_w_flat.clamp_min(1e-6).log() * attn_w_flat).sum(dim=-1).mean()
        max_entropy = math.log(attn_weights.shape[-1])
        return residual, {
            "cross_attn_entropy": (entropy / max_entropy).detach().reshape(1),
            "cross_attn_gamma": gamma.detach().reshape(1),
            "cross_attn_weight_std": attn_weights.detach().std().reshape(1),
            "cross_attn_residual_abs_mean": residual.detach().abs().mean(dim=(1, 2, 3)),
        }
