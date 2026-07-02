from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


class DEBELSpatialTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 192,
        spatial_token_hw: int = 8,
        summary_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.spatial_token_hw = int(spatial_token_hw)
        self.summary_tokens = int(summary_tokens)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + 1, d_model, 1, bias=False),
            nn.GroupNorm(_groups(d_model), d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
        )
        self.summary_head = nn.Conv2d(d_model, self.summary_tokens, 1)

    def forward(self, feat: torch.Tensor, anchor_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = feat.shape
        fg = torch.softmax(anchor_logits, dim=2)[:, :, 1:2]
        fg = F.interpolate(fg.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False).view(b, t, 1, h, w)
        z = torch.cat([feat, fg], dim=2).flatten(0, 1)
        z = F.adaptive_avg_pool2d(z, (self.spatial_token_hw, self.spatial_token_hw))
        token_map = self.project(z)
        _, d, th, tw = token_map.shape
        dense = token_map.flatten(2).transpose(1, 2).view(b, t, th * tw, d)
        attn_logits = self.summary_head(token_map).flatten(2)
        attn = torch.softmax(attn_logits, dim=-1)
        summary = torch.einsum("bkh,bch->bkc", attn, token_map.flatten(2)).view(b, t, self.summary_tokens, d)
        return torch.cat([summary, dense], dim=2), token_map.view(b, t, d, th, tw)
