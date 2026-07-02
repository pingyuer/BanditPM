from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedVideoBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        hidden = int(d_model * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        xs = x.reshape(b * t, n, c)
        xs = xs + self.spatial_attn(self.spatial_norm(xs), self.spatial_norm(xs), self.spatial_norm(xs), need_weights=False)[0]
        x = xs.view(b, t, n, c)
        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        xt = xt + self.temporal_attn(self.temporal_norm(xt), self.temporal_norm(xt), self.temporal_norm(xt), need_weights=False)[0]
        x = xt.view(b, n, t, c).permute(0, 2, 1, 3)
        return x + self.ffn(self.ffn_norm(x))


class FactorizedVideoTransformer(nn.Module):
    def __init__(self, d_model: int, heads: int, layers: int, mlp_ratio: float, dropout: float, max_time: int = 32, max_tokens: int = 256) -> None:
        super().__init__()
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_time, 1, d_model))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, max_tokens, d_model))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.blocks = nn.ModuleList([FactorizedVideoBlock(d_model, heads, mlp_ratio, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t, n, _ = tokens.shape
        x = tokens + self.temporal_pos[:, :t] + self.spatial_pos[:, :, :n]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
