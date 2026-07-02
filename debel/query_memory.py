from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SolverQueryDecoder(nn.Module):
    def __init__(self, feat_channels: int, d_model: int = 192, solver_queries: int = 8, heads: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.solver_queries = int(solver_queries)
        self.queries = nn.Parameter(torch.randn(self.solver_queries, d_model) * 0.02)
        self.query_bias = nn.Sequential(
            nn.Linear(feat_channels + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.solver_queries * d_model),
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, current_anchor: torch.Tensor, feat_map: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = feat_map.shape
        fg = torch.softmax(current_anchor, dim=2)[:, :, 1:2]
        fg = F.interpolate(fg.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False).view(b, t, 1, h, w)
        ctx = torch.cat([feat_map, fg], dim=2).mean(dim=(-2, -1))
        bias = self.query_bias(ctx).view(b, t, self.solver_queries, -1)
        query = self.queries.view(1, 1, self.solver_queries, -1) + bias
        query_bt = query.flatten(0, 1)
        mem = memory.reshape(b, t * memory.shape[2], memory.shape[3])
        mem_bt = mem[:, None].expand(b, t, mem.shape[1], mem.shape[2]).flatten(0, 1)
        state, attn = self.cross_attn(
            self.cross_norm(query_bt),
            self.memory_norm(mem_bt),
            self.memory_norm(mem_bt),
            need_weights=True,
            average_attn_weights=True,
        )
        state = self.out_norm(state + query_bt).view(b, t, self.solver_queries, -1)
        return state, attn.view(b, t, self.solver_queries, -1)


class BoundedGridSolver(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        d_model: int = 192,
        grid_head_channels: int = 64,
        max_disp: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_disp = float(max_disp)
        self.film = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feat_channels * 2),
        )
        self.norm = nn.GroupNorm(8 if feat_channels % 8 == 0 else 1, feat_channels)
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels + 1, grid_head_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if grid_head_channels % 8 == 0 else 1, grid_head_channels),
            nn.GELU(),
            nn.Conv2d(grid_head_channels, grid_head_channels, 3, padding=1, groups=grid_head_channels),
            nn.GELU(),
            nn.Conv2d(grid_head_channels, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, solver_state: torch.Tensor, feat_map: torch.Tensor, current_anchor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        b, t, c, h, w = feat_map.shape
        pooled = solver_state.mean(dim=2)
        gamma, beta = self.film(pooled).chunk(2, dim=-1)
        feat = self.norm(feat_map.flatten(0, 1))
        feat = feat * (1.0 + gamma.flatten(0, 1).unsqueeze(-1).unsqueeze(-1)) + beta.flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
        fg = torch.softmax(current_anchor, dim=2)[:, :, 1:2]
        fg = F.interpolate(fg.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False)
        raw = self.head(torch.cat([feat, fg], dim=1))
        raw = F.interpolate(raw, size=output_size, mode="bilinear", align_corners=False)
        return torch.tanh(raw).view(b, t, 2, output_size[0], output_size[1]) * self.max_disp


class WeakBoundaryResidual(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 2, alpha_max: float = 0.2) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0))
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, num_classes, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, decoder_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(decoder_feature), self.alpha()
