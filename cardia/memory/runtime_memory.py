from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import _group_count, _get_activation, RelationEncoder


class RuntimeMemory(nn.Module):
    """Gated runtime cardiac state, intentionally separate from anchor features."""

    def __init__(self, channels: int, hidden_dim: int | None = None, token_dim: int = 2, runtime_token_dim: int = 32, activation: str = "GELU") -> None:
        super().__init__()
        self.runtime_token_dim = int(runtime_token_dim)
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.state_norm = nn.GroupNorm(_group_count(channels), channels)
        self.relation = RelationEncoder(channels, hidden_dim, activation=activation)
        hidden = self.relation.out_dim
        act_cls = _get_activation(activation).__class__
        self.token_proj = nn.Sequential(
            nn.Conv2d(token_dim, hidden, kernel_size=1),
            act_cls(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.reset_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.candidate = nn.Sequential(
            nn.Conv2d(hidden + channels, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
            act_cls(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        token_in_dim = channels * 2 + token_dim + self.runtime_token_dim
        self.runtime_token_update = nn.Sequential(
            nn.Linear(token_in_dim, self.runtime_token_dim * 2),
            nn.LayerNorm(self.runtime_token_dim * 2),
            act_cls(),
            nn.Linear(self.runtime_token_dim * 2, self.runtime_token_dim * 2),
        )
        nn.init.constant_(self.update_gate.bias, -0.5)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        runtime_state_prev: torch.Tensor | None,
        area_token: torch.Tensor,
        runtime_token_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if runtime_state_prev is None:
            runtime_state_prev = anchor_feat_t.detach()
        if runtime_token_prev is None:
            runtime_token_prev = anchor_feat_t.new_zeros(anchor_feat_t.shape[0], self.runtime_token_dim)
        if runtime_state_prev.shape[-2:] != anchor_feat_t.shape[-2:]:
            runtime_state_prev = F.interpolate(runtime_state_prev, size=anchor_feat_t.shape[-2:], mode="bilinear", align_corners=False)
        token = area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype).view(anchor_feat_t.shape[0], -1, 1, 1)
        token = token.expand(-1, -1, anchor_feat_t.shape[-2], anchor_feat_t.shape[-1])
        anchor_norm = self.anchor_norm(anchor_feat_t)
        state_norm = self.state_norm(runtime_state_prev)
        relation = self.relation(anchor_norm, state_norm) + self.token_proj(token)
        reset = torch.sigmoid(self.reset_gate(relation))
        update = torch.sigmoid(self.update_gate(relation))
        candidate = self.candidate(torch.cat([relation, reset * runtime_state_prev], dim=1))
        runtime_state_t = (1.0 - update) * runtime_state_prev + update * candidate
        pooled = torch.cat(
            [
                self.anchor_norm(anchor_feat_t).mean(dim=(2, 3)),
                self.state_norm(runtime_state_t).mean(dim=(2, 3)),
                area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype),
                runtime_token_prev.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype),
            ],
            dim=1,
        )
        token_gate_raw, token_delta = self.runtime_token_update(pooled).chunk(2, dim=1)
        token_gate = torch.sigmoid(token_gate_raw)
        runtime_token_t = (1.0 - token_gate) * runtime_token_prev + token_gate * token_delta
        aux = {
            "runtime_update_mean": update.detach().mean(dim=(1, 2, 3)),
            "runtime_reset_mean": reset.detach().mean(dim=(1, 2, 3)),
            "runtime_state_norm": runtime_state_t.detach().flatten(1).norm(dim=1),
            "runtime_state_abs_mean": runtime_state_t.detach().abs().mean(dim=(1, 2, 3)),
            "runtime_state_rms": runtime_state_t.detach().pow(2).mean(dim=(1, 2, 3)).sqrt(),
            "runtime_token_abs_mean": runtime_token_t.detach().abs().mean(dim=1),
            "runtime_token_rms": runtime_token_t.detach().pow(2).mean(dim=1).sqrt(),
            "runtime_token_update_mean": token_gate.detach().mean(dim=1),
        }
        return runtime_state_t, runtime_token_t, aux
