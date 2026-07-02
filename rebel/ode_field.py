from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def _logit(p: float) -> float:
    p = min(max(float(p), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(p / (1.0 - p))


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.GroupNorm(_groups(dim), dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class BeliefODEField(nn.Module):
    def __init__(
        self,
        belief_dim: int,
        hidden_dim: int = 192,
        num_blocks: int = 3,
        max_offset_px: float = 6.0,
        offset_warmup_iters: int = 800,
        offset_warmup_start_ratio: float = 0.25,
        write_fast_init: float = 0.25,
        write_slow_init: float = 0.05,
        decay_fast_init: float = 0.75,
        decay_slow_init: float = 0.95,
    ) -> None:
        super().__init__()
        self.max_offset_px = float(max_offset_px)
        self.offset_warmup_iters = int(offset_warmup_iters)
        self.offset_warmup_start_ratio = float(offset_warmup_start_ratio)
        in_dim = belief_dim * 3 + 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ConvNeXtLiteBlock(hidden_dim) for _ in range(max(1, int(num_blocks)))])
        self.delta_obs_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.delta_mem_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.write_decay_head = nn.Conv2d(hidden_dim, 4, 3, padding=1)
        self._init_heads(write_fast_init, write_slow_init, decay_fast_init, decay_slow_init)

    def _init_heads(self, write_fast: float, write_slow: float, decay_fast: float, decay_slow: float) -> None:
        for head in (self.delta_obs_head, self.delta_mem_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.write_decay_head.weight)
        with torch.no_grad():
            self.write_decay_head.bias.copy_(
                torch.tensor([_logit(write_fast), _logit(write_slow), _logit(decay_fast), _logit(decay_slow)])
            )

    def _offset_scale(self, current_iter: int | None = None) -> float:
        if self.offset_warmup_iters <= 0:
            return self.max_offset_px
        step = 0 if current_iter is None else max(int(current_iter), 0)
        progress = min(step / float(self.offset_warmup_iters), 1.0)
        ratio = self.offset_warmup_start_ratio + (1.0 - self.offset_warmup_start_ratio) * progress
        return self.max_offset_px * ratio

    def forward(
        self,
        obs: torch.Tensor,
        working: torch.Tensor,
        stable: torch.Tensor,
        mask_prior: torch.Tensor,
        reliability: torch.Tensor,
        current_iter: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask_prior.shape[-2:] != obs.shape[-2:]:
            mask_prior = F.interpolate(mask_prior, size=obs.shape[-2:], mode="bilinear", align_corners=False)
        if reliability.shape[-2:] != obs.shape[-2:]:
            reliability = F.interpolate(reliability, size=obs.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([obs, working, stable, mask_prior, reliability], dim=1)
        h = self.blocks(self.stem(x))
        scale = self._offset_scale(current_iter)
        raw = self.write_decay_head(h)
        write_fast, write_slow, decay_fast, decay_slow = torch.sigmoid(raw).chunk(4, dim=1)
        return {
            "delta_obs": torch.tanh(self.delta_obs_head(h)) * scale,
            "delta_mem": torch.tanh(self.delta_mem_head(h)) * scale,
            "r_obs": torch.sigmoid(self.gate_head(h)),
            "write_fast": write_fast,
            "write_slow": write_slow,
            "decay_fast": decay_fast,
            "decay_slow": decay_slow,
        }
