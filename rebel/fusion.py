from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebel.ode_field import ConvNeXtLiteBlock, _groups


def _logit(p: float) -> float:
    p = min(max(float(p), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(p / (1.0 - p))


class BeliefLogitArbitration(nn.Module):
    """Reliability-aware final path for REBEL.

    The belief decoder is a candidate, not the only truth source. This module
    keeps the anatomical base path available and lets the model arbitrate
    between base, observation, belief prior, belief decoder and corrected logits.
    """

    candidate_names = ("base", "obs", "belief", "rebel", "corrected")

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        hidden_dim: int | None = None,
        init_base_bias: float = 1.2,
        init_obs_bias: float = 0.4,
        init_belief_bias: float = -0.2,
        init_rebel_bias: float = 0.0,
        init_corrected_bias: float = -0.4,
        init_temperature: float = 1.25,
        min_base_weight: float = 0.05,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or feature_dim)
        self.num_classes = int(num_classes)
        self.min_base_weight = float(min_base_weight)
        context_dim = feature_dim + len(self.candidate_names) + 2
        self.context = nn.Sequential(
            nn.Conv2d(context_dim, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            ConvNeXtLiteBlock(hidden),
            nn.Conv2d(hidden, len(self.candidate_names), 1),
        )
        self.static_logits = nn.Parameter(
            torch.tensor(
                [init_base_bias, init_obs_bias, init_belief_bias, init_rebel_bias, init_corrected_bias],
                dtype=torch.float32,
            )
        )
        self.raw_temperature = nn.Parameter(torch.tensor(_logit((init_temperature - 0.5) / 2.5), dtype=torch.float32))

    def temperature(self) -> torch.Tensor:
        return 0.5 + 2.5 * torch.sigmoid(self.raw_temperature)

    def _fg_prob(self, logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1:2]

    def forward(
        self,
        *,
        base_logits: torch.Tensor,
        obs_logits: torch.Tensor,
        belief_logits: torch.Tensor,
        rebel_logits: torch.Tensor,
        corrected_logits: torch.Tensor,
        rebel_feature: torch.Tensor,
        disagreement: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = rebel_logits.shape[-2:]
        candidates = {
            "base": base_logits,
            "obs": obs_logits,
            "belief": belief_logits,
            "rebel": rebel_logits,
            "corrected": corrected_logits,
        }
        aligned = []
        probs = []
        for name in self.candidate_names:
            item = candidates[name]
            if item.shape[-2:] != size:
                item = F.interpolate(item, size=size, mode="bilinear", align_corners=False)
            aligned.append(item)
            probs.append(self._fg_prob(item, size))
        if rebel_feature.shape[-2:] != size:
            rebel_feature = F.interpolate(rebel_feature, size=size, mode="bilinear", align_corners=False)
        if disagreement.shape[-2:] != size:
            disagreement = F.interpolate(disagreement, size=size, mode="bilinear", align_corners=False)
        if reliability.shape[-2:] != size:
            reliability = F.interpolate(reliability, size=size, mode="bilinear", align_corners=False)

        logits = self.context(torch.cat([rebel_feature, *probs, disagreement, reliability], dim=1))
        logits = logits + self.static_logits.view(1, -1, 1, 1)
        temperature = self.temperature().to(dtype=logits.dtype)
        weights = torch.softmax(logits / temperature, dim=1)
        if self.min_base_weight > 0:
            base_floor = weights.new_tensor(self.min_base_weight)
            weights = weights * (1.0 - base_floor)
            weights[:, 0:1] = weights[:, 0:1] + base_floor

        stacked = torch.stack(aligned, dim=1)
        final_logits = (weights.unsqueeze(2) * stacked).sum(dim=1)
        fg_stack = torch.stack(probs, dim=1)
        fused_fg = (weights.unsqueeze(2) * fg_stack).sum(dim=1)
        entropy = -(weights.clamp_min(1.0e-6) * weights.clamp_min(1.0e-6).log()).sum(dim=1)
        aux = {
            "arbitration_weights": weights,
            "arbitration_logits": logits,
            "arbitration_temperature": temperature.reshape(1),
            "arbitration_entropy": entropy,
            "arbitration_fused_fg": fused_fg,
        }
        for idx, name in enumerate(self.candidate_names):
            aux[f"arbitration_weight_{name}"] = weights[:, idx : idx + 1]
            aux[f"{name}_candidate_logits"] = aligned[idx]
        return final_logits, aux
