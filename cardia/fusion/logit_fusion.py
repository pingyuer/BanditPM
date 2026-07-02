from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory.helpers import _get_activation, _group_count


class RuntimeLogitFusion(nn.Module):
    """Pixel-wise fusion over base, dynamic, proposal, and memory-prior logits."""

    candidate_names = ("dynamic", "base", "proposal_top1", "proposal_mixture", "memory_prior")

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_dim: int | None = None,
        init_biases: list[float] | tuple[float, ...] | None = None,
        temperature_init: float = 1.0,
        temperature_min: float = 0.35,
        temperature_max: float = 4.0,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(feature_channels // 2, 16))
        act_cls = _get_activation(activation).__class__
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.feature_norm = nn.GroupNorm(_group_count(feature_channels), feature_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels + len(self.candidate_names), hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            act_cls(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(_group_count(hidden), hidden),
            act_cls(),
            nn.Conv2d(hidden, len(self.candidate_names), kernel_size=1),
        )
        if init_biases is None:
            init_biases = (1.0, 0.8, -0.2, -0.2, -0.6)
        if len(init_biases) != len(self.candidate_names):
            raise ValueError(f"RuntimeLogitFusion init_biases must have {len(self.candidate_names)} values.")
        self.candidate_bias = nn.Parameter(torch.tensor(init_biases, dtype=torch.float32))
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(max(float(temperature_init), 1.0e-3))))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _temperature(self) -> torch.Tensor:
        return self.raw_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    def forward(
        self,
        decoder_feature_t: torch.Tensor,
        dynamic_logits_t: torch.Tensor,
        base_logits_t: torch.Tensor,
        proposal_top1_logits_t: torch.Tensor,
        proposal_mixture_logits_t: torch.Tensor,
        memory_prior_logits_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N, H, W = dynamic_logits_t.shape
        candidates = [
            dynamic_logits_t,
            base_logits_t,
            proposal_top1_logits_t,
            proposal_mixture_logits_t,
            memory_prior_logits_t,
        ]
        fixed = []
        for item in candidates:
            if item.shape[-2:] != (H, W):
                item = F.interpolate(item.reshape(-1, 1, *item.shape[-2:]), size=(H, W), mode="bilinear", align_corners=False)
                item = item.view(B, -1, H, W)
            if item.shape[1] == 1 and N > 1:
                item = item.expand(-1, N, -1, -1)
            fixed.append(item)
        stack = torch.stack(fixed, dim=2)
        candidate_context = stack.detach().mean(dim=1)
        gate_in = torch.cat([self.feature_norm(decoder_feature_t), candidate_context], dim=1)
        gate_logits = self.gate(gate_in)
        temperature = self._temperature().to(device=gate_logits.device, dtype=gate_logits.dtype)
        weights = torch.softmax((gate_logits + self.candidate_bias.view(1, -1, 1, 1)) / temperature, dim=1)
        fused = (stack * weights[:, None]).sum(dim=2)
        weights_flat = weights.detach().flatten(2).float()
        entropy = -(weights_flat * weights_flat.clamp_min(1.0e-6).log()).sum(dim=1).mean(dim=1)
        entropy = entropy / math.log(float(len(self.candidate_names)))
        aux: dict[str, torch.Tensor] = {
            "logit_fusion_temperature": temperature.detach().reshape(1),
            "logit_fusion_entropy": entropy.to(dynamic_logits_t.dtype),
            "logit_fusion_fused_minus_base_abs_mean": (fused - base_logits_t).detach().abs().mean(dim=(1, 2, 3)),
            "logit_fusion_fused_minus_dynamic_abs_mean": (fused - dynamic_logits_t).detach().abs().mean(dim=(1, 2, 3)),
        }
        for idx, name in enumerate(self.candidate_names):
            aux[f"logit_fusion_weight_{name}"] = weights[:, idx].detach().mean(dim=(1, 2))
        return fused, aux
