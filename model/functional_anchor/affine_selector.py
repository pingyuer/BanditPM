from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


class AffineSelector(nn.Module):
    """Soft affine slot classifier with YOLO-like slot confidence."""

    def __init__(
        self,
        pooled_dim: int,
        query_dim: int,
        hidden_dim: int,
        num_slots: int,
        identity_slot_index: int,
        identity_bias: float,
        confidence_init: float,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.num_slots = int(num_slots)
        self.identity_slot_index = int(identity_slot_index)
        input_dim = int(pooled_dim) + 6 + int(query_dim) + 6 + 3 * int(num_slots)
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, query_dim),
        )
        self.slot_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_slots),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_slots),
        )
        with torch.no_grad():
            self.slot_head[-1].weight.zero_()
            self.slot_head[-1].bias.fill_(-0.5)
            self.slot_head[-1].bias[self.identity_slot_index] = float(identity_bias)
            self.confidence_head[-1].weight.zero_()
            self.confidence_head[-1].bias.fill_(_logit(confidence_init))

    def forward(
        self,
        anchor: dict[str, torch.Tensor],
        state: dict,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stats = anchor["anchor_stats"]
        pooled = anchor["pooled_features"]
        B, N = stats.shape[:2]
        device = stats.device
        dtype = stats.dtype
        prev_query = state.get("prev_query")
        if prev_query is None:
            prev_query = torch.zeros(B, N, self.query_dim, device=device, dtype=dtype)
        affine_summary = state["affine_state"].mean(dim=2)
        quality = state["slot_quality"]
        usage = state["usage"]
        velocity_norm = (state["velocity_state"].float().pow(2).mean(dim=-1) + 1.0e-8).sqrt().to(dtype=dtype)
        selector_in = torch.cat([pooled, stats, prev_query, affine_summary, quality, usage, velocity_norm], dim=-1)
        query = F.normalize(self.query_net(selector_in), dim=-1)
        slot_logits = self.slot_head(query)
        slot_weights = torch.softmax(slot_logits / temperature.clamp_min(1.0e-4), dim=-1)
        slot_confidence = torch.sigmoid(self.confidence_head(query))
        entropy = -(slot_weights * slot_weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        top_values = torch.topk(slot_weights, k=min(3, slot_weights.shape[-1]), dim=-1).values
        return slot_weights, {
            "query": query,
            "slot_logits": slot_logits,
            "slot_weights": slot_weights,
            "slot_confidence": slot_confidence,
            "slot_entropy": entropy,
            "slot_entropy_norm": entropy / math.log(max(self.num_slots, 2)),
            "effective_slot_number": entropy.exp(),
            "top1_slot_weight": top_values[..., 0].detach(),
            "top3_slot_weight_sum": top_values.sum(dim=-1).detach(),
            "identity_slot_usage": slot_weights[..., self.identity_slot_index].detach(),
        }
