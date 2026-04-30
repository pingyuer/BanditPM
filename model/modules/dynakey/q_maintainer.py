from __future__ import annotations

import torch
import torch.nn as nn

from model.modules.dynakey.ode_key_dictionary import ODEKeyDictionaryState


class DynaKeyQMaintainer(nn.Module):
    ACTION_KEEP = 0
    ACTION_UPDATE = 1
    ACTION_SPAWN = 2
    ACTION_SPLIT = 3
    ACTION_DELETE = 4

    def __init__(
        self,
        value_dim: int,
        bank_size: int,
        hidden_dim: int = 256,
        num_actions: int = 5,
        gamma: float = 0.95,
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.bank_size = int(bank_size)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.q_state_dim = 2 * self.value_dim + 16
        base = 2 * self.value_dim
        self.feature_index = {
            "l2_error": base,
            "cosine_error": base + 1,
            "has_target": base + 2,
            "max_weight": base + 3,
            "retrieval_entropy": base + 4,
            "has_match": base + 5,
            "nearest_distance": base + 6,
            "occupancy_ratio": base + 7,
            "mean_age": base + 8,
            "mean_usage": base + 9,
            "mean_error_ema": base + 10,
            "max_error_ema": base + 11,
            "selected_age": base + 12,
            "selected_usage": base + 13,
            "selected_error_ema": base + 14,
            "selected_valid": base + 15,
        }
        self.net = nn.Sequential(
            nn.Linear(self.q_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def _gather_slot(self, tensor: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
        while slot_idx.dim() < tensor.dim():
            slot_idx = slot_idx.unsqueeze(-1)
        index = slot_idx.expand(*tensor.shape[:2], 1, *tensor.shape[3:])
        return tensor.gather(2, index).squeeze(2)

    def build_q_state(
        self,
        z_BNC: torch.Tensor,
        z_next_pred_BNC: torch.Tensor,
        target_next_BNC: torch.Tensor | None,
        dictionary_state: ODEKeyDictionaryState,
        retrieval_aux: dict,
    ) -> torch.Tensor:
        delta_pred = z_next_pred_BNC - z_BNC
        if target_next_BNC is None:
            l2_error = torch.zeros(z_BNC.shape[:2], device=z_BNC.device, dtype=z_BNC.dtype)
            cosine_error = torch.zeros_like(l2_error)
            has_target = torch.zeros_like(l2_error)
        else:
            diff = z_next_pred_BNC - target_next_BNC
            l2_error = torch.mean(diff * diff, dim=-1)
            pred_norm = torch.nn.functional.normalize(z_next_pred_BNC, dim=-1)
            target_norm = torch.nn.functional.normalize(target_next_BNC, dim=-1)
            cosine_error = (1.0 - (pred_norm * target_norm).sum(dim=-1)).clamp_min(0.0)
            has_target = torch.ones_like(l2_error)

        weights = retrieval_aux.get("weights")
        if weights is None:
            weights = torch.zeros(*z_BNC.shape[:2], self.bank_size, device=z_BNC.device, dtype=z_BNC.dtype)
        max_weight = weights.max(dim=-1).values
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        has_match = retrieval_aux.get("has_match", dictionary_state.valid.any(dim=-1)).to(z_BNC.dtype)
        nearest_distance = retrieval_aux.get("nearest_distance", torch.zeros_like(max_weight)).to(z_BNC.dtype)

        valid = dictionary_state.valid
        valid_f = valid.float()
        denom = valid_f.sum(dim=-1).clamp_min(1.0)
        occupancy = valid_f.mean(dim=-1)
        mean_age = (dictionary_state.age * valid_f).sum(dim=-1) / denom
        mean_usage = (dictionary_state.usage * valid_f).sum(dim=-1) / denom
        mean_err = (dictionary_state.error_ema * valid_f).sum(dim=-1) / denom
        max_err = dictionary_state.error_ema.masked_fill(~valid, 0.0).max(dim=-1).values
        nearest_idx = retrieval_aux.get("nearest_idx", torch.zeros_like(max_weight, dtype=torch.long)).long()
        selected_age = self._gather_slot(dictionary_state.age.unsqueeze(-1), nearest_idx).squeeze(-1)
        selected_usage = self._gather_slot(dictionary_state.usage.unsqueeze(-1), nearest_idx).squeeze(-1)
        selected_err = self._gather_slot(dictionary_state.error_ema.unsqueeze(-1), nearest_idx).squeeze(-1)
        selected_valid = self._gather_slot(dictionary_state.valid.unsqueeze(-1), nearest_idx).squeeze(-1).float()

        scalars = torch.stack(
            [
                l2_error,
                cosine_error,
                has_target,
                max_weight,
                entropy,
                has_match,
                nearest_distance,
                occupancy,
                mean_age,
                mean_usage,
                mean_err,
                max_err,
                selected_age,
                selected_usage,
                selected_err,
                selected_valid,
            ],
            dim=-1,
        )
        q_state = torch.cat([z_BNC, delta_pred, scalars], dim=-1)
        return torch.nan_to_num(q_state, nan=0.0, posinf=1e6, neginf=-1e6)

    def action_mask(self, dictionary_state: ODEKeyDictionaryState, retrieval_aux: dict) -> torch.Tensor:
        valid_any = dictionary_state.valid.any(dim=-1)
        empty_any = (~dictionary_state.valid).any(dim=-1)
        mask = torch.zeros(*dictionary_state.valid.shape[:2], self.num_actions, device=dictionary_state.valid.device, dtype=torch.bool)
        mask[..., self.ACTION_KEEP] = True
        mask[..., self.ACTION_UPDATE] = valid_any
        mask[..., self.ACTION_SPAWN] = empty_any
        mask[..., self.ACTION_SPLIT] = valid_any & empty_any
        mask[..., self.ACTION_DELETE] = valid_any
        return mask

    def forward(self, q_state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        q_values = self.net(q_state)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, -1.0e4)
        return q_values

    def select_action(self, q_values: torch.Tensor, action_mask: torch.Tensor, mode: str = "greedy") -> torch.Tensor:
        masked = q_values.masked_fill(~action_mask, -1.0e4)
        if mode == "greedy":
            return masked.argmax(dim=-1)
        if mode == "sample":
            probs = torch.softmax(masked, dim=-1)
            return torch.distributions.Categorical(probs=probs).sample()
        raise ValueError(f"Unsupported DynaKey action selection mode={mode!r}")
