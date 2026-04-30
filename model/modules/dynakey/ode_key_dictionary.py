from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ODEKeyDictionaryState:
    center: torch.Tensor
    velocity: torch.Tensor
    scale: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    error_ema: torch.Tensor
    valid: torch.Tensor


class ODEKeyDictionary(nn.Module):
    """Small per-video dictionary of local Euler ODE experts."""

    def __init__(
        self,
        value_dim: int,
        bank_size: int,
        dt: float = 1.0,
        ema_alpha: float = 0.2,
        retrieval_temperature: float = 1.0,
        min_scale: float = 1e-3,
        max_velocity_norm: float = 10.0,
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.bank_size = int(bank_size)
        self.dt = float(dt)
        self.ema_alpha = float(ema_alpha)
        self.retrieval_temperature = float(retrieval_temperature)
        self.min_scale = float(min_scale)
        self.max_velocity_norm = float(max_velocity_norm)
        self._state: Optional[ODEKeyDictionaryState] = None

    @property
    def state(self) -> ODEKeyDictionaryState:
        if self._state is None:
            raise RuntimeError("ODEKeyDictionary state is not initialized")
        return self._state

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        shape = (batch_size, num_objects, self.bank_size)
        self._state = ODEKeyDictionaryState(
            center=torch.zeros(*shape, self.value_dim, device=device),
            velocity=torch.zeros(*shape, self.value_dim, device=device),
            scale=torch.ones(*shape, device=device),
            age=torch.zeros(*shape, device=device),
            usage=torch.zeros(*shape, device=device),
            error_ema=torch.zeros(*shape, device=device),
            valid=torch.zeros(*shape, dtype=torch.bool, device=device),
        )

    def clone_state(self) -> ODEKeyDictionaryState:
        state = self.state
        return ODEKeyDictionaryState(
            center=state.center.clone(),
            velocity=state.velocity.clone(),
            scale=state.scale.clone(),
            age=state.age.clone(),
            usage=state.usage.clone(),
            error_ema=state.error_ema.clone(),
            valid=state.valid.clone(),
        )

    def set_state(self, state: ODEKeyDictionaryState) -> None:
        self._state = state

    def _ensure_state_for(self, z_BNC: torch.Tensor) -> None:
        B, N, C = z_BNC.shape
        if C != self.value_dim:
            raise ValueError(f"Expected value_dim={self.value_dim}, got {C}")
        if self._state is None or self.state.center.shape[:2] != (B, N):
            self.reset_state(B, N, z_BNC.device)

    def _first_empty_slot(self) -> torch.Tensor:
        state = self.state
        empty = ~state.valid
        first = empty.float().argmax(dim=-1).long()
        has_empty = empty.any(dim=-1)
        fallback = state.usage.argmin(dim=-1).long()
        return torch.where(has_empty, first, fallback)

    def _gather_slot(self, tensor: torch.Tensor, slot_idx: torch.Tensor) -> torch.Tensor:
        while slot_idx.dim() < tensor.dim():
            slot_idx = slot_idx.unsqueeze(-1)
        index = slot_idx.expand(*tensor.shape[:2], 1, *tensor.shape[3:])
        return tensor.gather(2, index).squeeze(2)

    def _scatter_slot(self, tensor: torch.Tensor, slot_idx: torch.Tensor, value: torch.Tensor) -> None:
        while slot_idx.dim() < tensor.dim():
            slot_idx = slot_idx.unsqueeze(-1)
        index = slot_idx.expand(*tensor.shape[:2], 1, *tensor.shape[3:])
        tensor.scatter_(2, index, value.unsqueeze(2))

    def _clamp_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        norm = velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.max_velocity_norm / norm).clamp(max=1.0)
        return velocity * scale

    def retrieve(self, z_BNC: torch.Tensor) -> tuple[torch.Tensor, dict]:
        self._ensure_state_for(z_BNC)
        state = self.state
        z_norm = F.normalize(z_BNC, dim=-1).unsqueeze(2)
        center_norm = F.normalize(state.center, dim=-1)
        distance = torch.linalg.vector_norm(z_norm - center_norm, dim=-1)
        logits = -distance / max(self.retrieval_temperature, 1e-6)
        logits = logits.masked_fill(~state.valid, -1.0e9)
        has_match = state.valid.any(dim=-1)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(has_match.unsqueeze(-1), weights, torch.zeros_like(weights))
        nearest_idx = torch.where(
            has_match,
            logits.argmax(dim=-1),
            torch.zeros_like(logits.argmax(dim=-1)),
        )
        nearest_distance = self._gather_slot(distance.unsqueeze(-1), nearest_idx).squeeze(-1)
        nearest_distance = torch.where(has_match, nearest_distance, torch.zeros_like(nearest_distance))
        return weights, {
            "distance": distance,
            "has_match": has_match,
            "nearest_idx": nearest_idx,
            "nearest_distance": nearest_distance,
            "occupancy_ratio": state.valid.float().mean(dim=-1),
        }

    def predict(
        self,
        z_BNC: torch.Tensor,
        weights_BNK: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        self._ensure_state_for(z_BNC)
        if weights_BNK is None:
            weights_BNK, retrieval_aux = self.retrieve(z_BNC)
        else:
            retrieval_aux = {}
        state = self.state
        slot_pred = z_BNC.unsqueeze(2) + self.dt * state.velocity
        pred = (slot_pred * weights_BNK.unsqueeze(-1)).sum(dim=2)
        has_match = state.valid.any(dim=-1)
        pred = torch.where(has_match.unsqueeze(-1), pred, z_BNC)
        aux = {
            "slot_pred": slot_pred,
            "weights": weights_BNK,
            "used_identity_fallback": ~has_match,
        }
        aux.update(retrieval_aux)
        return pred, aux

    def spawn(
        self,
        z_BNC: torch.Tensor,
        velocity_BNC: Optional[torch.Tensor] = None,
        slot_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_state_for(z_BNC)
        state = self.state
        slot_idx = self._first_empty_slot() if slot_idx is None else slot_idx.long()
        velocity = torch.zeros_like(z_BNC) if velocity_BNC is None else velocity_BNC
        velocity = self._clamp_velocity(velocity.detach())
        self._scatter_slot(state.center, slot_idx, z_BNC.detach())
        self._scatter_slot(state.velocity, slot_idx, velocity)
        self._scatter_slot(state.scale, slot_idx, torch.ones_like(slot_idx, dtype=z_BNC.dtype))
        self._scatter_slot(state.age, slot_idx, torch.zeros_like(slot_idx, dtype=z_BNC.dtype))
        self._scatter_slot(state.usage, slot_idx, torch.ones_like(slot_idx, dtype=z_BNC.dtype))
        self._scatter_slot(state.error_ema, slot_idx, torch.zeros_like(slot_idx, dtype=z_BNC.dtype))
        self._scatter_slot(state.valid, slot_idx, torch.ones_like(slot_idx, dtype=torch.bool))
        return slot_idx

    def clone_slot(self, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> None:
        state = self.state
        src_idx = src_idx.long()
        dst_idx = dst_idx.long()
        for name in ("center", "velocity", "scale", "age", "usage", "error_ema", "valid"):
            tensor = getattr(state, name)
            value = self._gather_slot(tensor, src_idx)
            self._scatter_slot(tensor, dst_idx, value)

    def update(
        self,
        z_BNC: torch.Tensor,
        target_next_BNC: Optional[torch.Tensor],
        slot_idx: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        self._ensure_state_for(z_BNC)
        state = self.state
        slot_idx = slot_idx.long()
        alpha = self.ema_alpha
        if weight is not None:
            alpha_tensor = (alpha * weight).clamp(0.0, 1.0).to(z_BNC.device, z_BNC.dtype)
        else:
            alpha_tensor = torch.full(z_BNC.shape[:2], alpha, device=z_BNC.device, dtype=z_BNC.dtype)
        alpha_exp = alpha_tensor.unsqueeze(-1)

        old_center = self._gather_slot(state.center, slot_idx)
        new_center = (1.0 - alpha_exp) * old_center + alpha_exp * z_BNC.detach()
        self._scatter_slot(state.center, slot_idx, new_center)

        if target_next_BNC is not None:
            target_velocity = (target_next_BNC.detach() - z_BNC.detach()) / max(self.dt, 1e-6)
            target_velocity = self._clamp_velocity(target_velocity)
            old_velocity = self._gather_slot(state.velocity, slot_idx)
            new_velocity = (1.0 - alpha_exp) * old_velocity + alpha_exp * target_velocity
            self._scatter_slot(state.velocity, slot_idx, new_velocity)
            pred = z_BNC.detach() + self.dt * new_velocity
            err = torch.mean((pred - target_next_BNC.detach()) ** 2, dim=-1)
            old_err = self._gather_slot(state.error_ema.unsqueeze(-1), slot_idx).squeeze(-1)
            self._scatter_slot(state.error_ema, slot_idx, (1.0 - alpha_tensor) * old_err + alpha_tensor * err)

        old_usage = self._gather_slot(state.usage.unsqueeze(-1), slot_idx).squeeze(-1)
        self._scatter_slot(state.usage, slot_idx, old_usage + 1.0)
        self._scatter_slot(state.valid, slot_idx, torch.ones_like(slot_idx, dtype=torch.bool))

    def split(self, slot_idx: torch.Tensor, perturb_scale: float = 0.01) -> torch.Tensor:
        state = self.state
        dst_idx = self._first_empty_slot()
        self.clone_slot(slot_idx.long(), dst_idx)
        center = self._gather_slot(state.center, dst_idx)
        basis = torch.zeros_like(center)
        basis[..., 0] = float(perturb_scale)
        self._scatter_slot(state.center, dst_idx, center + basis)
        self._scatter_slot(state.age, dst_idx, torch.zeros_like(dst_idx, dtype=state.age.dtype))
        self._scatter_slot(state.usage, dst_idx, torch.ones_like(dst_idx, dtype=state.usage.dtype))
        return dst_idx

    def delete(self, slot_idx: torch.Tensor) -> None:
        state = self.state
        slot_idx = slot_idx.long()
        self._scatter_slot(state.center, slot_idx, torch.zeros(*slot_idx.shape, self.value_dim, device=state.center.device))
        self._scatter_slot(state.velocity, slot_idx, torch.zeros(*slot_idx.shape, self.value_dim, device=state.velocity.device))
        self._scatter_slot(state.scale, slot_idx, torch.zeros_like(slot_idx, dtype=state.scale.dtype))
        self._scatter_slot(state.age, slot_idx, torch.zeros_like(slot_idx, dtype=state.age.dtype))
        self._scatter_slot(state.usage, slot_idx, torch.zeros_like(slot_idx, dtype=state.usage.dtype))
        self._scatter_slot(state.error_ema, slot_idx, torch.zeros_like(slot_idx, dtype=state.error_ema.dtype))
        self._scatter_slot(state.valid, slot_idx, torch.zeros_like(slot_idx, dtype=torch.bool))

    def tick_age(self) -> None:
        state = self.state
        state.age = torch.where(state.valid, state.age + 1.0, state.age)

    def diagnostics(self) -> dict:
        state = self.state
        valid_float = state.valid.float()
        denom = valid_float.sum().clamp_min(1.0)
        return {
            "occupancy_ratio": valid_float.mean(dim=-1),
            "age_mean": (state.age * valid_float).sum() / denom,
            "usage_mean": (state.usage * valid_float).sum() / denom,
            "error_ema_mean": (state.error_ema * valid_float).sum() / denom,
        }
