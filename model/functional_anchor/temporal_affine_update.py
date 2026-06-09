from __future__ import annotations

import torch


class TemporalAffineUpdater:
    """Online temporal update for affine slot hypotheses.

    truncated_bptt_steps semantics:
    - -1: fully detached online filter.
    - 0: no truncation; keep BPTT through the clip.
    - N > 0: detach every N update steps.
    """

    def __init__(
        self,
        *,
        enable: bool,
        velocity_momentum: float,
        decay_min: float,
        truncated_bptt_steps: int,
    ) -> None:
        self.enable = bool(enable)
        self.velocity_momentum = float(velocity_momentum)
        self.decay_min = float(decay_min)
        self.truncated_bptt_steps = int(truncated_bptt_steps)

    def initial_state(self, batch_size: int, num_objects: int, num_slots: int, device, dtype) -> dict:
        shape = (batch_size, num_objects, num_slots)
        return {
            "affine_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "velocity_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "slot_quality": torch.full(shape, 0.5, device=device, dtype=dtype),
            "usage": torch.zeros(shape, device=device, dtype=dtype),
            "age": torch.zeros(shape, device=device, dtype=dtype),
            "prev_query": None,
            "prev_anchor_area": None,
            "step_count": 0,
        }

    def update(
        self,
        state: dict,
        slot_weights: torch.Tensor,
        slot_confidence: torch.Tensor,
        affine_delta: torch.Tensor,
        frame_quality: torch.Tensor,
        area_motion: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[dict, dict[str, torch.Tensor]]:
        affine_state = state["affine_state"]
        velocity_state = state["velocity_state"]
        slot_quality = state["slot_quality"]
        usage = state["usage"]
        age = state["age"]
        step_count = int(state.get("step_count", 0))
        fq = frame_quality.unsqueeze(-1).clamp(0.05, 1.0)
        motion = (0.5 + area_motion.unsqueeze(-1).clamp(0.0, 1.0))
        write = slot_weights * slot_confidence * fq * motion

        if self.truncated_bptt_steps < 0:
            allow_grad = False
        elif self.truncated_bptt_steps == 0:
            allow_grad = True
        else:
            allow_grad = (step_count + 1) % self.truncated_bptt_steps != 0
        delta = affine_delta if allow_grad else affine_delta.detach()
        if self.enable:
            next_velocity = self.velocity_momentum * velocity_state + (1.0 - self.velocity_momentum) * delta
            decay = torch.full_like(slot_weights, self.decay_min + (1.0 - self.decay_min) * 0.5)
            next_affine = decay.unsqueeze(-1) * affine_state + dt * write.unsqueeze(-1) * next_velocity
            next_affine = next_affine.clamp(-0.5, 0.5)
        else:
            next_velocity = torch.zeros_like(velocity_state)
            next_affine = affine_state
            write = torch.zeros_like(write)

        if not allow_grad:
            next_affine = next_affine.detach()
            next_velocity = next_velocity.detach()
        next_quality = (0.9 * slot_quality + 0.1 * write).detach().clamp(0.0, 1.0)
        update = (next_affine - affine_state).detach()
        next_state = {
            "affine_state": next_affine,
            "velocity_state": next_velocity,
            "slot_quality": next_quality,
            "usage": (usage + slot_weights.detach()).detach(),
            "age": (age + 1.0).detach(),
            "step_count": step_count + 1,
        }
        return next_state, {
            "write_strength": write.detach(),
            "write_strength_mean": write.detach().mean(),
            "decay_mean": decay.detach().mean() if self.enable else torch.zeros((), device=affine_state.device, dtype=affine_state.dtype),
            "memory_update_norm": update.float().pow(2).mean(dim=-1).sqrt().mean().to(dtype=affine_state.dtype),
            "velocity_norm": next_velocity.detach().float().pow(2).mean(dim=-1).sqrt().mean().to(dtype=affine_state.dtype),
            "online_state_detached": torch.tensor(float(not allow_grad), device=affine_state.device, dtype=affine_state.dtype),
            "truncated_bptt_steps": torch.tensor(float(self.truncated_bptt_steps), device=affine_state.device, dtype=affine_state.dtype),
        }
