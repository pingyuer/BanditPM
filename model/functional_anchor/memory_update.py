from __future__ import annotations

import torch
import torch.nn as nn


def _safe_rms_norm(x: torch.Tensor, dim=None, eps: float = 1.0e-8) -> torch.Tensor:
    raw = x.float().pow(2).mean(dim=dim)
    return (raw + eps).sqrt().to(dtype=x.dtype)


class MemoryUpdater:
    """Selective memory update with four-factor write strength.

    write_strength = retrieval_weight * frame_quality * morphology_motion * coverage_need

    Only updates affine/velocity/quality/usage/age (not function codes).
    """

    def __init__(
        self,
        memory_ema: float,
        velocity_momentum: float,
        enable: bool = True,
        truncated_bptt_steps: int = 0,
    ) -> None:
        self.memory_ema = float(memory_ema)
        self.velocity_momentum = float(velocity_momentum)
        self.enable = bool(enable)
        self.truncated_bptt_steps = int(truncated_bptt_steps)

    def initial_state(self, batch_size: int, num_objects: int, num_anchors: int, device, dtype) -> dict:
        shape = (batch_size, num_objects, num_anchors)
        return {
            "affine_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "velocity_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "quality": torch.full(shape, 0.5, device=device, dtype=dtype),
            "usage": torch.zeros(shape, device=device, dtype=dtype),
            "age": torch.zeros(shape, device=device, dtype=dtype),
            "step_count": 0,
            "prev_query": None,
            "prev_proposal": None,
            "prev_area": None,
        }

    def update(
        self,
        state: dict,
        weights: torch.Tensor,
        affine_delta: torch.Tensor,
        frame_quality: torch.Tensor,
        area_motion: torch.Tensor,
        coverage_gap: torch.Tensor,
        ode_dt: torch.Tensor,
    ) -> tuple[dict, dict[str, torch.Tensor]]:
        """
        Args:
            state: current memory state
            weights: [B, N, A] retrieval weights
            affine_delta: [B, N, A, 6] predicted deformation
            frame_quality: [B, N] per-frame quality (1 - uncertainty)
            area_motion: [B, N] |area_t - area_{t-1}|
            coverage_gap: [B, N] 1 - coverage_score
            ode_dt: scalar tensor

        Returns:
            next_state: updated memory state
            aux: dict with write diagnostics
        """
        quality = state["quality"]
        usage = state["usage"]
        age = state["age"]
        affine_state = state["affine_state"]
        velocity_state = state["velocity_state"]
        step_count = state.get("step_count", 0)

        # Four-factor write strength
        # 1. retrieval_weight: how much each anchor is active
        # 2. frame_quality: clear frames write more
        # 3. morphology_motion: changing morphology triggers write
        # 4. coverage_need: low coverage triggers write
        morph_motion = area_motion.unsqueeze(-1).clamp(0.0, 1.0)
        cov_need = coverage_gap.unsqueeze(-1).clamp(0.0, 1.0)
        fq = frame_quality.unsqueeze(-1).clamp(0.05, 1.0)

        write_strength = weights * fq * (0.3 + 0.4 * morph_motion + 0.3 * cov_need)

        # Velocity update (optionally with gradient for truncated BPTT)
        allow_grad = self.truncated_bptt_steps > 0 and step_count % self.truncated_bptt_steps == 0
        if self.enable:
            delta = affine_delta if allow_grad else affine_delta.detach()
            next_velocity = self.velocity_momentum * velocity_state + (1.0 - self.velocity_momentum) * delta
            next_affine = affine_state + ode_dt * write_strength.unsqueeze(-1) * next_velocity
        else:
            next_velocity = torch.zeros_like(velocity_state)
            next_affine = affine_state
            write_strength = torch.zeros_like(write_strength)

        next_affine = next_affine.detach()
        next_velocity = next_velocity.detach()

        # Quality EMA
        next_quality = (self.memory_ema * quality + (1.0 - self.memory_ema) * fq * weights).detach().clamp(0.0, 1.0)

        update_norm = _safe_rms_norm((ode_dt * write_strength.unsqueeze(-1) * next_velocity).detach(), dim=-1)

        next_state = {
            "affine_state": next_affine,
            "velocity_state": next_velocity,
            "quality": next_quality,
            "usage": (usage + weights.detach()).detach(),
            "age": (age + 1.0).detach(),
            "step_count": step_count + 1,
        }

        return next_state, {
            "write_strength": write_strength.detach(),
            "write_strength_mean": write_strength.detach().mean(),
            "write_strength_std": write_strength.detach().std(unbiased=False),
            "memory_update_norm": update_norm.mean(),
            "quality_weighted_write": (write_strength.detach() * fq.detach()).mean(),
            "affine_velocity_norm": _safe_rms_norm(next_velocity.detach(), dim=-1).mean(),
            "dead_anchor_ratio": ((usage + weights.detach()) < 1.0e-3).float().mean(),
        }
