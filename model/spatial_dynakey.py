from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpatialReadout:
    feature: torch.Tensor
    delta: torch.Tensor
    gate: torch.Tensor
    mask_prior: torch.Tensor
    weights: torch.Tensor
    phase: torch.Tensor
    aux: dict


class SpatialDynaKeyMemory(nn.Module):
    """Spatial-aware online memory for UNeXt-DynaKey.

    Slots keep a low-resolution feature map, a mask prototype, and a light
    phase descriptor. Retrieval combines global appearance, spatial structure,
    and phase consistency; the readout is a spatial map used by the refinement
    head instead of a broadcast global vector.
    """

    def __init__(
        self,
        value_dim: int,
        *,
        num_slots: int = 4,
        spatial_size: int = 16,
        ema_momentum: float = 0.9,
        temperature: float = 0.1,
        phase_weight: float = 1.0,
        spatial_weight: float = 1.0,
        shape_weight: float = 1.0,
        readout_scale: float = 0.1,
        confidence_threshold: float = 0.55,
        fg_ratio_min: float = 0.005,
        fg_ratio_max: float = 0.60,
        use_spatial_dynamics: bool = False,
        dynamics_momentum: float = 0.8,
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.num_slots = int(num_slots)
        self.spatial_size = int(spatial_size)
        self.ema_momentum = float(ema_momentum)
        self.temperature = float(temperature)
        self.phase_weight = float(phase_weight)
        self.spatial_weight = float(spatial_weight)
        self.shape_weight = float(shape_weight)
        self.readout_scale = float(readout_scale)
        self.confidence_threshold = float(confidence_threshold)
        self.fg_ratio_min = float(fg_ratio_min)
        self.fg_ratio_max = float(fg_ratio_max)
        self.use_spatial_dynamics = bool(use_spatial_dynamics)
        self.dynamics_momentum = float(dynamics_momentum)

        self.register_buffer("_spatial", torch.empty(0), persistent=False)
        self.register_buffer("_velocity", torch.empty(0), persistent=False)
        self.register_buffer("_global", torch.empty(0), persistent=False)
        self.register_buffer("_mask_proto", torch.empty(0), persistent=False)
        self.register_buffer("_phase", torch.empty(0), persistent=False)
        self.register_buffer("_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("_usage", torch.empty(0), persistent=False)
        self.register_buffer("_prev_area", torch.empty(0), persistent=False)
        self._last_update_aux: dict = {}

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (batch_size, num_objects, self.num_slots)
        self._spatial = torch.zeros(*shape, self.value_dim, self.spatial_size, self.spatial_size, device=device, dtype=dtype)
        self._velocity = torch.zeros_like(self._spatial)
        self._global = torch.zeros(*shape, self.value_dim, device=device, dtype=dtype)
        self._mask_proto = torch.zeros(*shape, self.spatial_size, self.spatial_size, device=device, dtype=dtype)
        self._phase = torch.zeros(*shape, 4, device=device, dtype=dtype)
        self._valid = torch.zeros(*shape, device=device, dtype=torch.bool)
        self._usage = torch.zeros(*shape, device=device, dtype=dtype)
        self._prev_area = torch.full((batch_size, num_objects), -1.0, device=device, dtype=dtype)
        self._last_update_aux = {}

    def _ensure_state(self, value_BNCHW: torch.Tensor) -> None:
        if self._spatial.numel() == 0:
            self.reset_state(value_BNCHW.shape[0], value_BNCHW.shape[1], value_BNCHW.device, value_BNCHW.dtype)

    def _phase_descriptor(
        self,
        prob_BNHW: torch.Tensor,
        *,
        frame_index: int = 0,
        total_frames: int = 1,
    ) -> torch.Tensor:
        prob = prob_BNHW.detach().float().clamp(1e-6, 1.0 - 1e-6)
        area = prob.mean(dim=(-2, -1))
        if self._prev_area.numel() == 0:
            area_delta = torch.zeros_like(area)
        else:
            prev = torch.where(self._prev_area >= 0.0, self._prev_area.float(), area)
            area_delta = area - prev
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        confidence = 1.0 - entropy.mean(dim=(-2, -1))
        denom = max(total_frames - 1, 1)
        norm_time = torch.full_like(area, float(frame_index) / float(denom))
        return torch.stack([area, area_delta, norm_time, confidence], dim=-1)

    def _resize_object_feature(self, feature_BCHW: torch.Tensor, num_objects: int) -> torch.Tensor:
        value = F.interpolate(
            feature_BCHW,
            size=(self.spatial_size, self.spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        return value.unsqueeze(1).expand(-1, num_objects, -1, -1, -1).contiguous()

    def _spatial_query(
        self,
        value_BNCHW: torch.Tensor,
        prob_BNHW: torch.Tensor,
        *,
        key_BCHW: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C, _, _ = value_BNCHW.shape
        if key_BCHW is None:
            value = F.interpolate(
                value_BNCHW.flatten(0, 1),
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            ).view(B, N, C, self.spatial_size, self.spatial_size)
        else:
            value = self._resize_object_feature(key_BCHW, N)
        mask = F.interpolate(
            prob_BNHW.flatten(0, 1).unsqueeze(1).float(),
            size=(self.spatial_size, self.spatial_size),
            mode="bilinear",
            align_corners=False,
        ).view(B, N, self.spatial_size, self.spatial_size)
        masked = value * mask.unsqueeze(2)
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        pooled = masked.sum(dim=(-2, -1)) / denom.squeeze(-1)
        fallback = value.mean(dim=(-2, -1))
        pooled = torch.where(denom.squeeze(-1) <= 1.0, fallback, pooled)
        return value, pooled, mask

    def _next_slot(self) -> torch.Tensor:
        invalid_score = torch.where(self._valid, torch.full_like(self._usage, 1.0e6), self._usage)
        free_slot = invalid_score.argmin(dim=-1)
        least_used = self._usage.argmin(dim=-1)
        has_free = (~self._valid).any(dim=-1)
        return torch.where(has_free, free_slot, least_used)

    def read(
        self,
        value_BNCHW: torch.Tensor,
        prob_BNHW: torch.Tensor,
        *,
        key_BCHW: torch.Tensor | None = None,
        pixfeat_BCHW: torch.Tensor | None = None,
        frame_index: int = 0,
        total_frames: int = 1,
        use_phase: bool = True,
        use_spatial_readout: bool = True,
    ) -> SpatialReadout:
        self._ensure_state(value_BNCHW)
        query_spatial, query_global, query_mask = self._spatial_query(
            value_BNCHW.detach(),
            prob_BNHW.detach(),
            key_BCHW=key_BCHW.detach() if key_BCHW is not None else None,
        )
        phase = self._phase_descriptor(prob_BNHW, frame_index=frame_index, total_frames=total_frames).to(value_BNCHW.dtype)

        qg = F.normalize(query_global, dim=-1)
        kg = F.normalize(self._global, dim=-1)
        feature_sim = (qg.unsqueeze(2) * kg).sum(dim=-1)

        qs = F.normalize(query_spatial.flatten(2), dim=-1)
        ss = F.normalize(self._spatial.flatten(3), dim=-1)
        spatial_sim = (qs.unsqueeze(2) * ss).sum(dim=-1)

        shape_dist = (query_mask.unsqueeze(2) - self._mask_proto).abs().mean(dim=(-2, -1))
        phase_dist = (phase.unsqueeze(2) - self._phase).abs().mean(dim=-1)
        logits = (feature_sim + self.spatial_weight * spatial_sim - self.shape_weight * shape_dist) / max(self.temperature, 1.0e-6)
        if use_phase:
            logits = logits - self.phase_weight * phase_dist
        logits = logits.masked_fill(~self._valid, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        has_valid = self._valid.any(dim=-1, keepdim=True)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))

        current_spatial = self._resize_object_feature(pixfeat_BCHW.detach(), value_BNCHW.shape[1]) if pixfeat_BCHW is not None else query_spatial
        memory_spatial = (weights[..., None, None, None] * (self._spatial + self._velocity)).sum(dim=2)
        memory_spatial = torch.where(has_valid.unsqueeze(-1).unsqueeze(-1), memory_spatial, current_spatial)
        mask_prior = (weights[..., None, None] * self._mask_proto).sum(dim=2)
        spatial_delta = memory_spatial - current_spatial
        gate_logits = mask_prior.unsqueeze(2) - spatial_delta.detach().abs().mean(dim=2, keepdim=True)
        spatial_gate = torch.sigmoid(gate_logits)
        if use_spatial_readout:
            spatial = current_spatial + self.readout_scale * spatial_gate * spatial_delta
        else:
            spatial = current_spatial
            spatial_delta = torch.zeros_like(spatial_delta)
            spatial_gate = torch.zeros_like(spatial_gate)
        top_slot = weights.argmax(dim=-1)
        aux = {
            "spatial_memory_valid_slots": self._valid.sum(dim=-1).detach(),
            "spatial_memory_weights": weights.detach(),
            "spatial_memory_entropy": (-(weights.clamp_min(1.0e-8).log() * weights).sum(dim=-1)).detach(),
            "spatial_memory_top_slot": top_slot.detach(),
            "phase_descriptor": phase.detach(),
            "phase_area": phase[..., 0].detach(),
            "phase_area_delta": phase[..., 1].detach(),
            "spatial_feature_similarity_mean": feature_sim.detach().mean(),
            "spatial_consistency_mean": spatial_sim.detach().mean(),
            "shape_consistency_mean": (-shape_dist).detach().mean(),
            "spatial_delta_norm": spatial_delta.detach().pow(2).mean(dim=(2, 3, 4)).sqrt(),
            "spatial_delta_hw_std": spatial_delta.detach().mean(dim=2).flatten(-2).std(dim=-1),
            "spatial_gate_mean": spatial_gate.detach().mean(),
            "spatial_gate_std": spatial_gate.detach().std(),
            "spatial_gate_max": spatial_gate.detach().max(),
            "key_BCHW_used": torch.tensor(1.0 if key_BCHW is not None else 0.0, device=value_BNCHW.device),
            "pixfeat_BCHW_used": torch.tensor(1.0 if pixfeat_BCHW is not None else 0.0, device=value_BNCHW.device),
        }
        aux.update(self._last_update_aux)
        return SpatialReadout(feature=spatial, delta=spatial_delta, gate=spatial_gate, mask_prior=mask_prior, weights=weights, phase=phase, aux=aux)

    def update(
        self,
        value_BNCHW: torch.Tensor,
        prob_BNHW: torch.Tensor,
        *,
        key_BCHW: torch.Tensor | None = None,
        frame_index: int = 0,
        total_frames: int = 1,
    ) -> dict:
        self._ensure_state(value_BNCHW)
        with torch.no_grad():
            spatial, pooled, mask_proto = self._spatial_query(
                value_BNCHW.detach(),
                prob_BNHW.detach(),
                key_BCHW=key_BCHW.detach() if key_BCHW is not None else None,
            )
            phase = self._phase_descriptor(prob_BNHW, frame_index=frame_index, total_frames=total_frames).to(value_BNCHW.dtype)
            confidence = phase[..., 3].float()
            fg_ratio = phase[..., 0].float()
            low_conf = confidence < self.confidence_threshold
            too_small = fg_ratio < self.fg_ratio_min
            too_large = fg_ratio > self.fg_ratio_max
            enabled = ~(low_conf | too_small | too_large)
            slot = self._next_slot()
            B, N = slot.shape
            for b in range(B):
                for n in range(N):
                    if not bool(enabled[b, n]):
                        continue
                    s = int(slot[b, n].item())
                    if self._valid[b, n, s]:
                        old_spatial = self._spatial[b, n, s].clone()
                        m = self.ema_momentum
                        new_spatial = m * old_spatial + (1.0 - m) * spatial[b, n]
                        if self.use_spatial_dynamics:
                            residual = spatial[b, n] - old_spatial
                            self._velocity[b, n, s] = self.dynamics_momentum * self._velocity[b, n, s] + (1.0 - self.dynamics_momentum) * residual
                        self._spatial[b, n, s] = new_spatial
                        self._global[b, n, s] = m * self._global[b, n, s] + (1.0 - m) * pooled[b, n]
                        self._mask_proto[b, n, s] = m * self._mask_proto[b, n, s] + (1.0 - m) * mask_proto[b, n]
                        self._phase[b, n, s] = m * self._phase[b, n, s] + (1.0 - m) * phase[b, n]
                    else:
                        self._spatial[b, n, s] = spatial[b, n]
                        self._global[b, n, s] = pooled[b, n]
                        self._mask_proto[b, n, s] = mask_proto[b, n]
                        self._phase[b, n, s] = phase[b, n]
                        self._valid[b, n, s] = True
                    self._usage[b, n, s] += 1.0
            self._prev_area = fg_ratio.to(self._prev_area.dtype)
            rejected = ~enabled
            aux = {
                "spatial_memory_update_rate": enabled.float().mean().detach(),
                "spatial_memory_fg_ratio_mean": fg_ratio.mean().detach(),
                "spatial_memory_confidence_mean": confidence.mean().detach(),
                "spatial_memory_rejected_count": rejected.float().sum().detach(),
                "spatial_memory_rejected_low_confidence": low_conf.float().sum().detach(),
                "spatial_memory_rejected_too_small": too_small.float().sum().detach(),
                "spatial_memory_rejected_too_large": too_large.float().sum().detach(),
                "spatial_dynamics_enabled": torch.tensor(float(self.use_spatial_dynamics), device=value_BNCHW.device),
            }
            self._last_update_aux = aux
            return aux


def segmentation_gain_reward(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    gt_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if gt_mask is None:
        return None
    gt = gt_mask.float()
    before = torch.sigmoid(before_logits).float()
    after = torch.sigmoid(after_logits).float()

    def dice(prob: torch.Tensor) -> torch.Tensor:
        inter = (prob * gt).sum(dim=(-2, -1))
        denom = prob.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
        return (2.0 * inter + 1.0) / (denom + 1.0)

    return dice(after) - dice(before)
