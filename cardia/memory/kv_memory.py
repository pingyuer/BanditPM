from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import _get_activation, _group_count


class CardiacKVMemory(nn.Module):
    """Mask-conditioned cardiac key-value memory with reliability-gated writes.

    This is deliberately lighter than GDKVM, but it gives CARDIA a real
    segmentation-conditioned memory state: key for retrieval, dense value for
    shape/feature readout, and a low-resolution mask prior. Write and decay are
    part of the state update rather than diagnostics only.
    """

    def __init__(
        self,
        channels: int,
        *,
        key_dim: int = 64,
        value_dim: int | None = None,
        runtime_token_dim: int = 32,
        token_dim: int = 2,
        hidden_dim: int = 128,
        write_bias: float = -1.0,
        decay_bias: float = 1.0,
        reliability_floor: float = 0.05,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim or channels)
        self.runtime_token_dim = int(runtime_token_dim)
        self.reliability_floor = float(reliability_floor)
        act_cls = _get_activation(activation).__class__
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.key_proj = nn.Sequential(
            nn.Conv2d(channels, self.key_dim, kernel_size=1),
            nn.GroupNorm(_group_count(self.key_dim), self.key_dim),
            act_cls(),
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(channels + 1, self.value_dim, kernel_size=1),
            nn.GroupNorm(_group_count(self.value_dim), self.value_dim),
            act_cls(),
            nn.Conv2d(self.value_dim, self.value_dim, kernel_size=3, padding=1, groups=self.value_dim),
            nn.GroupNorm(_group_count(self.value_dim), self.value_dim),
            act_cls(),
        )
        self.read_proj = nn.Sequential(
            nn.Conv2d(self.value_dim + 1, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
            act_cls(),
        )
        self.mask_prior_head = nn.Conv2d(self.value_dim, 1, kernel_size=1)
        gate_in = self.key_dim * 2 + token_dim + self.runtime_token_dim + 5
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, 3),
        )
        self.runtime_token_update = nn.Sequential(
            nn.Linear(channels + self.key_dim + token_dim + self.runtime_token_dim + 5, runtime_token_dim * 2),
            nn.LayerNorm(runtime_token_dim * 2),
            act_cls(),
            nn.Linear(runtime_token_dim * 2, runtime_token_dim * 2),
        )
        nn.init.constant_(self.gate_mlp[-1].bias[0], float(decay_bias))
        nn.init.constant_(self.gate_mlp[-1].bias[1], float(write_bias))
        nn.init.zeros_(self.gate_mlp[-1].bias[2])

    @staticmethod
    def _shape_reliability(prob: torch.Tensor, prev_mask: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        entropy = (prob * (1.0 - prob)).mean(dim=(1, 2, 3))
        sharpness = 1.0 - (4.0 * entropy).clamp(0.0, 1.0)
        area = prob.mean(dim=(1, 2, 3))
        area_ok = torch.exp(-((area - 0.08).abs() / 0.12).pow(2)).clamp(0.0, 1.0)
        dx = prob[:, :, :, 1:] - prob[:, :, :, :-1]
        dy = prob[:, :, 1:, :] - prob[:, :, :-1, :]
        boundary = 0.5 * (dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3)))
        boundary_ok = (boundary / 0.01).clamp(0.0, 1.0)
        if prev_mask is None:
            agreement = torch.ones_like(area) * 0.5
            delta_area_abs = torch.zeros_like(area)
        else:
            if prev_mask.shape[-2:] != prob.shape[-2:]:
                prev_mask = F.interpolate(prev_mask, size=prob.shape[-2:], mode="bilinear", align_corners=False)
            agreement = 1.0 - (prob - prev_mask).abs().mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
            delta_area_abs = (area - prev_mask.mean(dim=(1, 2, 3))).abs()
        reliability = (0.35 * sharpness + 0.25 * agreement + 0.2 * area_ok + 0.2 * boundary_ok).clamp(0.0, 1.0)
        aux = {
            "memory_reliability": reliability.detach(),
            "memory_base_entropy": entropy.detach(),
            "memory_boundary_quality": boundary_ok.detach(),
            "memory_area_ok": area_ok.detach(),
            "memory_current_agreement": agreement.detach(),
            "memory_delta_area_abs": delta_area_abs.detach(),
        }
        return reliability, aux

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        state_prev: dict[str, torch.Tensor] | None,
        mask_logits_t: torch.Tensor,
        area_token: torch.Tensor,
        runtime_token_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        B, _, H, W = anchor_feat_t.shape
        dtype = anchor_feat_t.dtype
        device = anchor_feat_t.device
        if runtime_token_prev is None:
            runtime_token_prev = anchor_feat_t.new_zeros(B, self.runtime_token_dim)
        prob = torch.sigmoid(mask_logits_t[:, :1].detach())
        if prob.shape[-2:] != (H, W):
            prob_low = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        else:
            prob_low = prob
        anchor = self.anchor_norm(anchor_feat_t)
        key_map = self.key_proj(anchor)
        current_key = F.normalize(key_map.mean(dim=(2, 3)), dim=1, eps=1.0e-6)
        current_value = self.value_proj(torch.cat([anchor, prob_low], dim=1))
        if state_prev is None:
            prev_key = torch.zeros(B, self.key_dim, device=device, dtype=dtype)
            prev_value = torch.zeros(B, self.value_dim, H, W, device=device, dtype=dtype)
            prev_mask = None
        else:
            prev_key = state_prev["key"].to(device=device, dtype=dtype)
            prev_value = state_prev["value"].to(device=device, dtype=dtype)
            prev_mask = state_prev["mask"].to(device=device, dtype=dtype)
            if prev_value.shape[-2:] != (H, W):
                prev_value = F.interpolate(prev_value, size=(H, W), mode="bilinear", align_corners=False)
            if prev_mask.shape[-2:] != (H, W):
                prev_mask = F.interpolate(prev_mask, size=(H, W), mode="bilinear", align_corners=False)
        reliability, reliability_aux = self._shape_reliability(prob_low, prev_mask)
        agreement = F.cosine_similarity(current_key, prev_key, dim=1).unsqueeze(1) if state_prev is not None else torch.zeros(B, 1, device=device, dtype=dtype)
        gate_features = torch.cat(
            [
                current_key,
                prev_key,
                area_token.to(device=device, dtype=dtype),
                runtime_token_prev.to(device=device, dtype=dtype),
                reliability[:, None],
                reliability_aux["memory_base_entropy"][:, None].to(dtype),
                reliability_aux["memory_boundary_quality"][:, None].to(dtype),
                reliability_aux["memory_current_agreement"][:, None].to(dtype),
                agreement,
            ],
            dim=1,
        )
        decay_raw, write_raw, read_raw = self.gate_mlp(gate_features).chunk(3, dim=1)
        decay = torch.sigmoid(decay_raw).view(B, 1, 1, 1)
        write = torch.sigmoid(write_raw).view(B, 1, 1, 1) * (self.reliability_floor + (1.0 - self.reliability_floor) * reliability.view(B, 1, 1, 1))
        read_gate = torch.sigmoid(read_raw).view(B, 1, 1, 1)
        denom = (decay + write).clamp_min(1.0e-4)
        value_t = (decay * prev_value + write * current_value) / denom
        mask_t = (decay * (prev_mask if prev_mask is not None else torch.zeros_like(prob_low)) + write * prob_low) / denom
        key_t = F.normalize((decay.flatten(1) * prev_key + write.flatten(1) * current_key), dim=1, eps=1.0e-6)
        memory_read = self.read_proj(torch.cat([value_t, mask_t], dim=1))
        context = anchor_feat_t + read_gate * memory_read
        token_input = torch.cat(
            [
                context.mean(dim=(2, 3)),
                key_t,
                area_token.to(device=device, dtype=dtype),
                runtime_token_prev.to(device=device, dtype=dtype),
                reliability[:, None],
                reliability_aux["memory_base_entropy"][:, None].to(dtype),
                reliability_aux["memory_boundary_quality"][:, None].to(dtype),
                reliability_aux["memory_current_agreement"][:, None].to(dtype),
                agreement,
            ],
            dim=1,
        )
        token_gate_raw, token_delta = self.runtime_token_update(token_input).chunk(2, dim=1)
        token_gate = torch.sigmoid(token_gate_raw)
        runtime_token_t = (1.0 - token_gate) * runtime_token_prev + token_gate * token_delta
        mask_prior_logits = self.mask_prior_head(value_t)
        state_t = {"key": key_t, "value": value_t, "mask": mask_t}
        aux = {
            **reliability_aux,
            "runtime_update_mean": write.detach().flatten(1).mean(dim=1),
            "runtime_reset_mean": decay.detach().flatten(1).mean(dim=1),
            "runtime_state_norm": value_t.detach().flatten(1).float().norm(dim=1),
            "runtime_state_abs_mean": value_t.detach().abs().mean(dim=(1, 2, 3)),
            "runtime_state_rms": value_t.detach().pow(2).mean(dim=(1, 2, 3)).sqrt(),
            "runtime_token_abs_mean": runtime_token_t.detach().abs().mean(dim=1),
            "runtime_token_rms": runtime_token_t.detach().pow(2).mean(dim=1).sqrt(),
            "runtime_token_update_mean": token_gate.detach().mean(dim=1),
            "memory_write_mean": write.detach().flatten(1).mean(dim=1),
            "memory_decay_mean": decay.detach().flatten(1).mean(dim=1),
            "memory_read_gate_mean": read_gate.detach().flatten(1).mean(dim=1),
            "memory_mask_prior_mean": mask_t.detach().mean(dim=(1, 2, 3)),
            "memory_mask_prior_logits": mask_prior_logits,
            "memory_readout_abs_mean": memory_read.detach().abs().mean(dim=(1, 2, 3)),
        }
        return context, state_t, runtime_token_t, aux
