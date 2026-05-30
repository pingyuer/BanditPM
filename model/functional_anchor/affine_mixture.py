from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_rms_norm(x: torch.Tensor, dim=None, eps: float = 1.0e-8) -> torch.Tensor:
    raw = x.float().pow(2).mean(dim=dim)
    value = torch.where(raw > 0.0, (raw + eps).sqrt(), torch.zeros_like(raw))
    return value.to(dtype=x.dtype)


class AffineMixtureGenerator(nn.Module):
    """Warp UNeXt anchor logits with a temporal affine slot bank."""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        num_slots: int,
        *,
        translate_limit: float = 0.15,
        scale_log_limit: float = 0.12,
        rotation_deg_limit: float = 10.0,
        shear_limit: float = 0.08,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.register_buffer(
            "limits",
            torch.tensor(
                [
                    translate_limit,
                    translate_limit,
                    scale_log_limit,
                    scale_log_limit,
                    math.radians(rotation_deg_limit),
                    shear_limit,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.affine_delta_head = nn.Sequential(
            nn.Linear(query_dim + 6 + num_slots + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.affine_delta_head[-1].weight)
        nn.init.zeros_(self.affine_delta_head[-1].bias)

    def _affine_matrix(self, affine: torch.Tensor) -> torch.Tensor:
        tx = affine[..., 0].clamp(-0.35, 0.35)
        ty = affine[..., 1].clamp(-0.35, 0.35)
        sx = affine[..., 2].clamp(-0.35, 0.35).exp()
        sy = affine[..., 3].clamp(-0.35, 0.35).exp()
        rot = affine[..., 4].clamp(-0.75, 0.75)
        shear = affine[..., 5].clamp(-0.25, 0.25)
        cos = rot.cos()
        sin = rot.sin()
        row0 = torch.stack([cos / sx, (-sin + shear) / sy, -tx], dim=-1)
        row1 = torch.stack([sin / sx, cos / sy, -ty], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def forward(
        self,
        base_logits: torch.Tensor,
        query: torch.Tensor,
        affine_state: torch.Tensor,
        slot_weights: torch.Tensor,
        slot_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, N, K, _ = affine_state.shape
        H, W = base_logits.shape[-2:]
        slot_eye = torch.eye(K, device=base_logits.device, dtype=base_logits.dtype).view(1, 1, K, K).expand(B, N, -1, -1)
        delta_in = torch.cat(
            [
                query.unsqueeze(2).expand(-1, -1, K, -1),
                affine_state,
                slot_eye,
                slot_confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_delta = self.affine_delta_head(delta_in)
        affine_delta = torch.tanh(raw_delta) * self.limits.to(device=raw_delta.device, dtype=raw_delta.dtype)
        candidate_state = affine_state + affine_delta
        theta = self._affine_matrix(candidate_state).flatten(0, 2)
        grid = F.affine_grid(theta, size=(B * N * K, 1, H, W), align_corners=False)
        anchors = base_logits.unsqueeze(2).expand(-1, -1, K, -1, -1).flatten(0, 2).unsqueeze(1)
        warped = F.grid_sample(anchors, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.view(B, N, K, H, W)
        mixture_logits = (slot_weights.unsqueeze(-1).unsqueeze(-1) * warped).sum(dim=2)
        slot_area = torch.sigmoid(warped).mean(dim=(-2, -1))
        return warped, mixture_logits, {
            "affine_delta": affine_delta,
            "affine_state_candidate": candidate_state,
            "slot_area": slot_area,
            "affine_delta_norm": _safe_rms_norm(affine_delta, dim=-1).mean().to(dtype=base_logits.dtype),
            "affine_state_norm": _safe_rms_norm(affine_state, dim=-1).mean().to(dtype=base_logits.dtype),
        }
