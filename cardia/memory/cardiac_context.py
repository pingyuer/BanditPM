from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import _get_activation


class CardiacContextEncoder(nn.Module):
    """Domain-robust cardiac phase/context token from predicted shape dynamics.

    The token is intentionally based on low-order geometry from the current
    anatomical anchor prediction instead of raw image appearance. This keeps the
    ODE controller closer to cardiac state and less tied to dataset-specific
    decoder feature statistics.
    """

    def __init__(
        self,
        token_dim: int = 32,
        *,
        hidden_dim: int = 64,
        detach_observation: bool = True,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.detach_observation = bool(detach_observation)
        act_cls = _get_activation(activation).__class__
        self.obs_dim = 10
        self.obs_proj = nn.Sequential(
            nn.Linear(self.obs_dim + self.token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, self.token_dim * 2),
        )
        nn.init.normal_(self.obs_proj[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.obs_proj[-1].bias)

    @staticmethod
    def _grid(batch: int, height: int, width: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        return xs.expand(batch, 1, height, width), ys.expand(batch, 1, height, width)

    def _observe(
        self,
        object_logits: torch.Tensor,
        prev_observation: torch.Tensor | None,
    ) -> torch.Tensor:
        prob = torch.sigmoid(object_logits[:, :1])
        if self.detach_observation:
            prob = prob.detach()
        B, _, H, W = prob.shape
        xs, ys = self._grid(B, H, W, prob.device, prob.dtype)
        mass = prob.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-4)
        area = prob.mean(dim=(2, 3))
        cx = (prob * xs).sum(dim=(2, 3)) / mass.flatten(1)
        cy = (prob * ys).sum(dim=(2, 3)) / mass.flatten(1)
        sx = ((prob * (xs - cx[:, :, None, None]).pow(2)).sum(dim=(2, 3)) / mass.flatten(1)).sqrt()
        sy = ((prob * (ys - cy[:, :, None, None]).pow(2)).sum(dim=(2, 3)) / mass.flatten(1)).sqrt()
        dx = prob[:, :, :, 1:] - prob[:, :, :, :-1]
        dy = prob[:, :, 1:, :] - prob[:, :, :-1, :]
        boundary_energy = 0.5 * (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3)))
        uncertainty = (prob * (1.0 - prob)).mean(dim=(2, 3))
        current = torch.cat([area, cx, cy, sx, sy, boundary_energy, uncertainty], dim=1)
        if prev_observation is None:
            prev_observation = torch.zeros(B, self.obs_dim, device=prob.device, dtype=prob.dtype)
            prev_observation[:, :7] = current.detach()
        delta_area = area - prev_observation[:, 0:1]
        delta_cx = cx - prev_observation[:, 1:2]
        delta_cy = cy - prev_observation[:, 2:3]
        return torch.cat([area, delta_area, cx, cy, delta_cx, delta_cy, sx, sy, boundary_energy, uncertainty], dim=1)

    def forward(
        self,
        object_logits: torch.Tensor,
        prev_token: torch.Tensor | None,
        prev_observation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B = object_logits.shape[0]
        if prev_token is None:
            prev_token = object_logits.new_zeros(B, self.token_dim)
        obs = self._observe(object_logits, prev_observation)
        gate_raw, delta = self.obs_proj(torch.cat([obs, prev_token], dim=1)).chunk(2, dim=1)
        gate = torch.sigmoid(gate_raw)
        token = (1.0 - gate) * prev_token + gate * delta
        aux = {
            "context_area": obs[:, 0].detach(),
            "context_delta_area": obs[:, 1].detach(),
            "context_centroid_x": obs[:, 2].detach(),
            "context_centroid_y": obs[:, 3].detach(),
            "context_delta_centroid_abs": obs[:, 4:6].detach().abs().mean(dim=1),
            "context_scale_x": obs[:, 6].detach(),
            "context_scale_y": obs[:, 7].detach(),
            "context_boundary_energy": obs[:, 8].detach(),
            "context_uncertainty": obs[:, 9].detach(),
            "context_token_rms": token.detach().pow(2).mean(dim=1).sqrt(),
            "context_update_mean": gate.detach().mean(dim=1),
        }
        return token, obs.detach(), aux
