from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.dynakey.ode_key_dictionary import ODEKeyDictionary
from model.modules.dynakey.q_maintainer import DynaKeyQMaintainer


class DynaKeyMemoryCore(nn.Module):
    """MemoryCore-compatible wrapper for local ODE dictionary readout."""

    def __init__(self, cfg, value_dim: int) -> None:
        super().__init__()
        cfg = cfg or {}
        self.value_dim = value_dim
        self.dictionary = ODEKeyDictionary(
            value_dim=value_dim,
            bank_size=int(cfg.get("BANK_SIZE", 4)),
            dt=float(cfg.get("DT", 1.0)),
            ema_alpha=float(cfg.get("EMA_ALPHA", 0.2)),
            retrieval_temperature=float(cfg.get("RETRIEVAL_TEMPERATURE", 1.0)),
        )
        self.q_maintainer = DynaKeyQMaintainer(
            value_dim=value_dim,
            bank_size=int(cfg.get("BANK_SIZE", 4)),
            hidden_dim=int(cfg.get("HIDDEN_DIM", 256)),
        )
        self.gate = nn.Parameter(torch.tensor(float(cfg.get("GATE_INIT", 1.0)), dtype=torch.float32))
        self._prev_z = None
        self._prev_nearest = None

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        self.dictionary.reset_state(batch_size, num_objects, device)
        self._prev_z = None
        self._prev_nearest = None

    def _pool_state(self, value_BNCHW: torch.Tensor, mask_BNHW: torch.Tensor | None) -> torch.Tensor:
        if mask_BNHW is None:
            return value_BNCHW.mean(dim=(-2, -1))
        mask = mask_BNHW.float()
        if mask.shape[-2:] != value_BNCHW.shape[-2:]:
            mask = F.interpolate(mask.flatten(0, 1).unsqueeze(1), size=value_BNCHW.shape[-2:], mode="area")
            mask = mask.view(value_BNCHW.shape[0], value_BNCHW.shape[1], *value_BNCHW.shape[-2:])
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        pooled = (value_BNCHW * mask.unsqueeze(2)).sum(dim=(-2, -1)) / denom.squeeze(-1)
        fallback = value_BNCHW.mean(dim=(-2, -1))
        return torch.where((denom.squeeze(-1) > 1e-5), pooled, fallback)

    def forward(
        self,
        value_BNCHW: torch.Tensor,
        key_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        mask_BNHW: torch.Tensor | None,
        policy_meta: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        del key_BCHW, pixfeat_BCHW
        z = self._pool_state(value_BNCHW, mask_BNHW)

        if self._prev_z is not None and self._prev_nearest is not None:
            self.dictionary.update(self._prev_z, z.detach(), self._prev_nearest)

        weights, retrieval_aux = self.dictionary.retrieve(z)
        if not retrieval_aux["has_match"].all():
            self.dictionary.spawn(z, torch.zeros_like(z))
            weights, retrieval_aux = self.dictionary.retrieve(z)

        z_next_pred, pred_aux = self.dictionary.predict(z, weights)
        action_mask = self.q_maintainer.action_mask(self.dictionary.state, retrieval_aux)
        q_state = self.q_maintainer.build_q_state(z, z_next_pred, None, self.dictionary.state, {**retrieval_aux, "weights": weights})
        q_values = self.q_maintainer(q_state, action_mask)
        actions = self.q_maintainer.select_action(q_values, action_mask, mode="greedy")

        delta = (z_next_pred - z).unsqueeze(-1).unsqueeze(-1)
        readout = value_BNCHW + self.gate.to(value_BNCHW.dtype) * delta
        nearest = retrieval_aux["nearest_idx"]
        self._prev_z = z.detach()
        self._prev_nearest = nearest.detach()
        self.dictionary.tick_age()

        hist = torch.nn.functional.one_hot(actions, num_classes=5).float().mean(dim=(0, 1))
        aux = {
            "weights": weights.detach(),
            "nearest_idx": nearest.detach(),
            "prediction_error": torch.zeros_like(retrieval_aux["has_match"], dtype=value_BNCHW.dtype),
            "occupancy_ratio": self.dictionary.state.valid.float().mean(dim=-1).detach(),
            "retrieval_entropy": (-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)).detach(),
            "actions": actions.detach(),
            "action_hist": hist.detach(),
            "q_values": q_values.detach(),
            "used_identity_fallback": pred_aux["used_identity_fallback"].detach(),
        }
        return readout, aux
