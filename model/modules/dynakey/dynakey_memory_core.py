from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.dynakey.counterfactual import compute_counterfactual_returns
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
            min_scale=float(cfg.get("MIN_SCALE", 1e-3)),
        )
        self.q_maintainer = DynaKeyQMaintainer(
            value_dim=value_dim,
            bank_size=int(cfg.get("BANK_SIZE", 4)),
            hidden_dim=int(cfg.get("HIDDEN_DIM", 256)),
        )
        self.gate = nn.Parameter(torch.tensor(float(cfg.get("GATE_INIT", 1.0)), dtype=torch.float32))
        self.policy_mode = str(cfg.get("POLICY_MODE", "fixed_residual")).lower()
        self.forced_action = str(cfg.get("FORCED_ACTION", "keep")).lower()
        self.residual_spawn_threshold = float(cfg.get("RESIDUAL_SPAWN_THRESHOLD", 0.05))
        self.split_eps = float(cfg.get("SPLIT_EPS", 0.01))
        self.split_scale_factor = float(cfg.get("SPLIT_SCALE_FACTOR", 0.7))
        self.enable_q_loss = bool(cfg.get("ENABLE_Q_LOSS", False))
        self.detach_q_state = bool(cfg.get("DETACH_Q_STATE", True))
        self._prev_z = None
        self._prev_pred = None
        self._prev_nearest = None
        self._prev_action = None

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        self.dictionary.reset_state(batch_size, num_objects, device)
        self._prev_z = None
        self._prev_pred = None
        self._prev_nearest = None
        self._prev_action = None

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

    def _action_id(self, name: str) -> int:
        return {
            "keep": DynaKeyQMaintainer.ACTION_KEEP,
            "update": DynaKeyQMaintainer.ACTION_UPDATE,
            "spawn": DynaKeyQMaintainer.ACTION_SPAWN,
            "split": DynaKeyQMaintainer.ACTION_SPLIT,
            "delete": DynaKeyQMaintainer.ACTION_DELETE,
        }.get(name, DynaKeyQMaintainer.ACTION_KEEP)

    def _forced_actions(self, z: torch.Tensor) -> torch.Tensor:
        return torch.full(z.shape[:2], self._action_id(self.forced_action), device=z.device, dtype=torch.long)

    def _fixed_residual_actions(self, residual_norm: torch.Tensor) -> torch.Tensor:
        count = self.dictionary.active_key_count()
        has_empty = (~self.dictionary.state.valid).any(dim=-1)
        high = residual_norm > self.residual_spawn_threshold
        actions = torch.full_like(count, DynaKeyQMaintainer.ACTION_KEEP, dtype=torch.long)
        actions = torch.where(high & has_empty, torch.full_like(actions, DynaKeyQMaintainer.ACTION_SPAWN), actions)
        actions = torch.where(high & ~has_empty & (count > 0), torch.full_like(actions, DynaKeyQMaintainer.ACTION_SPLIT), actions)
        actions = torch.where(~high & (count > 0), torch.full_like(actions, DynaKeyQMaintainer.ACTION_UPDATE), actions)
        return actions

    def _select_actions(
        self,
        z: torch.Tensor,
        residual_norm: torch.Tensor,
        residual: torch.Tensor,
        retrieval_aux: dict,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.policy_mode == "no_update":
            return torch.full(z.shape[:2], DynaKeyQMaintainer.ACTION_KEEP, device=z.device, dtype=torch.long), None
        if self.policy_mode == "forced":
            return self._forced_actions(z), None
        if self.policy_mode == "fixed_residual":
            return self._fixed_residual_actions(residual_norm), None
        if self.policy_mode == "q_greedy":
            pred = self._prev_pred if self._prev_pred is not None else z
            q_base = self._prev_z if self._prev_z is not None else z
            if self.detach_q_state:
                q_base = q_base.detach()
                pred = pred.detach()
                z_target = z.detach()
            else:
                z_target = z
            q_state = self.q_maintainer.build_q_state(
                q_base,
                pred,
                z_target,
                self.dictionary.state,
                {**retrieval_aux, "residual_norm": residual_norm},
            )
            q_values = self.q_maintainer(q_state, action_mask)
            return self.q_maintainer.select_action(q_values, action_mask, mode="greedy"), q_values
        return torch.full(z.shape[:2], DynaKeyQMaintainer.ACTION_KEEP, device=z.device, dtype=torch.long), None

    def _apply_actions(
        self,
        actions: torch.Tensor,
        prev_z: torch.Tensor,
        z: torch.Tensor,
        selected_slot: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        executed = actions.clone()
        transition_velocity = z.detach() - prev_z.detach()
        for action_id in range(5):
            mask = actions == action_id
            if not mask.any():
                continue
            if action_id == DynaKeyQMaintainer.ACTION_KEEP:
                continue
            if action_id == DynaKeyQMaintainer.ACTION_UPDATE:
                self.dictionary.update(prev_z, z, selected_slot, weight=mask.to(prev_z.dtype))
                continue
            if action_id == DynaKeyQMaintainer.ACTION_SPAWN:
                self.dictionary.spawn(z.detach(), transition_velocity)
                continue
            if action_id == DynaKeyQMaintainer.ACTION_SPLIT:
                self.dictionary.split(
                    selected_slot,
                    residual=residual,
                    split_eps=self.split_eps,
                    split_scale_factor=self.split_scale_factor,
                )
                continue
            if action_id == DynaKeyQMaintainer.ACTION_DELETE:
                before = self.dictionary.active_key_count()
                self.dictionary.delete(selected_slot)
                after = self.dictionary.active_key_count()
                executed = torch.where((mask & (after == before)), torch.full_like(executed, DynaKeyQMaintainer.ACTION_KEEP), executed)
        return executed

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

        if not self.dictionary.state.valid.any(dim=-1).all():
            self.dictionary.spawn(z, torch.zeros_like(z))

        if self._prev_pred is None:
            residual = torch.zeros_like(z)
            prediction_error = torch.zeros(z.shape[:2], device=z.device, dtype=z.dtype)
        else:
            residual = z.detach() - self._prev_pred.to(device=z.device, dtype=z.dtype)
            prediction_error = torch.mean(residual * residual, dim=-1)
        residual_norm = prediction_error

        weights, retrieval_aux = self.dictionary.retrieve(z)
        action_mask = self.q_maintainer.action_mask(self.dictionary.state, retrieval_aux)

        q_values = None
        q_target_action = None
        advantage_returns = None
        action_mask_for_loss = None
        if self._prev_z is not None and self._prev_nearest is not None:
            raw_returns, cf_aux = compute_counterfactual_returns(self.dictionary, self._prev_z, z.detach())
            advantage_returns = cf_aux["advantage_returns"].detach()
            q_target_action = cf_aux["best_action"].detach()
            q_pred = self._prev_pred if self._prev_pred is not None else self._prev_z
            q_base = self._prev_z.detach() if self.detach_q_state else self._prev_z
            q_target = z.detach() if self.detach_q_state else z
            q_state_for_loss = self.q_maintainer.build_q_state(
                q_base,
                q_pred.detach() if self.detach_q_state else q_pred,
                q_target,
                self.dictionary.state,
                {**retrieval_aux, "weights": weights, "residual_norm": residual_norm.detach()},
            )
            q_values_for_loss = self.q_maintainer(q_state_for_loss, action_mask)
            action_mask_for_loss = action_mask.detach()
            actions, q_values = self._select_actions(z, residual_norm, residual, retrieval_aux, action_mask)
            if q_values is None:
                q_values = q_values_for_loss
            if self.policy_mode != "forced":
                actions = torch.where(action_mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1), actions, torch.full_like(actions, DynaKeyQMaintainer.ACTION_KEEP))
            executed_actions = self._apply_actions(actions, self._prev_z, z, self._prev_nearest, residual)
        else:
            executed_actions = torch.full(z.shape[:2], DynaKeyQMaintainer.ACTION_SPAWN, device=z.device, dtype=torch.long)

        weights, retrieval_aux = self.dictionary.retrieve(z)
        z_next_pred, pred_aux = self.dictionary.predict(z, weights)

        delta = (z_next_pred - z).unsqueeze(-1).unsqueeze(-1)
        readout = value_BNCHW + self.gate.to(value_BNCHW.dtype) * delta
        nearest = retrieval_aux["nearest_idx"]
        self._prev_z = z.detach()
        self._prev_pred = z_next_pred.detach()
        self._prev_nearest = nearest.detach()
        self._prev_action = executed_actions.detach()
        self.dictionary.tick_age()

        hist = torch.nn.functional.one_hot(executed_actions, num_classes=5).float().mean(dim=(0, 1))
        entropy = (-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)).detach()
        aux = {
            "weights": weights.detach(),
            "nearest_idx": nearest.detach(),
            "active_key_count": self.dictionary.active_key_count().detach(),
            "prediction_error": prediction_error.detach(),
            "residual_norm": residual_norm.detach(),
            "occupancy_ratio": self.dictionary.state.valid.float().mean(dim=-1).detach(),
            "retrieval_entropy": entropy,
            "identity_fallback": pred_aux["used_identity_fallback"].detach(),
            "executed_action": executed_actions.detach(),
            "actions": executed_actions.detach(),
            "action_hist": hist.detach(),
            "policy_mode": self.policy_mode,
            "action_keep": hist[0].detach(),
            "action_update": hist[1].detach(),
            "action_spawn": hist[2].detach(),
            "action_split": hist[3].detach(),
            "action_delete": hist[4].detach(),
            "q_values": q_values,
            "q_target_action": q_target_action,
            "advantage_returns": advantage_returns,
            "action_mask": action_mask_for_loss,
            "used_identity_fallback": pred_aux["used_identity_fallback"].detach(),
        }
        return readout, aux
