from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

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
        forced_cfg = cfg.get("FORCED_ACTION", None)
        self.forced_action = None if forced_cfg in (None, "null", "") else str(forced_cfg).lower()
        if self.policy_mode != "forced" and self.forced_action is not None:
            warnings.warn("DynaKey FORCED_ACTION is ignored because POLICY_MODE is not 'forced'.", RuntimeWarning)
        self.residual_spawn_threshold = float(cfg.get("RESIDUAL_SPAWN_THRESHOLD", 0.05))
        self.split_eps = float(cfg.get("SPLIT_EPS", 0.01))
        self.split_scale_factor = float(cfg.get("SPLIT_SCALE_FACTOR", 0.7))
        self.enable_q_loss = bool(cfg.get("ENABLE_Q_LOSS", False))
        self.detach_q_state = bool(cfg.get("DETACH_Q_STATE", True))
        self._prev_z = None
        self._prev_pred = None
        self._prev_nearest = None
        self._prev_action = None
        self._prev_cf_state = None
        self._prev_q_state = None
        self._prev_q_values = None
        self._prev_action_mask = None
        self._prev_actions = None

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        self.dictionary.reset_state(batch_size, num_objects, device)
        self._prev_z = None
        self._prev_pred = None
        self._prev_nearest = None
        self._prev_action = None
        self._prev_cf_state = None
        self._prev_q_state = None
        self._prev_q_values = None
        self._prev_action_mask = None
        self._prev_actions = None

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
        return torch.full(z.shape[:2], self._action_id(self.forced_action or "keep"), device=z.device, dtype=torch.long)

    def _fixed_residual_actions(self, residual_norm: torch.Tensor) -> torch.Tensor:
        count = self.dictionary.active_key_count()
        has_empty = (~self.dictionary.state.valid).any(dim=-1)
        high = residual_norm > self.residual_spawn_threshold
        actions = torch.full_like(count, DynaKeyQMaintainer.ACTION_KEEP, dtype=torch.long)
        actions = torch.where(high & has_empty, torch.full_like(actions, DynaKeyQMaintainer.ACTION_SPAWN), actions)
        actions = torch.where(high & ~has_empty & (count > 0), torch.full_like(actions, DynaKeyQMaintainer.ACTION_SPLIT), actions)
        actions = torch.where(~high & (count > 0), torch.full_like(actions, DynaKeyQMaintainer.ACTION_UPDATE), actions)
        return actions

    def _select_actions_from_q_values(
        self,
        z: torch.Tensor,
        residual_norm: torch.Tensor,
        action_mask: torch.Tensor,
        q_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.policy_mode == "no_update":
            return torch.full(z.shape[:2], DynaKeyQMaintainer.ACTION_KEEP, device=z.device, dtype=torch.long), None
        if self.policy_mode == "forced":
            return self._forced_actions(z), None
        if self.policy_mode == "fixed_residual":
            return self._fixed_residual_actions(residual_norm), None
        if self.policy_mode == "q_greedy":
            if q_values is None:
                raise RuntimeError("q_greedy requires precomputed q_values")
            return self.q_maintainer.select_action_from_q_values(q_values, action_mask, mode="greedy"), q_values
        return torch.full(z.shape[:2], DynaKeyQMaintainer.ACTION_KEEP, device=z.device, dtype=torch.long), None

    def _build_q_decision(
        self,
        z: torch.Tensor,
        z_next_pred: torch.Tensor,
        residual_norm: torch.Tensor,
        retrieval_aux: dict,
        weights: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_z = z.detach() if self.detach_q_state else z
        q_pred = z_next_pred.detach() if self.detach_q_state else z_next_pred
        q_state = self.q_maintainer.build_q_state(
            q_z,
            q_pred,
            None,
            self.dictionary.state,
            {**retrieval_aux, "weights": weights, "residual_norm": residual_norm.detach()},
        )
        q_values = self.q_maintainer(q_state, action_mask)
        return q_state, q_values

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
                self.dictionary.update_masked(prev_z, z, selected_slot, enabled=mask)
                continue
            if action_id == DynaKeyQMaintainer.ACTION_SPAWN:
                self.dictionary.spawn_masked(z.detach(), transition_velocity, enabled=mask)
                continue
            if action_id == DynaKeyQMaintainer.ACTION_SPLIT:
                self.dictionary.split_masked(
                    selected_slot,
                    residual=residual,
                    enabled=mask,
                    split_eps=self.split_eps,
                    split_scale_factor=self.split_scale_factor,
                )
                continue
            if action_id == DynaKeyQMaintainer.ACTION_DELETE:
                before = self.dictionary.active_key_count()
                self.dictionary.delete_masked(selected_slot, enabled=mask)
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

        initialized = self.dictionary.state.valid.any(dim=-1)
        if not initialized.all():
            self.dictionary.spawn_masked(z, torch.zeros_like(z), enabled=~initialized)

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
        valid_q_samples = None
        invalid_q_targets = None
        q_loss_values = None
        q_loss_state = None
        if self._prev_z is not None and self._prev_nearest is not None:
            raw_returns, cf_aux = compute_counterfactual_returns(
                self.dictionary,
                self._prev_z,
                z.detach(),
                initial_state=self._prev_cf_state,
            )
            advantage_returns = cf_aux["advantage_returns"].detach()
            action_mask_for_loss = self._prev_action_mask
            if action_mask_for_loss is not None:
                masked_advantage = advantage_returns.masked_fill(~action_mask_for_loss, -1.0e4)
                valid_q_samples = action_mask_for_loss.any(dim=-1)
                q_target_action = masked_advantage.argmax(dim=-1)
                q_target_action = torch.where(
                    valid_q_samples,
                    q_target_action,
                    torch.full_like(q_target_action, DynaKeyQMaintainer.ACTION_KEEP),
                )
                invalid_q_targets = ~action_mask_for_loss.gather(-1, q_target_action.unsqueeze(-1)).squeeze(-1)
            q_loss_values = self._prev_q_values
            q_loss_state = self._prev_q_state

        pre_action_state = self.dictionary.clone_state()
        pre_action_pred, _ = self.dictionary.predict(z, weights)
        q_state, q_values = self._build_q_decision(z, pre_action_pred, residual_norm, retrieval_aux, weights, action_mask)

        if self._prev_z is not None and self._prev_nearest is not None:
            actions, _ = self._select_actions_from_q_values(z, residual_norm, action_mask, q_values)
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
        self._prev_cf_state = pre_action_state
        self._prev_q_state = q_state.detach()
        self._prev_q_values = q_values
        self._prev_action_mask = action_mask.detach()
        self._prev_actions = executed_actions.detach()
        self.dictionary.tick_age()

        one_hot_actions = torch.nn.functional.one_hot(executed_actions, num_classes=5).float()
        hist = one_hot_actions.mean(dim=(0, 1))
        action_counts = one_hot_actions.sum(dim=(0, 1))
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
            "action_counts": action_counts.detach(),
            "policy_mode": self.policy_mode,
            "forced_action": self.forced_action if self.policy_mode == "forced" else None,
            "action_keep": hist[0].detach(),
            "action_update": hist[1].detach(),
            "action_spawn": hist[2].detach(),
            "action_split": hist[3].detach(),
            "action_delete": hist[4].detach(),
            "q_values": q_loss_values,
            "q_state": q_loss_state,
            "q_target_action": q_target_action,
            "advantage_returns": advantage_returns,
            "action_mask": action_mask_for_loss,
            "valid_q_samples": valid_q_samples.detach() if valid_q_samples is not None else None,
            "invalid_q_targets": invalid_q_targets.detach() if invalid_q_targets is not None else None,
            "used_identity_fallback": pred_aux["used_identity_fallback"].detach(),
        }
        return readout, aux
