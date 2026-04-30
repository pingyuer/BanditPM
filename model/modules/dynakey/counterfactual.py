from __future__ import annotations

from typing import Optional

import torch

from model.modules.dynakey.ode_key_dictionary import ODEKeyDictionary, ODEKeyDictionaryState
from model.modules.dynakey.q_maintainer import DynaKeyQMaintainer


DEFAULT_ACTION_COSTS = torch.tensor([0.0, 0.02, 0.05, 0.08, 0.04], dtype=torch.float32)


def _copy_state(state: ODEKeyDictionaryState) -> ODEKeyDictionaryState:
    return ODEKeyDictionaryState(
        center=state.center.clone(),
        velocity=state.velocity.clone(),
        scale=state.scale.clone(),
        age=state.age.clone(),
        usage=state.usage.clone(),
        error_ema=state.error_ema.clone(),
        valid=state.valid.clone(),
    )


def _new_like(dictionary: ODEKeyDictionary, state: ODEKeyDictionaryState) -> ODEKeyDictionary:
    clone = ODEKeyDictionary(
        value_dim=dictionary.value_dim,
        bank_size=dictionary.bank_size,
        dt=dictionary.dt,
        ema_alpha=dictionary.ema_alpha,
        retrieval_temperature=dictionary.retrieval_temperature,
        min_scale=dictionary.min_scale,
        max_velocity_norm=dictionary.max_velocity_norm,
    )
    clone.set_state(_copy_state(state))
    return clone


def compute_counterfactual_returns(
    dictionary: ODEKeyDictionary,
    z_t_BNC: torch.Tensor,
    z_tp1_BNC: torch.Tensor,
    actions_BN: Optional[torch.Tensor] = None,
    rewards_cfg: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    del actions_BN
    rewards_cfg = rewards_cfg or {}
    costs = rewards_cfg.get("action_costs")
    if costs is None:
        costs_t = DEFAULT_ACTION_COSTS.to(device=z_t_BNC.device, dtype=z_t_BNC.dtype)
    else:
        costs_t = torch.tensor(costs, device=z_t_BNC.device, dtype=z_t_BNC.dtype)

    live_state = dictionary.clone_state()
    raw_returns = []
    errors = []

    for action in range(5):
        trial = _new_like(dictionary, live_state)
        weights, aux = trial.retrieve(z_t_BNC)
        nearest = aux["nearest_idx"]

        if action == DynaKeyQMaintainer.ACTION_KEEP:
            pass
        elif action == DynaKeyQMaintainer.ACTION_UPDATE:
            if trial.state.valid.any(dim=-1).any():
                trial.update(z_t_BNC, z_tp1_BNC, nearest)
        elif action == DynaKeyQMaintainer.ACTION_SPAWN:
            trial.spawn(z_t_BNC, z_tp1_BNC - z_t_BNC)
        elif action == DynaKeyQMaintainer.ACTION_SPLIT:
            valid_any = trial.state.valid.any(dim=-1)
            empty_any = (~trial.state.valid).any(dim=-1)
            if (valid_any & empty_any).any():
                trial.split(nearest, residual=z_tp1_BNC - z_t_BNC)
                weights_after, aux_after = trial.retrieve(z_t_BNC)
                trial.update(z_t_BNC, z_tp1_BNC, aux_after["nearest_idx"])
        elif action == DynaKeyQMaintainer.ACTION_DELETE:
            if trial.state.valid.any(dim=-1).any():
                trial.delete(nearest)

        weights_after, _ = trial.retrieve(z_t_BNC)
        pred, _ = trial.predict(z_t_BNC, weights_after)
        err = torch.mean((pred - z_tp1_BNC) ** 2, dim=-1)
        reward = -err - costs_t[action]
        raw_returns.append(reward)
        errors.append(err)

    raw_returns_BNA = torch.stack(raw_returns, dim=-1)
    errors_BNA = torch.stack(errors, dim=-1)
    keep_error = errors_BNA[..., DynaKeyQMaintainer.ACTION_KEEP]
    cost_delta = costs_t - costs_t[DynaKeyQMaintainer.ACTION_KEEP]
    advantage_returns = keep_error.unsqueeze(-1) - errors_BNA - cost_delta
    return raw_returns_BNA, {
        "raw_returns": raw_returns_BNA,
        "advantage_returns": advantage_returns,
        "action_errors": errors_BNA,
        "keep_error": keep_error,
        "prediction_error": errors_BNA,
        "action_cost": costs_t,
        "best_action": advantage_returns.argmax(dim=-1),
        "finite_mask": torch.isfinite(raw_returns_BNA) & torch.isfinite(advantage_returns),
    }
