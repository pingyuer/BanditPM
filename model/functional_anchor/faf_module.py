from __future__ import annotations

import torch
import torch.nn as nn

from model.functional_anchor.anchor_provider import AnchorProvider
from model.functional_anchor.affine_selector import AffineSelector
from model.functional_anchor.affine_mixture import AffineMixtureGenerator
from model.functional_anchor.confidence_fusion import ConfidenceFusion
from model.functional_anchor.dense_momentum import DenseMomentumWarp
from model.functional_anchor.temporal_affine_update import TemporalAffineUpdater


class FAFModule(nn.Module):
    """UNeXt-as-anchor FAF with a temporal affine mixture bank."""

    def __init__(
        self,
        *,
        feature_dims: dict[str, int],
        num_anchors: int,
        query_dim: int,
        code_dim: int,
        hidden_dim: int,
        basis_dim: int,
        anchor_size: int,
        residual_clip: float,
        trust_max: float,
        retrieval_temperature: float,
        memory_ema: float,
        enable_memory_update: bool = True,
        disable_trust_gate: bool = False,
        temperature_init: float = 1.0,
        temperature_warmup_iters: int = 1500,
        residual_scale_init: float = 0.0,
        residual_scale_max: float = 0.05,
        residual_warmup_iters: int = 1500,
        trust_warmup_iters: int = 1500,
        trust_min_warmup: float = 0.0,
        trust_curriculum_iters: int = 1500,
        ode_dt_init: float = 0.1,
        ode_dt_max: float = 0.6,
        ode_warmup_iters: int = 1500,
        velocity_momentum: float = 0.8,
        feature_modulation: dict | None = None,
        disable_proposal_in_residual: bool = False,
        disable_feature_modulation: bool = False,
        num_affine_slots: int | None = None,
        identity_slot_index: int = 0,
        affine_cfg: dict | None = None,
        selector_cfg: dict | None = None,
        confidence_cfg: dict | None = None,
        residual_cfg: dict | None = None,
        temporal_update_cfg: dict | None = None,
        dense_momentum_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        del code_dim, basis_dim, anchor_size, memory_ema, feature_modulation, disable_proposal_in_residual, disable_feature_modulation
        self.num_slots = int(num_affine_slots or num_anchors)
        self.num_anchors = self.num_slots
        self.identity_slot_index = int(identity_slot_index)
        pooled_dim = sum(feature_dims.values())
        dec_dim = feature_dims["dec"]

        affine_cfg = affine_cfg or {}
        selector_cfg = selector_cfg or {}
        confidence_cfg = confidence_cfg or {}
        residual_cfg = residual_cfg or {}
        temporal_update_cfg = temporal_update_cfg or {}
        dense_momentum_cfg = dense_momentum_cfg or {}

        self.temperature_init = float(selector_cfg.get("temperature_init", temperature_init))
        self.temperature_final = float(selector_cfg.get("temperature_final", retrieval_temperature))
        self.temperature_warmup_iters = int(selector_cfg.get("warmup_iters", temperature_warmup_iters))
        self.confidence_init = float(confidence_cfg.get("init", 0.10))
        self.confidence_max = float(confidence_cfg.get("max", trust_max))
        self.confidence_warmup_iters = int(confidence_cfg.get("warmup_iters", trust_warmup_iters))
        self.residual_scale_init = float(residual_cfg.get("init_scale", residual_scale_init))
        self.residual_scale_max = float(residual_cfg.get("max_scale", residual_scale_max))
        self.residual_warmup_iters = int(residual_cfg.get("warmup_iters", residual_warmup_iters))
        self.ode_dt_init = float(temporal_update_cfg.get("dt_init", ode_dt_init))
        self.ode_dt_max = float(temporal_update_cfg.get("dt_max", ode_dt_max))
        self.ode_warmup_iters = int(temporal_update_cfg.get("warmup_iters", ode_warmup_iters))
        self.disable_confidence = bool(disable_trust_gate) or not bool(confidence_cfg.get("enabled", True))
        self.disable_residual = not bool(residual_cfg.get("enabled", True))
        self.enable_memory_update = bool(enable_memory_update) and bool(temporal_update_cfg.get("enabled", True))
        self.use_dense_momentum = bool(dense_momentum_cfg.get("enabled", False))

        self.anchor_provider = AnchorProvider()
        self.selector = AffineSelector(
            pooled_dim=pooled_dim,
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_slots=self.num_slots,
            identity_slot_index=self.identity_slot_index,
            identity_bias=float(affine_cfg.get("init_identity_bias", selector_cfg.get("init_identity_bias", 2.0))),
            confidence_init=self.confidence_init,
        )
        limits = affine_cfg.get("limits", {})
        self.affine_mixture = AffineMixtureGenerator(
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_slots=self.num_slots,
            translate_limit=float(limits.get("translate", 0.15)),
            scale_log_limit=float(limits.get("scale_log", 0.12)),
            rotation_deg_limit=float(limits.get("rotation_deg", 10.0)),
            shear_limit=float(limits.get("shear", 0.08)),
        )
        self.temporal_updater = TemporalAffineUpdater(
            enable=self.enable_memory_update,
            velocity_momentum=float(temporal_update_cfg.get("velocity_momentum", velocity_momentum)),
            decay_min=float(temporal_update_cfg.get("decay_min", 0.85)),
            truncated_bptt_steps=int(temporal_update_cfg.get("truncated_bptt_steps", 0)),
        )
        self.dense_momentum = DenseMomentumWarp(
            decoder_dim=dec_dim,
            hidden_dim=int(dense_momentum_cfg.get("hidden_dim", hidden_dim)),
            flow_size=int(dense_momentum_cfg.get("flow_size", 16)),
            max_displacement=float(dense_momentum_cfg.get("max_displacement", 0.08)),
            integration_steps=int(dense_momentum_cfg.get("integration_steps", 4)),
        ) if self.use_dense_momentum else None
        self.fusion = ConfidenceFusion(
            dec_dim=dec_dim,
            hidden_dim=hidden_dim,
            confidence_init=self.confidence_init,
            residual_clip=float(residual_cfg.get("clip", residual_clip)),
        )

    def _scheduled(self, step, init: float, final: float, warmup_iters: int, device, dtype) -> torch.Tensor:
        if step is None or warmup_iters <= 0:
            return torch.tensor(final, device=device, dtype=dtype)
        step_value = float(step.detach().flatten()[0].item()) if torch.is_tensor(step) else float(step or 0)
        ratio = min(max(step_value / float(warmup_iters), 0.0), 1.0)
        return torch.tensor(init + ratio * (final - init), device=device, dtype=dtype)

    def initial_state(self, batch_size: int, num_objects: int, device, dtype) -> dict:
        return self.temporal_updater.initial_state(batch_size, num_objects, self.num_slots, device, dtype)

    def forward_step(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        state: dict,
        *,
        global_step=None,
        mode: str = "affine_mixture_safe",
    ) -> tuple[torch.Tensor, dict, dict]:
        dtype = base_logits.dtype
        device = base_logits.device
        temperature = self._scheduled(global_step, self.temperature_init, self.temperature_final, self.temperature_warmup_iters, device, dtype)
        confidence_max = self._scheduled(global_step, self.confidence_init, self.confidence_max, self.confidence_warmup_iters, device, dtype)
        residual_scale = self._scheduled(global_step, self.residual_scale_init, self.residual_scale_max, self.residual_warmup_iters, device, dtype)
        ode_dt = self._scheduled(global_step, self.ode_dt_init, self.ode_dt_max, self.ode_warmup_iters, device, dtype)

        anchor = self.anchor_provider(feats, base_logits)
        slot_weights, selector_aux = self.selector(anchor, state, temperature)
        if mode == "affine_identity_only":
            slot_weights = torch.zeros_like(slot_weights)
            slot_weights[..., self.identity_slot_index] = 1.0
            selector_aux["slot_weights"] = slot_weights
        warped, mixture_logits, mixture_aux = self.affine_mixture(
            base_logits,
            selector_aux["query"],
            state["affine_state"],
            slot_weights,
            selector_aux["slot_confidence"],
        )
        if mode == "affine_hard_top1":
            hard = torch.zeros_like(slot_weights)
            hard.scatter_(-1, slot_weights.argmax(dim=-1, keepdim=True), 1.0)
            mixture_logits = (hard.unsqueeze(-1).unsqueeze(-1) * warped).sum(dim=2)
        elif mode == "base_only":
            mixture_logits = base_logits
        dense_aux: dict[str, torch.Tensor] = {}
        dense_prewarp_logits = mixture_logits
        if self.dense_momentum is not None and mode not in {"base_only", "affine_identity_only"}:
            mixture_logits, dense_aux = self.dense_momentum(
                feats["dec"],
                base_logits,
                mixture_logits,
                anchor["base_uncertainty_map"],
                anchor["base_boundary_map"],
            )
        if mode == "affine_no_temporal":
            update_delta = torch.zeros_like(mixture_aux["affine_delta"])
        else:
            update_delta = mixture_aux["affine_delta"]

        prev_area = state.get("prev_anchor_area")
        area = anchor["query_area"]
        area_motion = (area - prev_area).abs().clamp(0.0, 0.25) * 4.0 if torch.is_tensor(prev_area) else torch.zeros_like(area)
        frame_quality = (1.0 - anchor["query_uncertainty"]).clamp(0.05, 1.0)
        next_state, update_aux = self.temporal_updater.update(
            state,
            slot_weights,
            selector_aux["slot_confidence"],
            update_delta,
            frame_quality,
            area_motion,
            ode_dt,
        )
        next_state["prev_query"] = selector_aux["query"].detach()
        next_state["prev_anchor_area"] = area.detach()

        final_logits, fusion_aux = self.fusion(
            feats["dec"],
            base_logits,
            mixture_logits,
            anchor["base_uncertainty_map"],
            anchor["base_boundary_map"],
            confidence_max=confidence_max,
            residual_scale=residual_scale,
            disable_confidence=(mode == "affine_no_confidence"),
            disable_residual=(self.disable_residual or mode == "affine_no_residual"),
        )
        if mode == "affine_mixture":
            final_logits = mixture_logits + fusion_aux["residual_logits"]
        elif mode in {"base_only", "affine_identity_only"}:
            final_logits = base_logits + fusion_aux["residual_logits"]

        slot_area = mixture_aux["slot_area"]
        aux = {
            "anchor_logits": base_logits,
            "base_object_logits": base_logits,
            "base_logits": base_logits,
            "unext_anchor_logits": base_logits,
            "warped_anchor_logits": warped,
            "mixture_logits": mixture_logits,
            "proposal_logits": mixture_logits,
            "affine_mixture_logits": dense_prewarp_logits,
            "final_logits": final_logits,
            "final_object_logits": final_logits,
            "slot_weights": slot_weights,
            "active_weights": slot_weights,
            "slot_logits": selector_aux["slot_logits"],
            "slot_confidence": selector_aux["slot_confidence"],
            "affine_delta": mixture_aux["affine_delta"],
            "affine_state": state["affine_state"].detach(),
            "affine_state_norm": mixture_aux["affine_state_norm"],
            "affine_delta_norm": mixture_aux["affine_delta_norm"],
            "affine_identity_logits": warped[:, :, self.identity_slot_index],
            "affine_top1_logits": warped.gather(
                2,
                slot_weights.argmax(dim=-1).view(*slot_weights.shape[:2], 1, 1, 1).expand(-1, -1, -1, *warped.shape[-2:]),
            ).squeeze(2),
            "slot_area": slot_area,
            "slot_usage_hist": slot_weights.detach().mean(dim=(0, 1)),
            "slot_area_diversity": slot_area.std(dim=-1, unbiased=False).mean(),
            "coverage_score": 1.0 - (1.0 - torch.sigmoid(warped).amax(dim=2).mean(dim=(-2, -1))).clamp(0.0, 1.0),
            "coverage_gap": (1.0 - torch.sigmoid(warped).amax(dim=2).mean(dim=(-2, -1))).clamp(0.0, 1.0),
            "retrieval_temperature": temperature.detach(),
            "selector_temperature": temperature.detach(),
            "confidence_max": confidence_max.detach(),
            "residual_scale": residual_scale.detach(),
            "ode_dt": ode_dt.detach(),
            "dense_momentum_enabled": torch.tensor(float(self.use_dense_momentum), device=device, dtype=dtype),
            "base_logits_abs_max": base_logits.detach().abs().amax(),
            "proposal_logits_abs_max": mixture_logits.detach().abs().amax(),
            "final_logits_abs_max": final_logits.detach().abs().amax(),
            "mode": "online",
        }
        aux.update(anchor)
        aux.update(selector_aux)
        aux.update(mixture_aux)
        aux.update(dense_aux)
        aux.update(update_aux)
        aux.update(fusion_aux)
        return final_logits, aux, next_state
