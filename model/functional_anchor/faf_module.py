from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.functional_anchor.field_memory import FieldMemory
from model.functional_anchor.retriever import Retriever
from model.functional_anchor.proposal_generator import ProposalGenerator
from model.functional_anchor.residual_refiner import ResidualRefiner
from model.functional_anchor.trust_gate import TrustGate
from model.functional_anchor.memory_update import MemoryUpdater


class FAFModule(nn.Module):
    """Functional Anchor Field — orchestrator.

    Data flow per frame:
        morphology query → soft retrieval → SDF proposal →
        proposal-conditioned residual → trust-gated correction → selective memory update
    """

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
        temperature_init: float = 0.7,
        temperature_warmup_iters: int = 500,
        residual_scale_init: float = 0.02,
        residual_scale_max: float = 0.12,
        residual_warmup_iters: int = 1500,
        trust_warmup_iters: int = 500,
        trust_min_warmup: float = 0.10,
        trust_curriculum_iters: int = 1500,
        ode_dt_init: float = 0.5,
        ode_dt_max: float = 1.0,
        ode_warmup_iters: int = 1500,
        velocity_momentum: float = 0.8,
        feature_modulation: dict | None = None,
        disable_proposal_in_residual: bool = False,
        disable_feature_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.residual_clip = float(residual_clip)
        self.trust_max = float(trust_max)
        self.temperature_init = float(temperature_init)
        self.temperature_final = float(retrieval_temperature)
        self.temperature_warmup_iters = int(temperature_warmup_iters)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_max = float(residual_scale_max)
        self.residual_warmup_iters = int(residual_warmup_iters)
        self.trust_warmup_iters = int(trust_warmup_iters)
        self.trust_min_warmup = float(trust_min_warmup)
        self.trust_curriculum_iters = int(trust_curriculum_iters)
        self.ode_dt_init = float(ode_dt_init)
        self.ode_dt_max = float(ode_dt_max)
        self.ode_warmup_iters = int(ode_warmup_iters)
        self.disable_trust_gate = bool(disable_trust_gate)
        self.disable_proposal_in_residual = bool(disable_proposal_in_residual)

        pooled_dim = sum(feature_dims.values())
        dec_dim = feature_dims["dec"]

        self.field_memory = FieldMemory(num_anchors, query_dim, code_dim, basis_dim, anchor_size, hidden_dim)
        self.retriever = Retriever(query_dim, hidden_dim, pooled_dim)
        self.proposal_generator = ProposalGenerator(query_dim, hidden_dim, num_anchors)
        self.residual_refiner = ResidualRefiner(dec_dim, hidden_dim, residual_clip)
        self.trust_gate_net = TrustGate(dec_dim, hidden_dim)
        self.memory_updater = MemoryUpdater(memory_ema, velocity_momentum, enable_memory_update)

    def _scheduled(self, step, init: float, final: float, warmup_iters: int, device, dtype) -> torch.Tensor:
        if step is None or warmup_iters <= 0:
            return torch.tensor(final, device=device, dtype=dtype)
        step_value = float(step.detach().flatten()[0].item()) if torch.is_tensor(step) else float(step or 0)
        ratio = min(max(step_value / float(warmup_iters), 0.0), 1.0)
        return torch.tensor(init + ratio * (final - init), device=device, dtype=dtype)

    def _trust_floor(self, step, device, dtype) -> torch.Tensor:
        if step is None or self.trust_curriculum_iters <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        step_value = float(step.detach().flatten()[0].item()) if torch.is_tensor(step) else float(step or 0)
        ratio = min(max(step_value / float(self.trust_curriculum_iters), 0.0), 1.0)
        return torch.tensor(self.trust_min_warmup * (1.0 - ratio), device=device, dtype=dtype)

    def initial_state(self, batch_size: int, num_objects: int, device, dtype) -> dict:
        state = self.memory_updater.initial_state(batch_size, num_objects, self.num_anchors, device, dtype)
        state["prev_query"] = None
        state["prev_proposal"] = None
        state["prev_area"] = None
        return state

    def forward_step(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        state: dict,
        *,
        global_step=None,
        canonical_sdf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict, dict]:
        B, N, H, W = base_logits.shape
        dtype = base_logits.dtype
        device = base_logits.device

        temperature = self._scheduled(global_step, self.temperature_init, self.temperature_final, self.temperature_warmup_iters, device, dtype)
        residual_scale = self._scheduled(global_step, self.residual_scale_init, self.residual_scale_max, self.residual_warmup_iters, device, dtype)
        trust_max = self._scheduled(global_step, 0.05, self.trust_max, self.trust_warmup_iters, device, dtype)
        trust_floor = self._trust_floor(global_step, device, dtype)
        ode_dt = self._scheduled(global_step, self.ode_dt_init, self.ode_dt_max, self.ode_warmup_iters, device, dtype)

        # 1. Retriever
        weights, retrieval_aux = self.retriever(
            feats, base_logits, self.field_memory.anchor_keys, state["quality"], state.get("prev_query"), temperature,
        )

        # 2. Proposal generator (uses cached canonical_sdf)
        if canonical_sdf is None:
            canonical_sdf = self.field_memory.decode_static_field()
        proposals, proposal_logits, proposal_aux = self.proposal_generator(
            retrieval_aux["query"], self.field_memory.anchor_keys, canonical_sdf,
            state["affine_state"], weights, (H, W),
        )

        # 3. Residual refiner
        raw_residual, bounded_residual, proposal_minus_base = self.residual_refiner(
            feats["dec"], base_logits, proposal_logits,
            retrieval_aux["base_uncertainty_map"], retrieval_aux["base_boundary_map"],
            disable_proposal=self.disable_proposal_in_residual,
        )
        residual = residual_scale * bounded_residual

        # 4. Trust gate
        residual_in_concat = torch.cat([
            feats["dec"].unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1),
            base_logits.flatten(0, 1).unsqueeze(1),
            proposal_logits.flatten(0, 1).unsqueeze(1),
            proposal_minus_base.flatten(0, 1).unsqueeze(1),
            retrieval_aux["base_uncertainty_map"].flatten(0, 1).unsqueeze(1),
            retrieval_aux["base_boundary_map"].flatten(0, 1).unsqueeze(1),
        ], dim=1)
        trust, gate = self.trust_gate_net(
            residual_in_concat, trust_max, trust_floor, disable=self.disable_trust_gate,
        )
        trust = trust.view(B, N, H, W)
        gate = gate.view(B, N, H, W)
        safety_residual = gate * trust * residual
        base_safety_logits = base_logits + safety_residual
        proposal_corrected_logits = proposal_logits + safety_residual

        # 5. Memory update
        frame_quality = (1.0 - retrieval_aux["query_uncertainty"]).clamp(0.05, 1.0)
        prev_area = state.get("prev_area")
        area = retrieval_aux["query_area"]
        area_motion = (area - prev_area).abs().clamp(0.0, 0.25) * 4.0 if torch.is_tensor(prev_area) else torch.zeros_like(area)
        coverage_gap = (1.0 - torch.sigmoid(proposals).amax(dim=2).mean(dim=(-2, -1))).clamp(0.0, 1.0)

        next_state, update_aux = self.memory_updater.update(
            state, weights, proposal_aux["affine_delta"], frame_quality, area_motion, coverage_gap, ode_dt,
        )
        next_state["prev_query"] = retrieval_aux["query"].detach()
        next_state["prev_proposal"] = proposal_logits.detach()
        next_state["prev_area"] = area.detach()

        # 6. Diagnostics
        proposal_coverage = 1.0 - coverage_gap
        anchor_area = proposal_aux["anchor_area"]
        anchor_area_means = anchor_area.mean(dim=(0, 1))
        anchor_area_separation = (anchor_area_means.max() - anchor_area_means.min()) if self.num_anchors > 1 else torch.zeros((), device=device, dtype=dtype)
        code_similarity = self.field_memory.code_pairwise_similarity()

        # Residual proposal sensitivity (debug only, expensive)
        residual_proposal_sensitivity = torch.zeros((), device=device, dtype=dtype)

        aux = {
            # Core outputs
            "safety_residual_logits": safety_residual,
            "base_safety_logits": base_safety_logits,
            "proposal_corrected_logits": proposal_corrected_logits,
            "proposal_logits": proposal_logits,
            "anchor_proposals": proposals,
            "active_weights": weights,
            "trust": trust,
            "gate": gate,
            "anchor_trust_map": trust,
            # Field info
            "function_codes": self.field_memory.anchor_function_codes.detach().view(1, 1, self.num_anchors, -1).expand(B, N, -1, -1),
            "canonical_sdf": canonical_sdf.detach(),
            "basis_weights": self.field_memory.get_basis_weights().detach(),
            # Coverage
            "coverage_score": proposal_coverage,
            "coverage_gap": coverage_gap,
            # Residual
            "residual_logits": residual,
            "residual_l1": residual.detach().abs().mean(),
            "residual_l2": residual.detach().pow(2).mean().sqrt(),
            "safety_residual_l1": safety_residual.detach().abs().mean(),
            "residual_clip_hit_ratio": (raw_residual.detach().abs() >= self.residual_clip * 0.99).float().mean(),
            "residual_proposal_sensitivity": residual_proposal_sensitivity,
            "residual_scale": residual_scale.detach(),
            # Trust
            "trust_mean": trust.detach().mean(),
            "trust_std": trust.detach().std(unbiased=False),
            "trust_floor": trust_floor.detach(),
            "gate_mean": gate.detach().mean(),
            "anchor_trust_ratio": trust.detach().mean(),
            "image_trust_ratio": (1.0 - trust.detach()).mean(),
            "trust_easy_mean": trust.detach().masked_select((retrieval_aux["base_uncertainty_map"].detach() < 0.35)).mean()
            if (retrieval_aux["base_uncertainty_map"].detach() < 0.35).any() else trust.detach().mean(),
            "trust_hard_mean": trust.detach().masked_select((retrieval_aux["base_uncertainty_map"].detach() >= 0.35)).mean()
            if (retrieval_aux["base_uncertainty_map"].detach() >= 0.35).any() else trust.detach().mean(),
            # Anchor diversity
            "anchor_area": anchor_area,
            "anchor_area_diversity": anchor_area.std(dim=-1, unbiased=False).mean(),
            "anchor_area_separation": anchor_area_separation,
            "anchor_pairwise_similarity": code_similarity,
            "anchor_function_diversity": 1.0 - code_similarity,
            "anchor_phase_purity_proxy": weights.max(dim=-1).values.detach(),
            # Scheduling
            "retrieval_temperature": temperature.detach(),
            "ode_dt": ode_dt.detach(),
            "trust_max": trust_max.detach(),
            # Affine
            "affine_delta": proposal_aux["affine_delta"],
            "affine_state": state["affine_state"].detach(),
            "affine_delta_norm": proposal_aux["affine_delta"].pow(2).mean(dim=-1).sqrt().mean(),
        }
        aux.update(retrieval_aux)
        aux.update(proposal_aux)
        aux.update(update_aux)

        return base_safety_logits, aux, next_state
