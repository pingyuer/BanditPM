from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


def _orthogonal_rows(num_rows: int, dim: int) -> torch.Tensor:
    basis = torch.empty(dim, dim)
    nn.init.orthogonal_(basis)
    if num_rows <= dim:
        return basis[:num_rows]
    repeats = math.ceil(num_rows / dim)
    return basis.repeat(repeats, 1)[:num_rows]


def _ellipse_sdf(size: int, rx: float, ry: float, cx: float = 0.0, cy: float = 0.0) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x = (xx - float(cx)) / max(float(rx), 1.0e-4)
    y = (yy - float(cy)) / max(float(ry), 1.0e-4)
    # Positive inside, negative outside; clamp keeps initial logits sane.
    sdf = (1.0 - torch.sqrt(x.square() + y.square()).clamp_min(1.0e-6)) * 2.0
    return sdf.clamp(-2.0, 2.0)


class FAFModule(nn.Module):
    """Lightweight functional anchor-affine field.

    The field is a small bank of canonical SDF anchors plus online affine ODE
    states. Dense per-anchor features and per-anchor residual heads are avoided.
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
    ) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.query_dim = int(query_dim)
        self.anchor_size = int(anchor_size)
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
        self.velocity_momentum = float(velocity_momentum)
        self.memory_ema = float(memory_ema)
        self.enable_memory_update = bool(enable_memory_update)
        self.disable_trust_gate = bool(disable_trust_gate)
        feature_modulation = feature_modulation or {}
        self.enable_feature_modulation = bool(feature_modulation.get("enabled", True))
        default_strengths = {"low": 0.0, "mid": 0.08, "high": 0.12, "dec": 0.06}
        strengths = feature_modulation.get("strengths", default_strengths) or default_strengths
        self.modulation_strengths = {
            level: float(strengths.get(level, default_strengths[level])) for level in ("low", "mid", "high", "dec")
        }

        pooled_dim = sum(feature_dims.values())
        self.query_net = nn.Sequential(
            nn.Linear(pooled_dim + 6 + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )
        self.anchor_keys = nn.Parameter(_orthogonal_rows(self.num_anchors, query_dim))
        anchors = [
            _ellipse_sdf(anchor_size, 0.66, 0.82, 0.00, 0.00),
            _ellipse_sdf(anchor_size, 0.50, 0.66, 0.03, 0.00),
            _ellipse_sdf(anchor_size, 0.34, 0.46, 0.00, 0.02),
            _ellipse_sdf(anchor_size, 0.52, 0.58, -0.14, 0.07),
            _ellipse_sdf(anchor_size, 0.44, 0.72, 0.12, -0.05),
        ]
        while len(anchors) < self.num_anchors:
            idx = len(anchors)
            radius = max(0.30, 0.60 - 0.04 * idx)
            anchors.append(_ellipse_sdf(anchor_size, radius, radius * 1.2, 0.05 * ((idx % 3) - 1), 0.0))
        self.canonical_sdf = nn.Parameter(torch.stack(anchors[: self.num_anchors]).unsqueeze(1))

        self.affine_delta_head = nn.Sequential(
            nn.Linear(query_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.affine_delta_head[-1].weight)
        nn.init.zeros_(self.affine_delta_head[-1].bias)

        dec_dim = feature_dims["dec"]
        self.residual_head = nn.Sequential(
            nn.Conv2d(dec_dim + 4, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.trust_gate = nn.Sequential(
            nn.Conv2d(dec_dim + 4, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )
        nn.init.normal_(self.trust_gate[-1].weight, mean=0.0, std=1.0e-3)
        with torch.no_grad():
            self.trust_gate[-1].bias[0] = _logit(0.15)
            self.trust_gate[-1].bias[1] = _logit(0.50)
        self.modulation_heads = nn.ModuleDict()
        for level, dim in feature_dims.items():
            self.modulation_heads[level] = nn.Sequential(
                nn.Conv2d(4, max(8, hidden_dim // 2), 3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(8, hidden_dim // 2), dim * 2, 1),
            )
            nn.init.normal_(self.modulation_heads[level][-1].weight, mean=0.0, std=5.0e-3)
            nn.init.zeros_(self.modulation_heads[level][-1].bias)

    def _scheduled(self, step, init: float, final: float, warmup_iters: int, device, dtype) -> torch.Tensor:
        if step is None:
            return torch.tensor(final, device=device, dtype=dtype)
        if warmup_iters <= 0:
            return torch.tensor(final, device=device, dtype=dtype)
        if torch.is_tensor(step):
            step_value = float(step.detach().flatten()[0].item())
        else:
            step_value = float(step or 0)
        ratio = min(max(step_value / float(warmup_iters), 0.0), 1.0)
        return torch.tensor(init + ratio * (final - init), device=device, dtype=dtype)

    def initial_state(self, batch_size: int, num_objects: int, device, dtype) -> dict[str, torch.Tensor | None]:
        shape = (batch_size, num_objects, self.num_anchors)
        return {
            "affine_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "velocity_state": torch.zeros(*shape, 6, device=device, dtype=dtype),
            "quality": torch.full(shape, 0.5, device=device, dtype=dtype),
            "usage": torch.zeros(shape, device=device, dtype=dtype),
            "age": torch.zeros(shape, device=device, dtype=dtype),
            "prev_query": None,
            "prev_proposal": None,
            "prev_area": None,
        }

    def _trust_floor(self, step, device, dtype) -> torch.Tensor:
        if step is None or self.trust_curriculum_iters <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        if torch.is_tensor(step):
            step_value = float(step.detach().flatten()[0].item())
        else:
            step_value = float(step or 0)
        ratio = min(max(step_value / float(self.trust_curriculum_iters), 0.0), 1.0)
        return torch.tensor(self.trust_min_warmup * (1.0 - ratio), device=device, dtype=dtype)

    def _build_feature_modulation(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        proposal_logits: torch.Tensor,
        query_aux: dict[str, torch.Tensor],
        trust_map: torch.Tensor,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, N = base_logits.shape[:2]
        base_prob = torch.sigmoid(base_logits).mean(dim=1, keepdim=True)
        proposal_prob = torch.sigmoid(proposal_logits).mean(dim=1, keepdim=True)
        uncertainty = query_aux["base_uncertainty_map"].mean(dim=1, keepdim=True)
        trust = trust_map.mean(dim=1, keepdim=True)
        seed = torch.cat([base_prob, proposal_prob, uncertainty, trust], dim=1)
        modulation: dict[str, dict[str, torch.Tensor]] = {}
        anchor_features: dict[str, torch.Tensor] = {}
        modulation_l1: dict[str, torch.Tensor] = {}
        for level, feat in feats.items():
            strength = self.modulation_strengths.get(level, 0.0)
            if not self.enable_feature_modulation or strength <= 0.0:
                shift = torch.zeros_like(feat)
                scale = torch.zeros_like(feat)
            else:
                resized_seed = F.interpolate(seed, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                raw = self.modulation_heads[level](resized_seed)
                scale_raw, shift_raw = raw.chunk(2, dim=1)
                scale = strength * torch.tanh(scale_raw)
                shift = strength * torch.tanh(shift_raw)
            modulation[level] = {"scale": scale, "shift": shift}
            anchor_features[level] = shift.unsqueeze(1).expand(B, N, -1, -1, -1)
            modulation_l1[level] = scale.abs().mean() + shift.abs().mean()
        return modulation, anchor_features, modulation_l1

    def _query(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        prev_query: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, N = base_logits.shape[:2]
        pooled = torch.cat([feats[level].mean(dim=(-2, -1)) for level in ("low", "mid", "high", "dec")], dim=1)
        pooled = pooled.unsqueeze(1).expand(-1, N, -1)
        prob = torch.sigmoid(base_logits)
        area = prob.mean(dim=(-2, -1)).unsqueeze(-1)
        uncertainty = (1.0 - (prob - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        uncertainty_mean = uncertainty.mean(dim=(-2, -1)).unsqueeze(-1)
        uncertainty_max = uncertainty.flatten(-2).amax(dim=-1).unsqueeze(-1)
        grad_y = F.pad((prob[..., 1:, :] - prob[..., :-1, :]).abs(), (0, 0, 0, 1))
        grad_x = F.pad((prob[..., :, 1:] - prob[..., :, :-1]).abs(), (0, 1, 0, 0))
        boundary = 0.5 * (grad_y.mean(dim=(-2, -1)) + grad_x.mean(dim=(-2, -1))).unsqueeze(-1)
        stats = torch.cat(
            [
                area,
                uncertainty_mean,
                uncertainty_max,
                boundary,
                prob.flatten(-2).amax(dim=-1).unsqueeze(-1),
                prob.flatten(-2).amin(dim=-1).unsqueeze(-1),
            ],
            dim=-1,
        )
        if prev_query is None:
            prev_query = torch.zeros(B, N, self.query_dim, device=base_logits.device, dtype=base_logits.dtype)
        query = F.normalize(self.query_net(torch.cat([pooled, stats, prev_query], dim=-1)), dim=-1)
        return query, {
            "query_area": area.detach().squeeze(-1),
            "query_uncertainty": uncertainty_mean.detach().squeeze(-1),
            "query_boundary_strength": boundary.detach().squeeze(-1),
            "base_uncertainty_map": uncertainty,
            "base_boundary_map": (grad_x + grad_y).clamp(0.0, 1.0),
        }

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

    def _warp_anchors(self, affine_state: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        B, N, A = affine_state.shape[:3]
        theta = self._affine_matrix(affine_state).flatten(0, 2)
        H, W = output_size
        grid = F.affine_grid(theta, size=(B * N * A, 1, H, W), align_corners=False)
        anchors = self.canonical_sdf.to(device=affine_state.device, dtype=affine_state.dtype)
        anchors = anchors.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1, -1).flatten(0, 2)
        proposals = F.grid_sample(anchors, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return proposals.view(B, N, A, H, W)

    def forward_step(
        self,
        feats: dict[str, torch.Tensor],
        base_logits: torch.Tensor,
        state: dict[str, torch.Tensor | dict | None],
        *,
        global_step=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor | dict | None]]:
        B, N, H, W = base_logits.shape
        dtype = base_logits.dtype
        device = base_logits.device
        temperature = self._scheduled(
            global_step, self.temperature_init, self.temperature_final, self.temperature_warmup_iters, device, dtype
        )
        residual_scale = self._scheduled(
            global_step, self.residual_scale_init, self.residual_scale_max, self.residual_warmup_iters, device, dtype
        )
        trust_max = self._scheduled(global_step, 0.05, self.trust_max, self.trust_warmup_iters, device, dtype)
        trust_floor = self._trust_floor(global_step, device, dtype)
        ode_dt = self._scheduled(global_step, self.ode_dt_init, self.ode_dt_max, self.ode_warmup_iters, device, dtype)

        query, query_aux = self._query(feats, base_logits, state.get("prev_query"))
        keys = F.normalize(self.anchor_keys.to(device=device, dtype=dtype), dim=-1)
        scores = torch.einsum("bnc,ac->bna", query, keys)
        quality = state["quality"]
        assert torch.is_tensor(quality)
        scores = scores + 0.5 * (quality - 0.5)
        weights = torch.softmax(scores / temperature.clamp_min(1.0e-4), dim=-1)
        entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        effective = entropy.exp()
        top_values = torch.topk(weights, k=min(3, self.num_anchors), dim=-1).values

        anchor_query = keys.view(1, 1, self.num_anchors, self.query_dim).expand(B, N, -1, -1)
        affine_state = state["affine_state"]
        velocity_state = state["velocity_state"]
        assert torch.is_tensor(affine_state) and torch.is_tensor(velocity_state)
        delta_in = torch.cat([query.unsqueeze(2).expand(-1, -1, self.num_anchors, -1), anchor_query, affine_state], dim=-1)
        raw_delta = self.affine_delta_head(delta_in)
        limits = torch.tensor(
            [0.08, 0.08, 0.05, 0.05, math.radians(8.0), 0.05],
            device=device,
            dtype=dtype,
        )
        affine_delta = torch.tanh(raw_delta) * limits

        frame_quality = (1.0 - query_aux["query_uncertainty"]).clamp(0.05, 1.0)
        prev_area = state.get("prev_area")
        area = query_aux["query_area"]
        if torch.is_tensor(prev_area):
            area_motion = (area - prev_area).abs().clamp(0.0, 0.25) * 4.0
        else:
            area_motion = torch.zeros_like(area)
        velocity_consistency = torch.exp(-(affine_delta.detach() - velocity_state.detach()).abs().mean(dim=-1))
        state_competition = (0.45 + 0.35 * area_motion.unsqueeze(-1) + 0.20 * velocity_consistency).clamp(0.05, 1.0)
        write_strength = weights * frame_quality.unsqueeze(-1) * state_competition
        if self.enable_memory_update:
            next_velocity = self.velocity_momentum * velocity_state + (1.0 - self.velocity_momentum) * affine_delta.detach()
            next_affine = affine_state + ode_dt * write_strength.unsqueeze(-1) * next_velocity
        else:
            next_velocity = torch.zeros_like(velocity_state)
            next_affine = affine_state
            write_strength = torch.zeros_like(write_strength)
        next_affine = next_affine.detach()
        next_velocity = next_velocity.detach()

        proposals = self._warp_anchors(affine_state + affine_delta, (H, W))
        proposal_logits = (weights.unsqueeze(-1).unsqueeze(-1) * proposals).sum(dim=2)
        base_prob = torch.sigmoid(base_logits)
        residual_in = torch.cat(
            [
                feats["dec"].unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1),
                base_logits.flatten(0, 1).unsqueeze(1),
                proposal_logits.flatten(0, 1).unsqueeze(1),
                query_aux["base_uncertainty_map"].flatten(0, 1).unsqueeze(1),
                query_aux["base_boundary_map"].flatten(0, 1).unsqueeze(1),
            ],
            dim=1,
        )
        raw_residual = self.residual_head(residual_in).view(B, N, H, W).clamp(-self.residual_clip, self.residual_clip)
        residual = residual_scale * torch.tanh(raw_residual)
        if self.disable_trust_gate:
            trust = torch.ones_like(base_logits)
            gate = torch.ones_like(base_logits)
        else:
            trust_gate = self.trust_gate(residual_in).view(B, N, 2, H, W)
            trust_raw = torch.sigmoid(trust_gate[:, :, 0])
            trust = trust_floor + (trust_max - trust_floor).clamp_min(0.0) * trust_raw
            gate = torch.sigmoid(trust_gate[:, :, 1])
        safety_residual = gate * trust * residual
        final_logits = base_logits + safety_residual
        feature_modulation, anchor_features, modulation_l1 = self._build_feature_modulation(
            feats, base_logits, proposal_logits, query_aux, trust
        )

        sim_coverage = (weights * scores.sigmoid()).sum(dim=-1)
        anchor_area = torch.sigmoid(proposals).mean(dim=(-2, -1))
        anchor_area_means = anchor_area.mean(dim=(0, 1))
        if self.num_anchors > 1:
            anchor_area_separation = anchor_area_means.max() - anchor_area_means.min()
        else:
            anchor_area_separation = torch.zeros((), device=device, dtype=dtype)
        function_vectors = self.canonical_sdf.flatten(1)
        function_vectors = F.normalize(function_vectors, dim=-1)
        pairwise = torch.matmul(function_vectors, function_vectors.transpose(0, 1))
        eye = torch.eye(self.num_anchors, device=device, dtype=torch.bool)
        pairwise_similarity = pairwise.masked_select(~eye).mean() if self.num_anchors > 1 else torch.zeros((), device=device, dtype=dtype)
        update_norm = (ode_dt * write_strength.unsqueeze(-1) * next_velocity).detach().pow(2).mean(dim=-1).sqrt()
        next_quality = (self.memory_ema * quality + (1.0 - self.memory_ema) * frame_quality.unsqueeze(-1) * weights).detach().clamp(0.0, 1.0)
        usage = state["usage"]
        age = state["age"]
        assert torch.is_tensor(usage) and torch.is_tensor(age)

        aux = {
            "query": query,
            "function_codes": self.canonical_sdf.flatten(1).view(1, 1, self.num_anchors, -1).expand(B, N, -1, -1),
            "active_weights": weights,
            "anchor_proposals": proposals,
            "anchor_logits": proposal_logits,
            "proposal_logits": proposal_logits,
            "aggregated_sdf": proposal_logits,
            "aggregated_residual": residual,
            "residual_logits": residual,
            "safety_residual_logits": safety_residual,
            "trust": trust,
            "gate": gate,
            "anchor_trust_map": trust,
            "feature_modulation": feature_modulation,
            "anchor_features": anchor_features,
            "coverage_score": sim_coverage,
            "coverage_gap": (1.0 - sim_coverage).clamp_min(0.0),
            "active_anchor_entropy": entropy,
            "active_anchor_entropy_norm": entropy / math.log(max(self.num_anchors, 2)),
            "effective_anchor_number": effective,
            "top1_anchor_weight": top_values[..., 0].detach(),
            "top3_anchor_weight_sum": top_values.sum(dim=-1).detach(),
            "anchor_area": anchor_area,
            "proposal_area_std": anchor_area.std(dim=-1, unbiased=False),
            "proposal_area_range": anchor_area.amax(dim=-1) - anchor_area.amin(dim=-1),
            "anchor_function_diversity": 1.0 - pairwise_similarity,
            "anchor_area_diversity": anchor_area.std(dim=-1, unbiased=False).mean(),
            "anchor_area_separation": anchor_area_separation,
            "anchor_pairwise_similarity": pairwise_similarity,
            "anchor_phase_purity_proxy": top_values[..., 0].detach(),
            "affine_delta": affine_delta,
            "affine_state": affine_state.detach(),
            "affine_delta_norm": affine_delta.pow(2).mean(dim=-1).sqrt().mean(),
            "affine_velocity_norm": next_velocity.detach().pow(2).mean(dim=-1).sqrt().mean(),
            "ode_velocity_norm": next_velocity.detach().pow(2).mean(dim=-1).sqrt().mean(),
            "write_strength": write_strength.detach(),
            "write_strength_mean": write_strength.detach().mean(),
            "write_strength_std": write_strength.detach().std(unbiased=False),
            "memory_update_norm": update_norm.mean(),
            "quality_weighted_write": (write_strength.detach() * frame_quality.detach().unsqueeze(-1)).mean(),
            "coverage_triggered_write_ratio": torch.zeros((), device=device, dtype=dtype),
            "recycled_anchor_ratio": torch.zeros((), device=device, dtype=dtype),
            "dead_anchor_ratio": ((usage + weights.detach()) < 1.0e-3).float().mean(),
            "trust_mean": trust.detach().mean(),
            "trust_std": trust.detach().std(unbiased=False),
            "trust_floor": trust_floor.detach(),
            "gate_mean": gate.detach().mean(),
            "anchor_trust_ratio": trust.detach().mean(),
            "image_trust_ratio": (1.0 - trust.detach()).mean(),
            "trust_easy_mean": trust.detach().masked_select((query_aux["base_uncertainty_map"].detach() < 0.35)).mean()
            if (query_aux["base_uncertainty_map"].detach() < 0.35).any()
            else trust.detach().mean(),
            "trust_hard_mean": trust.detach().masked_select((query_aux["base_uncertainty_map"].detach() >= 0.35)).mean()
            if (query_aux["base_uncertainty_map"].detach() >= 0.35).any()
            else trust.detach().mean(),
            "residual_l1": residual.detach().abs().mean(),
            "residual_l2": residual.detach().pow(2).mean().sqrt(),
            "safety_residual_l1": safety_residual.detach().abs().mean(),
            "residual_clip_hit_ratio": (raw_residual.detach().abs() >= self.residual_clip * 0.99).float().mean(),
            "residual_scale": residual_scale.detach(),
            "trust_max": trust_max.detach(),
            "retrieval_temperature": temperature.detach(),
            "ode_dt": ode_dt.detach(),
            "feature_modulation_l1": torch.stack(list(modulation_l1.values())).mean(),
            "feature_modulation_l1_low": modulation_l1["low"],
            "feature_modulation_l1_mid": modulation_l1["mid"],
            "feature_modulation_l1_high": modulation_l1["high"],
            "feature_modulation_l1_dec": modulation_l1["dec"],
            "memory_stats": {
                "quality": next_quality,
                "usage": (usage + weights.detach()).detach(),
                "age": (age + 1.0).detach(),
                "affine_state": next_affine,
                "velocity_state": next_velocity,
            },
        }
        for key in ("query_area", "query_uncertainty", "query_boundary_strength"):
            aux[key] = query_aux[key]
        next_state = {
            "affine_state": next_affine,
            "velocity_state": next_velocity,
            "quality": next_quality,
            "usage": (usage + weights.detach()).detach(),
            "age": (age + 1.0).detach(),
            "prev_query": query.detach(),
            "prev_proposal": proposal_logits.detach(),
            "prev_area": area.detach(),
        }
        return final_logits, aux, next_state
