from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


LEVELS = ("low", "mid", "high", "dec")


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _guidance_in_channels(mode: str) -> int:
    if mode == "warp_delta_boundary":
        return 3
    if mode == "warp_delta":
        return 2
    if mode == "warp_only":
        return 1
    raise ValueError(f"Unsupported guidance_input_mode: {mode}")


def mask_geometry_stats(mask_BNHW: torch.Tensor, prev_stats: torch.Tensor | None = None) -> torch.Tensor:
    """Soft LV geometry: area, center, spread, ratio, compactness, entropy, velocity."""
    B, N, H, W = mask_BNHW.shape
    mask = mask_BNHW.float().clamp(0.0, 1.0)
    area = mask.mean(dim=(-2, -1))
    ys = torch.linspace(0.0, 1.0, H, device=mask.device, dtype=mask.dtype).view(1, 1, H, 1)
    xs = torch.linspace(0.0, 1.0, W, device=mask.device, dtype=mask.dtype).view(1, 1, 1, W)
    mass = mask.sum(dim=(-2, -1)).clamp_min(1.0e-6)
    cx = (mask * xs).sum(dim=(-2, -1)) / mass
    cy = (mask * ys).sum(dim=(-2, -1)) / mass
    width = torch.sqrt(((xs - cx[..., None, None]) ** 2 * mask).sum(dim=(-2, -1)) / mass + 1.0e-6)
    height = torch.sqrt(((ys - cy[..., None, None]) ** 2 * mask).sum(dim=(-2, -1)) / mass + 1.0e-6)
    ratio = width / height.clamp_min(1.0e-6)
    compactness = area / (width * height * 4.0).clamp_min(1.0e-6)
    entropy = -(
        mask.clamp(1.0e-6, 1.0 - 1.0e-6) * mask.clamp(1.0e-6, 1.0 - 1.0e-6).log()
        + (1.0 - mask).clamp(1.0e-6, 1.0) * (1.0 - mask).clamp(1.0e-6, 1.0).log()
    ).mean(dim=(-2, -1))
    base = torch.stack([area, cx, cy, width, height, ratio, compactness, entropy], dim=-1)
    if prev_stats is None:
        velocity = torch.zeros(B, N, 3, device=mask.device, dtype=mask.dtype)
    else:
        velocity = torch.stack(
            [
                area - prev_stats[..., 0],
                cx - prev_stats[..., 1],
                cy - prev_stats[..., 2],
            ],
            dim=-1,
        )
    return torch.cat([base, velocity], dim=-1)


class CardiacStateEncoder(nn.Module):
    def __init__(self, feature_dims: dict[str, int], num_slots: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.stats_dim = 11
        self.prev_dim = self.stats_dim + self.num_slots + 4 + 24 + 6
        in_dim = sum(feature_dims.values()) + self.stats_dim + self.prev_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        base_mask: torch.Tensor,
        prev: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = base_mask.shape[:2]
        stats = mask_geometry_stats(base_mask, prev.get("geometry"))
        pooled = []
        for level in LEVELS:
            f = feats[level]
            pooled_f = f.mean(dim=(-2, -1)).unsqueeze(1).expand(-1, N, -1)
            pooled.append(pooled_f)
        prev_vec = torch.cat(
            [
                prev["geometry"],
                prev["slot_weights"],
                prev["confidence"],
                prev["affine"].flatten(start_dim=-2),
                prev["geometry_offset"],
            ],
            dim=-1,
        )
        return self.net(torch.cat([*pooled, stats, prev_vec], dim=-1)), stats


class SkipAwareAnchorBank(nn.Module):
    def __init__(self, num_slots: int, anchor_size: int, state_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.anchor_size = int(anchor_size)
        self.anchors = nn.ParameterDict(
            {
                level: nn.Parameter(torch.randn(num_slots, 1, anchor_size, anchor_size) * 0.02)
                for level in LEVELS
            }
        )
        self.motion_embed = nn.Parameter(torch.randn(num_slots, state_dim) * 0.02)
        self.condition = nn.Parameter(torch.randn(num_slots, condition_dim) * 0.02)
        self.affine_prior = nn.Parameter(torch.zeros(num_slots, len(LEVELS), 6))
        self.geometry_prior = nn.Parameter(torch.zeros(num_slots, 6))
        self.selector = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_slots),
        )

    def forward(self, state: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_score = torch.einsum("bnc,kc->bnk", state, self.motion_embed) / math.sqrt(max(state.shape[-1], 1))
        weights = torch.softmax(self.selector(state) + motion_score, dim=-1)
        anchors = {
            level: torch.einsum("bnk,kchw->bnchw", weights, self.anchors[level])
            for level in LEVELS
        }
        condition = torch.einsum("bnk,kc->bnc", weights, self.condition)
        affine_prior = torch.einsum("bnk,klp->bnlp", weights, self.affine_prior)
        geometry_prior = torch.einsum("bnk,kc->bnc", weights, self.geometry_prior)
        slot_confidence = (weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1).neg()
        slot_confidence = 1.0 - slot_confidence / math.log(max(self.num_slots, 2))
        return anchors, weights, condition, affine_prior, geometry_prior + 0.01 * slot_confidence.unsqueeze(-1)


class AffineOffsetRegressor(nn.Module):
    def __init__(self, state_dim: int, num_levels: int) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_levels * 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        raw = self.net(state).view(*state.shape[:2], self.num_levels, 6) + prior
        out = torch.empty_like(raw)
        out[..., 0:2] = 0.30 * torch.tanh(raw[..., 0:2])
        out[..., 2:4] = 1.0 + 0.20 * torch.tanh(raw[..., 2:4])
        out[..., 4:6] = 0.25 * torch.tanh(raw[..., 4:6])
        return out


class ODEResidualRefiner(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int, steps: int, gamma: float) -> None:
        super().__init__()
        self.steps = int(steps)
        self.gamma = float(gamma)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prior_logits: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        B, N, H, W = prior_logits.shape
        z = prior_logits.flatten(0, 1).unsqueeze(1)
        cond = self.cond_proj(condition.flatten(0, 1)).view(B * N, -1, 1, 1).expand(-1, -1, H, W)
        for _ in range(max(self.steps, 1)):
            z = z + self.gamma * torch.tanh(self.net(torch.cat([z, cond], dim=1)))
        return z.view(B, N, H, W)


class ConfidenceAssignment(nn.Module):
    def __init__(
        self,
        state_dim: int,
        *,
        prior_bias: float | None = None,
        base_bias: float | None = None,
        update_bias: float | None = None,
        boundary_bias: float | None = None,
        scale_bias: float | None = None,
        slot_bias: float | None = None,
        default_bias: float = -1.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 6, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, 9),
        )
        nn.init.constant_(self.net[-1].bias, float(default_bias))
        if any(v is not None for v in (prior_bias, base_bias, update_bias, boundary_bias, scale_bias, slot_bias)):
            bias = self.net[-1].bias
            with torch.no_grad():
                bias[0] = float(default_bias if prior_bias is None else prior_bias)
                bias[1] = float(default_bias if base_bias is None else base_bias)
                bias[2] = float(default_bias if update_bias is None else update_bias)
                bias[3] = float(default_bias if boundary_bias is None else boundary_bias)
                bias[4:8].fill_(float(default_bias if scale_bias is None else scale_bias))
                bias[8] = float(default_bias if slot_bias is None else slot_bias)

    def forward(
        self,
        state: torch.Tensor,
        base_prob: torch.Tensor,
        prior_prob: torch.Tensor,
        geom_offset: torch.Tensor,
        affine: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        entropy = -(
            base_prob.clamp(1.0e-6, 1.0 - 1.0e-6) * base_prob.clamp(1.0e-6, 1.0 - 1.0e-6).log()
            + (1.0 - base_prob).clamp(1.0e-6, 1.0) * (1.0 - base_prob).clamp(1.0e-6, 1.0).log()
        ).mean(dim=(-2, -1))
        disagreement = (base_prob - prior_prob).abs().mean(dim=(-2, -1))
        affine_mag = (affine[..., 0:2].abs().mean(dim=(-1, -2)) + (affine[..., 2:4] - 1.0).abs().mean(dim=(-1, -2)))
        geom_mag = geom_offset.abs().mean(dim=-1)
        stats = torch.stack([entropy, disagreement, affine_mag, geom_mag, base_prob.mean(dim=(-2, -1)), prior_prob.mean(dim=(-2, -1))], dim=-1)
        conf = torch.sigmoid(self.net(torch.cat([state, stats], dim=-1)))
        return {
            "prior": conf[..., 0],
            "base": conf[..., 1],
            "update": conf[..., 2],
            "boundary": conf[..., 3],
            "scale": conf[..., 4:8],
            "slot": conf[..., 8],
            "entropy": entropy,
            "disagreement": disagreement,
        }


class UNeXtAnchorODESegmenter(nn.Module):
    """UNeXt base segmenter with skip-aware Anchor-ODE temporal priors."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "anchor_ode", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_slots = int(_cfg_get(method_cfg, "num_slots", 8))
        self.state_dim = int(_cfg_get(method_cfg, "state_dim", 128))
        self.hidden_dim = int(_cfg_get(method_cfg, "hidden_dim", 128))
        self.anchor_size = int(_cfg_get(method_cfg, "anchor_size", 32))
        self.condition_dim = int(_cfg_get(method_cfg, "condition_dim", 64))
        self.ode_steps = int(_cfg_get(method_cfg, "ode_steps", 1))
        self.ode_gamma = float(_cfg_get(method_cfg, "ode_gamma", 0.1))
        self.prior_residual_clip = float(_cfg_get(method_cfg, "prior_residual_clip", 3.0))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        feature_dims = {
            "low": self.base_dim,
            "mid": self.base_dim * 2,
            "high": self.base_dim * 4,
            "dec": self.base_dim,
        }
        self.state_encoder = CardiacStateEncoder(feature_dims, self.num_slots, self.state_dim, self.hidden_dim)
        self.anchor_bank = SkipAwareAnchorBank(self.num_slots, self.anchor_size, self.state_dim, self.condition_dim)
        self.affine_regressor = AffineOffsetRegressor(self.state_dim, len(LEVELS))
        self.geometry_regressor = nn.Sequential(
            nn.Linear(self.state_dim + 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 6),
        )
        self.prior_projs = nn.ModuleDict({level: nn.Conv2d(1, feature_dims[level], kernel_size=1) for level in LEVELS})
        self.prior_head = nn.Sequential(
            nn.Conv2d(sum(feature_dims.values()) + 1, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        self.ode_refiner = ODEResidualRefiner(self.condition_dim, self.hidden_dim, self.ode_steps, self.ode_gamma)
        self.confidence = ConfidenceAssignment(self.state_dim)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _empty_prev(self, batch_size: int, num_objects: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "geometry": torch.zeros(batch_size, num_objects, 11, device=device, dtype=dtype),
            "slot_weights": torch.full((batch_size, num_objects, self.num_slots), 1.0 / self.num_slots, device=device, dtype=dtype),
            "confidence": torch.zeros(batch_size, num_objects, 4, device=device, dtype=dtype),
            "affine": torch.zeros(batch_size, num_objects, len(LEVELS), 6, device=device, dtype=dtype),
            "geometry_offset": torch.zeros(batch_size, num_objects, 6, device=device, dtype=dtype),
        }

    def _affine_matrix(self, params: torch.Tensor) -> torch.Tensor:
        tx, ty, sx, sy, rot, shear = params.unbind(dim=-1)
        cos = torch.cos(rot)
        sin = torch.sin(rot)
        a00 = sx * cos + shear * sin
        a01 = -sy * sin + shear * cos
        a10 = sx * sin
        a11 = sy * cos
        return torch.stack([a00, a01, tx, a10, a11, ty], dim=-1).view(-1, 2, 3)

    def _warp_anchor(self, anchor_BN1HW: torch.Tensor, affine_BN6: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        B, N = anchor_BN1HW.shape[:2]
        src = F.interpolate(anchor_BN1HW.flatten(0, 1), size=size, mode="bilinear", align_corners=False)
        grid = F.affine_grid(self._affine_matrix(affine_BN6), size=(B * N, 1, *size), align_corners=False)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False).view(B, N, *size)

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def _anchor_step(
        self,
        feat: dict[str, torch.Tensor],
        base_object_logits: torch.Tensor,
        prev: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = {"low": feat["low"], "mid": feat["mid"], "high": feat["high"], "dec": feat["decoder_feature"]}
        base_prob = torch.sigmoid(base_object_logits)
        state, stats = self.state_encoder(features, base_prob.detach(), prev)
        anchors, slot_weights, condition, affine_prior, geometry_prior = self.anchor_bank(state)
        affine = self.affine_regressor(state, affine_prior)
        geometry_offset = self.geometry_regressor(torch.cat([state, geometry_prior], dim=-1))
        geometry_pred = stats[..., :6] + geometry_offset

        warped = {}
        guided = []
        conf_seed = None
        for idx, level in enumerate(LEVELS):
            target_size = features[level].shape[-2:]
            warped[level] = self._warp_anchor(anchors[level], affine[..., idx, :], target_size)
            prior_feat = self.prior_projs[level](warped[level].flatten(0, 1).unsqueeze(1))
            B, N = base_object_logits.shape[:2]
            prior_feat = prior_feat.view(B, N, -1, *target_size).mean(dim=1)
            guided.append(F.interpolate(prior_feat, size=base_object_logits.shape[-2:], mode="bilinear", align_corners=False))
            if level == "dec":
                conf_seed = F.interpolate(warped[level].flatten(0, 1).unsqueeze(1), size=base_object_logits.shape[-2:], mode="bilinear", align_corners=False).view_as(base_object_logits)

        prior_in = torch.cat([*guided, base_prob.mean(dim=1, keepdim=True)], dim=1)
        prior_residual = self.prior_head(prior_in).expand(-1, base_object_logits.shape[1], -1, -1)
        prior_residual = prior_residual.clamp(min=-self.prior_residual_clip, max=self.prior_residual_clip)
        prior_logits = self.ode_refiner(conf_seed + prior_residual, condition)
        prior_prob = torch.sigmoid(prior_logits)
        conf = self.confidence(state, base_prob, prior_prob, geometry_offset, affine)
        object_logits = base_object_logits + conf["prior"].unsqueeze(-1).unsqueeze(-1) * (prior_logits - base_object_logits)

        next_prev = {
            "geometry": mask_geometry_stats(torch.sigmoid(object_logits), stats).detach(),
            "slot_weights": slot_weights.detach(),
            "confidence": torch.stack([conf["prior"], conf["base"], conf["update"], conf["boundary"]], dim=-1).detach(),
            "affine": affine.detach(),
            "geometry_offset": geometry_offset.detach(),
        }
        aux = {
            "method": "anchor_ode",
            "memory_type": "anchor_ode",
            "anchor_ode_aux": {
                "base_object_logits": base_object_logits,
                "prior_logits": prior_logits,
                "final_object_logits": object_logits,
                "slot_weights": slot_weights,
                "affine": affine,
                "affine_low": affine[..., 0, :],
                "affine_mid": affine[..., 1, :],
                "affine_high": affine[..., 2, :],
                "affine_dec": affine[..., 3, :],
                "warped_priors": warped,
                "geometry_pred": geometry_pred,
                "geometry_offset": geometry_offset,
                "confidence": torch.stack([conf["prior"], conf["base"], conf["update"], conf["boundary"]], dim=-1),
                "confidence_prior": conf["prior"],
                "confidence_base": conf["base"],
                "confidence_update": conf["update"],
                "confidence_boundary": conf["boundary"],
                "confidence_scale": conf["scale"],
                "base_prior_disagreement": conf["disagreement"],
                "mask_entropy": conf["entropy"],
                "dynamics_monitor": {
                    "area": next_prev["geometry"][..., 0],
                    "slot_usage": slot_weights.detach().mean(dim=(0, 1)),
                    "affine_abs_mean": affine.detach().abs().mean(dim=(-1, -2)),
                    "confidence_prior": conf["prior"].detach(),
                },
            },
        }
        return object_logits, prior_logits, next_prev, aux

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        prev = self._empty_prev(B, max_num_objects, images.device, images.dtype)
        out: Dict = {"num_objects": num_objects}

        for ti in range(T):
            feat = self.backbone(self._normalize(images[:, ti]))
            base_object_logits = feat["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)
            object_logits, prior_logits, prev, memory_aux = self._anchor_step(feat, base_object_logits, prev)
            logits, masks = self._object_logits_to_full(object_logits)
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": object_logits.detach(),
                "prior_logits": prior_logits.detach(),
            }
            out[f"memory_aux_{ti}"] = memory_aux
        return out


class CurrentAnchorStateEncoder(nn.Module):
    def __init__(self, feature_dims: dict[str, int], num_slots: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.stats_dim = 11
        self.prev_dim = self.stats_dim * 2 + self.num_slots + 4 + len(LEVELS) * 6 + 6 + 1
        in_dim = sum(feature_dims.values()) + self.stats_dim + self.prev_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        base_mask: torch.Tensor,
        prev: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = base_mask.shape[:2]
        valid = prev.get("valid", torch.ones(B, N, 1, device=base_mask.device, dtype=base_mask.dtype))
        bootstrap_geometry = mask_geometry_stats(base_mask, None)
        prev_base_geometry = valid * prev["base_geometry"] + (1.0 - valid) * bootstrap_geometry.detach()
        prev_final_geometry = valid * prev["final_geometry"] + (1.0 - valid) * bootstrap_geometry.detach()
        bootstrap_confidence = torch.tensor([0.25, 0.75, 0.4, 0.4], device=base_mask.device, dtype=base_mask.dtype).view(1, 1, 4)
        prev_confidence = valid * prev["confidence"] + (1.0 - valid) * bootstrap_confidence
        base_geometry = mask_geometry_stats(base_mask, prev_base_geometry)
        pooled = []
        for level in LEVELS:
            f = feats[level]
            pooled.append(f.mean(dim=(-2, -1)).unsqueeze(1).expand(-1, N, -1))
        prev_vec = torch.cat(
            [
                prev_base_geometry,
                prev_final_geometry,
                prev["slot_weights"],
                prev_confidence,
                prev["affine"].flatten(start_dim=-2),
                prev["geometry_delta"],
                prev["valid"],
            ],
            dim=-1,
        )
        return self.net(torch.cat([*pooled, base_geometry, prev_vec], dim=-1)), base_geometry


class TemporalAffineODEBank(nn.Module):
    def __init__(self, num_slots: int, state_dim: int, condition_dim: int, num_levels: int) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.num_levels = int(num_levels)
        self.condition = nn.Parameter(torch.randn(num_slots, condition_dim) * 0.02)
        self.motion_embed = nn.Parameter(torch.randn(num_slots, state_dim) * 0.02)
        self.affine_velocity = nn.Parameter(torch.zeros(num_slots, num_levels, 6))
        self.geometry_delta = nn.Parameter(torch.zeros(num_slots, 6))
        self.selector = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_slots),
        )
        nn.init.normal_(self.selector[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.selector[-1].bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_score = torch.einsum("bnc,kc->bnk", state, self.motion_embed) / math.sqrt(max(state.shape[-1], 1))
        weights = torch.softmax(self.selector(state) + motion_score, dim=-1)
        condition = torch.einsum("bnk,kc->bnc", weights, self.condition)
        selected_motion_embed = torch.einsum("bnk,kc->bnc", weights, self.motion_embed)
        affine_velocity = torch.einsum("bnk,klp->bnlp", weights, self.affine_velocity)
        geometry_delta = torch.einsum("bnk,kc->bnc", weights, self.geometry_delta)
        entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
        slot_confidence = 1.0 - entropy / math.log(max(self.num_slots, 2))
        return weights, condition, selected_motion_embed, affine_velocity, geometry_delta, slot_confidence


class ODEAffineMicroRegressor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        num_levels: int,
        max_translate: float,
        max_scale: float,
        max_rotate: float,
        steps: int,
        gamma: float,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.max_translate = float(max_translate)
        self.max_scale = float(max_scale)
        self.max_rotate = float(max_rotate)
        self.steps = int(steps)
        self.gamma = float(gamma)
        self.net = nn.Sequential(
            nn.Linear(state_dim + condition_dim + num_levels * 6, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, num_levels * 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, condition: torch.Tensor, prev_affine: torch.Tensor, slot_velocity: torch.Tensor) -> torch.Tensor:
        raw = slot_velocity
        for _ in range(max(self.steps, 1)):
            ode_in = torch.cat([state, condition, prev_affine.flatten(start_dim=-2)], dim=-1)
            raw = raw + self.gamma * self.net(ode_in).view(*state.shape[:2], self.num_levels, 6)
        out = torch.empty_like(raw)
        out[..., 0:2] = self.max_translate * torch.tanh(raw[..., 0:2])
        out[..., 2:4] = 1.0 + self.max_scale * torch.tanh(raw[..., 2:4])
        out[..., 4:6] = self.max_rotate * torch.tanh(raw[..., 4:6])
        return out


class UNeXtAnchorODEAffineSegmenter(nn.Module):
    """UNeXt current-mask anchors with temporal ODE affine micro-adjustments."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "anchor_ode", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_slots = int(_cfg_get(method_cfg, "num_slots", 8))
        self.state_dim = int(_cfg_get(method_cfg, "state_dim", 128))
        self.hidden_dim = int(_cfg_get(method_cfg, "hidden_dim", 128))
        self.condition_dim = int(_cfg_get(method_cfg, "condition_dim", 64))
        self.ode_steps = int(_cfg_get(method_cfg, "ode_steps", 1))
        self.ode_gamma = float(_cfg_get(method_cfg, "ode_gamma", 0.1))
        self.prior_residual_clip = float(_cfg_get(method_cfg, "prior_residual_clip", 2.0))
        self.gate_warmup_iters = int(_cfg_get(method_cfg, "gate_warmup_iters", 500))
        self.gate_init_bias = float(_cfg_get(method_cfg, "gate_init_bias", -4.0))
        self.guidance_input_mode = str(_cfg_get(method_cfg, "guidance_input_mode", "warp_delta_boundary"))
        self.guidance_fusion_mode = str(_cfg_get(method_cfg, "guidance_fusion_mode", "residual_add"))
        self.decoder_guidance_enabled = bool(_cfg_get(method_cfg, "decoder_guidance_enabled", True))
        self.skip_guidance_levels = tuple(_as_list(_cfg_get(method_cfg, "skip_guidance_levels", LEVELS), LEVELS))
        self.guidance_proj_hidden_dim = int(_cfg_get(method_cfg, "guidance_proj_hidden_dim", 0))
        if self.guidance_fusion_mode not in {"residual_add", "residual_concat"}:
            raise ValueError(f"Unsupported guidance_fusion_mode: {self.guidance_fusion_mode}")
        for level in self.skip_guidance_levels:
            if level not in LEVELS:
                raise ValueError(f"Unsupported skip guidance level: {level}")

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        feature_dims = {
            "low": self.base_dim,
            "mid": self.base_dim * 2,
            "high": self.base_dim * 4,
            "dec": self.base_dim,
        }
        self.state_encoder = CurrentAnchorStateEncoder(feature_dims, self.num_slots, self.state_dim, self.hidden_dim)
        self.ode_bank = TemporalAffineODEBank(self.num_slots, self.state_dim, self.condition_dim, len(LEVELS))
        self.affine_regressor = ODEAffineMicroRegressor(
            self.state_dim,
            self.condition_dim,
            len(LEVELS),
            max_translate=float(_cfg_get(method_cfg, "affine_max_translate", 0.12)),
            max_scale=float(_cfg_get(method_cfg, "affine_max_scale", 0.10)),
            max_rotate=float(_cfg_get(method_cfg, "affine_max_rotate", 0.15)),
            steps=self.ode_steps,
            gamma=self.ode_gamma,
        )
        self.geometry_regressor = nn.Sequential(
            nn.Linear(self.state_dim + 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 6),
        )
        nn.init.zeros_(self.geometry_regressor[-1].weight)
        nn.init.zeros_(self.geometry_regressor[-1].bias)
        guidance_channels = _guidance_in_channels(self.guidance_input_mode)
        self.guidance_projs = nn.ModuleDict()
        self.guidance_concat_projs = nn.ModuleDict()
        for level in LEVELS:
            if self.guidance_proj_hidden_dim > 0:
                self.guidance_projs[level] = nn.Sequential(
                    nn.Conv2d(guidance_channels, self.guidance_proj_hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(self.guidance_proj_hidden_dim, feature_dims[level], kernel_size=1),
                )
            else:
                self.guidance_projs[level] = nn.Conv2d(guidance_channels, feature_dims[level], kernel_size=1)
            if self.guidance_fusion_mode == "residual_concat":
                self.guidance_concat_projs[level] = nn.Sequential(
                    nn.Conv2d(feature_dims[level] * 2, feature_dims[level], kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(feature_dims[level], feature_dims[level], kernel_size=1),
                )
        self.gate_head = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(LEVELS)),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, self.gate_init_bias)
        self.confidence = ConfidenceAssignment(
            self.state_dim,
            prior_bias=_cfg_get(method_cfg, "confidence_prior_bias", None),
            base_bias=_cfg_get(method_cfg, "confidence_base_bias", None),
            update_bias=_cfg_get(method_cfg, "confidence_update_bias", None),
            boundary_bias=_cfg_get(method_cfg, "confidence_boundary_bias", None),
            scale_bias=_cfg_get(method_cfg, "confidence_scale_bias", None),
            slot_bias=_cfg_get(method_cfg, "confidence_slot_bias", None),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _empty_prev(self, batch_size: int, num_objects: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        affine = torch.zeros(batch_size, num_objects, len(LEVELS), 6, device=device, dtype=dtype)
        affine[..., 2:4] = 1.0
        return {
            "base_geometry": torch.zeros(batch_size, num_objects, 11, device=device, dtype=dtype),
            "final_geometry": torch.zeros(batch_size, num_objects, 11, device=device, dtype=dtype),
            "slot_weights": torch.full((batch_size, num_objects, self.num_slots), 1.0 / self.num_slots, device=device, dtype=dtype),
            "confidence": torch.zeros(batch_size, num_objects, 4, device=device, dtype=dtype),
            "affine": affine,
            "geometry_delta": torch.zeros(batch_size, num_objects, 6, device=device, dtype=dtype),
            "valid": torch.zeros(batch_size, num_objects, 1, device=device, dtype=dtype),
        }

    def _affine_matrix(self, params: torch.Tensor) -> torch.Tensor:
        tx, ty, sx, sy, rot, shear = params.unbind(dim=-1)
        cos = torch.cos(rot)
        sin = torch.sin(rot)
        a00 = sx * cos + shear * sin
        a01 = -sy * sin + shear * cos
        a10 = sx * sin
        a11 = sy * cos
        return torch.stack([a00, a01, tx, a10, a11, ty], dim=-1).view(-1, 2, 3)

    def _warp_anchor(self, anchor_BNHW: torch.Tensor, affine_BN6: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        B, N = anchor_BNHW.shape[:2]
        src = F.interpolate(anchor_BNHW.flatten(0, 1).unsqueeze(1), size=size, mode="bilinear", align_corners=False)
        grid = F.affine_grid(self._affine_matrix(affine_BN6), size=(B * N, 1, *size), align_corners=False)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False).view(B, N, *size)

    def _boundary(self, prob_BNHW: torch.Tensor) -> torch.Tensor:
        padded = F.pad(prob_BNHW.flatten(0, 1).unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        dx = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]).abs()
        dy = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]).abs()
        return (dx + dy).view_as(prob_BNHW)

    def _logit(self, prob: torch.Tensor) -> torch.Tensor:
        return torch.logit(prob.clamp(1.0e-4, 1.0 - 1.0e-4))

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def _fuse_logits(self, base_object_logits: torch.Tensor, guided_logits: torch.Tensor, prior_confidence: torch.Tensor) -> torch.Tensor:
        residual = (guided_logits - base_object_logits).clamp(min=-self.prior_residual_clip, max=self.prior_residual_clip)
        return base_object_logits + prior_confidence.unsqueeze(-1).unsqueeze(-1) * residual

    def _warmup_scale(self, data: Dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.training or self.gate_warmup_iters <= 0:
            return torch.ones((), device=device, dtype=dtype)
        current_iter = int(data.get("current_iter", self.gate_warmup_iters))
        return torch.tensor(min(max(current_iter / max(self.gate_warmup_iters, 1), 0.0), 1.0), device=device, dtype=dtype)

    def _build_guidance(
        self,
        warped_level: torch.Tensor,
        anchor_level: torch.Tensor,
        boundary_scale: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [warped_level]
        if self.guidance_input_mode in {"warp_delta_boundary", "warp_delta"}:
            pieces.append(warped_level - anchor_level)
        if self.guidance_input_mode == "warp_delta_boundary":
            pieces.append(self._boundary(warped_level) * boundary_scale.unsqueeze(-1).unsqueeze(-1))
        return torch.stack(pieces, dim=2).flatten(0, 1)

    def _fuse_guided_feature(
        self,
        level: str,
        feature: torch.Tensor,
        guidance: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if level not in self.skip_guidance_levels:
            return feature
        gated = gate * guidance
        if self.guidance_fusion_mode == "residual_add":
            return feature + gated.mean(dim=1)
        concat_guidance = gated.mean(dim=1)
        return self.guidance_concat_projs[level](torch.cat([feature, concat_guidance], dim=1))

    def _gated_history_update(
        self,
        prev: dict[str, torch.Tensor],
        candidate: dict[str, torch.Tensor],
        update_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in candidate.items():
            cand = value.detach()
            if key == "valid":
                out[key] = cand
                continue
            update_gate = torch.where(prev["valid"].squeeze(-1) > 0.0, update_confidence.detach(), torch.ones_like(update_confidence))
            gate = update_gate.unsqueeze(-1)
            prev_value = prev[key]
            while gate.dim() < cand.dim():
                gate = gate.unsqueeze(-1)
            out[key] = (1.0 - gate) * prev_value + gate * cand
        return out

    def _anchor_step(
        self,
        feat: dict[str, torch.Tensor],
        base_object_logits: torch.Tensor,
        prev: dict[str, torch.Tensor],
        data: Dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = {"low": feat["low"], "mid": feat["mid"], "high": feat["high"], "dec": feat["decoder_feature"]}
        base_prob = torch.sigmoid(base_object_logits)
        state, base_geometry = self.state_encoder(features, base_prob.detach(), prev)
        slot_weights, condition, selected_motion_embed, affine_velocity, geometry_delta_prior, slot_confidence = self.ode_bank(state)
        affine = self.affine_regressor(state, condition, prev["affine"], affine_velocity)
        geometry_delta = geometry_delta_prior + self.geometry_regressor(torch.cat([state, geometry_delta_prior], dim=-1))
        history_geometry = prev["final_geometry"][..., :6] + geometry_delta
        base_geometry_delta = base_geometry[..., :6] + geometry_delta
        geometry_pred = prev["valid"] * history_geometry + (1.0 - prev["valid"]) * base_geometry_delta

        anchors = {}
        warped = {}
        for idx, level in enumerate(LEVELS):
            target_size = features[level].shape[-2:]
            anchors[level] = F.interpolate(base_prob.flatten(0, 1).unsqueeze(1), size=target_size, mode="bilinear", align_corners=False).view(
                *base_prob.shape[:2], *target_size
            )
            warped[level] = self._warp_anchor(anchors[level], affine[..., idx, :], target_size)

        prior_prob = warped["dec"].clamp(1.0e-4, 1.0 - 1.0e-4)
        conf = self.confidence(state, base_prob, prior_prob, geometry_delta, affine)
        effective_slot_confidence = 0.5 * (conf["slot"] + slot_confidence)
        effective_prior_confidence = conf["prior"] * (0.5 + 0.5 * effective_slot_confidence)
        boundary_scale = 0.5 + conf["boundary"]
        scale_confidence = conf["scale"]
        gates = torch.sigmoid(self.gate_head(state)) * self._warmup_scale(data, base_prob.device, base_prob.dtype)
        gates = gates * (0.5 + scale_confidence)

        guided_features = {}
        for idx, level in enumerate(LEVELS):
            target_size = features[level].shape[-2:]
            guidance = self._build_guidance(warped[level], anchors[level], boundary_scale)
            guidance = self.guidance_projs[level](guidance)
            B, N = base_prob.shape[:2]
            guidance = guidance.view(B, N, -1, *target_size)
            gate = gates[..., idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            guided_features[level] = self._fuse_guided_feature(level, features[level], guidance, gate)

        decoded = self.backbone.decode(
            guided_features["low"],
            guided_features["mid"],
            guided_features["high"],
            base_object_logits.shape[-2:],
        )
        guided_decoder = decoded["decoder_feature"]
        if self.decoder_guidance_enabled and "dec" in self.skip_guidance_levels:
            dec_guidance = self._build_guidance(warped["dec"], anchors["dec"], boundary_scale)
            dec_guidance = self.guidance_projs["dec"](dec_guidance)
            B, N = base_prob.shape[:2]
            dec_guidance = dec_guidance.view(B, N, -1, *base_object_logits.shape[-2:])
            dec_gate = gates[..., LEVELS.index("dec")].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            guided_decoder = self._fuse_guided_feature("dec", guided_decoder, dec_guidance, dec_gate)
        guided_logits = self.backbone.logits_from_decoder_feature(guided_decoder)[:, 1:2].expand_as(base_object_logits)

        object_logits = self._fuse_logits(base_object_logits, guided_logits, effective_prior_confidence)
        final_prob = torch.sigmoid(object_logits)
        final_geometry = mask_geometry_stats(final_prob, prev.get("final_geometry"))

        confidence_vec = torch.stack([conf["prior"], conf["base"], conf["update"], conf["boundary"]], dim=-1)
        candidate_prev = {
            "base_geometry": base_geometry.detach(),
            "final_geometry": final_geometry.detach(),
            "slot_weights": slot_weights.detach(),
            "confidence": confidence_vec.detach(),
            "affine": affine.detach(),
            "geometry_delta": geometry_delta.detach(),
            "valid": torch.ones_like(prev["valid"]),
        }
        next_prev = self._gated_history_update(prev, candidate_prev, conf["update"])
        aux = {
            "method": "anchor_ode_v2",
            "memory_type": "anchor_ode_v2",
            "anchor_ode_aux": {
                "mode": "current_anchor_affine",
                "base_object_logits": base_object_logits,
                "prior_logits": self._logit(prior_prob),
                "guided_object_logits": guided_logits,
                "final_object_logits": object_logits,
                "selected_motion_embed": selected_motion_embed,
                "slot_weights": slot_weights,
                "slot_confidence": slot_confidence,
                "effective_slot_confidence": effective_slot_confidence,
                "affine": affine,
                "affine_low": affine[..., 0, :],
                "affine_mid": affine[..., 1, :],
                "affine_high": affine[..., 2, :],
                "affine_dec": affine[..., 3, :],
                "anchor_maps": anchors,
                "warped_priors": {level: self._logit(value.clamp(1.0e-4, 1.0 - 1.0e-4)) for level, value in warped.items()},
                "warped_anchor_low": warped["low"],
                "warped_anchor_mid": warped["mid"],
                "warped_anchor_high": warped["high"],
                "warped_anchor_dec": warped["dec"],
                "skip_gates": gates,
                "raw_skip_gates": torch.sigmoid(self.gate_head(state)),
                "guidance_input_mode": self.guidance_input_mode,
                "guidance_fusion_mode": self.guidance_fusion_mode,
                "decoder_guidance_enabled": torch.tensor(float(self.decoder_guidance_enabled), device=base_prob.device, dtype=base_prob.dtype),
                "geometry_pred": geometry_pred,
                "geometry_offset": geometry_delta,
                "geometry_delta": geometry_delta,
                "geometry_delta_prior": geometry_delta_prior,
                "base_geometry": base_geometry,
                "final_geometry": final_geometry,
                "confidence": confidence_vec,
                "confidence_prior": effective_prior_confidence,
                "raw_confidence_prior": conf["prior"],
                "confidence_base": conf["base"],
                "confidence_update": conf["update"],
                "confidence_boundary": conf["boundary"],
                "confidence_scale": scale_confidence,
                "confidence_slot": conf["slot"],
                "base_prior_disagreement": conf["disagreement"],
                "mask_entropy": conf["entropy"],
                "guided_base_residual_abs_mean": (guided_logits - base_object_logits).detach().abs().mean(),
                "final_base_residual_abs_mean": (object_logits - base_object_logits).detach().abs().mean(),
                "memory_update_rate": conf["update"].detach().mean(),
                "gate_mean": gates.detach().mean(),
                "residual_abs_mean": (object_logits - base_object_logits).detach().abs().mean(),
                "affine_reg": (affine[..., 0:2].pow(2).mean() + (affine[..., 2:4] - 1.0).pow(2).mean() + affine[..., 4:6].pow(2).mean()),
                "dynamics_monitor": {
                    "area": final_geometry[..., 0],
                    "slot_usage": slot_weights.detach().mean(dim=(0, 1)),
                    "affine_abs_mean": affine.detach().abs().mean(dim=(-1, -2)),
                    "confidence_prior": effective_prior_confidence.detach(),
                },
            },
        }
        return object_logits, guided_logits, next_prev, aux

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        prev = self._empty_prev(B, max_num_objects, images.device, images.dtype)
        out: Dict = {"num_objects": num_objects}

        for ti in range(T):
            feat = self.backbone(self._normalize(images[:, ti]))
            base_object_logits = feat["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)
            object_logits, guided_logits, prev, memory_aux = self._anchor_step(feat, base_object_logits, prev, data)
            logits, masks = self._object_logits_to_full(object_logits)
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits,
                "object_logits": object_logits.detach(),
                "prior_logits": guided_logits.detach(),
            }
            out[f"memory_aux_{ti}"] = memory_aux
        return out
