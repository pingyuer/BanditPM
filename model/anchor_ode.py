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
        weights = torch.softmax(self.selector(state), dim=-1)
        anchors = {
            level: torch.einsum("bnk,kchw->bnchw", weights, self.anchors[level])
            for level in LEVELS
        }
        condition = torch.einsum("bnk,kc->bnc", weights, self.condition)
        affine_prior = torch.einsum("bnk,klp->bnlp", weights, self.affine_prior)
        geometry_prior = torch.einsum("bnk,kc->bnc", weights, self.geometry_prior)
        slot_confidence = (weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1).neg()
        slot_confidence = 1.0 - slot_confidence / math.log(max(self.num_slots, 2))
        return anchors, weights, condition, affine_prior, geometry_prior + slot_confidence.unsqueeze(-1) * 0.0


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
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 6, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, 9),
        )
        nn.init.constant_(self.net[-1].bias, -1.5)

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
