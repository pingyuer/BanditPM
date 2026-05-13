from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


LEVELS = ("low", "mid", "high")


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _masked_mean(feature: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return feature.mean(dim=(-2, -1))
    mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return (feature * mask).sum(dim=(-2, -1)) / denom


def mask_geometry_stats(mask_BNHW: torch.Tensor, prev_stats: torch.Tensor | None = None) -> torch.Tensor:
    """Return shape stats used by S_t, including simple temporal velocities.

    Layout: area, cx, cy, width, height, entropy, area_v, cx_v, cy_v,
    scale_v, entropy_v.
    """
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
    entropy = -(
        mask.clamp(1.0e-6, 1.0 - 1.0e-6) * mask.clamp(1.0e-6, 1.0 - 1.0e-6).log()
        + (1.0 - mask).clamp(1.0e-6, 1.0).log() * (1.0 - mask)
    ).mean(dim=(-2, -1))
    base = torch.stack([area, cx, cy, width, height, entropy], dim=-1)
    if prev_stats is None:
        velocity = torch.zeros(B, N, 5, device=mask.device, dtype=mask.dtype)
    else:
        scale = 0.5 * (width + height)
        prev_scale = 0.5 * (prev_stats[..., 3] + prev_stats[..., 4])
        velocity = torch.stack(
            [
                area - prev_stats[..., 0],
                cx - prev_stats[..., 1],
                cy - prev_stats[..., 2],
                scale - prev_scale,
                entropy - prev_stats[..., 5],
            ],
            dim=-1,
        )
    return torch.cat([base, velocity], dim=-1)


class DelayODEBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_BNDHW: torch.Tensor, condition_BND: torch.Tensor, dt: float, gamma: float) -> torch.Tensor:
        B, N, D, H, W = z_BNDHW.shape
        z = z_BNDHW.flatten(0, 1)
        cond = condition_BND.flatten(0, 1).view(B * N, D, 1, 1).expand(-1, -1, H, W)
        dz = float(gamma) * torch.tanh(self.net(torch.cat([z, cond], dim=1))).view(B, N, D, H, W)
        return z_BNDHW + float(dt) * dz


class DelayODEMaskDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.high_to_mid = nn.Conv2d(dim, dim, kernel_size=1)
        self.mid_fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1), nn.GELU())
        self.low_fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1), nn.GELU())
        self.head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, latents: Dict[str, torch.Tensor], output_hw: tuple[int, int], enabled: Dict[str, bool]) -> torch.Tensor:
        low = latents["low"] if enabled["low"] else torch.zeros_like(latents["low"])
        mid = latents["mid"] if enabled["mid"] else torch.zeros_like(latents["mid"])
        high = latents["high"] if enabled["high"] else torch.zeros_like(latents["high"])
        B, N, D = low.shape[:3]
        high_f = self.high_to_mid(high.flatten(0, 1))
        high_f = F.interpolate(high_f, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        mid_f = self.mid_fuse(torch.cat([mid.flatten(0, 1), high_f], dim=1))
        mid_f = F.interpolate(mid_f, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_f = self.low_fuse(torch.cat([low.flatten(0, 1), mid_f], dim=1))
        logits = self.head(F.interpolate(low_f, size=output_hw, mode="bilinear", align_corners=False))
        return logits.view(B, N, *output_hw)


class DelayODEKeyMapSegmenter(nn.Module):
    """Predict-before-update multi-scale ODE key-map segmenter.

    Frame 0 is observation-only warm-up. For t>=1, current logits are decoded
    from previous recurrent state before the current image is encoded.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "delay_ode", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.num_slots = int(_cfg_get(method_cfg, "delay_ode_num_slots", 8))
        self.key_dim = int(_cfg_get(method_cfg, "delay_ode_key_dim", 64))
        self.value_dim = int(_cfg_get(method_cfg, "delay_ode_value_dim", 64))
        self.state_dim = int(_cfg_get(method_cfg, "delay_ode_state_dim", 128))
        self.temperature = float(_cfg_get(method_cfg, "delay_ode_temperature", 0.07))
        self.dt = float(_cfg_get(method_cfg, "delay_ode_dt", 1.0))
        self.steps = int(_cfg_get(method_cfg, "delay_ode_steps", 1))
        self.gamma = float(_cfg_get(method_cfg, "delay_ode_gamma", 0.1))
        self.keymap_ema = float(_cfg_get(method_cfg, "delay_ode_keymap_ema", 0.85))
        self.first_frame_init = str(_cfg_get(method_cfg, "delay_ode_first_frame_init", "init_head")).lower()
        self.allow_current_feature_for_current_mask = bool(_cfg_get(method_cfg, "delay_ode_allow_current_feature_for_current_mask", False))
        self.update_gate_max = float(_cfg_get(method_cfg, "delay_ode_update_gate_max", 0.5))
        self.supervise_first_frame = bool(_cfg_get(method_cfg, "delay_ode_supervise_first_frame", False))
        self.level_enabled = {
            "low": bool(_cfg_get(method_cfg, "delay_ode_use_low", True)),
            "mid": bool(_cfg_get(method_cfg, "delay_ode_use_mid", True)),
            "high": bool(_cfg_get(method_cfg, "delay_ode_use_high", True)),
        }

        self.lambda_slot_balance = float(
            _cfg_get(
                method_cfg,
                "delay_ode_lambda_slot_balance",
                _cfg_get(method_cfg, "delay_ode_lambda_selection_entropy", 0.001),
            )
        )
        self.lambda_gate_smooth = float(_cfg_get(method_cfg, "delay_ode_lambda_gate_smooth", 0.01))
        self.lambda_latent_smooth = float(_cfg_get(method_cfg, "delay_ode_lambda_latent_smooth", 0.01))
        self.lambda_state_smooth = float(_cfg_get(method_cfg, "delay_ode_lambda_state_smooth", 0.01))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=base_dim,
            value_dim=self.value_dim,
        )
        dims = {"low": base_dim, "mid": base_dim * 2, "high": base_dim * 4}
        self.feature_proj = nn.ModuleDict({level: nn.Conv2d(dims[level], self.value_dim, 1) for level in LEVELS})
        self.init_head = nn.Sequential(
            nn.Conv2d(self.value_dim, self.value_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.value_dim, 1, kernel_size=1),
        )
        self.desc_proj = nn.ModuleDict({level: nn.Linear(self.value_dim + self.state_dim, self.key_dim + self.value_dim) for level in LEVELS})
        self.query_proj = nn.ModuleDict({level: nn.Linear(self.state_dim, self.key_dim) for level in LEVELS})
        self.ode = nn.ModuleDict({level: DelayODEBlock(self.value_dim) for level in LEVELS})
        self.slot_key_embed = nn.ParameterDict({
            level: nn.Parameter(torch.randn(self.num_slots, self.key_dim) * 0.02) for level in LEVELS
        })
        self.slot_value_embed = nn.ParameterDict({
            level: nn.Parameter(torch.randn(self.num_slots, self.value_dim) * 0.02) for level in LEVELS
        })
        self.stats_dim = 11
        self.write_proj = nn.ModuleDict({level: nn.Linear(self.value_dim + self.state_dim + self.stats_dim, self.num_slots + 1) for level in LEVELS})
        self.init_state = nn.Sequential(
            nn.Linear(self.value_dim * 3 + self.stats_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.state_update = nn.GRUCell(self.value_dim * 3 + self.stats_dim + self.num_slots * 3, self.state_dim)
        self.decoder = DelayODEMaskDecoder(self.value_dim)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _features(self, frame_BTCHW: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(self._normalize(frame_BTCHW))
        return {
            "low": self.feature_proj["low"](feat["low"]),
            "mid": self.feature_proj["mid"](feat["mid"]),
            "high": self.feature_proj["high"](feat["high"]),
        }

    def _expand_objects(self, x_BCHW: torch.Tensor, num_objects: int) -> torch.Tensor:
        return x_BCHW.unsqueeze(1).expand(-1, num_objects, -1, -1, -1).contiguous()

    def _descriptors(self, feats: dict[str, torch.Tensor], num_objects: int, mask_BNHW: torch.Tensor | None) -> dict[str, torch.Tensor]:
        desc = {}
        for level in LEVELS:
            f = self._expand_objects(feats[level], num_objects)
            B, N, D, H, W = f.shape
            mask = None if mask_BNHW is None else mask_BNHW.flatten(0, 1).unsqueeze(1)
            pooled = _masked_mean(f.flatten(0, 1), mask).view(B, N, D)
            desc[level] = pooled
        return desc

    def _initial_mask(self, feats: dict[str, torch.Tensor], data: Dict, num_objects: int, output_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(feats.values())).shape[0]
        device = next(iter(feats.values())).device
        dtype = next(iter(feats.values())).dtype
        if self.first_frame_init == "zero":
            logits = torch.zeros(B, num_objects, *output_hw, device=device, dtype=dtype)
            return logits, torch.zeros_like(logits)
        if self.first_frame_init == "oracle_upperbound" and torch.is_tensor(data.get("ff_gt")):
            mask = data["ff_gt"][:, 0, :num_objects].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            logits = torch.logit(mask.clamp(1.0e-4, 1.0 - 1.0e-4))
            return logits, mask
        init_feature = F.interpolate(feats["low"], size=output_hw, mode="bilinear", align_corners=False)
        logits = self.init_head(init_feature).expand(-1, num_objects, -1, -1).contiguous()
        return logits, torch.sigmoid(logits)

    def _init_from_observation(self, feats: dict[str, torch.Tensor], init_mask: torch.Tensor):
        num_objects = init_mask.shape[1]
        stats = mask_geometry_stats(init_mask.detach(), None)
        desc = self._descriptors(feats, num_objects, init_mask.detach())
        state_in = torch.cat([desc["low"], desc["mid"], desc["high"], stats], dim=-1)
        state = self.init_state(state_in)
        latents = {level: self._expand_objects(feats[level], num_objects) for level in LEVELS}
        keys = {}
        values = {}
        for level in LEVELS:
            cand = self.desc_proj[level](torch.cat([desc[level], state], dim=-1))
            key, value = cand.split([self.key_dim, self.value_dim], dim=-1)
            keys[level] = key.unsqueeze(2) + self.slot_key_embed[level].view(1, 1, self.num_slots, self.key_dim)
            values[level] = value.unsqueeze(2) + self.slot_value_embed[level].view(1, 1, self.num_slots, self.value_dim)
        return state, latents, keys, values, stats

    def _select(self, state: torch.Tensor, keys: dict[str, torch.Tensor], values: dict[str, torch.Tensor]):
        weights = {}
        conditions = {}
        for level in LEVELS:
            query = self.query_proj[level](state).unsqueeze(2)
            logits = (query * keys[level]).sum(dim=-1) / math.sqrt(max(self.key_dim, 1))
            logits = logits / max(self.temperature, 1.0e-6)
            w = torch.softmax(logits, dim=-1)
            if not self.level_enabled[level]:
                condition = torch.zeros_like(values[level][..., 0, :])
            else:
                condition = (w.unsqueeze(-1) * values[level]).sum(dim=2)
            weights[level] = w
            conditions[level] = condition
        return weights, conditions

    def _advance(self, latents: dict[str, torch.Tensor], conditions: dict[str, torch.Tensor]):
        next_latents = {}
        for level in LEVELS:
            z = latents[level]
            sub_dt = self.dt / float(max(self.steps, 1))
            for _ in range(max(self.steps, 1)):
                if self.level_enabled[level]:
                    z = self.ode[level](z, conditions[level], sub_dt, self.gamma)
            next_latents[level] = z
        return next_latents

    def _update_state_and_keymap(
        self,
        feats: dict[str, torch.Tensor],
        pred_mask: torch.Tensor,
        prev_state: torch.Tensor,
        keys: dict[str, torch.Tensor],
        values: dict[str, torch.Tensor],
        prev_stats: torch.Tensor,
        selection_weights: dict[str, torch.Tensor],
    ):
        mask_detached = pred_mask.detach()
        stats = mask_geometry_stats(mask_detached, prev_stats.detach())
        desc = self._descriptors(feats, pred_mask.shape[1], mask_detached)
        selection_summary = torch.cat([selection_weights[level] for level in LEVELS], dim=-1)
        state_input = torch.cat([desc["low"], desc["mid"], desc["high"], stats, selection_summary], dim=-1)
        B, N = prev_state.shape[:2]
        state = self.state_update(state_input.flatten(0, 1), prev_state.flatten(0, 1)).view(B, N, self.state_dim)

        new_keys = {}
        new_values = {}
        gates = {}
        for level in LEVELS:
            cand = self.desc_proj[level](torch.cat([desc[level], state], dim=-1))
            cand_key, cand_value = cand.split([self.key_dim, self.value_dim], dim=-1)
            write_raw = self.write_proj[level](torch.cat([desc[level], state, stats], dim=-1))
            write_logits = write_raw[..., : self.num_slots]
            gate = torch.sigmoid(write_raw[..., self.num_slots:]) * self.update_gate_max
            write = torch.softmax(write_logits, dim=-1)
            effective_gate = gate * (1.0 - self.keymap_ema)
            gate_write = effective_gate * write
            if self.level_enabled[level]:
                new_keys[level] = keys[level] * (1.0 - gate_write.unsqueeze(-1)) + cand_key.unsqueeze(2) * gate_write.unsqueeze(-1)
                new_values[level] = values[level] * (1.0 - gate_write.unsqueeze(-1)) + cand_value.unsqueeze(2) * gate_write.unsqueeze(-1)
            else:
                new_keys[level] = keys[level]
                new_values[level] = values[level]
            gates[level] = gate
        effective_gates = {level: gates[level] * (1.0 - self.keymap_ema) for level in LEVELS}
        return state, new_keys, new_values, stats, gates, effective_gates

    def _update_latents(
        self,
        latents: dict[str, torch.Tensor],
        feats: dict[str, torch.Tensor],
        pred_mask: torch.Tensor,
        gates: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        updated = {}
        for level in LEVELS:
            obs = self._expand_objects(feats[level], pred_mask.shape[1])
            if obs.shape[-2:] != latents[level].shape[-2:]:
                obs = F.interpolate(
                    obs.flatten(0, 1),
                    size=latents[level].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view_as(latents[level])
            gate = gates[level].view(gates[level].shape[0], gates[level].shape[1], 1, 1, 1)
            if self.level_enabled[level]:
                updated[level] = latents[level] * (1.0 - gate) + obs * gate
            else:
                updated[level] = latents[level]
        return updated

    def _aggregate_object_logits(self, object_logits_BNHW: torch.Tensor):
        masks = torch.sigmoid(object_logits_BNHW)
        logits = aggregate(masks, dim=1)
        return logits, masks

    def _regularizers(self, weight_hist, gate_hist, state_hist, latent_hist):
        if not weight_hist:
            device = state_hist[0].device
            zero = torch.zeros((), device=device)
            return {
                "slot_balance": zero,
                "gate_smooth": zero,
                "latent_smooth": zero,
                "state_smooth": zero,
            }
        balance_terms = []
        for level in LEVELS:
            usage = torch.stack([item[level] for item in weight_hist], dim=2).mean(dim=(0, 1, 2))
            usage = usage / usage.sum().clamp_min(1.0e-8)
            uniform = torch.full_like(usage, 1.0 / max(usage.numel(), 1))
            balance_terms.append((usage.clamp_min(1.0e-8) * (usage.clamp_min(1.0e-8).log() - uniform.log())).sum())
        gate_terms = []
        if len(gate_hist) > 1:
            for a, b in zip(gate_hist[:-1], gate_hist[1:]):
                for level in LEVELS:
                    gate_terms.append((a[level] - b[level]).pow(2).mean())
        latent_terms = []
        if len(latent_hist) > 1:
            for a, b in zip(latent_hist[:-1], latent_hist[1:]):
                for level in LEVELS:
                    latent_terms.append((a[level] - b[level]).pow(2).mean())
        state_terms = [(a - b).pow(2).mean() for a, b in zip(state_hist[:-1], state_hist[1:])]
        zero = balance_terms[0].sum() * 0.0
        return {
            "slot_balance": torch.stack(balance_terms).mean() if balance_terms else zero,
            "gate_smooth": torch.stack(gate_terms).mean() if gate_terms else zero,
            "latent_smooth": torch.stack(latent_terms).mean() if latent_terms else zero,
            "state_smooth": torch.stack(state_terms).mean() if state_terms else zero,
        }

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        output_hw = images.shape[-2:]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        out: Dict = {"num_objects": num_objects}

        feats0 = self._features(images[:, 0])
        init_object_logits, init_masks = self._initial_mask(feats0, data, max_num_objects, output_hw)
        state, latents, keys, values, prev_stats = self._init_from_observation(feats0, init_masks)

        init_logits, init_fg_masks = self._aggregate_object_logits(init_object_logits)
        out["logits_0"] = init_logits
        out["masks_0"] = init_fg_masks
        out["aux_0"] = {"delay_ode_warmup_only": True, "init_object_logits": init_object_logits.detach()}

        weight_hist = []
        gate_hist = []
        state_hist = [state]
        latent_hist = [{level: latents[level] for level in LEVELS}]
        stats_hist = [prev_stats]
        effective_gate_hist = []

        warmup_aux = {
            "method": "delay_ode",
            "warmup_only": True,
            "supervise_first_frame": self.supervise_first_frame,
            "delay_ode_aux": {
                "keymap_weights": {level: torch.empty(B, max_num_objects, 0, self.num_slots, device=images.device) for level in LEVELS},
                "update_gates": {level: torch.empty(B, max_num_objects, 0, 1, device=images.device) for level in LEVELS},
                "states": torch.stack(state_hist, dim=2).detach(),
                "mask_stats": torch.stack(stats_hist, dim=2).detach(),
                "latents": {level: latents[level].detach() for level in LEVELS},
                "init_mode": self.first_frame_init,
                "current_feature_used_for_current_mask": False,
            },
        }
        out["memory_aux_0"] = warmup_aux

        for ti in range(1, T):
            weights, conditions = self._select(state, keys, values)
            next_latents = self._advance(latents, conditions)
            object_logits = self.decoder(next_latents, output_hw, self.level_enabled)
            logits, fg_masks = self._aggregate_object_logits(object_logits)

            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = fg_masks
            out[f"aux_{ti}"] = {
                "object_logits": object_logits.detach(),
                "delay_ode_warmup_only": False,
            }

            feats_t = self._features(images[:, ti])
            state, keys, values, prev_stats, gates, effective_gates = self._update_state_and_keymap(
                feats_t,
                fg_masks,
                state,
                keys,
                values,
                prev_stats,
                weights,
            )
            latents = self._update_latents(next_latents, feats_t, fg_masks, gates)
            weight_hist.append(weights)
            gate_hist.append(gates)
            effective_gate_hist.append(effective_gates)
            state_hist.append(state)
            latent_hist.append({level: latents[level] for level in LEVELS})
            stats_hist.append(prev_stats)
            regs = self._regularizers(weight_hist, gate_hist, state_hist, latent_hist)
            delay_aux = {
                "keymap_weights": {level: torch.stack([w[level] for w in weight_hist], dim=2).detach() for level in LEVELS},
                "update_gates": {level: torch.stack([g[level] for g in gate_hist], dim=2).detach() for level in LEVELS},
                "effective_update_gates": {level: torch.stack([g[level] for g in effective_gate_hist], dim=2).detach() for level in LEVELS},
                "states": torch.stack(state_hist, dim=2).detach(),
                "mask_stats": torch.stack(stats_hist, dim=2).detach(),
                "latents": {level: latents[level].detach() for level in LEVELS},
                "slot_balance": regs["slot_balance"],
                "gate_smooth": regs["gate_smooth"],
                "latent_smooth": regs["latent_smooth"],
                "state_smooth": regs["state_smooth"],
                "lambda_slot_balance": torch.tensor(self.lambda_slot_balance, device=images.device),
                "lambda_gate_smooth": torch.tensor(self.lambda_gate_smooth, device=images.device),
                "lambda_latent_smooth": torch.tensor(self.lambda_latent_smooth, device=images.device),
                "lambda_state_smooth": torch.tensor(self.lambda_state_smooth, device=images.device),
                "current_feature_used_for_current_mask": False,
                "first_frame_mode": self.first_frame_init,
                "conditional_ode_keymap": True,
                "ode_gamma": torch.tensor(self.gamma, device=images.device),
                "ode_sub_dt": torch.tensor(self.dt / float(max(self.steps, 1)), device=images.device),
                "keymap_ema": torch.tensor(self.keymap_ema, device=images.device),
            }
            out[f"memory_aux_{ti}"] = {
                "method": "delay_ode",
                "memory_type": "delay_ode",
                "warmup_only": False,
                "supervise_first_frame": self.supervise_first_frame,
                "delay_ode_aux": delay_aux,
            }
        return out
