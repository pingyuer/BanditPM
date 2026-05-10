from __future__ import annotations

from typing import Dict
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.memory_core import MemoryCore
from model.modules.unext import UNeXtBackbone
from model.memory_readout import MaskAwareMemoryReadout
from model.spatial_dynakey import SpatialDynaKeyMemory, segmentation_gain_reward
from utils.tensor_utils import aggregate


class UNeXtDynaKeySegmenter(nn.Module):
    """UNeXt-DynaKey video segmenter.

    This variant is a lightweight method inspired by linear/dynamic memory
    ideas, not a full LKVA/GDR/KPFF reproduction. UNeXt supplies the single-frame
    spatial prior, while DynaKey and optional mask-aware prototypes provide
    online temporal residual refinement under no-leak initialization protocols.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("unext_dynakey", {})
        self.allow_oracle_init_when_requested = bool(
            cfg.get(
                "allow_oracle_init_when_requested",
                cfg.get("use_first_frame_gt_init", True),
            )
        )
        self.use_dynakey = bool(model_cfg.get("use_dynakey", True))
        self.use_temporal_refine = bool(model_cfg.get("use_temporal_refine", True))
        self.dynakey_memory_mode = str(model_cfg.get("dynakey_memory_mode", model_cfg.get("memory_mode", "global"))).lower()
        self.use_spatial_memory = self.use_dynakey and self.dynakey_memory_mode == "spatial"
        self.use_phase_retrieval = bool(model_cfg.get("use_phase_retrieval", self.use_spatial_memory))
        self.dynamics_mode = str(model_cfg.get("dynamics_mode", "global" if self.use_dynakey else "none")).lower()
        self.readout_type = str(model_cfg.get("readout_type", "spatial_gate" if self.use_spatial_memory else "global_broadcast")).lower()
        self.q_policy_mode = str(model_cfg.get("q_policy_mode", "off")).lower()
        if self.q_policy_mode not in {"off", "diagnostic", "training"}:
            raise ValueError(f"Unsupported q_policy_mode={self.q_policy_mode}")
        self.enable_q_policy = self.q_policy_mode != "off"
        self.use_mask_memory = bool(model_cfg.get("use_mask_memory", model_cfg.get("use_memory_readout", self.use_dynakey)))
        self.value_dim = int(model_cfg.get("value_dim", 256))
        self.num_classes = int(model_cfg.get("num_classes", 2))
        self.in_channels = int(model_cfg.get("in_channels", 1))
        base_dim = int(model_cfg.get("base_dim", 32))
        residual_init = float(model_cfg.get("temporal_residual_init_scale", model_cfg.get("refine_alpha_init", 0.1)))
        residual_init = min(max(residual_init, 1e-4), 1.0 - 1e-4)
        self.temporal_residual_scale_logit = nn.Parameter(torch.tensor(math.log(residual_init / (1.0 - residual_init))))
        self.clamp_temporal_residual = bool(model_cfg.get("clamp_temporal_residual", True))
        self.temporal_residual_clip = float(model_cfg.get("temporal_residual_clip", 3.0))
        self._warned_large_residual = False

        tf_cfg = model_cfg.get("teacher_forcing_update_memory", {}) or {}
        self.teacher_forcing_enabled = bool(tf_cfg.get("enabled", False))
        self.teacher_forcing_start_prob = float(tf_cfg.get("start_prob", 0.5))
        self.teacher_forcing_end_prob = float(tf_cfg.get("end_prob", 0.0))
        self.teacher_forcing_warmup_iters = int(tf_cfg.get("warmup_iters", 300))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=base_dim,
            value_dim=self.value_dim,
        )
        self.memory_core = None
        if self.use_dynakey:
            self.memory_core = MemoryCore(
                value_dim=self.value_dim,
                key_dim=self.value_dim,
                prototype_value_cfg=None,
                temporal_memory_cfg=cfg.get("temporal_memory", None),
                memory_core_cfg=cfg.get("memory_core", None),
            )
        dec_dim = self.backbone.decoder_dim
        self.temporal_delta_proj = nn.Conv2d(self.value_dim, dec_dim, kernel_size=1)
        self.mask_memory_proj = nn.Conv2d(self.value_dim, dec_dim, kernel_size=1)
        refine_in_channels = dec_dim * 3 + 3
        self.temporal_refine_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, dec_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_dim, 1, kernel_size=1),
        )
        self.temporal_gate_head = nn.Sequential(
            nn.Conv2d(refine_in_channels, dec_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        gate_bias = float(model_cfg.get("temporal_gate_bias", -2.0))
        nn.init.constant_(self.temporal_gate_head[2].bias, gate_bias)
        self.spatial_memory_proj = nn.Conv2d(self.value_dim, dec_dim, kernel_size=1)
        self.q_policy_head = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        self.spatial_memory = SpatialDynaKeyMemory(
            self.value_dim,
            num_slots=int(model_cfg.get("spatial_memory_slots", model_cfg.get("mask_memory_slots", 4))),
            spatial_size=int(model_cfg.get("spatial_memory_size", 16)),
            ema_momentum=float(model_cfg.get("spatial_memory_ema", model_cfg.get("mask_memory_ema", 0.9))),
            temperature=float(model_cfg.get("spatial_memory_temperature", model_cfg.get("mask_memory_temperature", 0.1))),
            phase_weight=float(model_cfg.get("phase_retrieval_weight", 1.0)),
            spatial_weight=float(model_cfg.get("spatial_retrieval_weight", 1.0)),
            confidence_threshold=float(model_cfg.get("spatial_memory_confidence_threshold", model_cfg.get("mask_memory_confidence_threshold", 0.55))),
            fg_ratio_min=float(model_cfg.get("spatial_memory_fg_ratio_min", model_cfg.get("mask_memory_fg_ratio_min", 0.005))),
            fg_ratio_max=float(model_cfg.get("spatial_memory_fg_ratio_max", model_cfg.get("mask_memory_fg_ratio_max", 0.60))),
            use_spatial_dynamics=self.dynamics_mode == "spatial",
            dynamics_momentum=float(model_cfg.get("spatial_dynamics_momentum", 0.8)),
        )
        self.mask_memory = MaskAwareMemoryReadout(
            self.value_dim,
            num_slots=int(model_cfg.get("mask_memory_slots", 4)),
            ema_momentum=float(model_cfg.get("mask_memory_ema", 0.9)),
            temperature=float(model_cfg.get("mask_memory_temperature", 0.1)),
            mask_size=int(model_cfg.get("mask_memory_size", 16)),
            confidence_threshold=float(model_cfg.get("mask_memory_confidence_threshold", 0.55)),
            fg_ratio_min=float(model_cfg.get("mask_memory_fg_ratio_min", 0.005)),
            fg_ratio_max=float(model_cfg.get("mask_memory_fg_ratio_max", 0.60)),
            area_change_limit=model_cfg.get("mask_memory_area_change_limit", None),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_value(self, value_BCHW: torch.Tensor, num_objects: int) -> torch.Tensor:
        return value_BCHW.unsqueeze(1).expand(-1, num_objects, -1, -1, -1).contiguous()

    def _teacher_forcing_prob(self, data: Dict) -> float:
        if not self.training or not self.teacher_forcing_enabled:
            return 0.0
        current_iter = int(data.get("current_iter", 0))
        warmup = max(self.teacher_forcing_warmup_iters, 1)
        ratio = min(max(current_iter / warmup, 0.0), 1.0)
        return self.teacher_forcing_start_prob + ratio * (self.teacher_forcing_end_prob - self.teacher_forcing_start_prob)

    def _refine_object_logits(
        self,
        base_logits_B2HW: torch.Tensor,
        decoder_BCHW: torch.Tensor,
        temporal_delta_BNCHW: torch.Tensor,
        mask_BNHW: torch.Tensor,
        memory_readout_BNC11: torch.Tensor | None = None,
        mask_prior_BNHW: torch.Tensor | None = None,
        spatial_readout_BNCHW: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, N = mask_BNHW.shape[:2]
        fg_base = base_logits_B2HW[:, 1:2].expand(-1, N, -1, -1)
        if not self.use_temporal_refine:
            return fg_base, {
                "gate_mean": torch.tensor(0.0, device=fg_base.device),
                "gate_std": torch.tensor(0.0, device=fg_base.device),
                "residual_abs_mean": torch.tensor(0.0, device=fg_base.device),
                "base_logits_abs_mean": fg_base.detach().abs().mean(),
                "temporal_residual_scale": torch.tensor(0.0, device=fg_base.device),
            }

        target_hw = decoder_BCHW.shape[-2:]
        temporal = self.temporal_delta_proj(temporal_delta_BNCHW.flatten(0, 1))
        temporal = F.interpolate(temporal, size=target_hw, mode="bilinear", align_corners=False)
        decoder = decoder_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1)
        soft_mask = mask_BNHW.float()
        soft_mask = F.interpolate(soft_mask.flatten(0, 1).unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False)
        temporal = temporal * soft_mask

        if memory_readout_BNC11 is None:
            memory_map = torch.zeros_like(temporal)
        else:
            memory_map = self.mask_memory_proj(memory_readout_BNC11.flatten(0, 1))
            memory_map = F.interpolate(memory_map, size=target_hw, mode="bilinear", align_corners=False)
        if spatial_readout_BNCHW is not None and self.readout_type == "spatial_gate":
            spatial_map = self.spatial_memory_proj(spatial_readout_BNCHW.flatten(0, 1))
            spatial_map = F.interpolate(spatial_map, size=target_hw, mode="bilinear", align_corners=False)
            memory_map = memory_map + spatial_map
        if mask_prior_BNHW is None:
            mask_prior = torch.zeros_like(soft_mask)
        else:
            mask_prior = F.interpolate(
                mask_prior_BNHW.flatten(0, 1).unsqueeze(1).float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        base_prob = torch.sigmoid(fg_base).flatten(0, 1).unsqueeze(1)
        refine_in = torch.cat([decoder, temporal, memory_map, soft_mask, base_prob, mask_prior], dim=1)
        residual = self.temporal_refine_head(refine_in).view(B, N, *decoder_BCHW.shape[-2:])
        if self.clamp_temporal_residual:
            residual = residual.clamp(min=-self.temporal_residual_clip, max=self.temporal_residual_clip)
        gate = self.temporal_gate_head(refine_in).view(B, N, *decoder_BCHW.shape[-2:])
        scale = self.temporal_residual_scale_logit.sigmoid()
        aux = {
            "gate_mean": gate.detach().mean(),
            "gate_std": gate.detach().std(),
            "residual_abs_mean": residual.detach().abs().mean(),
            "base_logits_abs_mean": fg_base.detach().abs().mean(),
            "temporal_residual_scale": scale.detach(),
        }
        if bool(aux["residual_abs_mean"] > 2.0 * aux["base_logits_abs_mean"].clamp_min(1e-6)) and not self._warned_large_residual:
            warnings.warn("UNeXt-DynaKey temporal residual magnitude is much larger than base logits.", RuntimeWarning)
            self._warned_large_residual = True
        return fg_base + scale * gate * residual, aux

    def _aggregate_logits(self, object_logits_BNHW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits_BNHW)
        logits = aggregate(masks, dim=1)
        masks_all = torch.softmax(logits, dim=1)
        return logits, masks_all

    def forward(self, data: Dict) -> Dict:
        images_BTCHW = data["rgb"]
        B, T = images_BTCHW.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        out: Dict = {"num_objects": num_objects}
        if self.use_dynakey and self.memory_core is not None:
            self.memory_core.reset_state(B, max_num_objects, images_BTCHW.device)
        if self.use_mask_memory:
            self.mask_memory.reset_state(B, max_num_objects, images_BTCHW.device, images_BTCHW.dtype)
        if self.use_spatial_memory:
            self.spatial_memory.reset_state(B, max_num_objects, images_BTCHW.device, images_BTCHW.dtype)
        policy_meta = {
            "current_iter": int(data.get("current_iter", 0)),
            "current_epoch": int(data.get("current_epoch", 0)),
            "training": bool(self.training),
        }

        init_mode = str(data.get("init_mode", "oracle_gt"))
        init_mask = data["ff_gt"][:, 0, :max_num_objects].float()
        if init_mode != "oracle_gt" or not self.allow_oracle_init_when_requested:
            init_mask = torch.zeros_like(init_mask)
        last_masks = init_mask
        previous_value = None
        teacher_forcing_prob = self._teacher_forcing_prob(data)

        for ti in range(T):
            frame = self._normalize(images_BTCHW[:, ti])
            feat = self.backbone(frame)
            value_BNCHW = self._object_value(feat["value"], max_num_objects)
            memory_aux = {"memory_type": "none"}
            if self.use_dynakey and self.memory_core is not None:
                readout_BNCHW, memory_aux = self.memory_core(
                    value_BNCHW=value_BNCHW,
                    key_BCHW=feat["high_value"],
                    pixfeat_BCHW=feat["value"],
                    mask_BNHW=last_masks,
                    policy_meta=policy_meta,
                )
                temporal_delta = readout_BNCHW - value_BNCHW
            elif previous_value is not None:
                temporal_delta = previous_value - value_BNCHW
            else:
                temporal_delta = torch.zeros_like(value_BNCHW)

            memory_readout = None
            mask_prior = None
            spatial_readout = None
            spatial_phase = None
            if self.use_mask_memory:
                memory_readout, mask_prior, mask_mem_aux = self.mask_memory.read(value_BNCHW, last_masks)
                memory_aux.update(mask_mem_aux)
            if self.use_spatial_memory:
                spatial = self.spatial_memory.read(
                    value_BNCHW,
                    last_masks,
                    frame_index=ti,
                    total_frames=T,
                    use_phase=self.use_phase_retrieval,
                )
                spatial_readout = spatial.feature
                spatial_phase = spatial.phase
                if mask_prior is None:
                    mask_prior = spatial.mask_prior
                else:
                    mask_prior = 0.5 * mask_prior + 0.5 * spatial.mask_prior
                memory_aux.update(spatial.aux)
            object_logits, refine_aux = self._refine_object_logits(
                feat["logits"],
                feat["decoder_feature"],
                temporal_delta,
                last_masks,
                memory_readout,
                mask_prior,
                spatial_readout,
            )
            if self.enable_q_policy:
                fg_prob = torch.sigmoid(feat["logits"][:, 1:2]).expand(-1, max_num_objects, -1, -1)
                q_phase = spatial_phase
                if q_phase is None:
                    area = last_masks.float().mean(dim=(-2, -1))
                    q_phase = torch.stack(
                        [
                            area,
                            torch.zeros_like(area),
                            torch.full_like(area, float(ti) / float(max(T - 1, 1))),
                            1.0 - (-(fg_prob.clamp(1e-6, 1 - 1e-6) * fg_prob.clamp(1e-6, 1 - 1e-6).log()
                                      + (1 - fg_prob.clamp(1e-6, 1 - 1e-6)) * (1 - fg_prob.clamp(1e-6, 1 - 1e-6)).log())).mean(dim=(-2, -1)),
                        ],
                        dim=-1,
                    )
                valid_slots = memory_aux.get("spatial_memory_valid_slots")
                if torch.is_tensor(valid_slots):
                    valid_frac = (valid_slots.to(q_phase.dtype) / max(float(self.spatial_memory.num_slots), 1.0)).unsqueeze(-1)
                else:
                    valid_frac = torch.zeros_like(q_phase[..., :1])
                q_features = torch.cat(
                    [
                        q_phase,
                        valid_frac,
                        refine_aux["gate_mean"].expand_as(valid_frac),
                        refine_aux["residual_abs_mean"].expand_as(valid_frac),
                        refine_aux["base_logits_abs_mean"].expand_as(valid_frac),
                    ],
                    dim=-1,
                )
                q_values = self.q_policy_head(q_features)
                q_probs = torch.softmax(q_values, dim=-1)
                q_entropy = -(q_probs.clamp_min(1e-8).log() * q_probs).sum(dim=-1)
                memory_aux["spatial_q_values"] = q_values
                memory_aux["spatial_q_entropy"] = q_entropy.detach()
                memory_aux["spatial_q_policy_mode"] = self.q_policy_mode
            logits, masks_all = self._aggregate_logits(object_logits)
            last_masks = masks_all[:, 1:].detach()
            if self.enable_q_policy and self.q_policy_mode == "training" and "cls_gt" in data:
                gt_frame = data["cls_gt"][:, ti]
                if gt_frame.dim() == 4:
                    gt_frame = gt_frame.squeeze(1)
                gt_mask = gt_frame.unsqueeze(1).float().expand_as(object_logits)
                reward = segmentation_gain_reward(feat["logits"][:, 1:2].expand_as(object_logits), object_logits, gt_mask)
                label_valid = data.get("label_valid")
                valid = torch.ones(B, device=object_logits.device, dtype=torch.bool)
                if torch.is_tensor(label_valid) and label_valid.dim() >= 2:
                    valid = label_valid[:, ti].to(object_logits.device).bool()
                target = torch.where(reward > 0, torch.ones_like(reward, dtype=torch.long), torch.zeros_like(reward, dtype=torch.long))
                memory_aux["spatial_q_target_action"] = target
                memory_aux["spatial_q_reward"] = reward.detach()
                memory_aux["spatial_q_valid"] = valid[:, None].expand_as(target)
            if self.use_mask_memory:
                update_masks = last_masks
                if teacher_forcing_prob > 0.0 and "cls_gt" in data:
                    gt_frame = data["cls_gt"][:, ti]
                    if gt_frame.dim() == 4:
                        gt_frame = gt_frame.squeeze(1)
                    gt_mask = gt_frame.unsqueeze(1).float().expand_as(last_masks)
                    label_valid = data.get("label_valid")
                    valid = None
                    if torch.is_tensor(label_valid) and label_valid.dim() >= 2:
                        valid = label_valid[:, ti].view(-1, 1, 1, 1).to(last_masks.device)
                    use_tf = torch.rand((), device=last_masks.device) < teacher_forcing_prob
                    if valid is not None:
                        update_masks = torch.where(use_tf & valid, gt_mask, last_masks)
                    elif bool(use_tf):
                        update_masks = gt_mask
                update_aux = self.mask_memory.update(value_BNCHW, update_masks)
                memory_aux.update(update_aux)
            if self.use_spatial_memory:
                update_masks = last_masks
                if teacher_forcing_prob > 0.0 and "cls_gt" in data:
                    gt_frame = data["cls_gt"][:, ti]
                    if gt_frame.dim() == 4:
                        gt_frame = gt_frame.squeeze(1)
                    gt_mask = gt_frame.unsqueeze(1).float().expand_as(last_masks)
                    label_valid = data.get("label_valid")
                    valid = None
                    if torch.is_tensor(label_valid) and label_valid.dim() >= 2:
                        valid = label_valid[:, ti].view(-1, 1, 1, 1).to(last_masks.device)
                    use_tf = torch.rand((), device=last_masks.device) < teacher_forcing_prob
                    if valid is not None:
                        update_masks = torch.where(use_tf & valid, gt_mask, last_masks)
                    elif bool(use_tf):
                        update_masks = gt_mask
                spatial_update_aux = self.spatial_memory.update(value_BNCHW, update_masks, frame_index=ti, total_frames=T)
                memory_aux.update(spatial_update_aux)
            previous_value = value_BNCHW.detach()

            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks_all[:, 1:]
            out[f"aux_{ti}"] = {
                "base_foreground_logits": feat["logits"][:, 1:2].detach(),
                "object_logits": object_logits.detach(),
            }
            out[f"memory_aux_{ti}"] = memory_aux
            memory_aux.update(refine_aux)
            memory_aux["temporal_refine_enabled"] = self.use_temporal_refine
            memory_aux["dynakey_enabled"] = self.use_dynakey
            memory_aux["dynakey_memory_mode"] = self.dynakey_memory_mode
            memory_aux["spatial_memory_enabled"] = self.use_spatial_memory
            memory_aux["phase_retrieval_enabled"] = self.use_phase_retrieval
            memory_aux["readout_type"] = self.readout_type
            memory_aux["dynamics_mode"] = self.dynamics_mode
            memory_aux["q_policy_mode"] = self.q_policy_mode
            memory_aux["oracle_gt_init_allowed"] = self.allow_oracle_init_when_requested
            memory_aux["teacher_forcing_update_prob"] = torch.tensor(teacher_forcing_prob, device=images_BTCHW.device)
            if "dynakey_aux" in memory_aux:
                memory_aux["dynakey_aux"]["temporal_delta_norm"] = temporal_delta.detach().pow(2).mean(dim=(2, 3, 4))
                memory_aux["dynakey_aux"]["temporal_gate_alpha"] = self.temporal_residual_scale_logit.detach().sigmoid()
        return out
