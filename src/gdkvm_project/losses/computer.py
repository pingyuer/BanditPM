from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dpfr.losses import compute_dpfr_losses
from losses.base import ce_loss, dice_loss
from utils.frame_validity import build_default_endpoint_mask, mask_to_frame_ids, normalize_frame_validity_mask
from utils.point_features import calculate_uncertainty, get_uncertain_point_coords_with_randomness, point_sample
from utils.tensor_utils import cls_to_one_hot


class LossComputer(nn.Module):
    """Loss facade for the public GDKVM/DPFR project surface."""

    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = bool(stage_cfg.point_supervision)
        self.num_points = int(stage_cfg.train_num_points)
        self.oversample_ratio = float(stage_cfg.oversample_ratio)
        self.importance_sample_ratio = float(stage_cfg.importance_sample_ratio)

        aux_cfg = cfg.model.get("aux_loss", {}) if hasattr(cfg.model, "get") else {}
        self.sensory_weight = float(aux_cfg.get("sensory", {}).get("weight", 0.0))
        self.query_weight = float(aux_cfg.get("query", {}).get("weight", 0.0))

        self.model_name = str(cfg.model.get("name", "")).lower()
        self.is_dpfr = self.model_name in {"dpfr", "dual_prompt_flow_refinement"}
        self.dpfr_loss_cfg = cfg.get("loss", {}).get("dpfr", cfg.model.get("dpfr", {}).get("loss", {}))
        self.lambda_dpfr_final = float(self.dpfr_loss_cfg.get("lambda_final", self.dpfr_loss_cfg.get("final", 1.0)))
        self.lambda_dpfr_anchor = float(self.dpfr_loss_cfg.get("lambda_anchor", self.dpfr_loss_cfg.get("anchor", 0.3)))
        self.lambda_dpfr_prompt = float(self.dpfr_loss_cfg.get("lambda_prompt", self.dpfr_loss_cfg.get("prompt", 0.5)))
        self.lambda_dpfr_flow_seg = float(self.dpfr_loss_cfg.get("lambda_flow_seg", self.dpfr_loss_cfg.get("flow_seg", 0.2)))
        self.lambda_dpfr_flow_mag = float(self.dpfr_loss_cfg.get("lambda_flow_mag", self.dpfr_loss_cfg.get("flow_mag", 0.005)))
        self.lambda_dpfr_flow_smooth = float(self.dpfr_loss_cfg.get("lambda_flow_smooth", self.dpfr_loss_cfg.get("flow_smooth", 0.01)))
        self.lambda_dpfr_flow_temp = float(self.dpfr_loss_cfg.get("lambda_flow_temp", self.dpfr_loss_cfg.get("flow_temp", 0.01)))

    def _default_supervision_mask(self, batch_size: int, num_frames: int, device: torch.device) -> torch.Tensor:
        return build_default_endpoint_mask(batch_size, num_frames, device=device)

    def _resolve_supervision_mask(
        self,
        supervised_indices: torch.Tensor | None,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if supervised_indices is None:
            return self._default_supervision_mask(batch_size, num_frames, device)
        return normalize_frame_validity_mask(
            supervised_indices,
            batch_size=batch_size,
            total_frames=num_frames,
            device=device,
        )

    def _frame_ids_for_sample(self, supervised_mask: torch.Tensor, sample_idx: int) -> list[int]:
        return mask_to_frame_ids(supervised_mask[sample_idx])

    def mask_loss(self, logits: torch.Tensor, soft_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.point_supervision:
            return ce_loss(logits, soft_gt), dice_loss(logits.softmax(dim=1), soft_gt)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                logits,
                lambda x: calculate_uncertainty(x),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)
        point_logits = point_sample(logits, point_coords, align_corners=False)
        return ce_loss(point_logits, point_labels), dice_loss(point_logits.softmax(dim=1), point_labels)

    def frame_mask_loss(self, logits_TCHW: torch.Tensor, soft_gt_TCHW: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ce_items = []
        dice_items = []
        for ti in range(logits_TCHW.shape[0]):
            ce_t, dice_t = self.mask_loss(logits_TCHW[ti : ti + 1], soft_gt_TCHW[ti : ti + 1])
            ce_items.append(ce_t)
            dice_items.append(dice_t)
        return torch.stack(ce_items), torch.stack(dice_items)

    def compute(self, data: Dict[str, torch.Tensor], num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data["rgb"].shape[:2]
        losses = defaultdict(float)
        supervised_mask = self._resolve_supervision_mask(
            data.get("supervised_indices"),
            batch_size=batch_size,
            num_frames=num_frames,
            device=data["rgb"].device,
        )

        for bi in range(batch_size):
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            if not frame_ids:
                raise ValueError(f"Sample {bi} has no supervised frames")
            curr_num_obj = int(num_objects[bi])
            valid_slice = slice(None, curr_num_obj + 1)
            logits = torch.stack([data[f"logits_{ti}"][bi, valid_slice] for ti in frame_ids], dim=0)
            cls_gt = data["cls_gt"][bi, frame_ids]
            if cls_gt.dim() == 3:
                cls_gt = cls_gt.unsqueeze(1)
            soft_gt = cls_to_one_hot(cls_gt, curr_num_obj)
            frame_ce, frame_dice = self.frame_mask_loss(logits, soft_gt)
            losses["loss_ce"] += frame_ce.mean() / batch_size
            losses["loss_dice"] += frame_dice.mean() / batch_size
            losses["seg_quality"] += (-(frame_ce + frame_dice).mean()).detach() / batch_size
            if self.sensory_weight > 0.0 or self.query_weight > 0.0:
                self._add_gdkvm_aux_losses(losses, data, bi, frame_ids, valid_slice, soft_gt, batch_size)

        if self.is_dpfr:
            for key, value in compute_dpfr_losses(self, data, supervised_mask).items():
                losses[key] += value

        total_loss = torch.zeros((), device=data["rgb"].device, dtype=torch.float32)
        for key, value in losses.items():
            if not (torch.is_tensor(value) or isinstance(value, (float, int))):
                continue
            if self.is_dpfr:
                if key.startswith("dpfr_"):
                    total_loss = total_loss + value
            elif key.startswith("loss_") or key.startswith("aux_"):
                total_loss = total_loss + value
        losses["total_loss"] = total_loss
        return dict(losses)

    def _add_gdkvm_aux_losses(self, losses, data, bi, frame_ids, valid_slice, soft_gt, batch_size: int) -> None:
        aux_list = [data.get(f"aux_{ti}", {}) for ti in frame_ids]
        first_aux = aux_list[0] if aux_list else {}
        if self.sensory_weight > 0.0 and "sensory_logits" in first_aux:
            sensory_log = torch.stack([a["sensory_logits"][bi, valid_slice] for a in aux_list], dim=0)
            if sensory_log.shape[-2:] == soft_gt.shape[-2:]:
                ce_t, dice_t = self.mask_loss(sensory_log, soft_gt)
                losses["aux_sensory_ce"] += ce_t / batch_size * self.sensory_weight
                losses["aux_sensory_dice"] += dice_t / batch_size * self.sensory_weight
        if self.query_weight > 0.0 and "q_logits" in first_aux:
            num_levels = first_aux["q_logits"].shape[2]
            for level_idx in range(num_levels):
                query_log = torch.stack([a["q_logits"][bi, valid_slice, level_idx] for a in aux_list], dim=0)
                if query_log.shape[-2:] == soft_gt.shape[-2:]:
                    ce_t, dice_t = self.mask_loss(query_log, soft_gt)
                    losses[f"aux_query_ce_l{level_idx}"] += ce_t / batch_size * self.query_weight
                    losses[f"aux_query_dice_l{level_idx}"] += dice_t / batch_size * self.query_weight
