from __future__ import annotations

from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from utils.tensor_utils import aggregate, cls_to_one_hot
from utils.frame_validity import build_default_endpoint_mask, mask_to_frame_ids, normalize_frame_validity_mask

from .base import ce_loss, dice_loss
from .cardia import _compute_cardia_losses
from .geomaskformer import compute_geomaskformer_losses
from debel.losses import compute_debel_losses
from rebel.losses import compute_rebel_losses


class LossComputer(nn.Module):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

        unext_dynakey_cfg = cfg.model.get("unext_dynakey", {})
        self.enable_memory_only_loss = bool(unext_dynakey_cfg.get("memory_only_head_enabled", False))
        self.lambda_memory_only = float(unext_dynakey_cfg.get("lambda_memory_only", 0.0))

        cardia_cfg = cfg.model.get("cardia", {})
        self.is_cardia = str(cfg.model.get("name", "")).lower() in {"cardia", "unext_cardia"}
        self.lambda_cardia_base = float(cardia_cfg.get("lambda_cardia_base", 0.1))
        self.lambda_cardia_proposal_oracle = float(cardia_cfg.get("lambda_cardia_proposal_oracle", 0.2))
        self.lambda_cardia_proposal_top1 = float(cardia_cfg.get("lambda_cardia_proposal_top1", 0.0))
        legacy_selector = float(cardia_cfg.get("lambda_cardia_selector", 0.1))
        legacy_margin = float(cardia_cfg.get("lambda_cardia_selector_margin", 0.0))
        self.lambda_cardia_selector_global = float(cardia_cfg.get("lambda_cardia_selector_global", legacy_selector))
        self.lambda_cardia_selector_spatial = float(cardia_cfg.get("lambda_cardia_selector_spatial", 0.0))
        self.lambda_cardia_selector_margin_global = float(cardia_cfg.get("lambda_cardia_selector_margin_global", legacy_margin))
        self.lambda_cardia_selector_margin_spatial = float(cardia_cfg.get("lambda_cardia_selector_margin_spatial", 0.0))
        self.lambda_cardia_selector = self.lambda_cardia_selector_global + self.lambda_cardia_selector_spatial
        self.lambda_cardia_selector_margin = self.lambda_cardia_selector_margin_global + self.lambda_cardia_selector_margin_spatial
        self.cardia_selector_margin = float(cardia_cfg.get("selector_margin", 0.2))
        self.lambda_cardia_flow_smooth = float(cardia_cfg.get("lambda_cardia_flow_smooth", 0.005))
        self.lambda_cardia_stage3_flow_smooth = float(
            cardia_cfg.get("lambda_cardia_stage3_flow_smooth", self.lambda_cardia_flow_smooth)
        )
        self.lambda_cardia_stage2_flow_smooth = float(
            cardia_cfg.get("lambda_cardia_stage2_flow_smooth", self.lambda_cardia_flow_smooth)
        )
        self.cardia_flow_smooth_warmup_iters = int(cardia_cfg.get("flow_smooth_warmup_iters", 500))
        self.lambda_cardia_boundary_aux = float(cardia_cfg.get("lambda_cardia_boundary_aux", 0.02))
        self.lambda_cardia_memory_readout = float(cardia_cfg.get("lambda_cardia_memory_readout", 0.0))
        self.lambda_cardia_memory_readout_stage3 = float(cardia_cfg.get("lambda_cardia_memory_readout_stage3", 0.0))
        self.lambda_cardia_reliability_write = float(cardia_cfg.get("lambda_cardia_reliability_write", 0.0))
        self.cardia_boundary_dilation_kernel = int(cardia_cfg.get("boundary_dilation_kernel", 5))
        if self.cardia_boundary_dilation_kernel < 3 or self.cardia_boundary_dilation_kernel % 2 == 0:
            self.cardia_boundary_dilation_kernel = 5
        self.cardia_selector_temperature = float(cardia_cfg.get("selector_temperature", 0.1))
        self.cardia_proposal_softmin_temperature = float(cardia_cfg.get("proposal_softmin_temperature", 0.3))
        self.cardia_proposal_loss = str(cardia_cfg.get("proposal_loss", "softmin")).lower()
        self.cardia_base_after_iter = int(cardia_cfg.get("base_aux_after_iter", 800))
        self.cardia_base_after_weight = float(cardia_cfg.get("lambda_cardia_base_after", self.lambda_cardia_base))
        self.cardia_oracle_decay_start = int(cardia_cfg.get("oracle_decay_start_iter", 2500))
        self.cardia_oracle_after_weight = float(cardia_cfg.get("lambda_cardia_proposal_oracle_after", self.lambda_cardia_proposal_oracle))
        self.lambda_head_diversity = float(cardia_cfg.get("lambda_head_diversity", 0.0))
        self.lambda_cardia_multi_head_fused = float(cardia_cfg.get("lambda_cardia_multi_head_fused", 0.0))
        self.is_rebel = str(cfg.model.get("name", "")).lower() in {"rebel", "resampled_belief"}
        self.rebel_loss_cfg = cfg.get("loss", {}).get("rebel", cfg.model.get("rebel", {}).get("loss", {}))
        self.lambda_rebel_final = float(self.rebel_loss_cfg.get("final", 1.0))
        self.lambda_rebel_base_aux = float(self.rebel_loss_cfg.get("base_aux", 0.35))
        self.lambda_rebel_belief_prior = float(self.rebel_loss_cfg.get("belief_prior", 0.15))
        self.lambda_rebel_obs_aux = float(self.rebel_loss_cfg.get("obs_aux", 0.20))
        self.lambda_rebel_rebel_aux = float(self.rebel_loss_cfg.get("rebel_aux", 0.10))
        self.lambda_rebel_corrected_aux = float(self.rebel_loss_cfg.get("corrected_aux", 0.05))
        self.lambda_rebel_candidate_oracle = float(self.rebel_loss_cfg.get("candidate_oracle", 0.15))
        self.lambda_rebel_arbitration = float(self.rebel_loss_cfg.get("arbitration", 0.05))
        self.lambda_rebel_correction = float(self.rebel_loss_cfg.get("correction", 0.05))
        self.lambda_rebel_temporal = float(self.rebel_loss_cfg.get("temporal", 0.03))
        self.lambda_rebel_offset_smooth = float(self.rebel_loss_cfg.get("offset_smooth", 0.005))
        self.lambda_rebel_write_reg = float(self.rebel_loss_cfg.get("write_reg", 0.01))
        self.is_debel = str(cfg.model.get("name", "")).lower() == "debel"
        self.debel_loss_cfg = cfg.get("loss", {}).get("debel", cfg.model.get("debel", {}).get("loss", {}))
        self.lambda_debel_final = float(self.debel_loss_cfg.get("lambda_final", self.debel_loss_cfg.get("final", 1.0)))
        self.lambda_debel_anchor = float(self.debel_loss_cfg.get("lambda_anchor", self.debel_loss_cfg.get("anchor", 0.5)))
        self.lambda_debel_grid = float(self.debel_loss_cfg.get("lambda_grid", self.debel_loss_cfg.get("grid", 0.01)))
        self.lambda_debel_smooth = float(self.debel_loss_cfg.get("lambda_smooth", self.debel_loss_cfg.get("smooth", 0.02)))
        self.lambda_debel_temp = float(self.debel_loss_cfg.get("lambda_temp", self.debel_loss_cfg.get("temp", 0.01)))
        self.lambda_debel_area = float(self.debel_loss_cfg.get("lambda_area", self.debel_loss_cfg.get("area", 0.001)))
        self.lambda_debel_residual = float(self.debel_loss_cfg.get("lambda_residual", self.debel_loss_cfg.get("residual", 0.005)))
        self.is_geomaskformer = str(cfg.model.get("name", "")).lower() in {"geomaskformer", "geo_maskformer"}
        self.geomaskformer_loss_cfg = cfg.get("loss", {}).get("geomaskformer", cfg.model.get("geomaskformer", {}).get("loss", {}))
        self.lambda_geomaskformer_mask = float(self.geomaskformer_loss_cfg.get("mask", 1.0))
        self.lambda_geomaskformer_boundary = float(self.geomaskformer_loss_cfg.get("boundary", 0.2))
        self.lambda_geomaskformer_score = float(self.geomaskformer_loss_cfg.get("score", 0.5))
        self.lambda_geomaskformer_temporal = float(self.geomaskformer_loss_cfg.get("temporal", 0.03))
        self.lambda_geomaskformer_visible_reconstruction = float(
            self.geomaskformer_loss_cfg.get("visible_reconstruction", 0.0)
        )
        self.geomaskformer_topk_loss = int(self.geomaskformer_loss_cfg.get("topk", 4))
        self.geomaskformer_boundary_kernel = int(self.geomaskformer_loss_cfg.get("boundary_kernel", 5))
        if self.geomaskformer_boundary_kernel < 3 or self.geomaskformer_boundary_kernel % 2 == 0:
            self.geomaskformer_boundary_kernel = 5

    def _default_supervision_mask(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
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

    def mask_loss(
        self, logits: torch.Tensor, soft_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.point_supervision:
            loss_ce = ce_loss(logits, soft_gt)
            loss_dice = dice_loss(logits.softmax(dim=1), soft_gt)
            return loss_ce, loss_dice

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, lambda x: calculate_uncertainty(x),
                self.num_points, self.oversample_ratio, self.importance_sample_ratio
            )
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)

        point_logits = point_sample(logits, point_coords, align_corners=False)

        loss_ce = ce_loss(point_logits, point_labels)
        loss_dice = dice_loss(point_logits.softmax(dim=1), point_labels)

        return loss_ce, loss_dice

    def frame_mask_loss(
        self,
        logits_TCHW: torch.Tensor,
        soft_gt_TCHW: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_ce_list = []
        loss_dice_list = []
        for t in range(logits_TCHW.shape[0]):
            ce_t, dice_t = self.mask_loss(logits_TCHW[t:t+1], soft_gt_TCHW[t:t+1])
            loss_ce_list.append(ce_t)
            loss_dice_list.append(dice_t)
        return torch.stack(loss_ce_list), torch.stack(loss_dice_list)

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        supervised_mask = self._resolve_supervision_mask(
            data.get('supervised_indices'),
            batch_size=batch_size,
            num_frames=num_frames,
            device=data['rgb'].device,
        )

        for bi in range(batch_size):
            t_range = self._frame_ids_for_sample(supervised_mask, bi)
            if not t_range:
                raise ValueError(f"Sample {bi} has no supervised frames")
            curr_num_obj = num_objects[bi]
            valid_slice = slice(None, curr_num_obj + 1)

            logits = torch.stack(
                [data[f'logits_{ti}'][bi, valid_slice] for ti in t_range], dim=0
            )

            cls_gt = data['cls_gt'][bi, t_range]
            if cls_gt.dim() == 3:
                cls_gt = cls_gt.unsqueeze(1)
            soft_gt = cls_to_one_hot(cls_gt, curr_num_obj)

            frame_ce, frame_dice = self.frame_mask_loss(logits, soft_gt)
            loss_ce = frame_ce.mean()
            loss_dice = frame_dice.mean()
            losses['loss_ce'] += loss_ce / batch_size
            losses['loss_dice'] += loss_dice / batch_size
            losses['seg_quality'] += (-(frame_ce + frame_dice).mean()).detach() / batch_size

            aux_list = [data[f'aux_{ti}'] for ti in t_range]
            first_aux = aux_list[0]

            if 'sensory_logits' in first_aux:
                sensory_log = torch.stack(
                    [a['sensory_logits'][bi, valid_slice] for a in aux_list], dim=0
                )
                l_ce, l_dice = self.mask_loss(sensory_log, soft_gt)
                losses['aux_sensory_ce'] += l_ce / batch_size * self.sensory_weight
                losses['aux_sensory_dice'] += l_dice / batch_size * self.sensory_weight

            if 'q_logits' in first_aux:
                num_levels = first_aux['q_logits'].shape[2]

                for level_idx in range(num_levels):
                    query_log = torch.stack(
                        [a['q_logits'][bi, valid_slice, level_idx] for a in aux_list], dim=0
                    )

                    l_ce, l_dice = self.mask_loss(query_log, soft_gt)

                    losses[f'aux_query_ce_l{level_idx}'] += l_ce / batch_size * self.query_weight
                    losses[f'aux_query_dice_l{level_idx}'] += l_dice / batch_size * self.query_weight

            if self.enable_memory_only_loss and self.lambda_memory_only > 0:
                memory_logits_frames = []
                for a in aux_list:
                    mem_logits = a.get("memory_only_logits") if isinstance(a, dict) else None
                    if torch.is_tensor(mem_logits):
                        memory_logits_frames.append(mem_logits[bi, :curr_num_obj])
                if len(memory_logits_frames) == len(t_range):
                    memory_obj = torch.stack(memory_logits_frames, dim=0)
                    memory_masks = torch.sigmoid(memory_obj)
                    memory_logits = aggregate(memory_masks, dim=1)
                    l_ce, l_dice = self.frame_mask_loss(memory_logits, soft_gt)
                    losses["aux_memory_only_ce"] += l_ce.mean() / batch_size * self.lambda_memory_only
                    losses["aux_memory_only_dice"] += l_dice.mean() / batch_size * self.lambda_memory_only

        cardia_terms = _compute_cardia_losses(self, data, supervised_mask)
        for k, v in cardia_terms.items():
            losses[k] += v
        if self.is_rebel:
            for k, v in compute_rebel_losses(self, data, supervised_mask).items():
                losses[k] += v
        if self.is_debel:
            for k, v in compute_debel_losses(self, data, supervised_mask).items():
                losses[k] += v
        if self.is_geomaskformer:
            for k, v in compute_geomaskformer_losses(self, data, supervised_mask).items():
                losses[k] += v

        total_loss = torch.zeros((), device=data["rgb"].device, dtype=torch.float32)
        for key, value in losses.items():
            if not (torch.is_tensor(value) or isinstance(value, (float, int))):
                continue
            if key.startswith("loss_") or key.startswith("aux_") or key.startswith("rebel_") or key.startswith("debel_") or key in {"dynakey_q_total", "spatial_q_total"}:
                total_loss = total_loss + value
        losses['total_loss'] = total_loss

        return losses
