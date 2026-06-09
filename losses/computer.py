from typing import List, Dict, Tuple
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from utils.tensor_utils import aggregate, cls_to_one_hot
from utils.frame_validity import build_default_endpoint_mask, mask_to_frame_ids, normalize_frame_validity_mask

@torch.jit.script
def ce_loss(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, soft_gt, reduction='none')
    return loss.sum(0).mean()


@torch.jit.script
def dice_loss(mask: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    mask = mask[:, 1:].flatten(start_dim=2).contiguous()
    gt = soft_gt[:, 1:].float().flatten(start_dim=2).contiguous()
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum(0).mean()


class LossComputer(nn.Module):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight
        bpm_cfg = cfg.model.temporal_memory.get("bpm", {})
        self.enable_policy_ce_loss = bool(bpm_cfg.get("ENABLE_POLICY_CE_LOSS", bpm_cfg.get("ENABLE_POLICY_LOSS", False)))
        self.enable_rl_loss = bool(bpm_cfg.get("ENABLE_RL_LOSS", False))
        self.lambda_policy_ce = float(bpm_cfg.get("LAMBDA_POLICY_CE", bpm_cfg.get("POLICY_LOSS_WEIGHT", 0.0)))
        self.lambda_rl = float(bpm_cfg.get("LAMBDA_RL", 0.0))
        self.lambda_entropy = float(bpm_cfg.get("LAMBDA_ENTROPY", 0.0))
        self.rl_on_supervised_only = bool(bpm_cfg.get("RL_ON_SUPERVISED_FRAMES_ONLY", True))
        self.adv_clamp = float(bpm_cfg.get("ADV_CLAMP", 1.0))
        self.rl_baseline_momentum = float(bpm_cfg.get("RL_BASELINE_MOMENTUM", 0.95))
        self.register_buffer("action_reward_baseline", torch.zeros(4, dtype=torch.float32), persistent=True)
        dynakey_cfg = cfg.model.get("memory_core", {}).get("dynakey", {})
        self.enable_dynakey_q_loss = bool(dynakey_cfg.get("ENABLE_Q_LOSS", False))
        self.lambda_dynakey_q_ce = float(dynakey_cfg.get("LAMBDA_Q_CE", 1.0))
        self.lambda_dynakey_q_adv = float(dynakey_cfg.get("LAMBDA_Q_ADV", 0.0))
        self.dynakey_advantage_clamp = float(dynakey_cfg.get("ADVANTAGE_CLAMP", 5.0))
        unext_dynakey_cfg = cfg.model.get("unext_dynakey", {})
        self.spatial_q_policy_mode = str(unext_dynakey_cfg.get("q_policy_mode", "off")).lower()
        self.enable_spatial_q_loss = bool(unext_dynakey_cfg.get("enable_q_loss", self.spatial_q_policy_mode == "training"))
        self.lambda_spatial_q_ce = float(unext_dynakey_cfg.get("lambda_q_ce", 1.0))
        self.enable_memory_only_loss = bool(unext_dynakey_cfg.get("memory_only_head_enabled", False))
        self.lambda_memory_only = float(unext_dynakey_cfg.get("lambda_memory_only", 0.0))
        delay_ode_cfg = cfg.model.get("delay_ode", {})
        self.is_delay_ode = str(cfg.model.get("name", "")).lower() == "delay_ode"
        self.delay_ode_supervise_first_frame = bool(delay_ode_cfg.get("delay_ode_supervise_first_frame", False))
        self.lambda_delay_ode_slot_balance = float(
            delay_ode_cfg.get(
                "delay_ode_lambda_slot_balance",
                delay_ode_cfg.get("delay_ode_lambda_selection_entropy", 0.0),
            )
        )
        self.lambda_delay_ode_gate_smooth = float(delay_ode_cfg.get("delay_ode_lambda_gate_smooth", 0.0))
        self.lambda_delay_ode_latent_smooth = float(delay_ode_cfg.get("delay_ode_lambda_latent_smooth", 0.0))
        self.lambda_delay_ode_state_smooth = float(delay_ode_cfg.get("delay_ode_lambda_state_smooth", 0.0))
        self.lambda_delay_ode_phase_slot_usage = float(delay_ode_cfg.get("delay_ode_lambda_phase_slot_usage", 0.0))
        self.lambda_delay_ode_motion_scale_smooth = float(delay_ode_cfg.get("delay_ode_lambda_motion_scale_smooth", 0.0))
        self.lambda_delay_ode_latent_decode = {
            "low": float(delay_ode_cfg.get("delay_ode_lambda_latent_decode_low", 0.0)),
            "mid": float(delay_ode_cfg.get("delay_ode_lambda_latent_decode_mid", 0.0)),
            "high": float(delay_ode_cfg.get("delay_ode_lambda_latent_decode_high", 0.0)),
        }
        self.lambda_delay_ode_boundary = float(delay_ode_cfg.get("delay_ode_lambda_boundary", 0.0))
        anchor_ode_cfg = cfg.model.get("anchor_ode", {})
        self.is_anchor_ode = str(cfg.model.get("name", "")).lower() in {
            "anchor_ode",
            "unext_anchor_ode",
            "unextanchorode",
            "anchor_ode_v2",
            "unext_anchor_ode_affine",
            "unextanchorodeaffine",
        }
        self.lambda_anchor_ode_base_seg = float(anchor_ode_cfg.get("lambda_base_seg", 0.0))
        self.lambda_anchor_ode_guided_seg = float(anchor_ode_cfg.get("lambda_guided_seg", 0.0))
        self.lambda_anchor_ode_prior = float(anchor_ode_cfg.get("lambda_prior", 0.0))
        self.lambda_anchor_ode_warp_prior = float(anchor_ode_cfg.get("lambda_warp_prior", 0.0))
        self.lambda_anchor_ode_multiscale_prior = float(anchor_ode_cfg.get("lambda_multiscale_prior", 0.0))
        self.lambda_anchor_ode_geo = float(anchor_ode_cfg.get("lambda_geo", 0.0))
        self.lambda_anchor_ode_temp_geo = float(anchor_ode_cfg.get("lambda_temp_geo", 0.0))
        self.lambda_anchor_ode_conf = float(anchor_ode_cfg.get("lambda_conf", 0.0))
        self.lambda_anchor_ode_slot_balance = float(anchor_ode_cfg.get("lambda_slot_balance", 0.0))
        self.lambda_anchor_ode_affine_reg = float(anchor_ode_cfg.get("lambda_affine_reg", 0.0))
        functional_cfg = cfg.model.get("functional_anchor", {})
        self.is_functional_anchor = str(cfg.model.get("name", "")).lower() == "functional_anchor"
        self.lambda_functional_anchor_anchor = float(functional_cfg.get("lambda_anchor", 0.5))
        self.lambda_functional_anchor_base = float(functional_cfg.get("lambda_base_seg", 0.1))
        self.lambda_functional_anchor_residual_l1 = float(functional_cfg.get("lambda_residual_smallness", 0.02))
        self.lambda_functional_anchor_boundary = float(functional_cfg.get("lambda_boundary_residual", 0.1))
        self.lambda_functional_anchor_phase = float(functional_cfg.get("lambda_phase_consistency", 0.02))
        self.lambda_functional_anchor_temp = float(functional_cfg.get("lambda_anchor_temporal", 0.02))
        self.lambda_functional_anchor_slot_order = float(functional_cfg.get("lambda_slot_area_order", 0.01))
        self.lambda_functional_anchor_phase_slot = float(functional_cfg.get("lambda_phase_slot_correlation", 0.01))
        self.lambda_functional_anchor_trust_l1 = float(functional_cfg.get("trust_l1_weight", 0.0))
        self.lambda_functional_anchor_trust_entropy = float(functional_cfg.get("trust_entropy_weight", 0.0))
        self.lambda_functional_anchor_ode_raw_delta = float(functional_cfg.get("lambda_ode_raw_delta", 0.0))
        faf_cfg = cfg.model.get("unext_faf", {})
        self.is_faf = str(cfg.model.get("name", "")).lower() in {
            "unext_faf",
            "unext-faf",
            "faf",
            "unext_ode_affine",
            "unext-ode-affine",
            "ode_affine",
        }
        self.lambda_faf_mixture = float(faf_cfg.get("lambda_faf_mixture", 0.2))
        self.lambda_faf_oracle = float(faf_cfg.get("lambda_faf_oracle", 0.2))
        self.lambda_faf_top1 = float(faf_cfg.get("lambda_faf_top1", 0.0))
        self.lambda_faf_selector = float(faf_cfg.get("lambda_faf_selector", 0.1))
        self.lambda_faf_confidence = float(faf_cfg.get("lambda_faf_confidence", 0.05))
        self.lambda_faf_base = float(faf_cfg.get("lambda_faf_base", 1.0))
        self.lambda_faf_coverage = float(faf_cfg.get("lambda_faf_coverage", 0.05))
        self.lambda_faf_sparse = float(faf_cfg.get("lambda_faf_sparse", 0.002))
        self.lambda_faf_diversity = float(faf_cfg.get("lambda_faf_diversity", 0.005))
        self.lambda_faf_temporal = float(faf_cfg.get("lambda_faf_temporal", 0.02))
        self.lambda_faf_write = float(faf_cfg.get("lambda_faf_write", 0.001))
        self.lambda_faf_residual_smallness = float(faf_cfg.get("lambda_faf_residual_smallness", 0.05))
        self.lambda_faf_affine = float(faf_cfg.get("lambda_faf_affine", 0.0))
        self.lambda_faf_velocity = float(faf_cfg.get("lambda_faf_velocity", 0.0))
        self.lambda_faf_feature_modulation = float(faf_cfg.get("lambda_faf_feature_modulation", 0.001))
        self.lambda_faf_dense_flow = float(faf_cfg.get("lambda_faf_dense_flow", 0.0))
        self.lambda_faf_dense_smooth = float(faf_cfg.get("lambda_faf_dense_smooth", 0.0))
        selector_cfg = faf_cfg.get("selector", {})
        self.faf_assignment_temperature = float(selector_cfg.get("assignment_temperature", 0.15)) if hasattr(selector_cfg, "get") else 0.15
        self.faf_residual_smallness_start_iter = int(faf_cfg.get("residual_smallness_start_iter", 0))
        gar_cfg = cfg.model.get("unext_gar", {})
        self.is_gar = str(cfg.model.get("name", "")).lower() in {
            "unext_gar",
            "grid_anchor_router",
            "gar",
        }
        self.lambda_gar_base = float(gar_cfg.get("lambda_gar_base", 0.1))
        self.lambda_gar_proposal_oracle = float(gar_cfg.get("lambda_gar_proposal_oracle", 0.2))
        self.lambda_gar_selector = float(gar_cfg.get("lambda_gar_selector", 0.1))
        self.lambda_gar_flow_smooth = float(gar_cfg.get("lambda_gar_flow_smooth", 0.005))
        self.lambda_gar_boundary_aux = float(gar_cfg.get("lambda_gar_boundary_aux", 0.02))
        self.gar_selector_temperature = float(gar_cfg.get("selector_temperature", 0.1))
        self.gar_proposal_softmin_temperature = float(gar_cfg.get("proposal_softmin_temperature", 0.3))
        self.gar_proposal_loss = str(gar_cfg.get("proposal_loss", "best")).lower()
        self.gar_base_after_iter = int(gar_cfg.get("base_aux_after_iter", 800))
        self.gar_base_after_weight = float(gar_cfg.get("lambda_gar_base_after", self.lambda_gar_base))
        self.gar_oracle_decay_start = int(gar_cfg.get("oracle_decay_start_iter", 2500))
        self.gar_oracle_after_weight = float(gar_cfg.get("lambda_gar_proposal_oracle_after", self.lambda_gar_proposal_oracle))
        cardia_cfg = cfg.model.get("cardia", {})
        self.is_cardia = str(cfg.model.get("name", "")).lower() in {"cardia", "unext_cardia"}
        self.lambda_cardia_base = float(cardia_cfg.get("lambda_cardia_base", 0.1))
        self.lambda_cardia_proposal_oracle = float(cardia_cfg.get("lambda_cardia_proposal_oracle", 0.2))
        self.lambda_cardia_selector = float(cardia_cfg.get("lambda_cardia_selector", 0.1))
        self.lambda_cardia_flow_smooth = float(cardia_cfg.get("lambda_cardia_flow_smooth", 0.005))
        self.lambda_cardia_boundary_aux = float(cardia_cfg.get("lambda_cardia_boundary_aux", 0.02))
        self.cardia_selector_temperature = float(cardia_cfg.get("selector_temperature", 0.1))
        self.cardia_proposal_softmin_temperature = float(cardia_cfg.get("proposal_softmin_temperature", 0.3))
        self.cardia_proposal_loss = str(cardia_cfg.get("proposal_loss", "softmin")).lower()
        self.cardia_base_after_iter = int(cardia_cfg.get("base_aux_after_iter", 800))
        self.cardia_base_after_weight = float(cardia_cfg.get("lambda_cardia_base_after", self.lambda_cardia_base))
        self.cardia_oracle_decay_start = int(cardia_cfg.get("oracle_decay_start_iter", 2500))
        self.cardia_oracle_after_weight = float(cardia_cfg.get("lambda_cardia_proposal_oracle_after", self.lambda_cardia_proposal_oracle))

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
        if self.is_delay_ode and not self.delay_ode_supervise_first_frame and num_frames > 0:
            supervised_mask[:, 0] = False

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

            bpm_aux_list = [self._slice_aux_for_sample(data.get(f'bpm_aux_{ti}'), bi, batch_size) for ti in t_range]
            rl_terms = self._compute_policy_and_rl_losses(
                bpm_aux_list=bpm_aux_list,
                frame_seg_loss=(frame_ce + frame_dice).detach(),
                device=logits.device,
            )
            for k, v in rl_terms.items():
                losses[k] += v / batch_size if torch.is_tensor(v) else v / batch_size

        dynakey_q_terms = self._compute_dynakey_q_loss(data)
        for k, v in dynakey_q_terms.items():
            losses[k] += v
        spatial_q_terms = self._compute_spatial_q_loss(data)
        for k, v in spatial_q_terms.items():
            losses[k] += v
        delay_ode_terms = self._compute_delay_ode_regularizers(data)
        for k, v in delay_ode_terms.items():
            losses[k] += v
        anchor_ode_terms = self._compute_anchor_ode_losses(data, supervised_mask)
        for k, v in anchor_ode_terms.items():
            losses[k] += v
        functional_anchor_terms = self._compute_functional_anchor_losses(data, supervised_mask)
        for k, v in functional_anchor_terms.items():
            losses[k] += v
        faf_terms = self._compute_faf_losses(data, supervised_mask)
        for k, v in faf_terms.items():
            losses[k] += v
        gar_terms = self._compute_gar_losses(data, supervised_mask)
        for k, v in gar_terms.items():
            losses[k] += v
        cardia_terms = self._compute_cardia_losses(data, supervised_mask)
        for k, v in cardia_terms.items():
            losses[k] += v

        total_loss = torch.zeros((), device=data["rgb"].device, dtype=torch.float32)
        for key, value in losses.items():
            if not (torch.is_tensor(value) or isinstance(value, (float, int))):
                continue
            if key.startswith("loss_") or key.startswith("aux_") or key in {"dynakey_q_total", "spatial_q_total"} or key in {"rl_loss", "entropy_reg"}:
                total_loss = total_loss + value
        losses['total_loss'] = total_loss

        if self.enable_policy_ce_loss and self.lambda_policy_ce > 0:
            policy_loss = self._compute_policy_loss(data)
            if policy_loss is not None:
                losses['policy_ce'] = policy_loss * self.lambda_policy_ce
                losses['total_loss'] = losses['total_loss'] + losses['policy_ce']

        return losses

    def _compute_faf_losses(
        self,
        data: Dict[str, torch.Tensor],
        supervised_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_faf or "cls_gt" not in data:
            return {}

        batch_size = data["rgb"].shape[0]
        device = data["rgb"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        out: Dict[str, torch.Tensor] = {}
        oracle_terms = []
        top1_terms = []
        mixture_terms = []
        selector_terms = []
        confidence_terms = []
        base_terms = []
        coverage_terms = []
        sparse_terms = []
        diversity_terms = []
        temporal_terms = []
        write_terms = []
        residual_terms = []
        affine_terms = []
        velocity_terms = []
        modulation_terms = []
        dense_flow_terms = []
        dense_smooth_terms = []
        current_iter = int(data.get("global_step", data.get("current_iter", 0)) or 0)
        residual_smallness_weight = (
            self.lambda_faf_residual_smallness
            if current_iter >= self.faf_residual_smallness_start_iter
            else 0.0
        )

        for bi in range(batch_size):
            curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            prev_proposal = None
            for ti in frame_ids:
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("faf_aux") if isinstance(memory_aux, dict) else None
                if not isinstance(aux, dict):
                    continue
                soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)
                gt_mask = self._gt_object_masks(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

                warped = aux.get("warped_anchor_logits")
                slot_weights = aux.get("slot_weights")
                slot_logits = aux.get("slot_logits")
                slot_confidence = aux.get("slot_confidence")
                per_slot_dice = None
                if torch.is_tensor(warped):
                    proposals = warped[bi : bi + 1, :curr_num_obj]
                    gt = gt_mask[:, :curr_num_obj].to(device=proposals.device, dtype=proposals.dtype).unsqueeze(2)
                    bce = F.binary_cross_entropy_with_logits(proposals, gt.expand_as(proposals), reduction="none").mean(dim=(-2, -1))
                    pred = torch.sigmoid(proposals)
                    inter = (pred * gt).sum(dim=(-2, -1))
                    denom = pred.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
                    per_slot_dice = ((2.0 * inter + 1.0) / (denom + 1.0)).detach()
                    if self.lambda_faf_oracle > 0:
                        oracle_terms.append(bce.min(dim=-1).values.mean())
                    if torch.is_tensor(slot_weights) and self.lambda_faf_top1 > 0:
                        top1_idx = slot_weights[bi, 0].argmax().item()
                        if top1_idx < proposals.shape[2]:
                            top1_terms.append(bce[:, :, top1_idx].mean())
                    if torch.is_tensor(slot_logits) and self.lambda_faf_selector > 0:
                        logits = slot_logits[bi : bi + 1, :curr_num_obj]
                        target = torch.softmax(per_slot_dice / max(self.faf_assignment_temperature, 1.0e-4), dim=-1)
                        selector_terms.append(F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"))
                    if torch.is_tensor(slot_confidence) and self.lambda_faf_confidence > 0:
                        conf = slot_confidence[bi : bi + 1, :curr_num_obj]
                        target = per_slot_dice.clamp(0.0, 1.0)
                        with torch.amp.autocast(device_type=conf.device.type, enabled=False):
                            confidence_loss = F.binary_cross_entropy(conf.float(), target.float())
                        confidence_terms.append(confidence_loss.to(dtype=conf.dtype))

                mixture_obj = aux.get("mixture_logits", aux.get("proposal_logits"))
                if torch.is_tensor(mixture_obj) and self.lambda_faf_mixture > 0:
                    mixture_obj = mixture_obj[bi : bi + 1, :curr_num_obj]
                    mixture_binary_logits = aggregate(torch.sigmoid(mixture_obj), dim=1)
                    ce, dice = self.mask_loss(mixture_binary_logits, soft_gt)
                    mixture_terms.append(ce + dice)
                    if prev_proposal is not None and self.lambda_faf_temporal > 0:
                        temporal_terms.append((torch.sigmoid(mixture_obj) - prev_proposal).abs().mean())
                    prev_proposal = torch.sigmoid(mixture_obj).detach()

                base_obj = aux.get("base_object_logits")
                if torch.is_tensor(base_obj) and self.lambda_faf_base > 0:
                    base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                    base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                    ce, dice = self.mask_loss(base_logits, soft_gt)
                    base_terms.append(ce + dice)

                for src, bucket, weight in (
                    ("coverage_gap", coverage_terms, self.lambda_faf_coverage),
                    ("slot_entropy", sparse_terms, self.lambda_faf_sparse),
                    ("slot_area_diversity", diversity_terms, self.lambda_faf_diversity),
                    ("memory_update_norm", write_terms, self.lambda_faf_write),
                    ("residual_logits", residual_terms, residual_smallness_weight),
                    ("affine_delta_norm", affine_terms, self.lambda_faf_affine),
                    ("velocity_norm", velocity_terms, self.lambda_faf_velocity),
                    ("feature_modulation_l1", modulation_terms, self.lambda_faf_feature_modulation),
                    ("dense_flow_abs_mean", dense_flow_terms, self.lambda_faf_dense_flow),
                    ("dense_flow_smoothness", dense_smooth_terms, self.lambda_faf_dense_smooth),
                ):
                    value = aux.get(src)
                    if torch.is_tensor(value) and weight > 0:
                        item = value[bi : bi + 1] if value.dim() > 0 and value.shape[0] > bi else value
                        if src == "residual_logits":
                            item = item[:, :curr_num_obj]
                            bucket.append(item.float().abs().mean())
                        else:
                            bucket.append(item.float().mean())
        for name, terms, weight in (
            ("oracle", oracle_terms, self.lambda_faf_oracle),
            ("top1", top1_terms, self.lambda_faf_top1),
            ("mixture", mixture_terms, self.lambda_faf_mixture),
            ("selector", selector_terms, self.lambda_faf_selector),
            ("confidence", confidence_terms, self.lambda_faf_confidence),
            ("base", base_terms, self.lambda_faf_base),
            ("coverage", coverage_terms, self.lambda_faf_coverage),
            ("sparse", sparse_terms, self.lambda_faf_sparse),
            ("diversity", diversity_terms, self.lambda_faf_diversity),
            ("temporal", temporal_terms, self.lambda_faf_temporal),
            ("write", write_terms, self.lambda_faf_write),
            ("residual_smallness", residual_terms, residual_smallness_weight),
            ("affine", affine_terms, self.lambda_faf_affine),
            ("velocity", velocity_terms, self.lambda_faf_velocity),
            ("feature_modulation", modulation_terms, self.lambda_faf_feature_modulation),
            ("dense_flow", dense_flow_terms, self.lambda_faf_dense_flow),
            ("dense_smooth", dense_smooth_terms, self.lambda_faf_dense_smooth),
        ):
            if terms:
                raw = torch.stack(terms).mean()
                out[f"raw_faf_{name}"] = raw.detach()
                out[f"aux_faf_{name}"] = raw * weight
        if not out:
            out["aux_faf_zero"] = zero
        return out

    def _compute_gar_losses(
        self,
        data: Dict[str, torch.Tensor],
        supervised_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_gar or "cls_gt" not in data:
            return {}

        batch_size = data["rgb"].shape[0]
        device = data["rgb"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        current_iter = int(data.get("current_iter", 0))
        lambda_gar_base = self.lambda_gar_base
        if current_iter >= self.gar_base_after_iter:
            lambda_gar_base = self.gar_base_after_weight
        lambda_gar_proposal_oracle = self.lambda_gar_proposal_oracle
        if current_iter >= self.gar_oracle_decay_start:
            lambda_gar_proposal_oracle = self.gar_oracle_after_weight
        base_terms = []
        proposal_terms = []
        selector_terms = []
        smooth_terms = []
        boundary_terms = []

        for bi in range(batch_size):
            curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            for ti in frame_ids:
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("gar_aux") if isinstance(memory_aux, dict) else None
                if not isinstance(aux, dict):
                    continue
                soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

                base_obj = aux.get("base_object_logits")
                if torch.is_tensor(base_obj) and lambda_gar_base > 0:
                    base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                    base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                    ce, dice = self.mask_loss(base_logits, soft_gt)
                    base_terms.append(ce + dice)

                proposals = aux.get("proposal_logits")
                head_losses = []
                if torch.is_tensor(proposals) and (lambda_gar_proposal_oracle > 0 or self.lambda_gar_selector > 0):
                    proposals = proposals[bi : bi + 1, :curr_num_obj]
                    for ki in range(proposals.shape[2]):
                        prop_obj = proposals[:, :, ki]
                        prop_logits = aggregate(torch.sigmoid(prop_obj), dim=1)
                        ce, dice = self.mask_loss(prop_logits, soft_gt)
                        head_losses.append(ce + dice)
                    if head_losses and lambda_gar_proposal_oracle > 0:
                        stacked = torch.stack(head_losses)
                        if self.gar_proposal_loss == "softmin":
                            tau = max(self.gar_proposal_softmin_temperature, 1.0e-4)
                            weights = torch.softmax(-stacked.detach() / tau, dim=0)
                            proposal_terms.append((weights * stacked).sum())
                        else:
                            proposal_terms.append(stacked.min())
                    if head_losses and self.lambda_gar_selector > 0:
                        selector_logits = aux.get("selector_logits")
                        if torch.is_tensor(selector_logits):
                            logits = selector_logits[bi : bi + 1, 0]
                            stacked = torch.stack(head_losses).detach()
                            target = torch.softmax(-stacked / max(self.gar_selector_temperature, 1.0e-4), dim=0).unsqueeze(0)
                            selector_terms.append(F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"))

                for src in ("stage2_flow_smooth", "stage3_flow_smooth"):
                    value = aux.get(src)
                    if torch.is_tensor(value) and self.lambda_gar_flow_smooth > 0:
                        item = value[bi : bi + 1] if value.dim() > 0 and value.shape[0] > bi else value
                        smooth_terms.append(item.float().mean())

                boundary_logits = aux.get("boundary_logits")
                if torch.is_tensor(boundary_logits) and self.lambda_gar_boundary_aux > 0:
                    gt_fg = (data["cls_gt"][bi, ti : ti + 1].float() > 0).float()
                    if gt_fg.dim() == 3:
                        gt_fg = gt_fg.unsqueeze(1)
                    if gt_fg.shape[-2:] != boundary_logits.shape[-2:]:
                        gt_fg = F.interpolate(gt_fg, size=boundary_logits.shape[-2:], mode="nearest")
                    dil = F.max_pool2d(gt_fg, kernel_size=3, stride=1, padding=1)
                    ero = 1.0 - F.max_pool2d(1.0 - gt_fg, kernel_size=3, stride=1, padding=1)
                    boundary = (dil - ero).clamp(0.0, 1.0)
                    pred = boundary_logits[bi : bi + 1]
                    boundary_terms.append(F.binary_cross_entropy_with_logits(pred.float(), boundary.to(device=pred.device, dtype=pred.dtype).float()))

        out: Dict[str, torch.Tensor] = {}
        if base_terms:
            raw = torch.stack(base_terms).mean()
            out["raw_gar_base"] = raw.detach()
            out["aux_gar_base"] = raw * lambda_gar_base
        if proposal_terms:
            raw = torch.stack(proposal_terms).mean()
            out["raw_gar_proposal_oracle"] = raw.detach()
            out["aux_gar_proposal_oracle"] = raw * lambda_gar_proposal_oracle
        if selector_terms:
            raw = torch.stack(selector_terms).mean()
            out["raw_gar_selector"] = raw.detach()
            out["aux_gar_selector"] = raw * self.lambda_gar_selector
        if smooth_terms:
            raw = torch.stack(smooth_terms).mean()
            out["raw_gar_flow_smooth"] = raw.detach()
            out["aux_gar_flow_smooth"] = raw * self.lambda_gar_flow_smooth
        if boundary_terms:
            raw = torch.stack(boundary_terms).mean()
            out["raw_gar_boundary_aux"] = raw.detach()
            out["aux_gar_boundary_aux"] = raw * self.lambda_gar_boundary_aux
        if not out:
            out["aux_gar_zero"] = zero
        return out

    def _compute_cardia_losses(
        self,
        data: Dict[str, torch.Tensor],
        supervised_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_cardia or "cls_gt" not in data:
            return {}

        batch_size = data["rgb"].shape[0]
        device = data["rgb"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        current_iter = int(data.get("current_iter", 0))
        lambda_base = self.lambda_cardia_base if current_iter < self.cardia_base_after_iter else self.cardia_base_after_weight
        lambda_oracle = (
            self.lambda_cardia_proposal_oracle
            if current_iter < self.cardia_oracle_decay_start
            else self.cardia_oracle_after_weight
        )
        base_terms = []
        proposal_terms = []
        selector_terms = []
        smooth_terms = []
        boundary_terms = []

        for bi in range(batch_size):
            curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            for ti in frame_ids:
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("cardia_aux") if isinstance(memory_aux, dict) else None
                if not isinstance(aux, dict):
                    continue
                soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

                base_obj = aux.get("base_object_logits")
                if torch.is_tensor(base_obj) and lambda_base > 0:
                    base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                    base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                    ce, dice = self.mask_loss(base_logits, soft_gt)
                    base_terms.append(ce + dice)

                proposals = aux.get("proposal_logits")
                head_losses = []
                if torch.is_tensor(proposals) and (lambda_oracle > 0 or self.lambda_cardia_selector > 0):
                    proposals = proposals[bi : bi + 1, :curr_num_obj]
                    for ki in range(proposals.shape[2]):
                        prop_obj = proposals[:, :, ki]
                        prop_logits = aggregate(torch.sigmoid(prop_obj), dim=1)
                        ce, dice = self.mask_loss(prop_logits, soft_gt)
                        head_losses.append(ce + dice)
                    if head_losses and lambda_oracle > 0:
                        stacked = torch.stack(head_losses)
                        if self.cardia_proposal_loss == "softmin":
                            tau = max(self.cardia_proposal_softmin_temperature, 1.0e-4)
                            weights = torch.softmax(-stacked.detach() / tau, dim=0)
                            proposal_terms.append((weights * stacked).sum())
                        else:
                            proposal_terms.append(stacked.min())
                    if head_losses and self.lambda_cardia_selector > 0:
                        selector_logits = aux.get("selector_logits")
                        if torch.is_tensor(selector_logits):
                            logits = selector_logits[bi : bi + 1, 0]
                            stacked = torch.stack(head_losses).detach()
                            target = torch.softmax(-stacked / max(self.cardia_selector_temperature, 1.0e-4), dim=0).unsqueeze(0)
                            selector_terms.append(F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"))

                for src in ("stage2_flow_smooth", "stage3_flow_smooth"):
                    value = aux.get(src)
                    if torch.is_tensor(value) and self.lambda_cardia_flow_smooth > 0:
                        item = value[bi : bi + 1] if value.dim() > 0 and value.shape[0] > bi else value
                        smooth_terms.append(item.float().mean())

                boundary_logits = aux.get("boundary_logits")
                if torch.is_tensor(boundary_logits) and self.lambda_cardia_boundary_aux > 0:
                    gt_fg = (data["cls_gt"][bi, ti : ti + 1].float() > 0).float()
                    if gt_fg.dim() == 3:
                        gt_fg = gt_fg.unsqueeze(1)
                    if gt_fg.shape[-2:] != boundary_logits.shape[-2:]:
                        gt_fg = F.interpolate(gt_fg, size=boundary_logits.shape[-2:], mode="nearest")
                    dil = F.max_pool2d(gt_fg, kernel_size=3, stride=1, padding=1)
                    ero = 1.0 - F.max_pool2d(1.0 - gt_fg, kernel_size=3, stride=1, padding=1)
                    boundary = (dil - ero).clamp(0.0, 1.0)
                    pred = boundary_logits[bi : bi + 1]
                    boundary_terms.append(F.binary_cross_entropy_with_logits(pred.float(), boundary.to(device=pred.device, dtype=pred.dtype).float()))

        out: Dict[str, torch.Tensor] = {}
        if base_terms:
            raw = torch.stack(base_terms).mean()
            out["raw_cardia_base"] = raw.detach()
            out["aux_cardia_base"] = raw * lambda_base
        if proposal_terms:
            raw = torch.stack(proposal_terms).mean()
            out["raw_cardia_proposal_oracle"] = raw.detach()
            out["aux_cardia_proposal_oracle"] = raw * lambda_oracle
        if selector_terms:
            raw = torch.stack(selector_terms).mean()
            out["raw_cardia_selector"] = raw.detach()
            out["aux_cardia_selector"] = raw * self.lambda_cardia_selector
        if smooth_terms:
            raw = torch.stack(smooth_terms).mean()
            out["raw_cardia_flow_smooth"] = raw.detach()
            out["aux_cardia_flow_smooth"] = raw * self.lambda_cardia_flow_smooth
        if boundary_terms:
            raw = torch.stack(boundary_terms).mean()
            out["raw_cardia_boundary_aux"] = raw.detach()
            out["aux_cardia_boundary_aux"] = raw * self.lambda_cardia_boundary_aux
        if not out:
            out["aux_cardia_zero"] = zero
        return out

    def _compute_delay_ode_regularizers(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.is_delay_ode:
            return {}
        latest_aux = None
        for key in sorted(data.keys()):
            if not key.startswith("memory_aux_"):
                continue
            aux = data.get(key)
            if isinstance(aux, dict) and isinstance(aux.get("delay_ode_aux"), dict):
                latest_aux = aux["delay_ode_aux"]
        if latest_aux is None:
            return {}
        terms = {}
        total = None
        mapping = {
            "slot_balance": ("aux_delay_ode_slot_balance", self.lambda_delay_ode_slot_balance),
            "phase_slot_usage": ("aux_delay_ode_phase_slot_usage", self.lambda_delay_ode_phase_slot_usage),
            "gate_smooth": ("aux_delay_ode_gate_smooth", self.lambda_delay_ode_gate_smooth),
            "latent_smooth": ("aux_delay_ode_latent_smooth", self.lambda_delay_ode_latent_smooth),
            "state_smooth": ("aux_delay_ode_state_smooth", self.lambda_delay_ode_state_smooth),
            "motion_scale_smooth": ("aux_delay_ode_motion_scale_smooth", self.lambda_delay_ode_motion_scale_smooth),
        }
        for src, (dst, weight) in mapping.items():
            value = latest_aux.get(src)
            if not torch.is_tensor(value) or weight <= 0:
                continue
            weighted = value * float(weight)
            terms[dst] = weighted
            total = weighted if total is None else total + weighted
        if total is not None:
            terms["aux_delay_ode_total"] = total
        decode_terms = self._compute_delay_ode_decode_losses(data, latest_aux)
        for key, value in decode_terms.items():
            terms[key] = value
            if key.startswith("aux_delay_ode_"):
                terms["aux_delay_ode_total"] = value if "aux_delay_ode_total" not in terms else terms["aux_delay_ode_total"] + value
        return terms

    def _soft_geometry6(self, mask_BNHW: torch.Tensor) -> torch.Tensor:
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
        return torch.stack([area, cx, cy, width, height, ratio], dim=-1)

    def _gt_object_masks(self, cls_gt_THW: torch.Tensor, num_objects: int) -> torch.Tensor:
        if cls_gt_THW.dim() == 4:
            cls_gt_THW = cls_gt_THW.squeeze(1)
        masks = [(cls_gt_THW == obj_id).float() for obj_id in range(1, num_objects + 1)]
        if not masks:
            return torch.zeros(cls_gt_THW.shape[0], 0, *cls_gt_THW.shape[-2:], device=cls_gt_THW.device)
        return torch.stack(masks, dim=1)

    def _compute_anchor_ode_losses(
        self,
        data: Dict[str, torch.Tensor],
        supervised_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_anchor_ode:
            return {}
        if "cls_gt" not in data:
            return {}

        batch_size, num_frames = data["rgb"].shape[:2]
        out: Dict[str, torch.Tensor] = {}
        device = data["rgb"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        prior_terms = []
        base_terms = []
        guided_terms = []
        multi_terms = []
        geo_terms = []
        conf_terms = []
        temp_geo_terms = []
        slot_terms = []
        affine_reg_terms = []

        for bi in range(batch_size):
            curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            prev_pred_geo = None
            prev_gt_geo = None
            for ti in frame_ids:
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("anchor_ode_aux") if isinstance(memory_aux, dict) else None
                if not isinstance(aux, dict):
                    continue
                gt_mask = self._gt_object_masks(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)
                soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

                base_obj = aux.get("base_object_logits")
                if torch.is_tensor(base_obj) and self.lambda_anchor_ode_base_seg > 0:
                    base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                    base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                    ce, dice = self.mask_loss(base_logits, soft_gt)
                    base_terms.append(ce + dice)

                prior_obj = aux.get("prior_logits")
                if torch.is_tensor(prior_obj) and self.lambda_anchor_ode_prior > 0:
                    prior_obj = prior_obj[bi : bi + 1, :curr_num_obj]
                    prior_logits = aggregate(torch.sigmoid(prior_obj), dim=1)
                    ce, dice = self.mask_loss(prior_logits, soft_gt)
                    prior_terms.append(ce + dice)

                guided_obj = aux.get("guided_object_logits")
                if torch.is_tensor(guided_obj) and self.lambda_anchor_ode_guided_seg > 0:
                    guided_obj = guided_obj[bi : bi + 1, :curr_num_obj]
                    guided_logits = aggregate(torch.sigmoid(guided_obj), dim=1)
                    ce, dice = self.mask_loss(guided_logits, soft_gt)
                    guided_terms.append(ce + dice)

                warped = aux.get("warped_priors")
                multi_weight = self.lambda_anchor_ode_multiscale_prior
                if str(aux.get("mode", "")) == "current_anchor_affine":
                    multi_weight = max(multi_weight, self.lambda_anchor_ode_warp_prior)
                if isinstance(warped, dict) and multi_weight > 0:
                    for value in warped.values():
                        if not torch.is_tensor(value):
                            continue
                        prior_obj_s = F.interpolate(
                            value[bi : bi + 1, :curr_num_obj].flatten(0, 1).unsqueeze(1),
                            size=gt_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).view(1, curr_num_obj, *gt_mask.shape[-2:])
                        prior_logits_s = aggregate(torch.sigmoid(prior_obj_s), dim=1)
                        ce, dice = self.mask_loss(prior_logits_s, soft_gt)
                        multi_terms.append(ce + dice)

                affine_reg = aux.get("affine_reg")
                if torch.is_tensor(affine_reg) and self.lambda_anchor_ode_affine_reg > 0:
                    affine_reg_terms.append(affine_reg)

                geom_pred = aux.get("geometry_pred")
                gt_geo = self._soft_geometry6(gt_mask)
                if torch.is_tensor(geom_pred) and self.lambda_anchor_ode_geo > 0:
                    pred = geom_pred[bi : bi + 1, :curr_num_obj, :6]
                    geo_terms.append(F.smooth_l1_loss(pred, gt_geo.to(device=pred.device, dtype=pred.dtype)))
                    if prev_pred_geo is not None and prev_gt_geo is not None and self.lambda_anchor_ode_temp_geo > 0:
                        temp_geo_terms.append(
                            F.smooth_l1_loss(
                                pred - prev_pred_geo,
                                gt_geo.to(device=pred.device, dtype=pred.dtype) - prev_gt_geo,
                            )
                        )
                    prev_pred_geo = pred
                    prev_gt_geo = gt_geo.to(device=pred.device, dtype=pred.dtype)

                conf = aux.get("confidence_prior")
                if torch.is_tensor(conf) and torch.is_tensor(prior_obj) and self.lambda_anchor_ode_conf > 0:
                    with torch.no_grad():
                        base_for_conf = aux.get("base_object_logits")
                        guided_for_conf = aux.get("guided_object_logits", aux.get("prior_logits"))
                        base_prob = torch.sigmoid(base_for_conf[bi : bi + 1, :curr_num_obj]) if torch.is_tensor(base_for_conf) else None
                        prior_prob = torch.sigmoid(guided_for_conf[bi : bi + 1, :curr_num_obj])
                        gt_float = gt_mask.to(device=prior_prob.device, dtype=prior_prob.dtype)
                        prior_err = (prior_prob - gt_float).abs().mean(dim=(-2, -1))
                        if base_prob is not None:
                            base_err = (base_prob.to(device=prior_prob.device, dtype=prior_prob.dtype) - gt_float).abs().mean(dim=(-2, -1))
                            target = torch.sigmoid(8.0 * (base_err - prior_err)).clamp(0.0, 1.0)
                        else:
                            target = torch.exp(-4.0 * prior_err).clamp(0.0, 1.0)
                    pred_conf = conf[bi : bi + 1, :curr_num_obj].to(device=target.device, dtype=target.dtype)
                    conf_terms.append(F.smooth_l1_loss(pred_conf, target))

            slot_weights = []
            for ti in range(num_frames):
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("anchor_ode_aux") if isinstance(memory_aux, dict) else None
                if isinstance(aux, dict) and torch.is_tensor(aux.get("slot_weights")):
                    slot_weights.append(aux["slot_weights"][bi, :curr_num_obj])
            if slot_weights and self.lambda_anchor_ode_slot_balance > 0:
                usage = torch.stack(slot_weights, dim=0).mean(dim=(0, 1))
                usage = usage / usage.sum().clamp_min(1.0e-8)
                uniform = torch.full_like(usage, 1.0 / max(usage.numel(), 1))
                slot_terms.append((usage.clamp_min(1.0e-8) * (usage.clamp_min(1.0e-8).log() - uniform.log())).sum())

        if base_terms:
            out["aux_anchor_ode_base"] = torch.stack(base_terms).mean() * self.lambda_anchor_ode_base_seg
        if prior_terms:
            out["aux_anchor_ode_prior"] = torch.stack(prior_terms).mean() * self.lambda_anchor_ode_prior
        if guided_terms:
            out["aux_anchor_ode_guided"] = torch.stack(guided_terms).mean() * self.lambda_anchor_ode_guided_seg
        if multi_terms:
            weight = max(self.lambda_anchor_ode_multiscale_prior, self.lambda_anchor_ode_warp_prior)
            out["aux_anchor_ode_multiscale_prior"] = torch.stack(multi_terms).mean() * weight
        if geo_terms:
            out["aux_anchor_ode_geo"] = torch.stack(geo_terms).mean() * self.lambda_anchor_ode_geo
        if temp_geo_terms:
            out["aux_anchor_ode_temp_geo"] = torch.stack(temp_geo_terms).mean() * self.lambda_anchor_ode_temp_geo
        if conf_terms:
            out["aux_anchor_ode_conf"] = torch.stack(conf_terms).mean() * self.lambda_anchor_ode_conf
        if slot_terms:
            out["aux_anchor_ode_slot_balance"] = torch.stack(slot_terms).mean() * self.lambda_anchor_ode_slot_balance
        if affine_reg_terms:
            out["aux_anchor_ode_affine_reg"] = torch.stack(affine_reg_terms).mean() * self.lambda_anchor_ode_affine_reg
        if not out:
            out["aux_anchor_ode_zero"] = zero
        return out

    def _boundary_weight(self, gt_mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(gt_mask.float(), kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-gt_mask.float(), kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _compute_functional_anchor_losses(
        self,
        data: Dict[str, torch.Tensor],
        supervised_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_functional_anchor or "cls_gt" not in data:
            return {}

        batch_size, num_frames = data["rgb"].shape[:2]
        out: Dict[str, torch.Tensor] = {}
        device = data["rgb"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        anchor_terms = []
        base_terms = []
        residual_terms = []
        boundary_terms = []
        phase_terms = []
        temporal_terms = []
        slot_order_terms = []
        phase_slot_terms = []
        trust_l1_terms = []
        trust_entropy_terms = []
        ode_raw_delta_terms = []

        for bi in range(batch_size):
            curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
            frame_ids = self._frame_ids_for_sample(supervised_mask, bi)
            prev_anchor_prob = None
            prev_anchor_area = None
            for ti in frame_ids:
                memory_aux = data.get(f"memory_aux_{ti}")
                aux = memory_aux.get("functional_anchor_aux") if isinstance(memory_aux, dict) else None
                if not isinstance(aux, dict):
                    continue
                gt_mask = self._gt_object_masks(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)
                soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

                anchor_obj = aux.get("anchor_logits")
                if torch.is_tensor(anchor_obj) and self.lambda_functional_anchor_anchor > 0:
                    anchor_obj = anchor_obj[bi : bi + 1, :curr_num_obj]
                    anchor_logits = aggregate(torch.sigmoid(anchor_obj), dim=1)
                    ce, dice = self.mask_loss(anchor_logits, soft_gt)
                    anchor_terms.append(ce + dice)
                    anchor_prob = torch.sigmoid(anchor_obj)
                    anchor_area = anchor_prob.mean(dim=(-2, -1))
                    if prev_anchor_prob is not None and self.lambda_functional_anchor_temp > 0:
                        temporal_terms.append((anchor_prob - prev_anchor_prob).abs().mean())
                    if prev_anchor_area is not None and self.lambda_functional_anchor_phase > 0:
                        phase_terms.append((anchor_area - prev_anchor_area).abs().mean())
                    prev_anchor_prob = anchor_prob
                    prev_anchor_area = anchor_area

                base_obj = aux.get("base_object_logits")
                if torch.is_tensor(base_obj) and self.lambda_functional_anchor_base > 0:
                    base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                    base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                    ce, dice = self.mask_loss(base_logits, soft_gt)
                    base_terms.append(ce + dice)

                residual = aux.get("residual_logits")
                if torch.is_tensor(residual) and self.lambda_functional_anchor_residual_l1 > 0:
                    residual_terms.append(residual[bi : bi + 1, :curr_num_obj].abs().mean())
                trust = aux.get("anchor_trust_map", aux.get("anchor_trust"))
                if torch.is_tensor(trust):
                    trust_item = trust[bi : bi + 1].float().clamp(1.0e-6, 1.0 - 1.0e-6)
                    if self.lambda_functional_anchor_trust_l1 > 0:
                        trust_l1_terms.append(trust_item.mean())
                    if self.lambda_functional_anchor_trust_entropy > 0:
                        trust_entropy_terms.append(
                            -(trust_item * trust_item.log() + (1.0 - trust_item) * (1.0 - trust_item).log()).mean()
                        )

                final_obj = aux.get("final_object_logits")
                if torch.is_tensor(final_obj) and self.lambda_functional_anchor_boundary > 0:
                    final_obj = final_obj[bi : bi + 1, :curr_num_obj]
                    final_logits = aggregate(torch.sigmoid(final_obj), dim=1)
                    fg = final_logits[:, 1:2]
                    gt_fg = gt_mask[:, :1].to(device=fg.device, dtype=fg.dtype)
                    weight = 1.0 + 4.0 * self._boundary_weight(gt_fg)
                    boundary_terms.append(F.binary_cross_entropy_with_logits(fg, gt_fg, weight=weight))

                slot_violation = aux.get("slot_area_order_violation")
                if torch.is_tensor(slot_violation) and self.lambda_functional_anchor_slot_order > 0:
                    slot_order_terms.append(slot_violation.float().mean())
                slot_weights = aux.get("slot_weights")
                phase_descriptor = aux.get("phase_descriptor")
                if torch.is_tensor(slot_weights) and torch.is_tensor(phase_descriptor) and self.lambda_functional_anchor_phase_slot > 0:
                    norm_time = phase_descriptor[bi : bi + 1, :curr_num_obj, 0]
                    ed_target = (norm_time <= 0.125).float()
                    es_target = ((norm_time - 0.5).abs() <= 0.125).float()
                    phase_slot_terms.append(F.mse_loss(slot_weights[bi : bi + 1, :curr_num_obj, 0], ed_target))
                    phase_slot_terms.append(F.mse_loss(slot_weights[bi : bi + 1, :curr_num_obj, 2], es_target))
                z_delta = aux.get("z_delta")
                if torch.is_tensor(z_delta) and self.lambda_functional_anchor_ode_raw_delta > 0:
                    ode_raw_delta_terms.append(z_delta[bi : bi + 1, :curr_num_obj].pow(2).mean())

        if anchor_terms:
            raw = torch.stack(anchor_terms).mean()
            out["raw_functional_anchor_anchor"] = raw.detach()
            out["aux_functional_anchor_anchor"] = raw * self.lambda_functional_anchor_anchor
        if base_terms:
            raw = torch.stack(base_terms).mean()
            out["raw_functional_anchor_base"] = raw.detach()
            out["aux_functional_anchor_base"] = raw * self.lambda_functional_anchor_base
        if residual_terms:
            raw = torch.stack(residual_terms).mean()
            out["raw_functional_anchor_residual_l1"] = raw.detach()
            out["aux_functional_anchor_residual_l1"] = raw * self.lambda_functional_anchor_residual_l1
        if boundary_terms:
            raw = torch.stack(boundary_terms).mean()
            out["raw_functional_anchor_boundary_residual"] = raw.detach()
            out["aux_functional_anchor_boundary_residual"] = raw * self.lambda_functional_anchor_boundary
        if phase_terms:
            raw = torch.stack(phase_terms).mean()
            out["raw_functional_anchor_phase_consistency"] = raw.detach()
            out["aux_functional_anchor_phase_consistency"] = raw * self.lambda_functional_anchor_phase
        if temporal_terms:
            raw = torch.stack(temporal_terms).mean()
            out["raw_functional_anchor_anchor_temporal"] = raw.detach()
            out["aux_functional_anchor_anchor_temporal"] = raw * self.lambda_functional_anchor_temp
        if slot_order_terms:
            raw = torch.stack(slot_order_terms).mean()
            out["raw_functional_anchor_slot_area_order"] = raw.detach()
            out["aux_functional_anchor_slot_area_order"] = raw * self.lambda_functional_anchor_slot_order
        if phase_slot_terms:
            raw = torch.stack(phase_slot_terms).mean()
            out["raw_functional_anchor_phase_slot_correlation"] = raw.detach()
            out["aux_functional_anchor_phase_slot_correlation"] = raw * self.lambda_functional_anchor_phase_slot
        if trust_l1_terms:
            raw = torch.stack(trust_l1_terms).mean()
            out["raw_functional_anchor_trust_l1"] = raw.detach()
            out["aux_functional_anchor_trust_l1"] = raw * self.lambda_functional_anchor_trust_l1
        if trust_entropy_terms:
            raw = torch.stack(trust_entropy_terms).mean()
            out["raw_functional_anchor_trust_entropy"] = raw.detach()
            out["aux_functional_anchor_trust_entropy"] = raw * self.lambda_functional_anchor_trust_entropy
        if ode_raw_delta_terms:
            raw = torch.stack(ode_raw_delta_terms).mean()
            out["raw_functional_anchor_ode_raw_delta"] = raw.detach()
            out["aux_functional_anchor_ode_raw_delta"] = raw * self.lambda_functional_anchor_ode_raw_delta
        if not out:
            out["aux_functional_anchor_zero"] = zero
        return out

    def _compute_delay_ode_decode_losses(self, data: Dict[str, torch.Tensor], latest_aux: Dict) -> Dict[str, torch.Tensor]:
        if "cls_gt" not in data:
            return {}
        cls_gt = data["cls_gt"]
        if cls_gt.shape[1] <= 1:
            return {}
        target_t = cls_gt[:, -1]
        out: Dict[str, torch.Tensor] = {}
        latent_logits = latest_aux.get("latent_decode_logits", {})
        if isinstance(latent_logits, dict):
            for level, weight in self.lambda_delay_ode_latent_decode.items():
                obj_logits = latent_logits.get(level)
                if weight <= 0 or not torch.is_tensor(obj_logits):
                    continue
                masks = torch.sigmoid(obj_logits)
                logits = aggregate(masks, dim=1)
                total = torch.zeros((), device=logits.device, dtype=logits.dtype)
                valid_count = 0
                for bi in range(logits.shape[0]):
                    num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else masks.shape[1]
                    soft_gt = cls_to_one_hot(target_t[bi : bi + 1], num_obj)
                    ce, dice = self.mask_loss(logits[bi : bi + 1, : num_obj + 1], soft_gt)
                    total = total + ce + dice
                    valid_count += 1
                if valid_count:
                    out[f"aux_delay_ode_latent_decode_{level}"] = total / valid_count * weight
        boundary_logits = latest_aux.get("boundary_logits")
        if self.lambda_delay_ode_boundary > 0 and torch.is_tensor(boundary_logits):
            fg = (target_t.float() > 0).float()
            if fg.ndim == 3:
                fg = fg.unsqueeze(1)
            dil = F.max_pool2d(fg, kernel_size=3, stride=1, padding=1)
            ero = -F.max_pool2d(-fg, kernel_size=3, stride=1, padding=1)
            boundary = (dil - ero).clamp(0.0, 1.0)
            pred = boundary_logits[:, : boundary.shape[1]]
            out["aux_delay_ode_boundary"] = F.binary_cross_entropy_with_logits(pred, boundary.to(device=pred.device, dtype=pred.dtype)) * self.lambda_delay_ode_boundary
        return out

    def _compute_dynakey_q_loss(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enable_dynakey_q_loss:
            return {}

        q_values_list = []
        labels_list = []
        advantage_list = []
        mask_list = []

        for key in sorted(data.keys()):
            if not key.startswith("memory_aux_"):
                continue
            memory_aux = data.get(key)
            if not isinstance(memory_aux, dict):
                continue
            aux = memory_aux.get("dynakey_aux")
            if not isinstance(aux, dict):
                continue
            q_values = aux.get("q_values")
            target = aux.get("q_target_action")
            advantage = aux.get("advantage_returns")
            action_mask = aux.get("action_mask")
            if q_values is None or target is None or advantage is None:
                continue
            if not torch.is_tensor(q_values) or not q_values.requires_grad:
                continue
            q_values_list.append(q_values.flatten(0, 1))
            labels_list.append(target.flatten().long())
            advantage_list.append(advantage.to(device=q_values.device, dtype=q_values.dtype).flatten(0, 1))
            if action_mask is None:
                mask_list.append(torch.ones_like(q_values, dtype=torch.bool).flatten(0, 1))
            else:
                mask_list.append(action_mask.to(device=q_values.device).bool().flatten(0, 1))

        if not q_values_list:
            return {}

        q_values = torch.cat(q_values_list, dim=0)
        labels = torch.cat(labels_list, dim=0).to(device=q_values.device)
        advantages = torch.cat(advantage_list, dim=0).clamp(
            -self.dynakey_advantage_clamp,
            self.dynakey_advantage_clamp,
        )
        action_mask = torch.cat(mask_list, dim=0)
        labels = labels.clamp(0, q_values.shape[-1] - 1)
        valid_label = action_mask.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        q_values_masked = q_values.masked_fill(~action_mask, -1.0e4)

        out: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=q_values.device, dtype=q_values.dtype)
        out["dynakey_q_valid_samples"] = valid_label.float().sum().detach()
        out["dynakey_q_invalid_targets"] = (~valid_label).float().sum().detach()
        if not valid_label.any():
            zero = q_values.sum() * 0.0
            out["dynakey_q_ce"] = zero
            out["dynakey_q_adv"] = zero
            out["dynakey_q_total"] = zero
            return out

        q_values_valid = q_values[valid_label]
        q_values_masked_valid = q_values_masked[valid_label]
        labels_valid = labels[valid_label]
        advantages_valid = advantages[valid_label]
        action_mask_valid = action_mask[valid_label]
        if self.lambda_dynakey_q_ce > 0:
            ce = F.cross_entropy(q_values_masked_valid, labels_valid)
            out["dynakey_q_ce"] = ce * self.lambda_dynakey_q_ce
            total = total + out["dynakey_q_ce"]
        if self.lambda_dynakey_q_adv > 0:
            adv_target = advantages_valid.masked_fill(~action_mask_valid, 0.0)
            valid_count = action_mask_valid.float().sum().clamp_min(1.0)
            adv_loss = (((q_values_valid - adv_target) ** 2) * action_mask_valid.float()).sum() / valid_count
            out["dynakey_q_adv"] = adv_loss * self.lambda_dynakey_q_adv
            total = total + out["dynakey_q_adv"]
        out["dynakey_q_total"] = total
        return out

    def _compute_spatial_q_loss(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enable_spatial_q_loss or self.spatial_q_policy_mode != "training":
            return {}
        q_values_list = []
        labels_list = []
        valid_list = []
        for key in sorted(data.keys()):
            if not key.startswith("memory_aux_"):
                continue
            memory_aux = data.get(key)
            if not isinstance(memory_aux, dict):
                continue
            q_values = memory_aux.get("spatial_q_values")
            target = memory_aux.get("spatial_q_target_action")
            valid = memory_aux.get("spatial_q_valid")
            if q_values is None or target is None:
                continue
            if not torch.is_tensor(q_values) or not q_values.requires_grad:
                continue
            q_values_list.append(q_values.flatten(0, 1))
            labels_list.append(target.flatten().long().to(q_values.device))
            if torch.is_tensor(valid):
                valid_list.append(valid.flatten().bool().to(q_values.device))
            else:
                valid_list.append(torch.ones(target.numel(), device=q_values.device, dtype=torch.bool))
        if not q_values_list:
            return {}
        q_values = torch.cat(q_values_list, dim=0)
        labels = torch.cat(labels_list, dim=0).clamp(0, q_values.shape[-1] - 1)
        valid = torch.cat(valid_list, dim=0)
        zero = q_values.sum() * 0.0
        out: Dict[str, torch.Tensor] = {
            "spatial_q_valid_samples": valid.float().sum().detach(),
        }
        if not valid.any():
            out["spatial_q_ce"] = zero
            out["spatial_q_total"] = zero
            return out
        ce = F.cross_entropy(q_values[valid], labels[valid]) * self.lambda_spatial_q_ce
        out["spatial_q_ce"] = ce
        out["spatial_q_total"] = ce
        return out

    def _slice_aux_for_sample(
        self,
        aux: Dict[str, torch.Tensor] | None,
        sample_idx: int,
        batch_size: int,
    ) -> Dict[str, torch.Tensor] | None:
        if aux is None:
            return None

        sliced = {}
        for key, value in aux.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
                sliced[key] = value[sample_idx:sample_idx + 1]
            else:
                sliced[key] = value
        return sliced

    def _compute_policy_and_rl_losses(
        self,
        bpm_aux_list: List[Dict[str, torch.Tensor] | None],
        frame_seg_loss: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not bpm_aux_list:
            return {}

        rl_loss_terms = []
        entropy_terms = []
        reward_logs = defaultdict(list)

        for frame_idx, aux in enumerate(bpm_aux_list):
            if aux is None or "policy_actions" not in aux:
                continue

            action = aux["policy_actions"].flatten()
            log_prob = aux.get("log_prob")
            entropy = aux.get("entropy")
            learned_mask = aux.get("policy_is_learned")
            if log_prob is None or entropy is None or learned_mask is None:
                continue

            seg_quality = -frame_seg_loss[frame_idx].detach()
            log_prob = log_prob.flatten()
            entropy = entropy.flatten()
            learned_mask = learned_mask.flatten().bool()
            action_cost_vec = aux["action_cost"].to(device=device, dtype=seg_quality.dtype)

            for sample_idx, action_id in enumerate(action.tolist()):
                if not learned_mask[sample_idx]:
                    continue
                baseline = self.action_reward_baseline[action_id].detach().to(device=device)
                action_cost = action_cost_vec[action_id]
                centered_reward = seg_quality - baseline - action_cost
                advantage = centered_reward.clamp(-self.adv_clamp, self.adv_clamp)
                if self.enable_rl_loss and self.lambda_rl > 0:
                    rl_loss_terms.append(-advantage * log_prob[sample_idx])
                if self.lambda_entropy > 0:
                    entropy_terms.append(entropy[sample_idx])

                reward_logs[f"reward_{action_id}"].append(centered_reward.detach())
                reward_logs[f"advantage_{action_id}"].append(advantage.detach())
                reward_logs["entropy"].append(entropy[sample_idx].detach())
                reward_logs["rule_agreement"].append(aux["action_agreement"].flatten()[sample_idx].detach())

                with torch.no_grad():
                    old = self.action_reward_baseline[action_id]
                    self.action_reward_baseline[action_id] = (
                        self.rl_baseline_momentum * old
                        + (1.0 - self.rl_baseline_momentum) * seg_quality.float().cpu()
                    )

        out = {}
        if rl_loss_terms:
            out["rl_loss"] = torch.stack(rl_loss_terms).mean() * self.lambda_rl
        if entropy_terms:
            out["entropy_reg"] = -torch.stack(entropy_terms).mean() * self.lambda_entropy

        action_names = ["keep", "refine", "replace", "spawn"]
        for idx, name in enumerate(action_names):
            if reward_logs[f"reward_{idx}"]:
                out[f"reward_{name}"] = torch.stack(reward_logs[f"reward_{idx}"]).mean()
            if reward_logs[f"advantage_{idx}"]:
                out[f"advantage_{name}"] = torch.stack(reward_logs[f"advantage_{idx}"]).mean()
            out[f"baseline_{name}"] = self.action_reward_baseline[idx].detach().to(device)
        if reward_logs["entropy"]:
            out["policy_entropy"] = torch.stack(reward_logs["entropy"]).mean()
        if reward_logs["rule_agreement"]:
            out["policy_rule_agreement"] = torch.stack(reward_logs["rule_agreement"]).mean()
        return out

    def _compute_policy_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        supervised_indices = data.get('supervised_indices')
        batch_size = data['rgb'].shape[0]
        logits_list = []
        labels_list = []
        supervised_mask = None
        if supervised_indices is not None:
            supervised_mask = self._resolve_supervision_mask(
                supervised_indices,
                batch_size=batch_size,
                num_frames=data['rgb'].shape[1],
                device=data['rgb'].device,
            )
        for bi in range(batch_size):
            if supervised_mask is None:
                frame_ids = sorted(int(k.split('_')[-1]) for k in data.keys() if k.startswith('bpm_aux_'))
            else:
                frame_ids = self._frame_ids_for_sample(supervised_mask, bi)

            for ti in frame_ids:
                aux = self._slice_aux_for_sample(data.get(f'bpm_aux_{ti}'), bi, batch_size)
                if aux is None:
                    continue
                if 'policy_logits' not in aux or 'policy_labels' not in aux:
                    continue
                logits = aux['policy_logits'].flatten(start_dim=0, end_dim=1)
                labels = aux['policy_labels'].flatten()
                logits_list.append(logits)
                labels_list.append(labels)

        if not logits_list:
            return None

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return F.cross_entropy(logits, labels)
