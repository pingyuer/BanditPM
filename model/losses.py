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
        self.is_anchor_ode = str(cfg.model.get("name", "")).lower() in {"anchor_ode", "unext_anchor_ode", "unextanchorode"}
        self.lambda_anchor_ode_prior = float(anchor_ode_cfg.get("lambda_prior", 0.0))
        self.lambda_anchor_ode_multiscale_prior = float(anchor_ode_cfg.get("lambda_multiscale_prior", 0.0))
        self.lambda_anchor_ode_geo = float(anchor_ode_cfg.get("lambda_geo", 0.0))
        self.lambda_anchor_ode_temp_geo = float(anchor_ode_cfg.get("lambda_temp_geo", 0.0))
        self.lambda_anchor_ode_conf = float(anchor_ode_cfg.get("lambda_conf", 0.0))
        self.lambda_anchor_ode_slot_balance = float(anchor_ode_cfg.get("lambda_slot_balance", 0.0))

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
        multi_terms = []
        geo_terms = []
        conf_terms = []
        temp_geo_terms = []
        slot_terms = []

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

                prior_obj = aux.get("prior_logits")
                if torch.is_tensor(prior_obj) and self.lambda_anchor_ode_prior > 0:
                    prior_obj = prior_obj[bi : bi + 1, :curr_num_obj]
                    prior_logits = aggregate(torch.sigmoid(prior_obj), dim=1)
                    ce, dice = self.mask_loss(prior_logits, soft_gt)
                    prior_terms.append(ce + dice)

                warped = aux.get("warped_priors")
                if isinstance(warped, dict) and self.lambda_anchor_ode_multiscale_prior > 0:
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
                        prior_prob = torch.sigmoid(aux["prior_logits"][bi : bi + 1, :curr_num_obj])
                        err = (prior_prob - gt_mask.to(device=prior_prob.device, dtype=prior_prob.dtype)).abs().mean(dim=(-2, -1))
                        target = torch.exp(-4.0 * err).clamp(0.0, 1.0)
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

        if prior_terms:
            out["aux_anchor_ode_prior"] = torch.stack(prior_terms).mean() * self.lambda_anchor_ode_prior
        if multi_terms:
            out["aux_anchor_ode_multiscale_prior"] = torch.stack(multi_terms).mean() * self.lambda_anchor_ode_multiscale_prior
        if geo_terms:
            out["aux_anchor_ode_geo"] = torch.stack(geo_terms).mean() * self.lambda_anchor_ode_geo
        if temp_geo_terms:
            out["aux_anchor_ode_temp_geo"] = torch.stack(temp_geo_terms).mean() * self.lambda_anchor_ode_temp_geo
        if conf_terms:
            out["aux_anchor_ode_conf"] = torch.stack(conf_terms).mean() * self.lambda_anchor_ode_conf
        if slot_terms:
            out["aux_anchor_ode_slot_balance"] = torch.stack(slot_terms).mean() * self.lambda_anchor_ode_slot_balance
        if not out:
            out["aux_anchor_ode_zero"] = zero
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
