from typing import List, Dict, Tuple
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn.functional as F

from utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from utils.tensor_utils import cls_to_one_hot

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


class LossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight
        bpm_cfg = cfg.model.temporal_memory.get("bpm", {})
        self.policy_loss_weight = float(bpm_cfg.get("POLICY_LOSS_WEIGHT", 0.0))
        self.enable_policy_loss = bool(bpm_cfg.get("ENABLE_POLICY_LOSS", False))

    def mask_loss(
        self, logits: torch.Tensor, soft_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.point_supervision

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

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        supervised_indices = data.get('supervised_indices')
        if supervised_indices is None:
            t_range = [0, num_frames - 1] if num_frames > 1 else [0]
        else:
            t_range = supervised_indices.tolist()

        for bi in range(batch_size):
            curr_num_obj = num_objects[bi]
            valid_slice = slice(None, curr_num_obj + 1)

            logits = torch.stack(
                [data[f'logits_{ti}'][bi, valid_slice] for ti in t_range], dim=0
            )

            cls_gt = data['cls_gt'][bi, t_range]
            soft_gt = cls_to_one_hot(cls_gt, curr_num_obj)

            loss_ce, loss_dice = self.mask_loss(logits, soft_gt)
            losses['loss_ce'] += loss_ce / batch_size
            losses['loss_dice'] += loss_dice / batch_size

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

        losses['total_loss'] = sum(losses.values())

        if self.enable_policy_loss and self.policy_loss_weight > 0:
            policy_loss = self._compute_policy_loss(data)
            if policy_loss is not None:
                losses['policy_ce'] = policy_loss * self.policy_loss_weight
                losses['total_loss'] = losses['total_loss'] + losses['policy_ce']

        return losses

    def _compute_policy_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        supervised_indices = data.get('supervised_indices')
        if supervised_indices is None:
            frame_ids = [k for k in data.keys() if k.startswith('bpm_aux_')]
            if not frame_ids:
                return None
            t_range = sorted(int(k.split('_')[-1]) for k in frame_ids)
        else:
            t_range = supervised_indices.tolist()

        logits_list = []
        labels_list = []
        for ti in t_range:
            aux = data.get(f'bpm_aux_{ti}')
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
