from typing import List, Dict, Tuple
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from utils.tensor_utils import cls_to_one_hot
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

            bpm_aux_list = [self._slice_aux_for_sample(data.get(f'bpm_aux_{ti}'), bi, batch_size) for ti in t_range]
            rl_terms = self._compute_policy_and_rl_losses(
                bpm_aux_list=bpm_aux_list,
                frame_seg_loss=(frame_ce + frame_dice).detach(),
                device=logits.device,
            )
            for k, v in rl_terms.items():
                losses[k] += v / batch_size if torch.is_tensor(v) else v / batch_size

        total_loss = torch.zeros((), device=data["rgb"].device, dtype=torch.float32)
        for key, value in losses.items():
            if not (torch.is_tensor(value) or isinstance(value, (float, int))):
                continue
            if key.startswith("loss_") or key.startswith("aux_") or key in {"rl_loss", "entropy_reg"}:
                total_loss = total_loss + value
        losses['total_loss'] = total_loss

        if self.enable_policy_ce_loss and self.lambda_policy_ce > 0:
            policy_loss = self._compute_policy_loss(data)
            if policy_loss is not None:
                losses['policy_ce'] = policy_loss * self.lambda_policy_ce
                losses['total_loss'] = losses['total_loss'] + losses['policy_ce']

        return losses

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
