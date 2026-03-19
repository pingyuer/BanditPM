from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PrototypeBankState:
    """
    Explicit prototype bank state.

    Shapes:
        proto: [B, N, K, C]
        age: [B, N, K]
        usage: [B, N, K]
        conf: [B, N, K]
        valid: [B, N, K]
    """

    proto: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    conf: torch.Tensor
    valid: torch.Tensor


class PolicyHead(nn.Module):
    """Lightweight policy head for one-step contextual bandit decisions."""

    def __init__(self, context_dim: int, hidden_dim: int, num_actions: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class BanditPrototypeManager(nn.Module):
    """
    Explicit prototype bank with bandit-style light policy updates.

    This module maintains a small prototype bank per object and returns a
    prototype-conditioned dense feature map with the same shape as the input.

    Input shapes:
        value_BNCHW: [B, N, C, H, W]
        frame_feat_BCHW: [B, C, H, W]
        mask_BNHW: [B, N, H_img, W_img] or [B, N, H, W]

    Output shapes:
        conditioned_BNCHW: [B, N, C, H, W]
        aux: dict of logging tensors
    """

    ACTION_KEEP = 0
    ACTION_REFINE = 1
    ACTION_REPLACE = 2
    ACTION_SPAWN = 3
    SOURCE_RULE = 0
    SOURCE_LEARNED_SAMPLE = 1
    SOURCE_LEARNED_GREEDY = 2
    SOURCE_MIXED_RULE = 3

    def __init__(self, cfg, value_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.value_dim = value_dim
        self.bank_size = int(cfg.BANK_SIZE)
        self.proto_alpha = float(getattr(cfg, "REFINE_EMA_ALPHA", cfg.PROTO_ALPHA))
        self.readout_temperature = float(getattr(cfg, "READOUT_TEMPERATURE", 1.0))
        self.default_action = int(getattr(cfg, "DEFAULT_ACTION", self.ACTION_REFINE))
        spawn_mode = str(getattr(cfg, "SPAWN_WITHOUT_EMPTY_SLOT", "replace_fallback")).lower()
        self.spawn_replace_when_full = spawn_mode != "forbid"
        self.use_rule_based_policy = bool(cfg.USE_RULE_BASED_POLICY)
        self.use_learned_policy = bool(getattr(cfg, "USE_LEARNED_POLICY", False))
        self.sim_threshold_high = float(cfg.SIM_THRESHOLD_HIGH)
        self.sim_threshold_low = float(cfg.SIM_THRESHOLD_LOW)
        self.debug_mode = bool(getattr(cfg, "DEBUG_MODE", False))
        self.mask_pooling = str(getattr(cfg, "PROTO_POOLING", "mask")).lower()
        self.fusion_type = str(getattr(cfg, "FUSION_TYPE", "add")).lower()
        self.hidden_dim = int(getattr(cfg, "POLICY_HIDDEN_DIM", value_dim))
        self.exec_policy = str(getattr(cfg, "EXEC_POLICY", "rule")).lower()
        self.policy_warmup_epochs = int(getattr(cfg, "POLICY_WARMUP_EPOCHS", 0))
        self.sample_in_train = bool(getattr(cfg, "SAMPLE_ACTIONS_IN_TRAIN", True))
        self.exec_greedy_on_eval = bool(getattr(cfg, "EXEC_GREEDY_ON_EVAL", True))
        self.epsilon_rule_mix_init = float(getattr(cfg, "EPSILON_RULE_MIX_INIT", 1.0))
        self.epsilon_rule_mix_final = float(getattr(cfg, "EPSILON_RULE_MIX_FINAL", 0.0))
        self.epsilon_rule_mix_epochs = int(getattr(cfg, "EPSILON_RULE_MIX_EPOCHS", 1))
        self.victim_weight_age = float(getattr(cfg, "VICTIM_WEIGHT_AGE", 1.0))
        self.victim_weight_usage = float(getattr(cfg, "VICTIM_WEIGHT_USAGE", 1.0))
        self.victim_weight_conf = float(getattr(cfg, "VICTIM_WEIGHT_CONF", 1.0))

        action_costs = getattr(cfg, "ACTION_COSTS", {})
        self.action_costs = {
            self.ACTION_KEEP: float(action_costs.get("keep", 0.0)),
            self.ACTION_REFINE: float(action_costs.get("refine", 0.05)),
            self.ACTION_REPLACE: float(action_costs.get("replace", 0.1)),
            self.ACTION_SPAWN: float(action_costs.get("spawn", 0.12)),
        }

        # Context = pooled frame feat + sim[K] + slot stats[4K] + similarity summary[3]
        #         + bank globals[5] + uncertainty[5]
        context_dim = value_dim + self.bank_size + self.bank_size * 4 + 13
        self.policy_head = PolicyHead(context_dim=context_dim, hidden_dim=self.hidden_dim)

        if self.fusion_type == "concat":
            self.fuse_proj = nn.Conv2d(value_dim * 2, value_dim, kernel_size=1)
        else:
            self.fuse_proj = None
            self.proto_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.frame_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._state: Optional[PrototypeBankState] = None
        self._prev_mask: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        shape_bnk = (batch_size, num_objects, self.bank_size)
        self._state = PrototypeBankState(
            proto=torch.zeros(*shape_bnk, self.value_dim, device=device),
            age=torch.zeros(*shape_bnk, device=device),
            usage=torch.zeros(*shape_bnk, device=device),
            conf=torch.zeros(*shape_bnk, device=device),
            valid=torch.zeros(*shape_bnk, dtype=torch.bool, device=device),
        )
        self._prev_mask = None

    def _ensure_state(self, value_BNCHW: torch.Tensor) -> PrototypeBankState:
        B, N, C = value_BNCHW.shape[:3]
        if self._state is None or self._state.proto.shape[:3] != (B, N, self.bank_size):
            self.reset_state(batch_size=B, num_objects=N, device=value_BNCHW.device)
        return self._state

    def _resize_masks(self, mask_BNHW: Optional[torch.Tensor], target_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if mask_BNHW is None:
            return None
        if mask_BNHW.shape[-2:] == target_hw:
            return mask_BNHW.float()
        return F.interpolate(mask_BNHW.float(), size=target_hw, mode="area")

    def _compute_candidate_proto(
        self,
        value_BNCHW: torch.Tensor,
        mask_BNHW: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = value_BNCHW.shape
        mask_small = self._resize_masks(mask_BNHW, target_hw=(H, W))
        flat_value = value_BNCHW.flatten(start_dim=3)  # [B, N, C, HW]

        if mask_small is not None and self.mask_pooling == "mask":
            flat_mask = mask_small.flatten(start_dim=2).unsqueeze(2)  # [B, N, 1, HW]
            denom = flat_mask.sum(dim=-1).clamp_min(1e-6)
            cand_proto = (flat_value * flat_mask).sum(dim=-1) / denom
            mask_strength = flat_mask.mean(dim=-1).squeeze(2)
            fallback_proto = flat_value.mean(dim=-1)
            use_fallback = denom.squeeze(2) <= 1e-5
            cand_proto = torch.where(use_fallback.unsqueeze(-1), fallback_proto, cand_proto)
        else:
            cand_proto = flat_value.mean(dim=-1)
            mask_strength = value_BNCHW.new_ones(B, N)

        cand_proto = F.normalize(cand_proto, dim=-1)
        return cand_proto, mask_strength

    def _compute_uncertainty_features(
        self,
        mask_BNHW: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask_BNHW is None:
            return torch.zeros(1, 1, 5, device=self._state.proto.device if self._state is not None else None)

        mask = mask_BNHW.float().clamp(1e-4, 1 - 1e-4)
        entropy = -(mask * torch.log(mask) + (1.0 - mask) * torch.log(1.0 - mask))
        fg_prob = mask.mean(dim=(-2, -1))
        fg_area = (mask > 0.5).float().mean(dim=(-2, -1))
        grad_x = torch.abs(mask[..., 1:, :] - mask[..., :-1, :]).mean(dim=(-2, -1))
        grad_y = torch.abs(mask[..., :, 1:] - mask[..., :, :-1]).mean(dim=(-2, -1))
        boundary_uncertainty = 0.5 * (grad_x + grad_y)

        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            temporal_instability = torch.zeros_like(fg_prob)
        else:
            prev_bin = (self._prev_mask > 0.5).float()
            curr_bin = (mask > 0.5).float()
            inter = (prev_bin * curr_bin).sum(dim=(-2, -1))
            union = prev_bin.sum(dim=(-2, -1)) + curr_bin.sum(dim=(-2, -1)) - inter
            iou = inter / union.clamp_min(1e-6)
            temporal_instability = 1.0 - iou

        feats = torch.stack(
            [
                entropy.mean(dim=(-2, -1)),
                fg_prob,
                fg_area,
                boundary_uncertainty,
                temporal_instability,
            ],
            dim=-1,
        )
        return feats

    def _compute_context(
        self,
        cand_proto_BNC: torch.Tensor,
        frame_feat_BCHW: torch.Tensor,
        state: PrototypeBankState,
        uncertainty_BNU: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, C = cand_proto_BNC.shape
        norm_bank = F.normalize(state.proto, dim=-1)
        sim_BNK = (cand_proto_BNC.unsqueeze(2) * norm_bank).sum(dim=-1)
        sim_BNK = torch.where(state.valid, sim_BNK, sim_BNK.new_full(sim_BNK.shape, -1.0))

        g_t_BC = frame_feat_BCHW.mean(dim=(2, 3))
        g_t_BNC = g_t_BC.unsqueeze(1).expand(-1, N, -1)
        age_norm = state.age / (state.age.max().detach().clamp_min(1.0))
        usage_norm = state.usage / (state.usage.max().detach().clamp_min(1.0))
        bank_stats_BNK4 = torch.stack(
            [
                state.valid.float(),
                age_norm,
                usage_norm,
                state.conf,
            ],
            dim=-1,
        )
        top2_sim = torch.topk(sim_BNK, k=min(2, self.bank_size), dim=-1).values
        top1 = top2_sim[..., 0]
        top2 = top2_sim[..., 1] if top2_sim.shape[-1] > 1 else torch.zeros_like(top1)
        sim_margin = top1 - top2
        occupancy_ratio = state.valid.float().mean(dim=-1)
        oldest_age = age_norm.max(dim=-1).values
        least_used = usage_norm.min(dim=-1).values
        mean_conf = state.conf.mean(dim=-1)
        max_conf = state.conf.max(dim=-1).values
        global_stats = torch.stack([occupancy_ratio, oldest_age, least_used, mean_conf, max_conf], dim=-1)
        context = torch.cat(
            [
                g_t_BNC,
                sim_BNK,
                bank_stats_BNK4.flatten(start_dim=2),
                torch.stack([top1, top2, sim_margin], dim=-1),
                global_stats,
                uncertainty_BNU,
            ],
            dim=-1,
        )
        target_slot = torch.where(
            state.valid.any(dim=-1),
            sim_BNK.argmax(dim=-1),
            torch.zeros(B, N, dtype=torch.long, device=cand_proto_BNC.device),
        )
        max_sim = sim_BNK.gather(dim=-1, index=target_slot.unsqueeze(-1)).squeeze(-1)
        max_sim = torch.where(state.valid.any(dim=-1), max_sim, max_sim.new_zeros(max_sim.shape))
        context_stats = {
            "top1_sim": top1,
            "top2_sim": top2,
            "sim_margin": sim_margin,
            "occupancy_ratio": occupancy_ratio,
            "oldest_age": oldest_age,
            "least_used": least_used,
            "mean_conf": mean_conf,
            "max_conf": max_conf,
            "mask_entropy_mean": uncertainty_BNU[..., 0],
            "fg_prob_mean": uncertainty_BNU[..., 1],
            "fg_area_ratio": uncertainty_BNU[..., 2],
            "boundary_uncertainty": uncertainty_BNU[..., 3],
            "temporal_instability": uncertainty_BNU[..., 4],
        }
        return context, sim_BNK, max_sim, context_stats

    def _rule_action(
        self,
        max_sim_BN: torch.Tensor,
        state: PrototypeBankState,
    ) -> torch.Tensor:
        has_free_slot = (~state.valid).any(dim=-1)
        action = torch.full_like(max_sim_BN, fill_value=self.ACTION_REFINE, dtype=torch.long)
        action = torch.where(max_sim_BN > self.sim_threshold_high, action.new_full(action.shape, self.ACTION_KEEP), action)
        action = torch.where(
            (max_sim_BN <= self.sim_threshold_high) & (max_sim_BN >= self.sim_threshold_low),
            action.new_full(action.shape, self.ACTION_REFINE),
            action,
        )
        low_sim = max_sim_BN < self.sim_threshold_low
        action = torch.where(low_sim & has_free_slot, action.new_full(action.shape, self.ACTION_SPAWN), action)
        action = torch.where(low_sim & (~has_free_slot), action.new_full(action.shape, self.ACTION_REPLACE), action)
        return action

    def _select_worst_slot(self, state: PrototypeBankState) -> torch.Tensor:
        invalid_bonus = (~state.valid).float() * 1000.0
        age_norm = state.age / state.age.max().clamp_min(1.0)
        usage_norm = state.usage / state.usage.max().clamp_min(1.0)
        conf_norm = state.conf
        worst_score = (
            self.victim_weight_age * age_norm
            - self.victim_weight_usage * usage_norm
            - self.victim_weight_conf * conf_norm
            + invalid_bonus
        )
        return worst_score.argmax(dim=-1)

    def _compute_mix_epsilon(self, current_epoch: int) -> float:
        total = max(self.epsilon_rule_mix_epochs, 1)
        ratio = min(max(current_epoch, 0) / total, 1.0)
        return self.epsilon_rule_mix_init + ratio * (self.epsilon_rule_mix_final - self.epsilon_rule_mix_init)

    def _select_action(
        self,
        action_logits_BNA: torch.Tensor,
        rule_action_BN: torch.Tensor,
        current_epoch: int,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=action_logits_BNA)
        sampled_action = dist.sample()
        greedy_action = action_logits_BNA.argmax(dim=-1)

        # Warmup stage: execute rule actions but still expose logits for CE imitation.
        if current_epoch < self.policy_warmup_epochs:
            chosen = rule_action_BN
            source = torch.full_like(rule_action_BN, self.SOURCE_RULE)
            is_learned = torch.zeros_like(rule_action_BN, dtype=torch.bool)
        elif self.exec_policy == "rule":
            chosen = rule_action_BN
            source = torch.full_like(rule_action_BN, self.SOURCE_RULE)
            is_learned = torch.zeros_like(rule_action_BN, dtype=torch.bool)
        elif self.exec_policy == "mixed":
            epsilon = self._compute_mix_epsilon(current_epoch)
            mix_rule = torch.rand_like(rule_action_BN.float()) < epsilon if training else torch.zeros_like(rule_action_BN, dtype=torch.bool)
            learned_action = sampled_action if (training and self.sample_in_train) else greedy_action
            chosen = torch.where(mix_rule, rule_action_BN, learned_action)
            source = torch.where(
                mix_rule,
                torch.full_like(rule_action_BN, self.SOURCE_MIXED_RULE),
                torch.full_like(rule_action_BN, self.SOURCE_LEARNED_SAMPLE if (training and self.sample_in_train) else self.SOURCE_LEARNED_GREEDY),
            )
            is_learned = ~mix_rule
        elif self.exec_policy == "learned" and self.use_learned_policy:
            chosen = sampled_action if (training and self.sample_in_train) else greedy_action
            source = torch.full_like(rule_action_BN, self.SOURCE_LEARNED_SAMPLE if (training and self.sample_in_train) else self.SOURCE_LEARNED_GREEDY)
            is_learned = torch.ones_like(rule_action_BN, dtype=torch.bool)
        else:
            chosen = torch.full_like(rule_action_BN, self.default_action)
            source = torch.full_like(rule_action_BN, self.SOURCE_RULE)
            is_learned = torch.zeros_like(rule_action_BN, dtype=torch.bool)

        log_prob = dist.log_prob(chosen)
        entropy = dist.entropy()
        return chosen, log_prob, entropy, source, is_learned

    def _update_state(
        self,
        state: PrototypeBankState,
        cand_proto_BNC: torch.Tensor,
        action_BN: torch.Tensor,
        target_slot_BN: torch.Tensor,
        max_sim_BN: torch.Tensor,
        mask_strength_BN: torch.Tensor,
    ) -> Tuple[PrototypeBankState, torch.Tensor]:
        proto = state.proto.clone()
        age = state.age.clone() + 1.0
        usage = state.usage.clone()
        conf = state.conf.clone()
        valid = state.valid.clone()
        worst_slot_BN = self._select_worst_slot(state)

        B, N, C = cand_proto_BNC.shape
        batch_ids = torch.arange(B, device=cand_proto_BNC.device)[:, None].expand(B, N)
        obj_ids = torch.arange(N, device=cand_proto_BNC.device)[None, :].expand(B, N)

        chosen_slot = target_slot_BN.clone()

        spawn_mask = action_BN == self.ACTION_SPAWN
        if spawn_mask.any():
            free_slot = (~valid).float().argmax(dim=-1)
            chosen_slot = torch.where((~valid).any(dim=-1), free_slot, chosen_slot)
            if self.spawn_replace_when_full:
                chosen_slot = torch.where((~valid).any(dim=-1), chosen_slot, worst_slot_BN)

        replace_mask = action_BN == self.ACTION_REPLACE
        chosen_slot = torch.where(replace_mask, worst_slot_BN, chosen_slot)

        refine_mask = action_BN == self.ACTION_REFINE
        update_mask = refine_mask | replace_mask | spawn_mask

        if refine_mask.any():
            idx = refine_mask
            slot = chosen_slot[idx]
            old_proto = proto[batch_ids[idx], obj_ids[idx], slot]
            new_proto = (1.0 - self.proto_alpha) * old_proto + self.proto_alpha * cand_proto_BNC[idx]
            proto[batch_ids[idx], obj_ids[idx], slot] = F.normalize(new_proto, dim=-1)
            valid[batch_ids[idx], obj_ids[idx], slot] = True
            conf[batch_ids[idx], obj_ids[idx], slot] = torch.clamp(
                0.5 * conf[batch_ids[idx], obj_ids[idx], slot] + 0.5 * max_sim_BN[idx],
                0.0,
                1.0,
            )

        replace_like_mask = replace_mask | spawn_mask
        if replace_like_mask.any():
            idx = replace_like_mask
            slot = chosen_slot[idx]
            proto[batch_ids[idx], obj_ids[idx], slot] = cand_proto_BNC[idx]
            age[batch_ids[idx], obj_ids[idx], slot] = 0.0
            usage[batch_ids[idx], obj_ids[idx], slot] = 1.0
            conf[batch_ids[idx], obj_ids[idx], slot] = torch.clamp(mask_strength_BN[idx], 0.0, 1.0)
            valid[batch_ids[idx], obj_ids[idx], slot] = True

        keep_mask = action_BN == self.ACTION_KEEP
        touched_mask = update_mask | keep_mask
        if touched_mask.any():
            idx = touched_mask
            slot = chosen_slot[idx]
            age[batch_ids[idx], obj_ids[idx], slot] = 0.0
            usage[batch_ids[idx], obj_ids[idx], slot] = usage[batch_ids[idx], obj_ids[idx], slot] + 1.0

        valid_float = valid.float()
        age = age * valid_float
        usage = usage * valid_float
        conf = conf * valid_float
        new_state = PrototypeBankState(proto=proto, age=age, usage=usage, conf=conf, valid=valid)
        return new_state, chosen_slot

    def _readout(self, cand_proto_BNC: torch.Tensor, state: PrototypeBankState) -> torch.Tensor:
        norm_bank = F.normalize(state.proto, dim=-1)
        sim_BNK = (cand_proto_BNC.unsqueeze(2) * norm_bank).sum(dim=-1)
        sim_BNK = sim_BNK / max(self.readout_temperature, 1e-6)
        sim_BNK = torch.where(state.valid, sim_BNK, sim_BNK.new_full(sim_BNK.shape, -1e4))
        attn_BNK = F.softmax(sim_BNK, dim=-1)
        fallback = cand_proto_BNC
        proto_readout = torch.einsum("bnk,bnkc->bnc", attn_BNK, state.proto)
        has_valid = state.valid.any(dim=-1, keepdim=True)
        proto_readout = torch.where(has_valid, proto_readout, fallback)
        return proto_readout

    def _fuse(
        self,
        value_BNCHW: torch.Tensor,
        proto_readout_BNC: torch.Tensor,
        frame_feat_BCHW: torch.Tensor,
    ) -> torch.Tensor:
        proto_map = proto_readout_BNC.unsqueeze(-1).unsqueeze(-1).expand_as(value_BNCHW)
        frame_map = frame_feat_BCHW.unsqueeze(1).expand_as(value_BNCHW)
        if self.fusion_type == "concat":
            B, N, C, H, W = value_BNCHW.shape
            fused = torch.cat([value_BNCHW + self.frame_gate * frame_map, proto_map], dim=2)
            fused = self.fuse_proj(fused.flatten(start_dim=0, end_dim=1))
            return fused.view(B, N, C, H, W)
        return value_BNCHW + self.proto_gate * proto_map + self.frame_gate * frame_map

    def forward(
        self,
        value_BNCHW: torch.Tensor,
        frame_feat_BCHW: torch.Tensor,
        mask_BNHW: Optional[torch.Tensor] = None,
        policy_meta: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self._ensure_state(value_BNCHW)
        cand_proto_BNC, mask_strength_BN = self._compute_candidate_proto(value_BNCHW, mask_BNHW)
        uncertainty_BNU = self._compute_uncertainty_features(mask_BNHW).to(value_BNCHW.device)
        context_BNF, sim_BNK, max_sim_BN, context_stats = self._compute_context(
            cand_proto_BNC,
            frame_feat_BCHW,
            state,
            uncertainty_BNU,
        )

        flat_context = context_BNF.view(-1, context_BNF.shape[-1])
        action_logits_BNA = self.policy_head(flat_context).view(*context_BNF.shape[:2], 4)
        rule_action_BN = self._rule_action(max_sim_BN, state)
        policy_meta = policy_meta or {}
        current_epoch = int(policy_meta.get("current_epoch", 0))
        is_training = bool(policy_meta.get("training", self.training))
        action_BN, log_prob_BN, entropy_BN, source_BN, learned_mask_BN = self._select_action(
            action_logits_BNA,
            rule_action_BN,
            current_epoch=current_epoch,
            training=is_training,
        )

        target_slot_BN = torch.where(
            state.valid.any(dim=-1),
            sim_BNK.argmax(dim=-1),
            torch.zeros_like(rule_action_BN),
        )
        state, chosen_slot_BN = self._update_state(
            state=state,
            cand_proto_BNC=cand_proto_BNC,
            action_BN=action_BN,
            target_slot_BN=target_slot_BN,
            max_sim_BN=max_sim_BN,
            mask_strength_BN=mask_strength_BN,
        )
        self._state = state
        if mask_BNHW is not None:
            self._prev_mask = mask_BNHW.detach().clone()

        proto_readout_BNC = self._readout(cand_proto_BNC, state)
        conditioned_BNCHW = self._fuse(value_BNCHW, proto_readout_BNC, frame_feat_BCHW)

        aux = {
            "policy_logits": action_logits_BNA,
            "policy_labels": rule_action_BN,
            "policy_actions": action_BN,
            "chosen_action": action_BN,
            "log_prob": log_prob_BN,
            "entropy": entropy_BN,
            "action_source_id": source_BN,
            "policy_is_learned": learned_mask_BN,
            "rule_action": rule_action_BN,
            "target_slot": target_slot_BN,
            "chosen_slot": chosen_slot_BN,
            "max_sim": max_sim_BN,
            "sim_vec": sim_BNK,
            "cand_proto": cand_proto_BNC,
            "proto_readout": proto_readout_BNC,
            "bank_proto": state.proto.detach(),
            "bank_age": state.age.detach(),
            "bank_usage": state.usage.detach(),
            "bank_conf": state.conf.detach(),
            "bank_valid": state.valid.detach(),
            "action_cost": torch.tensor(
                [
                    self.action_costs[self.ACTION_KEEP],
                    self.action_costs[self.ACTION_REFINE],
                    self.action_costs[self.ACTION_REPLACE],
                    self.action_costs[self.ACTION_SPAWN],
                ],
                device=value_BNCHW.device,
                dtype=value_BNCHW.dtype,
            ),
            "action_agreement": (action_BN == rule_action_BN).float(),
            "spawn_count": (action_BN == self.ACTION_SPAWN).float(),
            "replace_count": (action_BN == self.ACTION_REPLACE).float(),
        }
        aux.update(context_stats)
        if self.debug_mode:
            aux["debug_text"] = {
                "actions": action_BN.detach().cpu(),
                "slots": chosen_slot_BN.detach().cpu(),
            }
        return conditioned_BNCHW, aux
