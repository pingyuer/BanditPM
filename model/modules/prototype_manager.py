from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, cfg, value_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.value_dim = value_dim
        self.bank_size = int(cfg.BANK_SIZE)
        self.proto_alpha = float(cfg.PROTO_ALPHA)
        self.readout_temperature = float(getattr(cfg, "READOUT_TEMPERATURE", 1.0))
        self.default_action = int(getattr(cfg, "DEFAULT_ACTION", self.ACTION_REFINE))
        self.spawn_replace_when_full = bool(getattr(cfg, "SPAWN_REPLACE_WHEN_FULL", True))
        self.use_rule_based_policy = bool(cfg.USE_RULE_BASED_POLICY)
        self.use_learned_policy = bool(getattr(cfg, "USE_LEARNED_POLICY", False))
        self.policy_loss_weight = float(getattr(cfg, "POLICY_LOSS_WEIGHT", 1.0))
        self.sim_threshold_high = float(cfg.SIM_THRESHOLD_HIGH)
        self.sim_threshold_low = float(cfg.SIM_THRESHOLD_LOW)
        self.debug_mode = bool(getattr(cfg, "DEBUG_MODE", False))
        self.mask_pooling = str(getattr(cfg, "PROTO_POOLING", "mask")).lower()
        self.fusion_type = str(getattr(cfg, "FUSION_TYPE", "add")).lower()
        self.hidden_dim = int(getattr(cfg, "POLICY_HIDDEN_DIM", value_dim))

        action_costs = getattr(cfg, "ACTION_COSTS", {})
        self.action_costs = {
            self.ACTION_KEEP: float(action_costs.get("keep", 0.0)),
            self.ACTION_REFINE: float(action_costs.get("refine", 0.05)),
            self.ACTION_REPLACE: float(action_costs.get("replace", 0.1)),
            self.ACTION_SPAWN: float(action_costs.get("spawn", 0.2)),
        }

        # Context = g_t[C] + sim_vec[K] + bank_stats[3K]
        context_dim = value_dim + self.bank_size + self.bank_size * 3
        self.policy_head = PolicyHead(context_dim=context_dim, hidden_dim=self.hidden_dim)

        if self.fusion_type == "concat":
            self.fuse_proj = nn.Conv2d(value_dim * 2, value_dim, kernel_size=1)
        else:
            self.fuse_proj = None
            self.proto_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.frame_gate = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._state: Optional[PrototypeBankState] = None

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        shape_bnk = (batch_size, num_objects, self.bank_size)
        self._state = PrototypeBankState(
            proto=torch.zeros(*shape_bnk, self.value_dim, device=device),
            age=torch.zeros(*shape_bnk, device=device),
            usage=torch.zeros(*shape_bnk, device=device),
            conf=torch.zeros(*shape_bnk, device=device),
            valid=torch.zeros(*shape_bnk, dtype=torch.bool, device=device),
        )

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

    def _compute_context(
        self,
        cand_proto_BNC: torch.Tensor,
        frame_feat_BCHW: torch.Tensor,
        state: PrototypeBankState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = cand_proto_BNC.shape
        norm_bank = F.normalize(state.proto, dim=-1)
        sim_BNK = (cand_proto_BNC.unsqueeze(2) * norm_bank).sum(dim=-1)
        sim_BNK = torch.where(state.valid, sim_BNK, sim_BNK.new_full(sim_BNK.shape, -1.0))

        g_t_BC = frame_feat_BCHW.mean(dim=(2, 3))
        g_t_BNC = g_t_BC.unsqueeze(1).expand(-1, N, -1)
        bank_stats_BNK3 = torch.stack(
            [
                state.age / (state.age.max().detach().clamp_min(1.0)),
                state.usage / (state.usage.max().detach().clamp_min(1.0)),
                state.conf,
            ],
            dim=-1,
        )
        context = torch.cat(
            [
                g_t_BNC,
                sim_BNK,
                bank_stats_BNK3.flatten(start_dim=2),
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
        return context, sim_BNK, max_sim

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
        worst_score = state.age - state.usage - state.conf + invalid_bonus
        return worst_score.argmax(dim=-1)

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self._ensure_state(value_BNCHW)
        cand_proto_BNC, mask_strength_BN = self._compute_candidate_proto(value_BNCHW, mask_BNHW)
        context_BNF, sim_BNK, max_sim_BN = self._compute_context(cand_proto_BNC, frame_feat_BCHW, state)

        flat_context = context_BNF.view(-1, context_BNF.shape[-1])
        action_logits_BNA = self.policy_head(flat_context).view(*context_BNF.shape[:2], 4)
        rule_action_BN = self._rule_action(max_sim_BN, state)

        if self.use_rule_based_policy:
            action_BN = rule_action_BN
        elif self.use_learned_policy:
            action_BN = action_logits_BNA.argmax(dim=-1)
        else:
            action_BN = torch.full_like(rule_action_BN, fill_value=self.default_action)

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

        proto_readout_BNC = self._readout(cand_proto_BNC, state)
        conditioned_BNCHW = self._fuse(value_BNCHW, proto_readout_BNC, frame_feat_BCHW)

        aux = {
            "policy_logits": action_logits_BNA,
            "policy_labels": rule_action_BN,
            "policy_actions": action_BN,
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
        }
        if self.debug_mode:
            aux["debug_text"] = {
                "actions": action_BN.detach().cpu(),
                "slots": chosen_slot_BN.detach().cpu(),
            }
        return conditioned_BNCHW, aux
