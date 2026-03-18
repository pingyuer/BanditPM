from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.modules.prototype_temporal_state import PrototypeTemporalState
from model.modules.prototype_value_bank import PrototypeValueBank
from model.modules.prototype_value_fuser import PrototypeValueFuser


class PrototypeValueHead(nn.Module):
    """
    Prototype-guided value augmentation head.

    Shapes:
        feat_t: [B, C, H, W] or [B, N, C, H, W]
        v_orig: [B, N, C, H, W]
        v_out:  [B, N, C, H, W]
    """

    def __init__(self, cfg, input_dim: int, value_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable = bool(cfg.enable)
        self.module_mode = str(cfg.module_mode).lower()
        self.mode = str(cfg.mode).lower()
        self.feature_source = str(cfg.feature_source).lower()
        self.replace_ratio = float(cfg.replace_ratio)
        self.debug_cfg = cfg.debug

        bank_cfg = cfg.bank
        self.input_dim = input_dim
        self.value_dim = value_dim
        self.bank_dim = bank_cfg.dim
        self.input_proj = nn.Identity() if input_dim == self.bank_dim else nn.Linear(input_dim, self.bank_dim)
        self.output_proj = nn.Identity() if self.bank_dim == value_dim else nn.Linear(self.bank_dim, value_dim)
        self.bank = PrototypeValueBank(
            num_proto=bank_cfg.num_proto,
            dim=bank_cfg.dim,
            temperature=bank_cfg.temperature,
            normalize=bank_cfg.normalize,
            init_mode=bank_cfg.init_mode,
            topk=bank_cfg.topk,
        )

        fuse_cfg = cfg.fuse
        self.fuser = PrototypeValueFuser(
            mode=fuse_cfg.type,
            value_dim=value_dim,
            proto_dim=value_dim,
            hidden_dim=fuse_cfg.hidden_dim,
        )

        temporal_cfg = cfg.temporal
        self.use_slow_state = bool(temporal_cfg.enable_slow_state) or self.module_mode == "slow"
        self.temporal_state = PrototypeTemporalState(
            num_proto=bank_cfg.num_proto,
            momentum=temporal_cfg.momentum,
            learnable_momentum=temporal_cfg.learnable_momentum,
            detach_prev_state=temporal_cfg.detach_prev_state,
        )
        self.reset_per_video = bool(temporal_cfg.reset_per_video)
        self._prev_state: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self._prev_state = self.temporal_state.reset(batch_size=batch_size, device=device)

    def _pool_feature(self, feat_t: torch.Tensor) -> torch.Tensor:
        if feat_t.ndim == 5:
            return feat_t.mean(dim=(1, 3, 4))
        if feat_t.ndim == 4:
            return feat_t.mean(dim=(2, 3))
        if feat_t.ndim == 2:
            return feat_t
        raise ValueError(f"Unsupported prototype feature shape={tuple(feat_t.shape)}")

    def _replace_value(self, v_orig: torch.Tensor, v_proto: torch.Tensor) -> torch.Tensor:
        proto_map = v_proto[:, None, :, None, None].expand_as(v_orig)
        ratio = max(0.0, min(1.0, self.replace_ratio))
        return (1.0 - ratio) * v_orig + ratio * proto_map

    def forward(
        self,
        feat_t: torch.Tensor,
        v_orig: torch.Tensor,
        state_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable:
            return v_orig, {}

        feat_vec = self._pool_feature(feat_t)
        feat_vec = self.input_proj(feat_vec)
        assign_prob, proto_value, aux = self.bank(feat_vec)

        temporal_state = None
        proto_for_value = proto_value
        if self.use_slow_state:
            prev_state = state_cache if state_cache is not None else self._prev_state
            temporal_state = self.temporal_state(assign_prob, prev_state=prev_state)
            self._prev_state = temporal_state
            proto_for_value = torch.matmul(temporal_state, self.bank.codebook)

        proto_for_value = self.output_proj(proto_for_value)

        if self.mode == "replace":
            v_out = self._replace_value(v_orig, proto_for_value)
        else:
            v_out = self.fuser(v_orig, proto_for_value)

        aux_out = {
            "p_t": assign_prob,
            "v_proto": proto_for_value,
        }
        if temporal_state is not None:
            aux_out["T_t"] = temporal_state
        aux_out.update(aux)
        return v_out, aux_out
