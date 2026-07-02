"""Unified temporal memory core used by the GDKVM shell."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.modules.banditpm_core import BanditPMCore
from model.modules.dynakey import DynaKeyMemoryCore
from model.modules.gdr_core import GDRCore
from model.modules.prototype_manager import BanditPrototypeManager
from model.modules.prototype_value_head import PrototypeValueHead


class MemoryCore(nn.Module):
    """Collect GDR/BPM/prototype paths behind one replaceable interface."""

    def __init__(
        self,
        value_dim: int,
        key_dim: int,
        prototype_value_cfg=None,
        temporal_memory_cfg=None,
        memory_core_cfg=None,
    ) -> None:
        super().__init__()
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.prototype_value_cfg = prototype_value_cfg
        self.temporal_memory_cfg = temporal_memory_cfg
        self.memory_core_cfg = memory_core_cfg

        self.memory_type = self._resolve_memory_type(memory_core_cfg, temporal_memory_cfg)
        self.prototype_value_head = None
        if prototype_value_cfg is not None and bool(prototype_value_cfg.enable):
            self.prototype_value_head = PrototypeValueHead(
                prototype_value_cfg,
                input_dim=value_dim,
                value_dim=value_dim,
            )

        self.gdr_core = GDRCore(value_dim=value_dim)
        self.prototype_manager = None
        self.bpm_key_adapter = None
        self.banditpm_core = None
        self.dynakey_core = None

        temporal_type = self.memory_type
        if temporal_type in {"bpm"}:
            bpm_cfg = temporal_memory_cfg.bpm if temporal_memory_cfg is not None else None
            if bpm_cfg is not None and bool(bpm_cfg.get("ENABLE", True)):
                self.prototype_manager = BanditPrototypeManager(
                    bpm_cfg,
                    value_dim=value_dim,
                )
                self.bpm_key_adapter = nn.Conv2d(key_dim, value_dim, kernel_size=1)
            self.gdr_core.freeze_parameters()
        elif temporal_type in {"none", "identity", "static_proto", "banditpm_placeholder"}:
            self.gdr_core.freeze_parameters()
            if temporal_type == "banditpm_placeholder":
                self.banditpm_core = BanditPMCore()
        elif temporal_type == "dynakey":
            dynakey_cfg = None
            if memory_core_cfg is not None:
                dynakey_cfg = memory_core_cfg.get("dynakey", None)
            if dynakey_cfg is None and temporal_memory_cfg is not None:
                dynakey_cfg = temporal_memory_cfg.get("dynakey", None)
            self.dynakey_core = DynaKeyMemoryCore(dynakey_cfg, value_dim=value_dim)
            self.gdr_core.freeze_parameters()

    def _resolve_memory_type(self, memory_core_cfg, temporal_memory_cfg) -> str:
        if temporal_memory_cfg is not None:
            temporal_type = str(temporal_memory_cfg.get("type", "gdr")).lower()
            mapping = {
                "gdr": "original_gdr",
                "none": "none",
                "bpm": "bpm",
                "dynakey": "dynakey",
            }
            temporal_core_type = mapping.get(temporal_type, temporal_type)
        else:
            temporal_core_type = "original_gdr"

        if memory_core_cfg is not None:
            core_type = str(memory_core_cfg.get("type", temporal_core_type)).lower()
            if core_type == "original_gdr" and temporal_core_type != "original_gdr":
                core_type = temporal_core_type
        else:
            core_type = temporal_core_type
        return core_type

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device) -> None:
        self.gdr_core.reset_state(batch_size=batch_size, num_objects=num_objects, device=device)
        if self.prototype_value_head is not None and self.prototype_value_head.reset_per_video:
            self.prototype_value_head.reset_state(batch_size=batch_size, device=device)
        if self.prototype_manager is not None:
            self.prototype_manager.reset_state(
                batch_size=batch_size,
                num_objects=num_objects,
                device=device,
            )
        if self.banditpm_core is not None:
            self.banditpm_core.reset_state(
                batch_size=batch_size,
                num_objects=num_objects,
                device=device,
            )
        if self.dynakey_core is not None:
            self.dynakey_core.reset_state(
                batch_size=batch_size,
                num_objects=num_objects,
                device=device,
            )

    def _select_proto_feature(
        self,
        feature_source: str,
        value_BNCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
    ) -> torch.Tensor:
        if feature_source in {"value", "value_mid"}:
            return value_BNCHW
        if feature_source in {"encoder", "shared"}:
            return pixfeat_BCHW
        raise ValueError(f"Unsupported prototype feature_source={feature_source!r}")

    def _apply_prototype_value(
        self,
        value_BNCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        aux: Dict,
    ) -> torch.Tensor:
        if self.prototype_value_head is None:
            return value_BNCHW

        feat_t = self._select_proto_feature(
            self.prototype_value_head.feature_source,
            value_BNCHW,
            pixfeat_BCHW,
        )
        value_out, proto_aux = self.prototype_value_head(feat_t=feat_t, v_orig=value_BNCHW)
        if proto_aux:
            aux["proto_value_aux"] = proto_aux
        return value_out

    def _get_bpm_frame_feature(
        self,
        pixfeat_BCHW: torch.Tensor,
        key_BCHW: torch.Tensor,
    ) -> torch.Tensor:
        if self.bpm_key_adapter is None:
            return pixfeat_BCHW
        return pixfeat_BCHW + self.bpm_key_adapter(key_BCHW)

    def forward(
        self,
        value_BNCHW: torch.Tensor,
        key_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        mask_BNHW: Optional[torch.Tensor],
        policy_meta: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        aux: Dict = {"memory_type": self.memory_type}
        value_BNCHW = self._apply_prototype_value(value_BNCHW, pixfeat_BCHW, aux)

        if self.memory_type == "original_gdr":
            readout_BNCHW, gdr_aux = self.gdr_core(
                value_BNCHW=value_BNCHW,
                key_BCHW=key_BCHW,
            )
            if gdr_aux:
                aux["gdr_aux"] = gdr_aux
            return readout_BNCHW, aux

        if self.memory_type == "bpm":
            if self.prototype_manager is None:
                return value_BNCHW, aux
            frame_feat_BCHW = self._get_bpm_frame_feature(pixfeat_BCHW, key_BCHW)
            readout_BNCHW, bpm_aux = self.prototype_manager(
                value_BNCHW=value_BNCHW,
                frame_feat_BCHW=frame_feat_BCHW,
                mask_BNHW=mask_BNHW,
                policy_meta=policy_meta,
            )
            aux.update(bpm_aux)
            return readout_BNCHW, aux

        if self.memory_type == "banditpm_placeholder" and self.banditpm_core is not None:
            readout_BNCHW, placeholder_aux = self.banditpm_core(
                value_BNCHW=value_BNCHW,
                key_BCHW=key_BCHW,
                pixfeat_BCHW=pixfeat_BCHW,
                mask_BNHW=mask_BNHW,
                policy_meta=policy_meta,
            )
            aux.update(placeholder_aux)
            return readout_BNCHW, aux

        if self.memory_type == "dynakey" and self.dynakey_core is not None:
            readout_BNCHW, dynakey_aux = self.dynakey_core(
                value_BNCHW=value_BNCHW,
                key_BCHW=key_BCHW,
                pixfeat_BCHW=pixfeat_BCHW,
                mask_BNHW=mask_BNHW,
                policy_meta=policy_meta,
            )
            aux["dynakey_aux"] = dynakey_aux
            return readout_BNCHW, aux

        # "none", "identity", and "static_proto" currently share the same readout.
        return value_BNCHW, aux
