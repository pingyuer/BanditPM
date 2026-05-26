from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.functional_anchor.faf_module import FAFModule
from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class UNeXtFAF(nn.Module):
    """UNeXt with an online Functional Anchor Field residual correction path."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "unext_faf", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_anchors = int(_cfg_get(method_cfg, "num_anchors", 8))
        self.query_dim = int(_cfg_get(method_cfg, "query_dim", 64))
        self.code_dim = int(_cfg_get(method_cfg, "code_dim", 64))
        self.hidden_dim = int(_cfg_get(method_cfg, "hidden_dim", 128))
        self.basis_dim = int(_cfg_get(method_cfg, "basis_dim", 8))
        self.anchor_size = int(_cfg_get(method_cfg, "anchor_size", 32))
        self.residual_clip = float(_cfg_get(method_cfg, "residual_clip", 0.5))
        self.trust_max = float(_cfg_get(method_cfg, "trust_max", 0.6))
        self.retrieval_temperature = float(_cfg_get(method_cfg, "retrieval_temperature", 0.2))
        self.memory_ema = float(_cfg_get(method_cfg, "memory_ema", 0.9))
        self.enable_memory_update = bool(_cfg_get(method_cfg, "enable_memory_update", True))
        self.disable_trust_gate = bool(_cfg_get(method_cfg, "disable_trust_gate", False))
        self.temperature_init = float(_cfg_get(method_cfg, "temperature_init", 0.7))
        self.temperature_warmup_iters = int(_cfg_get(method_cfg, "temperature_warmup_iters", 500))
        residual_scale_cfg = _cfg_get(method_cfg, "residual_scale", {})
        self.residual_scale_init = float(_cfg_get(residual_scale_cfg, "init", 0.02)) if hasattr(residual_scale_cfg, "get") else 0.02
        self.residual_scale_max = float(_cfg_get(residual_scale_cfg, "max", 0.12)) if hasattr(residual_scale_cfg, "get") else 0.12
        self.residual_warmup_iters = int(_cfg_get(residual_scale_cfg, "warmup_iters", 1500)) if hasattr(residual_scale_cfg, "get") else 1500
        self.trust_warmup_iters = int(_cfg_get(method_cfg, "trust_warmup_iters", 500))
        ode_cfg = _cfg_get(method_cfg, "ode_update", {})
        self.ode_dt_init = float(_cfg_get(ode_cfg, "dt_init", 0.5)) if hasattr(ode_cfg, "get") else 0.5
        self.ode_dt_max = float(_cfg_get(ode_cfg, "dt_max", 1.0)) if hasattr(ode_cfg, "get") else 1.0
        self.ode_warmup_iters = int(_cfg_get(ode_cfg, "warmup_iters", 1500)) if hasattr(ode_cfg, "get") else 1500
        self.velocity_momentum = float(_cfg_get(ode_cfg, "velocity_momentum", 0.8)) if hasattr(ode_cfg, "get") else 0.8
        self.mode = str(_cfg_get(method_cfg, "mode", "online")).lower()
        if self.mode != "online":
            raise ValueError("UNeXtFAF currently supports only online mode")

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        feature_dims = {
            "low": self.base_dim,
            "mid": self.base_dim * 2,
            "high": self.base_dim * 4,
            "dec": self.base_dim,
        }
        self.faf = FAFModule(
            feature_dims=feature_dims,
            num_anchors=self.num_anchors,
            query_dim=self.query_dim,
            code_dim=self.code_dim,
            hidden_dim=self.hidden_dim,
            basis_dim=self.basis_dim,
            anchor_size=self.anchor_size,
            residual_clip=self.residual_clip,
            trust_max=self.trust_max,
            retrieval_temperature=self.retrieval_temperature,
            memory_ema=self.memory_ema,
            enable_memory_update=self.enable_memory_update,
            disable_trust_gate=self.disable_trust_gate,
            temperature_init=self.temperature_init,
            temperature_warmup_iters=self.temperature_warmup_iters,
            residual_scale_init=self.residual_scale_init,
            residual_scale_max=self.residual_scale_max,
            residual_warmup_iters=self.residual_warmup_iters,
            trust_warmup_iters=self.trust_warmup_iters,
            ode_dt_init=self.ode_dt_init,
            ode_dt_max=self.ode_dt_max,
            ode_warmup_iters=self.ode_warmup_iters,
            velocity_momentum=self.velocity_momentum,
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        state = self.faf.initial_state(B, max_num_objects, images.device, images.dtype)
        out: Dict = {"num_objects": num_objects}
        global_step = data.get("global_step", data.get("current_iter", 0))

        for ti in range(T):
            feat = self.backbone(self._normalize(images[:, ti]))
            feats = {
                "low": feat["low"],
                "mid": feat["mid"],
                "high": feat["high"],
                "dec": feat["decoder_feature"],
            }
            base_object_logits = feat["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)
            final_object_logits, faf_aux, state = self.faf.forward_step(
                feats,
                base_object_logits,
                state,
                global_step=global_step,
            )
            logits, masks = self._object_logits_to_full(final_object_logits)

            faf_aux = {
                **faf_aux,
                "base_object_logits": base_object_logits,
                "final_object_logits": final_object_logits,
                "base_logits": base_object_logits,
                "final_logits": final_object_logits,
                "mode": "online",
                "trust_mean": faf_aux["trust_mean"],
                "anchor_trust_ratio": faf_aux["anchor_trust_ratio"],
            }
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": final_object_logits.detach(),
                "anchor_logits": faf_aux["anchor_logits"].detach(),
            }
            out[f"memory_aux_{ti}"] = {"faf_aux": faf_aux}
        return out
