from __future__ import annotations

from pathlib import Path
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
    """UNeXt with Functional Anchor Field — single decode path.

    Default path: encode → decode → base logits → FAF proposal → final logits.
    Feature modulation is disabled by default (ablation only).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "unext_faf", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_anchors = int(_cfg_get(method_cfg, "num_anchors", 8))
        self.num_affine_slots = int(_cfg_get(method_cfg, "num_affine_slots", self.num_anchors))
        self.identity_slot_index = int(_cfg_get(method_cfg, "identity_slot_index", 0))
        self.query_dim = int(_cfg_get(method_cfg, "query_dim", 64))
        self.code_dim = int(_cfg_get(method_cfg, "code_dim", 64))
        self.hidden_dim = int(_cfg_get(method_cfg, "hidden_dim", 128))
        self.basis_dim = int(_cfg_get(method_cfg, "basis_dim", 8))
        self.anchor_size = int(_cfg_get(method_cfg, "anchor_size", 32))
        self.residual_clip = float(_cfg_get(method_cfg, "residual_clip", 0.25))
        self.trust_max = float(_cfg_get(method_cfg, "trust_max", 0.35))
        self.retrieval_temperature = float(_cfg_get(method_cfg, "retrieval_temperature", 0.25))
        self.memory_ema = float(_cfg_get(method_cfg, "memory_ema", 0.9))
        self.enable_memory_update = bool(_cfg_get(method_cfg, "enable_memory_update", True))
        self.disable_trust_gate = bool(_cfg_get(method_cfg, "disable_trust_gate", False))
        self.disable_proposal_in_residual = bool(_cfg_get(method_cfg, "disable_proposal_in_residual", False))
        self.prediction_mode = str(_cfg_get(method_cfg, "prediction_mode", "affine_mixture_safe")).lower()
        self.allowed_prediction_modes = {
            "affine_mixture_safe",
            "affine_mixture",
            "base_only",
            "affine_identity_only",
            "affine_hard_top1",
            "affine_no_temporal",
            "affine_no_confidence",
            "affine_no_residual",
        }
        if self.prediction_mode not in self.allowed_prediction_modes:
            raise ValueError(f"Unsupported UNeXtFAF prediction_mode: {self.prediction_mode}")
        self.temperature_init = float(_cfg_get(method_cfg, "temperature_init", 0.7))
        self.temperature_warmup_iters = int(_cfg_get(method_cfg, "temperature_warmup_iters", 500))
        residual_scale_cfg = _cfg_get(method_cfg, "residual_scale", {})
        self.residual_scale_init = float(_cfg_get(residual_scale_cfg, "init", 0.01)) if hasattr(residual_scale_cfg, "get") else 0.01
        self.residual_scale_max = float(_cfg_get(residual_scale_cfg, "max", 0.08)) if hasattr(residual_scale_cfg, "get") else 0.08
        self.residual_warmup_iters = int(_cfg_get(residual_scale_cfg, "warmup_iters", 1500)) if hasattr(residual_scale_cfg, "get") else 1500
        self.trust_warmup_iters = int(_cfg_get(method_cfg, "trust_warmup_iters", 500))
        self.trust_min_warmup = float(_cfg_get(method_cfg, "trust_min_warmup", 0.10))
        self.trust_curriculum_iters = int(_cfg_get(method_cfg, "trust_curriculum_iters", 1500))
        ode_cfg = _cfg_get(method_cfg, "ode_update", {})
        self.ode_dt_init = float(_cfg_get(ode_cfg, "dt_init", 0.2)) if hasattr(ode_cfg, "get") else 0.2
        self.ode_dt_max = float(_cfg_get(ode_cfg, "dt_max", 0.8)) if hasattr(ode_cfg, "get") else 0.8
        self.ode_warmup_iters = int(_cfg_get(ode_cfg, "warmup_iters", 1500)) if hasattr(ode_cfg, "get") else 1500
        self.velocity_momentum = float(_cfg_get(ode_cfg, "velocity_momentum", 0.8)) if hasattr(ode_cfg, "get") else 0.8
        self.truncated_bptt_steps = int(_cfg_get(ode_cfg, "truncated_bptt_steps", 0)) if hasattr(ode_cfg, "get") else 0
        self.require_pretrained_unext = bool(_cfg_get(method_cfg, "require_pretrained_unext", False))
        self.pretrained_unext_path = _cfg_get(method_cfg, "pretrained_unext_path", None)
        self.pretrained_unext_strict_backbone = bool(_cfg_get(method_cfg, "pretrained_unext_strict_backbone", False))
        feature_mod_cfg = _cfg_get(method_cfg, "feature_modulation", {})
        self.disable_feature_modulation = not bool(_cfg_get(feature_mod_cfg, "enabled", False)) if hasattr(feature_mod_cfg, "get") else True

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        self._load_pretrained_unext_if_requested()
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
            trust_min_warmup=self.trust_min_warmup,
            trust_curriculum_iters=self.trust_curriculum_iters,
            ode_dt_init=self.ode_dt_init,
            ode_dt_max=self.ode_dt_max,
            ode_warmup_iters=self.ode_warmup_iters,
            velocity_momentum=self.velocity_momentum,
            feature_modulation=feature_mod_cfg,
            disable_proposal_in_residual=self.disable_proposal_in_residual,
            disable_feature_modulation=self.disable_feature_modulation,
            num_affine_slots=self.num_affine_slots,
            identity_slot_index=self.identity_slot_index,
            affine_cfg=_cfg_get(method_cfg, "affine", {}),
            selector_cfg=_cfg_get(method_cfg, "selector", {}),
            confidence_cfg=_cfg_get(method_cfg, "confidence", {}),
            residual_cfg=_cfg_get(method_cfg, "residual", {}),
            temporal_update_cfg=_cfg_get(method_cfg, "temporal_update", _cfg_get(method_cfg, "ode_update", {})),
        )

    def _load_pretrained_unext_if_requested(self) -> None:
        path_value = self.pretrained_unext_path
        if path_value in (None, "", "null"):
            if self.require_pretrained_unext:
                raise FileNotFoundError(
                    "model.unext_faf.require_pretrained_unext=true but pretrained_unext_path is empty. "
                    "Run Stage 0 UNeXt anchor warmup first and set model.unext_faf.pretrained_unext_path."
                )
            return
        path = Path(str(path_value)).expanduser()
        if not path.exists():
            if self.require_pretrained_unext:
                raise FileNotFoundError(f"Pretrained UNeXt checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported pretrained checkpoint format: {path}")
        backbone_state = {}
        target = self.backbone.state_dict()
        for key, value in state.items():
            clean = key[7:] if key.startswith("module.") else key
            if clean.startswith("backbone."):
                clean = clean[len("backbone."):]
            elif clean.startswith("model.backbone."):
                clean = clean[len("model.backbone."):]
            else:
                continue
            if clean in target and tuple(target[clean].shape) == tuple(value.shape):
                backbone_state[clean] = value
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if self.pretrained_unext_strict_backbone and (missing or unexpected):
            raise RuntimeError(f"Strict UNeXt checkpoint load failed: missing={missing}, unexpected={unexpected}")
        if self.require_pretrained_unext and not backbone_state:
            raise RuntimeError(f"No compatible UNeXt backbone weights found in {path}")

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def _fuse_object_logits(self, base_logits: torch.Tensor, faf_aux: dict) -> torch.Tensor:
        del base_logits
        return faf_aux["final_logits"]

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        state = self.faf.initial_state(B, max_num_objects, images.device, images.dtype)
        out: Dict = {"num_objects": num_objects}
        global_step = data.get("global_step", data.get("current_iter"))

        for ti in range(T):
            image = self._normalize(images[:, ti])

            # Single encode + decode (no second decode for modulation)
            encoded = self.backbone.encode(image)
            decoded = self.backbone.decode(encoded["low"], encoded["mid"], encoded["high"], image.shape[-2:])
            feat = {**encoded, **decoded}
            feats = {
                "low": feat["low"],
                "mid": feat["mid"],
                "high": feat["high"],
                "dec": feat["decoder_feature"],
            }
            base_object_logits = feat["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)

            final_candidate_logits, faf_aux, state = self.faf.forward_step(
                feats, base_object_logits, state, global_step=global_step, mode=self.prediction_mode,
            )

            final_object_logits = self._fuse_object_logits(base_object_logits, faf_aux)
            logits, masks = self._object_logits_to_full(final_object_logits)

            faf_aux = {
                **faf_aux,
                "base_object_logits": base_object_logits,
                "anchor_logits": base_object_logits,
                "final_object_logits": final_object_logits,
                "base_logits": base_object_logits,
                "final_logits": final_object_logits,
                "safety_base_logits": final_candidate_logits,
                "prediction_mode": self.prediction_mode,
                "mode": "online",
            }
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": final_object_logits.detach(),
                "anchor_logits": base_object_logits.detach(),
                "mixture_logits": faf_aux["mixture_logits"].detach(),
            }
            out[f"memory_aux_{ti}"] = {"faf_aux": faf_aux}
        return out
