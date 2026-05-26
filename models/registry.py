from __future__ import annotations

import torch

from utils.registry import Registry
from model.gdkvm01 import GDKVM
from model.delay_ode import DelayODEKeyMapSegmenter
from model.anchor_ode import UNeXtAnchorODEAffineSegmenter, UNeXtAnchorODESegmenter
from model.functional_anchor import FunctionalAnchorSegmenter
from model.unext_faf import UNeXtFAF
from model.unext_dynakey import UNeXtDynaKeySegmenter


MODEL_REGISTRY = Registry("model")


def _model_cfg(cfg):
    return cfg.get("model", cfg) if hasattr(cfg, "get") else cfg


def _allow_oracle_init(model_cfg) -> bool:
    return bool(
        model_cfg.get(
            "allow_oracle_init_when_requested",
            model_cfg.get("use_first_frame_gt_init", True),
        )
    )


@MODEL_REGISTRY.register("gdkvm")
@MODEL_REGISTRY.register("banditpm")
def build_gdkvm(cfg, *, device: torch.device | str):
    model_cfg = _model_cfg(cfg)
    return GDKVM(
        use_first_frame_gt_init=_allow_oracle_init(model_cfg),
        prototype_value_cfg=model_cfg.get("prototype_value", None),
        temporal_memory_cfg=model_cfg.get("temporal_memory", None),
        memory_core_cfg=model_cfg.get("memory_core", None),
        use_kpff=bool(model_cfg.get("use_kpff", True)),
    ).to(device)


@MODEL_REGISTRY.register("kpff")
def build_kpff(cfg, *, device: torch.device | str):
    model_cfg = _model_cfg(cfg)
    return GDKVM(
        use_first_frame_gt_init=_allow_oracle_init(model_cfg),
        prototype_value_cfg=None,
        temporal_memory_cfg={"type": "none"},
        memory_core_cfg={"type": "none"},
        use_kpff=True,
    ).to(device)


@MODEL_REGISTRY.register("unext_fusion")
@MODEL_REGISTRY.register("unext_dynakey")
@MODEL_REGISTRY.register("dynakey_unext")
@MODEL_REGISTRY.register("unextdynakey")
def build_unext_fusion(cfg, *, device: torch.device | str):
    return UNeXtDynaKeySegmenter(_model_cfg(cfg)).to(device)


@MODEL_REGISTRY.register("delay_ode")
def build_delay_ode(cfg, *, device: torch.device | str):
    return DelayODEKeyMapSegmenter(_model_cfg(cfg)).to(device)


@MODEL_REGISTRY.register("anchor_ode")
@MODEL_REGISTRY.register("unext_anchor_ode")
@MODEL_REGISTRY.register("unextanchorode")
def build_anchor_ode(cfg, *, device: torch.device | str):
    return UNeXtAnchorODESegmenter(_model_cfg(cfg)).to(device)


@MODEL_REGISTRY.register("anchor_ode_v2")
@MODEL_REGISTRY.register("unext_anchor_ode_affine")
@MODEL_REGISTRY.register("unextanchorodeaffine")
def build_anchor_ode_v2(cfg, *, device: torch.device | str):
    return UNeXtAnchorODEAffineSegmenter(_model_cfg(cfg)).to(device)


@MODEL_REGISTRY.register("functional_anchor")
def build_functional_anchor(cfg, *, device: torch.device | str):
    return FunctionalAnchorSegmenter(_model_cfg(cfg)).to(device)


@MODEL_REGISTRY.register("unext_faf")
@MODEL_REGISTRY.register("unext-faf")
@MODEL_REGISTRY.register("faf")
def build_unext_faf(cfg, *, device: torch.device | str):
    return UNeXtFAF(_model_cfg(cfg)).to(device)


def build_model(cfg, *, device: torch.device | str):
    model_cfg = _model_cfg(cfg)
    if not str(model_cfg.get("name", "")).strip():
        model_cfg.name = "gdkvm"
    return MODEL_REGISTRY.build(cfg, device=device)
