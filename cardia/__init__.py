from .cardia import CARDIA
from .registry import MODEL_REGISTRY, BACKBONE_REGISTRY

import torch


def _model_cfg(cfg):
    return cfg.get("model", cfg) if hasattr(cfg, "get") else cfg


@MODEL_REGISTRY.register("cardia")
@MODEL_REGISTRY.register("unext_cardia")
def build_cardia(cfg, *, device: torch.device | str):
    return CARDIA(_model_cfg(cfg)).to(device)


def build_model(cfg, *, device: torch.device | str):
    model_cfg = _model_cfg(cfg)
    if not str(model_cfg.get("name", "")).strip():
        model_cfg.name = "cardia"
    return MODEL_REGISTRY.build(cfg, device=device)


__all__ = ["CARDIA", "build_model", "MODEL_REGISTRY", "BACKBONE_REGISTRY"]
