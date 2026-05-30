from __future__ import annotations

from typing import Any

import torch.nn as nn


def count_parameters(module: nn.Module | None) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return sum(param.numel() for param in module.parameters())


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def infer_unext_capacity(model: nn.Module, cfg=None) -> dict[str, Any]:
    module = _unwrap_model(model)
    backbone = getattr(module, "backbone", None)
    faf = getattr(module, "faf", None)
    faf_selector = getattr(faf, "selector", None) if faf is not None else None
    faf_affine = getattr(faf, "affine_mixture", None) if faf is not None else None
    faf_temporal = getattr(faf, "temporal_updater", None) if faf is not None else None
    faf_fusion = getattr(faf, "fusion", None) if faf is not None else None
    method_params = count_parameters(faf)
    if method_params == 0 and backbone is not None:
        method_params = count_parameters(module) - count_parameters(backbone)

    base_dim = getattr(module, "base_dim", None)
    value_dim = getattr(module, "value_dim", None)
    num_anchors = getattr(module, "num_anchors", None)
    query_dim = getattr(module, "query_dim", None)
    code_dim = getattr(module, "code_dim", None)
    hidden_dim = getattr(module, "hidden_dim", None)

    model_cfg = _cfg_get(cfg, "model", cfg)
    for section_name in ("unext_faf", "unext_dynakey", "functional_anchor", "anchor_ode", "delay_ode"):
        section = _cfg_get(model_cfg, section_name, None)
        if section is None:
            continue
        base_dim = base_dim if base_dim is not None else _cfg_get(section, "base_dim", None)
        value_dim = value_dim if value_dim is not None else _cfg_get(section, "value_dim", None)
        num_anchors = num_anchors if num_anchors is not None else _cfg_get(section, "num_anchors", None)
        query_dim = query_dim if query_dim is not None else _cfg_get(section, "query_dim", None)
        code_dim = code_dim if code_dim is not None else _cfg_get(section, "code_dim", None)
        hidden_dim = hidden_dim if hidden_dim is not None else _cfg_get(section, "hidden_dim", None)

    channels = [int(base_dim), int(base_dim) * 2, int(base_dim) * 4] if base_dim is not None else []
    total_params = count_parameters(module)
    backbone_params = count_parameters(backbone)
    return {
        "parameters_total": total_params,
        "parameters_backbone": backbone_params,
        "parameters_method": method_params,
        "parameters_m_total": total_params / 1.0e6,
        "parameters_m_backbone": backbone_params / 1.0e6,
        "parameters_m_method": method_params / 1.0e6,
        "parameters_m_faf": count_parameters(faf) / 1.0e6,
        "parameters_m_faf_selector": count_parameters(faf_selector) / 1.0e6,
        "parameters_m_faf_affine_mixture": count_parameters(faf_affine) / 1.0e6,
        "parameters_m_faf_temporal_update": count_parameters(faf_temporal) / 1.0e6,
        "parameters_m_faf_fusion": count_parameters(faf_fusion) / 1.0e6,
        "unext_base_dim": int(base_dim) if base_dim is not None else None,
        "unext_channels": channels,
        "value_dim": int(value_dim) if value_dim is not None else None,
        "num_anchors": int(num_anchors) if num_anchors is not None else None,
        "query_dim": int(query_dim) if query_dim is not None else None,
        "code_dim": int(code_dim) if code_dim is not None else None,
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
    }
