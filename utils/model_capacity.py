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
    if all(hasattr(module, name) for name in ("image_tokenizer", "mask_tokenizer", "transformer", "pixel_decoder", "proposal_decoder")):
        image_params = count_parameters(getattr(module, "image_tokenizer", None))
        mask_params = count_parameters(getattr(module, "mask_tokenizer", None))
        transformer_params = count_parameters(getattr(module, "transformer", None))
        pixel_params = count_parameters(getattr(module, "pixel_decoder", None))
        proposal_params = count_parameters(getattr(module, "proposal_decoder", None))
        proposal_params += count_parameters(getattr(module, "full_res_refiner", None))
        proposal_params += count_parameters(getattr(module, "full_res_proposal_head", None))
        proposal_params += count_parameters(getattr(module, "variant_refiner", None))
        total_params = count_parameters(module)
        geomask_cfg = _cfg_get(_cfg_get(cfg, "model", cfg), "geomaskformer", {})
        dim = _cfg_get(geomask_cfg, "dim", getattr(module, "dim", None))
        base_dim = _cfg_get(geomask_cfg, "base_channels", getattr(module, "base_dim", None))
        return {
            "parameters_total": total_params,
            "parameters_backbone": image_params,
            "parameters_method": total_params - image_params,
            "parameters_m_total": total_params / 1.0e6,
            "parameters_m_backbone": image_params / 1.0e6,
            "parameters_m_method": (total_params - image_params) / 1.0e6,
            "parameters_m_encoder": image_params / 1.0e6,
            "parameters_m_memory": 0.0,
            "parameters_m_ode": 0.0,
            "parameters_m_decoder": (pixel_params + proposal_params) / 1.0e6,
            "parameters_m_transformer": transformer_params / 1.0e6,
            "parameters_m_tokenizer": (image_params + mask_params) / 1.0e6,
            "parameters_m_faf": 0.0,
            "parameters_m_faf_selector": 0.0,
            "parameters_m_faf_affine_mixture": 0.0,
            "parameters_m_faf_temporal_update": 0.0,
            "parameters_m_faf_fusion": 0.0,
            "backbone_name": "geomaskformer",
            "unext_base_dim": int(base_dim) if base_dim is not None else None,
            "unext_channels": [int(base_dim), int(base_dim) * 2, int(base_dim) * 4] if base_dim is not None else [],
            "value_dim": int(dim) if dim is not None else None,
            "num_anchors": int(_cfg_get(geomask_cfg, "num_queries", getattr(module, "num_queries", 0)) or 0),
            "query_dim": int(dim) if dim is not None else None,
            "code_dim": None,
            "hidden_dim": int(dim) if dim is not None else None,
        }
    if hasattr(module, "frame_net") and hasattr(module, "tokenizer") and hasattr(module, "grid_solver"):
        backbone_params = count_parameters(getattr(module, "frame_net", None))
        transformer_params = count_parameters(getattr(module, "video_encoder", None))
        tokenizer_params = count_parameters(getattr(module, "tokenizer", None))
        query_params = count_parameters(getattr(module, "query_decoder", None))
        grid_params = count_parameters(getattr(module, "grid_solver", None))
        residual_params = count_parameters(getattr(module, "boundary_residual", None))
        total_params = count_parameters(module)
        method_params = total_params - backbone_params
        debel_cfg = _cfg_get(_cfg_get(cfg, "model", cfg), "debel", {})
        backbone_cfg = _cfg_get(debel_cfg, "backbone", {})
        base_dim = _cfg_get(backbone_cfg, "base_dim", _cfg_get(debel_cfg, "base_dim", None))
        d_model = _cfg_get(debel_cfg, "d_model", None)
        return {
            "parameters_total": total_params,
            "parameters_backbone": backbone_params,
            "parameters_method": method_params,
            "parameters_m_total": total_params / 1.0e6,
            "parameters_m_backbone": backbone_params / 1.0e6,
            "parameters_m_method": method_params / 1.0e6,
            "parameters_m_debel": method_params / 1.0e6,
            "parameters_m_transformer": transformer_params / 1.0e6,
            "parameters_m_grid_head": grid_params / 1.0e6,
            "parameters_m_query_memory": query_params / 1.0e6,
            "parameters_m_tokenizer": tokenizer_params / 1.0e6,
            "parameters_m_residual": residual_params / 1.0e6,
            "parameters_m_faf": 0.0,
            "parameters_m_faf_selector": 0.0,
            "parameters_m_faf_affine_mixture": 0.0,
            "parameters_m_faf_temporal_update": 0.0,
            "parameters_m_faf_fusion": 0.0,
            "backbone_name": str(_cfg_get(backbone_cfg, "name", "official")),
            "unext_base_dim": int(base_dim) if base_dim is not None else None,
            "unext_channels": [int(base_dim), int(base_dim) * 2, int(base_dim) * 4] if base_dim is not None else [],
            "value_dim": int(d_model) if d_model is not None else None,
            "num_anchors": None,
            "query_dim": int(d_model) if d_model is not None else None,
            "code_dim": None,
            "hidden_dim": int(d_model) if d_model is not None else None,
        }
    if hasattr(module, "encoder") and hasattr(module, "ode") and hasattr(module, "memory") and hasattr(module, "decoder"):
        encoder_params = count_parameters(getattr(module, "encoder", None))
        ode_params = count_parameters(getattr(module, "ode", None))
        memory_params = count_parameters(getattr(module, "memory", None))
        decoder_params = count_parameters(getattr(module, "decoder", None)) + count_parameters(getattr(module, "correction", None))
        obs_params = count_parameters(getattr(module, "obs_head", None))
        total_params = count_parameters(module)
        method_params = total_params - encoder_params
        rebel_cfg = _cfg_get(_cfg_get(cfg, "model", cfg), "rebel", {})
        backbone_cfg = _cfg_get(rebel_cfg, "backbone", {})
        base_dim = _cfg_get(backbone_cfg, "base_dim", None)
        belief_dim = _cfg_get(rebel_cfg, "belief_dim", None)
        return {
            "parameters_total": total_params,
            "parameters_backbone": encoder_params,
            "parameters_method": method_params,
            "parameters_m_total": total_params / 1.0e6,
            "parameters_m_backbone": encoder_params / 1.0e6,
            "parameters_m_method": method_params / 1.0e6,
            "parameters_m_encoder": encoder_params / 1.0e6,
            "parameters_m_rebel": method_params / 1.0e6,
            "parameters_m_memory": memory_params / 1.0e6,
            "parameters_m_ode": ode_params / 1.0e6,
            "parameters_m_decoder": decoder_params / 1.0e6,
            "parameters_m_faf": 0.0,
            "parameters_m_faf_selector": 0.0,
            "parameters_m_faf_affine_mixture": 0.0,
            "parameters_m_faf_temporal_update": 0.0,
            "parameters_m_faf_fusion": 0.0,
            "parameters_m_obs_head": obs_params / 1.0e6,
            "backbone_name": str(_cfg_get(backbone_cfg, "name", "official")),
            "unext_base_dim": int(base_dim) if base_dim is not None else None,
            "unext_channels": [int(base_dim), int(base_dim) * 2, int(base_dim) * 4] if base_dim is not None else [],
            "value_dim": int(belief_dim) if belief_dim is not None else None,
            "num_anchors": None,
            "query_dim": None,
            "code_dim": None,
            "hidden_dim": int(_cfg_get(_cfg_get(rebel_cfg, "ode", {}), "hidden_dim", 0) or 0),
        }
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
    backbone_name = getattr(module, "backbone_name", None)
    value_dim = getattr(module, "value_dim", None)
    num_anchors = getattr(module, "num_anchors", None)
    query_dim = getattr(module, "query_dim", None)
    code_dim = getattr(module, "code_dim", None)
    hidden_dim = getattr(module, "hidden_dim", None)

    model_cfg = _cfg_get(cfg, "model", cfg)
    for section_name in ("unext_faf", "unext_gar", "cardia", "unext_dynakey", "functional_anchor", "anchor_ode", "delay_ode"):
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
        "backbone_name": str(backbone_name) if backbone_name is not None else "",
        "unext_base_dim": int(base_dim) if base_dim is not None else None,
        "unext_channels": channels,
        "value_dim": int(value_dim) if value_dim is not None else None,
        "num_anchors": int(num_anchors) if num_anchors is not None else None,
        "query_dim": int(query_dim) if query_dim is not None else None,
        "code_dim": int(code_dim) if code_dim is not None else None,
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
    }
