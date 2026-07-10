from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _is_no_decay(name: str, ndim: int) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return ndim <= 1 or leaf == "bias" or "norm" in name.lower()


def get_parameter_groups(model, stage_cfg, print_log: bool = False):
    """Build optimizer groups for the public GDKVM/DPFR surface."""
    base_lr = float(stage_cfg.learning_rate)
    weight_decay = float(stage_cfg.weight_decay)
    embed_weight_decay = float(stage_cfg.get("embed_weight_decay", 0.0))
    backbone_lr_ratio = float(stage_cfg.get("backbone_lr_ratio", 1.0))
    unext_lr_ratio = float(stage_cfg.get("unext_lr_ratio", backbone_lr_ratio))
    dpfr_lr_ratio = float(stage_cfg.get("dpfr_lr_ratio", 1.0))

    groups = {
        "backbone": {"params": [], "lr": base_lr * unext_lr_ratio, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": base_lr * unext_lr_ratio, "weight_decay": 0.0},
        "method": {"params": [], "lr": base_lr * dpfr_lr_ratio, "weight_decay": weight_decay},
        "method_no_decay": {"params": [], "lr": base_lr * dpfr_lr_ratio, "weight_decay": 0.0},
        "embedding": {"params": [], "lr": base_lr, "weight_decay": embed_weight_decay},
        "other": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "other_no_decay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
    }

    backbone_prefixes = ("backbone.", "frame_net.", "encoder.backbone.")
    method_prefixes = (
        "prompt_encoder.",
        "prompt_heads.",
        "flow_head.",
        "final_fusion.",
        "memory_core.",
        "prototype_manager.",
        "key_proj.",
        "query_proj.",
        "value_proj.",
        "temporal_memory.",
    )
    embedding_suffixes = ("summary_pos.weight", "query_init.weight", "query_emb.weight", "obj_pe.weight")

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or param in seen:
            continue
        seen.add(param)
        clean_name = name[7:] if name.startswith("module.") else name
        no_decay = _is_no_decay(clean_name, param.ndim)
        if clean_name.startswith(backbone_prefixes):
            key = "backbone_no_decay" if no_decay else "backbone"
        elif clean_name.startswith(method_prefixes):
            key = "method_no_decay" if no_decay else "method"
        elif clean_name.endswith(embedding_suffixes):
            key = "embedding"
        else:
            key = "other_no_decay" if no_decay else "other"
        groups[key]["params"].append(param)
        if print_log:
            log.info("%s counted as %s parameter", clean_name, key)

    return [
        {"name": name, **group}
        for name, group in groups.items()
        if group["params"]
    ]
