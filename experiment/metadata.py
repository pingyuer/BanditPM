from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from omegaconf import DictConfig


def resolve_model_name(cfg: DictConfig) -> str:
    return str(cfg.get("model_name", cfg.model.get("name", "BanditPM")))


def resolve_git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        return "nogit"


def resolve_git_metadata() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        short = "nogit"
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip())
    except Exception:
        dirty = False
    return {"git_commit": commit, "git_short": short, "git_dirty": dirty}


def resolve_mlflow_experiment_name(cfg: DictConfig) -> str:
    mlflow_cfg = cfg.get("mlflow", {})
    configured = mlflow_cfg.get("experiment_name", None) if hasattr(mlflow_cfg, "get") else None
    if configured:
        return str(configured)

    model_name = resolve_model_name(cfg).lower()
    exp_id = str(cfg.get("exp_id", "")).lower()
    memory_cfg = cfg.get("model", {}).get("memory_core", {}) if hasattr(cfg.get("model", {}), "get") else {}
    memory_type = str(memory_cfg.get("type", "")).lower() if hasattr(memory_cfg, "get") else ""
    unext_cfg = cfg.get("model", {}).get("unext_dynakey", {}) if hasattr(cfg.get("model", {}), "get") else {}
    unext_uses_dynakey = bool(unext_cfg.get("use_dynakey", False)) if hasattr(unext_cfg, "get") else False
    uses_dynakey = memory_type == "dynakey" or unext_uses_dynakey

    if model_name == "functional_anchor" or "functional_anchor" in exp_id:
        return "functional_anchor"
    if model_name.startswith("anchor_ode") or "anchor_ode" in exp_id:
        return "anchor_ode"
    if uses_dynakey or "dynakey" in exp_id:
        return "dynakey"
    if model_name in {"gdkvm", "banditpm"}:
        return "gdkvm"
    if model_name == "kpff":
        return "kpff"
    if model_name == "unext_fusion":
        return "dynakey" if uses_dynakey else "unext_fusion"
    if model_name == "delay_ode":
        return "delay_ode"
    if "ablation" in exp_id and "anchor_ode" in exp_id:
        return "ablation_anchor_ode"
    if model_name in {"unext", "unext_only", "baseline_unext"} or "unext_only" in exp_id:
        return "unext_baseline"
    return model_name or "experiment"


def _cfg_get_nested(cfg, path: str, default=None):
    value = cfg
    for part in path.split("."):
        if not hasattr(value, "get"):
            return default
        value = value.get(part, default)
        if value is default:
            return default
    return value


def build_mlflow_metadata(cfg: DictConfig, *, world_size: int) -> tuple[dict, dict]:
    git_info = resolve_git_metadata()
    method_family = resolve_mlflow_experiment_name(cfg)
    model_name = resolve_model_name(cfg)
    dataset_name = str(cfg.get("dataset_name", "dataset"))
    protocol_name = str(cfg.get("data", {}).get("protocol_name", "unknown"))
    mlflow_cfg = cfg.get("mlflow", {})
    stage = str(mlflow_cfg.get("stage", "full")) if hasattr(mlflow_cfg, "get") else "full"
    run_type = str(mlflow_cfg.get("run_type", "train")) if hasattr(mlflow_cfg, "get") else "train"
    eval_cfg = cfg.get("evaluation", {})
    post_cfg = eval_cfg.get("postprocess", {}) if hasattr(eval_cfg, "get") else {}
    tags = {
        "project": "tahara-3d",
        "method": method_family,
        "model": model_name,
        "dataset": dataset_name,
        "protocol": protocol_name,
        "run_type": run_type,
        "stage": stage,
        "exp_id": str(cfg.get("exp_id", "experiment")),
        "seed": int(cfg.get("seed", 42)),
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "ddp_world_size": int(world_size),
        "has_ema": bool(_cfg_get_nested(cfg, "ema.enabled", False) or _cfg_get_nested(cfg, "main_training.ema_enabled", False)),
        "has_tta": bool(eval_cfg.get("tta_enabled", False)) if hasattr(eval_cfg, "get") else False,
        "has_postprocess": bool(post_cfg.get("enabled", eval_cfg.get("postprocess_enabled", False))) if hasattr(post_cfg, "get") else False,
    }
    stage_cfg = cfg.get("main_training", {})
    loss_cfg = cfg.get("loss", cfg.get("losses", {}))
    model_cfg = cfg.get("model", {})
    anchor_cfg = model_cfg.get("anchor_ode", model_cfg.get("memory_core", {})) if hasattr(model_cfg, "get") else {}
    functional_cfg = model_cfg.get("functional_anchor", {}) if hasattr(model_cfg, "get") else {}
    if method_family == "functional_anchor" and hasattr(functional_cfg, "get"):
        tags["prediction_mode"] = str(functional_cfg.get("prediction_mode", "base_primary"))
        tags["training_stage"] = str(functional_cfg.get("training_stage", stage_cfg.get("training_stage", "joint_residual") if hasattr(stage_cfg, "get") else "joint_residual"))
        if cfg.get("ablation_type", None):
            tags["ablation_type"] = str(cfg.get("ablation_type"))
    params = {
        "model.name": model_name,
        "dataset.name": dataset_name,
        "dataset.resolution": cfg.get("resolution", _cfg_get_nested(cfg, "data.resolution", "")),
        "dataset.sequence_length": stage_cfg.get("seq_length", _cfg_get_nested(cfg, "data.sequence_length", "")) if hasattr(stage_cfg, "get") else "",
        "train.lr": stage_cfg.get("learning_rate", None),
        "train.batch_size": stage_cfg.get("batch_size", None),
        "train.optimizer": stage_cfg.get("optimizer", cfg.get("optimizer", "")),
        "train.scheduler": stage_cfg.get("scheduler", cfg.get("scheduler", "")),
        "train.max_iter": stage_cfg.get("num_iterations", None),
        "loss.dice_weight": loss_cfg.get("dice_weight", loss_cfg.get("lambda_dice", None)) if hasattr(loss_cfg, "get") else None,
        "loss.bce_weight": loss_cfg.get("bce_weight", loss_cfg.get("lambda_bce", None)) if hasattr(loss_cfg, "get") else None,
        "loss.boundary_weight": loss_cfg.get("boundary_weight", loss_cfg.get("lambda_boundary", None)) if hasattr(loss_cfg, "get") else None,
        "anchor_ode.state_dim": anchor_cfg.get("state_dim", None) if hasattr(anchor_cfg, "get") else None,
        "anchor_ode.num_slots": anchor_cfg.get("num_slots", None) if hasattr(anchor_cfg, "get") else None,
        "anchor_ode.gate_init_bias": anchor_cfg.get("gate_init_bias", None) if hasattr(anchor_cfg, "get") else None,
        "anchor_ode.prior_residual_clip": anchor_cfg.get("prior_residual_clip", None) if hasattr(anchor_cfg, "get") else None,
        "anchor_ode.affine_max_translate": anchor_cfg.get("affine_max_translate", None) if hasattr(anchor_cfg, "get") else None,
        "anchor_ode.affine_max_scale": anchor_cfg.get("affine_max_scale", None) if hasattr(anchor_cfg, "get") else None,
        "functional_anchor.state_dim": functional_cfg.get("state_dim", None) if hasattr(functional_cfg, "get") else None,
        "functional_anchor.num_slots": functional_cfg.get("num_slots", None) if hasattr(functional_cfg, "get") else None,
        "functional_anchor.phase_dim": functional_cfg.get("phase_dim", None) if hasattr(functional_cfg, "get") else None,
        "functional_anchor.prediction_mode": functional_cfg.get("prediction_mode", None) if hasattr(functional_cfg, "get") else None,
        "functional_anchor.residual_clip": functional_cfg.get("residual_clip", None) if hasattr(functional_cfg, "get") else None,
        "postprocess.enabled": tags["has_postprocess"],
        "postprocess.min_area": post_cfg.get("min_area", None) if hasattr(post_cfg, "get") else None,
        "eval.threshold": eval_cfg.get("threshold", eval_cfg.get("default_threshold", 0.5)) if hasattr(eval_cfg, "get") else 0.5,
        "seed": int(cfg.get("seed", 42)),
    }
    return tags, params


def resolve_mlflow_run_name(
    cfg: DictConfig,
    *,
    timestamp: str | None = None,
    git_hash: str | None = None,
) -> str:
    mlflow_cfg = cfg.get("mlflow", {})
    configured = mlflow_cfg.get("run_name", None) if hasattr(mlflow_cfg, "get") else None
    if configured:
        return str(configured)
    model_name = resolve_model_name(cfg)
    dataset_name = str(cfg.get("dataset_name", "dataset"))
    protocol = str(cfg.get("data", {}).get("protocol_name", "protocol"))
    mlflow_cfg = cfg.get("mlflow", {})
    run_type = str(mlflow_cfg.get("run_type", "train")) if hasattr(mlflow_cfg, "get") else "train"
    seed = int(cfg.get("seed", 42))
    timestamp = timestamp or datetime.datetime.now().strftime("%m%d-%H%M")
    git_hash = git_hash or resolve_git_short_hash()
    return f"{model_name}_{dataset_name}_{protocol}_{run_type}_s{seed}_{timestamp}_{git_hash}"
