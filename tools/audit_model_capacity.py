#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import build_model
from utils.model_capacity import infer_unext_capacity


def _load_config(config_name: str, overrides: list[str] | None = None):
    name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    with initialize_config_dir(version_base="1.3.2", config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name=name, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def _dummy_forward(model: torch.nn.Module, cfg, *, frames: int) -> dict:
    crop_size = cfg.main_training.get("crop_size", [128, 128])
    height, width = int(crop_size[0]), int(crop_size[1])
    batch = {
        "rgb": torch.rand(1, int(frames), 1, height, width),
        "cls_gt": torch.zeros(1, int(frames), 1, height, width, dtype=torch.long),
        "ff_gt": torch.zeros(1, 1, 1, height, width, dtype=torch.long),
        "selector": torch.ones(1, 1),
        "label_valid": torch.ones(1, int(frames), dtype=torch.bool),
        "eval_valid": torch.ones(1, int(frames), dtype=torch.bool),
        "info": {"num_objects": torch.ones(1, dtype=torch.long)},
        "current_iter": 0,
        "global_step": 0,
        "init_mode": "pred_or_zero",
    }
    with torch.no_grad():
        out = model(batch)
    logits_key = "logits_0"
    masks_key = "masks_0"
    return {
        "input_rgb_shape": list(batch["rgb"].shape),
        "target_mask_shape": list(batch["cls_gt"].shape),
        "logits_shape": list(out[logits_key].shape) if logits_key in out else None,
        "masks_shape": list(out[masks_key].shape) if masks_key in out else None,
        "warped_anchor_logits_shape": list(out.get("memory_aux_0", {}).get("faf_aux", {}).get("warped_anchor_logits", torch.empty(0)).shape),
        "mixture_logits_shape": list(out.get("memory_aux_0", {}).get("faf_aux", {}).get("mixture_logits", torch.empty(0)).shape),
        "output_keys_preview": sorted(list(out.keys()))[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit UNeXt/FAF model capacity from a Hydra config.")
    parser.add_argument("--config", required=True, help="Hydra config name, e.g. faf_camus.yaml")
    parser.add_argument("--dummy-forward", action="store_true", help="Run a small dummy forward and print shapes.")
    parser.add_argument(
        "--allow-missing-pretrained",
        action="store_true",
        help="Temporarily disable require_pretrained_unext for capacity/dummy audits.",
    )
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()

    overrides = ["model.unext_faf.require_pretrained_unext=false"] if args.allow_missing_pretrained else []
    cfg = _load_config(args.config, overrides=overrides)
    model = build_model(cfg, device="cpu")
    capacity = infer_unext_capacity(model, cfg)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    result = {
        "config": args.config,
        "model_name": str(cfg.get("model_name", "")),
        "resolved_model_config": model_cfg,
        "capacity": capacity,
    }
    if args.dummy_forward:
        result["dummy_forward"] = _dummy_forward(model, cfg, frames=args.frames)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
