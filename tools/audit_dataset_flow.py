#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.utils.data as data
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.registry import resolve_dataset_class_from_cfg


def _load_config(config_name: str):
    config_dir = REPO_ROOT / "config"
    name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir)):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    return cfg


def _stage_cfg_for_split(cfg, split: str):
    return cfg.main_training if split == "train" else cfg.eval_stage


def _build_dataset(cfg, split: str):
    _, dataset_cls = resolve_dataset_class_from_cfg(cfg)
    stage_cfg = _stage_cfg_for_split(cfg, split)
    data_cfg = cfg.get("data", {})
    return dataset_cls(
        filepath=os.path.expanduser(str(cfg.data_path)),
        mode=split,
        seq_length=stage_cfg.seq_length if "seq_length" in stage_cfg else cfg.main_training.seq_length,
        max_num_obj=stage_cfg.num_objects if "num_objects" in stage_cfg else cfg.main_training.num_objects,
        size=int(stage_cfg.crop_size[0]),
        augmentation=cfg.get("augmentation", {}) if split == "train" else {},
        lv_class_id=data_cfg.get("lv_class_id", None) if hasattr(data_cfg, "get") else None,
    )


def _eval_indices(batch: dict, cfg) -> torch.Tensor:
    rgb = batch["rgb"]
    total_frames = rgb.shape[1]
    eval_cfg = cfg.get("evaluation", {})
    frame_scope = str(eval_cfg.get("frame_scope", "supervised_only"))
    if frame_scope == "all_available":
        source = batch.get("eval_valid", batch.get("label_valid"))
    else:
        source = batch.get("label_valid")
    if source is None:
        mask = torch.ones((rgb.shape[0], total_frames), dtype=torch.bool)
    else:
        mask = source.bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(rgb.shape[0], -1)
    if bool(eval_cfg.get("exclude_init_frame", False)):
        init_idx = int(eval_cfg.get("init_frame_index", 0))
        if 0 <= init_idx < total_frames:
            mask = mask.clone()
            mask[:, init_idx] = False
    return mask


def _tensor_unique(values: torch.Tensor) -> list[int | float]:
    out = []
    for value in torch.unique(values.detach().cpu()).tolist():
        as_float = float(value)
        out.append(int(as_float) if as_float.is_integer() else as_float)
    return out


def _summarize_split(cfg, split: str, *, num_samples: int, show_batch: bool) -> tuple[dict, list[str]]:
    dataset = _build_dataset(cfg, split)
    stage_cfg = _stage_cfg_for_split(cfg, split)
    expected_hw = tuple(int(v) for v in stage_cfg.crop_size)
    sample_count = len(dataset)
    issues: list[str] = []
    image_shapes = set()
    mask_shapes = set()
    mask_values = set()
    fg_ratios = []
    supervised_empty = 0
    supervised_total = 0
    label_hist = Counter()
    eval_hist = Counter()
    first_frame_has_gt = 0
    image_min = None
    image_max = None
    image_dtype = None
    first_eval_indices = []

    if sample_count == 0:
        issues.append("sample_count_is_zero")
        return {
            "split": split,
            "sample_count": sample_count,
            "expected_crop_size": list(expected_hw),
        }, issues

    n = min(int(num_samples), sample_count)
    for idx in range(n):
        sample = dataset[idx]
        rgb = sample["rgb"]
        mask = sample["cls_gt"]
        label_valid = sample.get("label_valid", torch.ones(rgb.shape[0], dtype=torch.bool)).bool()
        eval_mask = _eval_indices(
            {
                "rgb": rgb.unsqueeze(0),
                "label_valid": label_valid.unsqueeze(0),
                "eval_valid": sample.get("eval_valid", label_valid).bool().unsqueeze(0),
            },
            cfg,
        )[0]
        image_shapes.add(tuple(int(v) for v in rgb.shape))
        mask_shapes.add(tuple(int(v) for v in mask.shape))
        mask_values.update(int(v) for v in torch.unique(mask).detach().cpu().tolist())
        label_hist[int(label_valid.sum().item())] += 1
        eval_hist[int(eval_mask.sum().item())] += 1
        if bool(mask[0].max().item() > 0):
            first_frame_has_gt += 1
        image_dtype = str(rgb.dtype)
        cur_min = float(rgb.min().item())
        cur_max = float(rgb.max().item())
        image_min = cur_min if image_min is None else min(image_min, cur_min)
        image_max = cur_max if image_max is None else max(image_max, cur_max)
        for frame_idx in torch.nonzero(label_valid, as_tuple=False).flatten().tolist():
            frame_mask = mask[int(frame_idx)]
            fg = float((frame_mask > 0).float().mean().item())
            fg_ratios.append(fg)
            supervised_total += 1
            if fg == 0.0:
                supervised_empty += 1
        if idx == 0:
            first_eval_indices = torch.nonzero(eval_mask, as_tuple=False).flatten().tolist()

    batch_summary = {}
    if show_batch:
        loader = data.DataLoader(dataset, batch_size=min(4, sample_count), shuffle=False, num_workers=0)
        batch = next(iter(loader))
        batch_summary = {
            "rgb_shape": list(batch["rgb"].shape),
            "cls_gt_shape": list(batch["cls_gt"].shape),
            "rgb_min": float(batch["rgb"].min().item()),
            "rgb_max": float(batch["rgb"].max().item()),
            "cls_gt_unique": _tensor_unique(batch["cls_gt"]),
            "label_valid_hist": dict(sorted(Counter(int(v) for v in batch["label_valid"].bool().sum(dim=1).tolist()).items())),
        }

    empty_ratio = supervised_empty / max(supervised_total, 1)
    fg_tensor = torch.tensor(fg_ratios, dtype=torch.float32) if fg_ratios else torch.tensor([])
    summary = {
        "split": split,
        "sample_count": sample_count,
        "expected_crop_size": list(expected_hw),
        "image_shape_unique": sorted([list(v) for v in image_shapes]),
        "mask_shape_unique": sorted([list(v) for v in mask_shapes]),
        "image_dtype": image_dtype,
        "image_range": [image_min, image_max],
        "mask_unique_values": sorted(mask_values),
        "foreground_ratio": {
            "mean": float(fg_tensor.mean().item()) if fg_ratios else 0.0,
            "std": float(fg_tensor.std(unbiased=False).item()) if fg_ratios else 0.0,
            "min": float(fg_tensor.min().item()) if fg_ratios else 0.0,
            "max": float(fg_tensor.max().item()) if fg_ratios else 0.0,
        },
        "empty_mask_ratio": empty_ratio,
        "label_valid_count_histogram": dict(sorted(label_hist.items())),
        "first_frame_has_gt_ratio": first_frame_has_gt / max(n, 1),
        "eval_protocol": {
            "frame_scope": str(cfg.get("evaluation", {}).get("frame_scope", "supervised_only")),
            "exclude_init_frame": bool(cfg.get("evaluation", {}).get("exclude_init_frame", False)),
            "eval_protocol": str(cfg.get("evaluation", {}).get("eval_protocol", "")),
            "eval_indices_first_sample": first_eval_indices,
            "eval_valid_count_histogram": dict(sorted(eval_hist.items())),
        },
        "first_batch": batch_summary,
    }

    if not mask_values or max(mask_values) <= 0:
        issues.append("mask_unique_has_no_foreground")
    if empty_ratio > 0.5:
        issues.append(f"empty_mask_ratio_high:{empty_ratio:.3f}")
    for shape in image_shapes:
        if shape[-2:] != expected_hw:
            issues.append(f"rgb_spatial_size_mismatch:{shape[-2:]}!={expected_hw}")
    for shape in mask_shapes:
        if shape[-2:] != expected_hw:
            issues.append(f"mask_spatial_size_mismatch:{shape[-2:]}!={expected_hw}")
    if str(cfg.get("dataset_name", "")).lower() in {"echonet", "echo"}:
        if str(cfg.get("evaluation", {}).get("frame_scope", "supervised_only")) == "supervised_only":
            if not any(count > 0 for count in eval_hist):
                issues.append("echonet_supervised_only_has_no_eval_frame")
    return summary, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resolved dataset flow for CAMUS/EchoNet.")
    parser.add_argument("--config", required=True, help="Hydra config name, e.g. faf_camus.yaml")
    parser.add_argument("--dataset", default=None, help="Optional expected dataset name.")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--show-batch", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_name, _ = resolve_dataset_class_from_cfg(cfg)
    if args.dataset and args.dataset.lower() not in {dataset_name.lower(), str(cfg.get("dataset_name", "")).lower()}:
        print(f"expected dataset={args.dataset}, resolved dataset={dataset_name}", file=sys.stderr)
        return 2

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    output = {
        "resolved_config": {
            "dataset": dataset_name,
            "data_path": os.path.expanduser(str(cfg.data_path)),
            "processed_root": str(cfg.get("processed_root", "")),
            "seq_length": int(cfg.main_training.seq_length),
            "main_training_crop_size": list(cfg.main_training.crop_size),
            "eval_stage_crop_size": list(cfg.eval_stage.crop_size),
            "evaluation": {
                "frame_scope": str(cfg.get("evaluation", {}).get("frame_scope", "supervised_only")),
                "exclude_init_frame": bool(cfg.get("evaluation", {}).get("exclude_init_frame", False)),
                "eval_protocol": str(cfg.get("evaluation", {}).get("eval_protocol", "")),
            },
        },
        "splits": {},
    }
    all_issues: list[str] = []
    for split in splits:
        summary, issues = _summarize_split(cfg, split, num_samples=args.num_samples, show_batch=args.show_batch)
        output["splits"][split] = summary
        all_issues.extend([f"{split}:{issue}" for issue in issues])

    print(json.dumps(output, indent=2, sort_keys=True))
    if all_issues:
        print("AUDIT_FAILED " + json.dumps(all_issues, ensure_ascii=False), file=sys.stderr)
        return 1
    print("AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
