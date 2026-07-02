from __future__ import annotations

import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from monai.metrics import (
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)


def binary_overlap_metrics(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.float()
    gt = gt.float()
    inter = float((pred * gt).sum().item())
    pred_sum = float(pred.sum().item())
    gt_sum = float(gt.sum().item())
    union = pred_sum + gt_sum - inter
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0, 1.0
    dice = (2.0 * inter) / max(pred_sum + gt_sum, 1e-6)
    iou = inter / max(union, 1e-6)
    return dice, iou


def binary_boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask.float() > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return binary_overlap_metrics(binary_boundary_mask(pred), binary_boundary_mask(gt))[0]


def postprocess_binary_mask(mask: torch.Tensor, cfg) -> torch.Tensor:
    eval_cfg = cfg.get("evaluation", {})
    post_cfg = eval_cfg.get("postprocess", {})
    if isinstance(post_cfg, dict) or hasattr(post_cfg, "get"):
        default = str(cfg.get("model", {}).get("name", "")).lower() in {"anchor_ode_v2", "unext_anchor_ode_affine"}
        enabled = bool(post_cfg.get("enabled", default))
    else:
        enabled = bool(post_cfg)
    if not enabled:
        return mask
    try:
        from scipy import ndimage
    except Exception:
        return mask

    min_size = int(post_cfg.get("min_size", 16)) if hasattr(post_cfg, "get") else 16
    keep_largest = bool(post_cfg.get("largest_component", True)) if hasattr(post_cfg, "get") else True
    fill_holes = bool(post_cfg.get("fill_holes", True)) if hasattr(post_cfg, "get") else True
    remove_small = bool(post_cfg.get("remove_small_objects", True)) if hasattr(post_cfg, "get") else True
    binary_closing = bool(post_cfg.get("binary_closing", True)) if hasattr(post_cfg, "get") else True
    structure = np.ones((3, 3), dtype=bool)
    device = mask.device
    dtype = mask.dtype
    arr = mask.detach().cpu().numpy().astype(bool)
    out = np.zeros_like(arr, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    flat_out = out.reshape(-1, out.shape[-2], out.shape[-1])
    for idx, item in enumerate(flat):
        if keep_largest:
            labels, num = ndimage.label(item, structure=structure)
            if num > 0:
                counts = np.bincount(labels.ravel())
                counts[0] = 0
                largest = int(counts.argmax())
                item = labels == largest
        if fill_holes:
            item = ndimage.binary_fill_holes(item)
        if remove_small and min_size > 1:
            labels, num = ndimage.label(item, structure=structure)
            if num > 0:
                counts = np.bincount(labels.ravel())
                keep = counts >= min_size
                keep[0] = False
                item = keep[labels]
        if binary_closing:
            item = ndimage.binary_closing(item, structure=structure)
        flat_out[idx] = item.astype(np.float32)
    return torch.as_tensor(out, device=device, dtype=dtype)


def surface_metrics_single(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.float()
    gt = gt.float()
    pred_sum = float(pred.sum().item())
    gt_sum = float(gt.sum().item())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 0.0, 0.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        max_dim = float(max(pred.shape[-2], pred.shape[-1], gt.shape[-2], gt.shape[-1]))
        return max_dim, max_dim

    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
    hd_metric(y_pred=pred, y=gt)
    assd_metric(y_pred=pred, y=gt)
    hd95 = hd_metric.aggregate()
    assd = assd_metric.aggregate()
    hd95 = hd95.item() if isinstance(hd95, torch.Tensor) else float(hd95)
    assd = assd.item() if isinstance(assd, torch.Tensor) else float(assd)
    if not np.isfinite(hd95):
        hd95 = float(max(pred.shape[-2], pred.shape[-1]))
    if not np.isfinite(assd):
        assd = float(max(pred.shape[-2], pred.shape[-1]))
    return hd95, assd


def grad_norm_for_prefixes(trainer, prefixes: tuple[str, ...]) -> float | None:
    total = 0.0
    found = False
    for name, param in trainer.model_without_ddp.named_parameters():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if param.grad is None:
            continue
        found = True
        total += float(param.grad.detach().float().pow(2).sum().item())
    if not found:
        return None
    return total ** 0.5


def resize_to_original(pred: torch.Tensor, gt: torch.Tensor, original_hw):
    target_h = int(original_hw[0])
    target_w = int(original_hw[1])
    if pred.shape[-2:] == (target_h, target_w) and gt.shape[-2:] == (target_h, target_w):
        return pred, gt
    pred_up = F.interpolate(pred.float(), size=(target_h, target_w), mode="nearest")
    gt_up = F.interpolate(gt.float(), size=(target_h, target_w), mode="nearest")
    return pred_up, gt_up
