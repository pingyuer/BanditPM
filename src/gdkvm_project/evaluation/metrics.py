from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from dpfr.grid import flow_smoothness, out_of_bound_ratio
from gdkvm_project.utils import Registry


METRIC_COLLECTOR_REGISTRY = Registry("metric_collector")


def align_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align B,T,C,H,W logits to the target mask spatial size for fair metrics."""
    if logits.dim() != 5:
        raise ValueError(f"Expected logits with shape B,T,C,H,W, got {tuple(logits.shape)}")
    if target.dim() not in {4, 5}:
        raise ValueError(f"Expected target with shape B,T,H,W or B,T,1,H,W, got {tuple(target.shape)}")
    target_hw = target.shape[-2:]
    if logits.shape[-2:] == target_hw:
        return logits
    batch, time, channels = logits.shape[:3]
    aligned = F.interpolate(
        logits.flatten(0, 1).float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return aligned.view(batch, time, channels, *target_hw)


def binary_dice_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = pred.bool()
    target = target.bool()
    inter = float((pred & target).sum().item())
    pred_sum = float(pred.sum().item())
    target_sum = float(target.sum().item())
    union = float((pred | target).sum().item())
    dice = (2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    iou = (inter + 1.0) / (union + 1.0)
    return dice, iou


def _foreground_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor) -> float | None:
    if not torch.is_tensor(logits):
        return None
    pred = torch.softmax(logits.float(), dim=2)[:, :, 1] >= 0.5
    target = target.bool()
    values = []
    for bi in range(pred.shape[0]):
        for ti in torch.nonzero(frame_mask[bi], as_tuple=False).flatten().tolist():
            p = pred[bi, ti]
            y = target[bi, ti]
            inter = float((p & y).sum().item())
            denom = float(p.sum().item() + y.sum().item())
            values.append((2.0 * inter + 1.0) / (denom + 1.0))
    return float(sum(values) / len(values)) if values else None


def _scalar(value) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


@METRIC_COLLECTOR_REGISTRY.register("dpfr")
@METRIC_COLLECTOR_REGISTRY.register("dual_prompt_flow_refinement")
def collect_dpfr_diagnostics(batch: dict, output: dict, supervised_indices: torch.Tensor | None = None) -> dict[str, float]:
    """Collect DPFR anchor/prompt/flow diagnostics from model outputs."""
    if supervised_indices is None:
        supervised_indices = batch.get("supervised_indices")
    if not torch.is_tensor(supervised_indices):
        return {}
    target = batch.get("cls_gt")
    if not torch.is_tensor(target):
        return {}
    if target.dim() == 5:
        target = target.squeeze(2)

    metrics: dict[str, float] = {}
    for key, output_key in (
        ("dpfr/final_dice", "final_logits"),
        ("dpfr/anchor_dice", "anchor_logits"),
        ("dpfr/prompt_dice", "prompt_logits"),
        ("dpfr/flow_dice", "flow_logits"),
    ):
        value = _foreground_dice_from_logits(output.get(output_key), target, supervised_indices)
        if value is not None:
            metrics[key] = value
    if "dpfr/final_dice" in metrics and "dpfr/anchor_dice" in metrics:
        metrics["dpfr/final_minus_anchor_dice"] = metrics["dpfr/final_dice"] - metrics["dpfr/anchor_dice"]
    if "dpfr/prompt_dice" in metrics and "dpfr/anchor_dice" in metrics:
        metrics["dpfr/prompt_minus_anchor_dice"] = metrics["dpfr/prompt_dice"] - metrics["dpfr/anchor_dice"]
    if "dpfr/flow_dice" in metrics and "dpfr/prompt_dice" in metrics:
        metrics["dpfr/flow_minus_prompt_dice"] = metrics["dpfr/flow_dice"] - metrics["dpfr/prompt_dice"]

    flow = output.get("flow_grid")
    if torch.is_tensor(flow):
        flat_flow = flow.flatten(0, 1).detach().float()
        metrics["dpfr/flow_abs_mean"] = float(flat_flow.abs().mean().item())
        metrics["dpfr/flow_abs_max"] = float(flat_flow.abs().amax().item())
        metrics["dpfr/flow_smoothness"] = float(flow_smoothness(flat_flow).detach().item())
        metrics["dpfr/flow_out_of_bound_ratio"] = float(out_of_bound_ratio(flat_flow).detach().item())

    aux = output.get("aux", {})
    if isinstance(aux, dict):
        aux_map = {
            "dpfr/prompt/modulation_abs_mean": "dpfr/prompt_modulation_abs_mean",
            "dpfr/mask_prompt/gt_ratio": "dpfr/mask_prompt_gt_ratio",
            "dpfr/mask_prompt/mask_ratio": "dpfr/mask_prompt_mask_ratio",
            "dpfr/mask_prompt/anchor_ratio": "dpfr/mask_prompt_anchor_ratio",
            "dpfr/fusion/prompt_gate_mean": "dpfr/fusion_prompt_gate",
            "dpfr/fusion/flow_gate_mean": "dpfr/fusion_flow_gate",
            "dpfr/fusion/final_anchor_delta_abs_mean": "dpfr/final_anchor_delta_abs_mean",
            "dpfr/fusion/prompt_anchor_delta_abs_mean": "dpfr/prompt_anchor_delta_abs_mean",
            "dpfr/fusion/flow_prompt_delta_abs_mean": "dpfr/flow_prompt_delta_abs_mean",
        }
        for raw_key, metric_key in aux_map.items():
            value = _scalar(aux.get(raw_key))
            if value is not None:
                metrics[metric_key] = value
    return metrics
