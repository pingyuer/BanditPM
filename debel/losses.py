from __future__ import annotations

from collections import defaultdict

import torch

from losses.base import dice_loss
from rebel.losses import dice_ce
from debel.grid import flow_smoothness
from utils.tensor_utils import cls_to_one_hot


def _fg_area(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1].sum(dim=(-2, -1))


def foreground_dice(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1:].flatten(start_dim=2)
    gt = soft_gt[:, 1:].float().flatten(start_dim=2)
    return ((2.0 * (prob * gt).sum(-1) + 1.0) / (prob.sum(-1) + gt.sum(-1) + 1.0)).mean().detach()


def compute_debel_losses(loss_computer, data, supervised_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = loss_computer.debel_loss_cfg
    weights = {
        "final": float(cfg.get("lambda_final", cfg.get("final", 1.0))),
        "anchor": float(cfg.get("lambda_anchor", cfg.get("anchor", 0.5))),
        "grid": float(cfg.get("lambda_grid", cfg.get("grid", 0.01))),
        "smooth": float(cfg.get("lambda_smooth", cfg.get("smooth", 0.02))),
        "temp": float(cfg.get("lambda_temp", cfg.get("temp", 0.01))),
        "area": float(cfg.get("lambda_area", cfg.get("area", 0.001))),
        "residual": float(cfg.get("lambda_residual", cfg.get("residual", 0.005))),
    }
    losses = defaultdict(lambda: torch.zeros((), device=data["rgb"].device))
    batch_size = data["rgb"].shape[0]
    anchor_dice = []
    warped_dice = []
    final_dice = []
    for bi in range(batch_size):
        t_range = loss_computer._frame_ids_for_sample(supervised_mask, bi)
        if not t_range:
            continue
        num_obj = int(data.get("num_objects", [1] * batch_size)[bi]) if not torch.is_tensor(data.get("num_objects")) else int(data["num_objects"][bi])
        valid_slice = slice(None, num_obj + 1)
        cls_gt = data["cls_gt"][bi, t_range]
        if cls_gt.dim() == 3:
            cls_gt = cls_gt.unsqueeze(1)
        soft_gt = cls_to_one_hot(cls_gt, num_obj)
        final = torch.stack([data[f"logits_{ti}"][bi, valid_slice] for ti in t_range], dim=0)
        anchor = torch.stack([data[f"aux_{ti}"]["anchor_logits"][bi, valid_slice] for ti in t_range], dim=0)
        warped = torch.stack([data[f"aux_{ti}"]["warped_logits"][bi, valid_slice] for ti in t_range], dim=0)
        losses["debel_final"] += dice_ce(final, soft_gt) / batch_size * weights["final"]
        losses["debel_anchor"] += dice_ce(anchor, soft_gt) / batch_size * weights["anchor"]
        anchor_dice.append(foreground_dice(anchor, soft_gt))
        warped_dice.append(foreground_dice(warped, soft_gt))
        final_dice.append(foreground_dice(final, soft_gt))
    delta = data.get("delta_grids")
    if torch.is_tensor(delta):
        total_delta = delta.sum(dim=2) if delta.dim() == 6 else delta
        losses["debel_grid"] += total_delta.abs().mean() * weights["grid"]
        losses["debel_smooth"] += flow_smoothness(total_delta.flatten(0, 1)) * weights["smooth"]
        if total_delta.shape[1] > 1:
            losses["debel_temp"] += (total_delta[:, 1:] - total_delta[:, :-1]).abs().mean() * weights["temp"]
        else:
            losses["debel_temp"] += total_delta.sum() * 0.0
    logits = data.get("logits")
    if torch.is_tensor(logits) and logits.shape[1] > 2:
        area = _fg_area(logits.flatten(0, 1)).view(logits.shape[0], logits.shape[1])
        accel = area[:, 2:] - 2.0 * area[:, 1:-1] + area[:, :-2]
        losses["debel_area"] += accel.abs().mean() * weights["area"]
    elif torch.is_tensor(logits):
        losses["debel_area"] += logits.sum() * 0.0
    residual = data.get("residual_logits")
    if torch.is_tensor(residual):
        losses["debel_residual"] += residual.abs().mean() * weights["residual"]
    if anchor_dice:
        losses["debel/anchor_dice"] = torch.stack(anchor_dice).mean()
        losses["debel/warped_dice"] = torch.stack(warped_dice).mean()
        losses["debel/final_dice"] = torch.stack(final_dice).mean()
        losses["debel/warped_minus_anchor_dice"] = losses["debel/warped_dice"] - losses["debel/anchor_dice"]
        losses["debel/final_minus_warped_dice"] = losses["debel/final_dice"] - losses["debel/warped_dice"]
    return dict(losses)
