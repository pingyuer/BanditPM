from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _binary_dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def _binary_dice_score(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - _binary_dice_loss(prob, target)


def _boundary_band(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_geometry_loss(logits: torch.Tensor, target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pred_boundary = _boundary_band(torch.sigmoid(logits), kernel_size=kernel_size)
    target_boundary = _boundary_band(target.float(), kernel_size=kernel_size)
    return _binary_dice_loss(pred_boundary, target_boundary)


def _centroid(prob: torch.Tensor) -> torch.Tensor:
    b, _, h, w = prob.shape
    ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype).reshape(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype).reshape(1, 1, 1, w)
    mass = prob.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
    cx = (prob * xs).sum(dim=(2, 3), keepdim=True) / mass
    cy = (prob * ys).sum(dim=(2, 3), keepdim=True) / mass
    return torch.cat([cx.reshape(b, 1), cy.reshape(b, 1)], dim=1)


def _second_order_smooth(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] < 3:
        return values.new_zeros(())
    accel = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
    return accel.abs().mean()


def compute_geomaskformer_losses(lc, data: Dict[str, torch.Tensor], supervised_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    if not getattr(lc, "is_geomaskformer", False):
        return {}
    proposal_logits = data.get("proposal_logits")
    quality_scores = data.get("quality_scores")
    if not torch.is_tensor(proposal_logits) or not torch.is_tensor(quality_scores):
        return {}

    batch_size, num_frames, num_queries = proposal_logits.shape[:3]
    gt = (data["cls_gt"].float() > 0).float()
    if gt.shape[-2:] != proposal_logits.shape[-2:]:
        gt_for_loss = F.interpolate(
            gt.reshape(batch_size * num_frames, 1, gt.shape[-2], gt.shape[-1]),
            size=proposal_logits.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, num_frames, 1, proposal_logits.shape[-2], proposal_logits.shape[-1])
    else:
        gt_for_loss = gt

    topk = min(int(lc.geomaskformer_topk_loss), int(num_queries))
    mask_terms = []
    boundary_terms = []
    score_terms = []
    top1_scores = []
    oracle_scores = []
    best_ids = []

    for bi in range(batch_size):
        frame_ids = lc._frame_ids_for_sample(supervised_mask, bi)
        for ti in frame_ids:
            target = gt_for_loss[bi, ti : ti + 1]
            logits_k = proposal_logits[bi, ti]
            prob_k = torch.sigmoid(logits_k).unsqueeze(1)
            target_k = target.expand(num_queries, -1, -1, -1)
            with torch.no_grad():
                dice_k = _binary_dice_score(prob_k, target_k)
                best = torch.topk(dice_k, k=topk, dim=0).indices
                top1_idx = torch.sigmoid(quality_scores[bi, ti]).argmax()
                top1_scores.append(dice_k[top1_idx].detach())
                oracle_scores.append(dice_k[best[0]].detach())
                best_ids.append(best[0].detach())
            selected_logits = logits_k[best].unsqueeze(1)
            selected_target = target.expand(topk, -1, -1, -1)
            bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="none").mean(dim=(1, 2, 3))
            dice = _binary_dice_loss(torch.sigmoid(selected_logits), selected_target)
            mask_terms.append((bce + dice).mean())
            boundary_terms.append(boundary_geometry_loss(selected_logits, selected_target, lc.geomaskformer_boundary_kernel).mean())
            quality_target = dice_k.detach().clamp(0.0, 1.0)
            quality_prob = torch.sigmoid(quality_scores[bi, ti])
            score_terms.append(F.smooth_l1_loss(quality_prob, quality_target))

    out: Dict[str, torch.Tensor] = {}
    device = data["rgb"].device
    if mask_terms:
        raw_mask = torch.stack(mask_terms).mean()
        raw_boundary = torch.stack(boundary_terms).mean()
        raw_score = torch.stack(score_terms).mean()
        out["raw_geomaskformer_bestofk_mask"] = raw_mask.detach()
        out["raw_geomaskformer_boundary"] = raw_boundary.detach()
        out["raw_geomaskformer_score"] = raw_score.detach()
        out["aux_geomaskformer_bestofk_mask"] = raw_mask * lc.lambda_geomaskformer_mask
        out["aux_geomaskformer_boundary"] = raw_boundary * lc.lambda_geomaskformer_boundary
        out["aux_geomaskformer_score"] = raw_score * lc.lambda_geomaskformer_score
        out["geomaskformer/proposal_top1_dice"] = torch.stack(top1_scores).mean()
        out["geomaskformer/proposal_oracle_topk_dice"] = torch.stack(oracle_scores).mean()
        out["geomaskformer/best_query_mean"] = torch.stack(best_ids).float().mean()
    if lc.lambda_geomaskformer_temporal > 0 and torch.is_tensor(data.get("logits")):
        fg = torch.softmax(data["logits"], dim=2)[:, :, 1:2]
        area = fg.mean(dim=(2, 3, 4))
        cent = torch.stack([_centroid(fg[:, ti]) for ti in range(fg.shape[1])], dim=1)
        temporal = _second_order_smooth(area) + _second_order_smooth(cent[..., 0]) + _second_order_smooth(cent[..., 1])
        out["raw_geomaskformer_temporal"] = temporal.detach()
        out["aux_geomaskformer_temporal"] = temporal * lc.lambda_geomaskformer_temporal
    if not out:
        out["aux_geomaskformer_bestofk_mask"] = torch.zeros((), device=device)
    return out
