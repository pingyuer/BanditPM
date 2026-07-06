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


def _linear_warmup_scale(data: Dict[str, torch.Tensor], warmup_iters: int) -> float:
    if warmup_iters <= 0:
        return 1.0
    step = data.get("global_step", data.get("current_iter", None))
    if torch.is_tensor(step):
        step = int(step.detach().flatten()[0].item())
    if step is None:
        return 1.0
    return max(0.0, min(1.0, float(step + 1) / float(warmup_iters)))


def _bestofk_mask_loss_for_logits(
    proposal_logits: torch.Tensor,
    gt_for_loss: torch.Tensor,
    supervised_mask: torch.Tensor,
    lc,
) -> torch.Tensor:
    batch_size, num_frames, num_queries = proposal_logits.shape[:3]
    if gt_for_loss.shape[-2:] != proposal_logits.shape[-2:]:
        gt_for_loss = F.interpolate(
            gt_for_loss.reshape(batch_size * num_frames, 1, gt_for_loss.shape[-2], gt_for_loss.shape[-1]),
            size=proposal_logits.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, num_frames, 1, proposal_logits.shape[-2], proposal_logits.shape[-1])
    topk = max(1, min(int(lc.geomaskformer_topk_loss), int(num_queries)))
    terms = []
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
            selected_logits = logits_k[best].unsqueeze(1)
            selected_target = target.expand(topk, -1, -1, -1)
            bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="none").mean(dim=(1, 2, 3))
            dice = _binary_dice_loss(torch.sigmoid(selected_logits), selected_target)
            terms.append((bce + dice).mean())
    if not terms:
        return proposal_logits.new_zeros(())
    return torch.stack(terms).mean()


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

    topk = max(1, min(int(lc.geomaskformer_topk_loss), int(num_queries)))
    mask_terms = []
    boundary_terms = []
    score_terms = []
    ranking_terms = []
    diversity_terms = []
    visible_reconstruction_terms = []
    top1_scores = []
    oracle_scores = []
    oracle_topk_mean_scores = []
    oracle_top4_scores = []
    oracle_top5_best_scores = []
    oracle_top5_mean_scores = []
    cover85 = []
    cover90 = []
    oracle_top10_scores = []
    best_ids = []
    all_best_ids = []
    all_score_quality = []
    all_oracle_quality = []

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
                oracle_topk_mean_scores.append(torch.topk(dice_k, k=topk, dim=0).values.mean().detach())
                oracle_top4_scores.append(torch.topk(dice_k, k=min(4, num_queries), dim=0).values.mean().detach())
                top5_values = torch.topk(dice_k, k=min(5, num_queries), dim=0).values
                oracle_top5_best_scores.append(top5_values.max().detach())
                oracle_top5_mean_scores.append(top5_values.mean().detach())
                cover85.append((top5_values.max() >= 0.85).float().detach())
                cover90.append((top5_values.max() >= 0.90).float().detach())
                oracle_top10_scores.append(torch.topk(dice_k, k=min(10, num_queries), dim=0).values.mean().detach())
                best_ids.append(best[0].detach())
                all_best_ids.append(best[0].detach())
                all_score_quality.append(torch.sigmoid(quality_scores[bi, ti]).detach())
                all_oracle_quality.append(dice_k.detach())
            selected_logits = logits_k[best].unsqueeze(1)
            selected_target = target.expand(topk, -1, -1, -1)
            bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="none").mean(dim=(1, 2, 3))
            dice = _binary_dice_loss(torch.sigmoid(selected_logits), selected_target)
            mask_terms.append((bce + dice).mean())
            boundary_terms.append(boundary_geometry_loss(selected_logits, selected_target, lc.geomaskformer_boundary_kernel).mean())
            if topk > 1:
                selected_prob = torch.sigmoid(selected_logits).flatten(1)
                selected_prob = F.normalize(selected_prob - selected_prob.mean(dim=1, keepdim=True), dim=1)
                sim = selected_prob @ selected_prob.t()
                off_diag = sim[~torch.eye(topk, dtype=torch.bool, device=sim.device)]
                diversity_terms.append(off_diag.clamp_min(0.0).mean())
            quality_target = dice_k.detach().clamp(0.0, 1.0)
            quality_prob = torch.sigmoid(quality_scores[bi, ti])
            score_terms.append(F.smooth_l1_loss(quality_prob, quality_target))
            dice_diff = quality_target[:, None] - quality_target[None, :]
            score_diff = quality_scores[bi, ti][:, None] - quality_scores[bi, ti][None, :]
            valid_pairs = dice_diff.abs() > 0.02
            if valid_pairs.any():
                ranking_terms.append(F.softplus(-score_diff[valid_pairs] * dice_diff[valid_pairs].sign()).mean())

    visible_reconstruction_weight = float(getattr(lc, "lambda_geomaskformer_visible_reconstruction", 0.0))
    mask_visibility = data.get("mask_visibility")
    label_valid = data.get("label_valid")
    if visible_reconstruction_weight > 0 and torch.is_tensor(mask_visibility) and torch.is_tensor(label_valid):
        visible_supervised = (mask_visibility.to(device=proposal_logits.device) > 0) & label_valid.to(device=proposal_logits.device).bool()
        for bi in range(batch_size):
            for ti in torch.nonzero(visible_supervised[bi], as_tuple=False).flatten().tolist():
                target = gt_for_loss[bi, ti : ti + 1]
                logits_k = proposal_logits[bi, ti]
                prob_k = torch.sigmoid(logits_k).unsqueeze(1)
                target_k = target.expand(num_queries, -1, -1, -1)
                with torch.no_grad():
                    best = torch.topk(_binary_dice_score(prob_k, target_k), k=topk, dim=0).indices
                selected_logits = logits_k[best].unsqueeze(1)
                selected_target = target.expand(topk, -1, -1, -1)
                bce = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction="none").mean(dim=(1, 2, 3))
                dice = _binary_dice_loss(torch.sigmoid(selected_logits), selected_target)
                visible_reconstruction_terms.append((bce + dice).mean())

    out: Dict[str, torch.Tensor] = {}
    device = data["rgb"].device
    if mask_terms:
        raw_mask = torch.stack(mask_terms).mean()
        raw_boundary = torch.stack(boundary_terms).mean()
        raw_score = torch.stack(score_terms).mean()
        raw_ranking = torch.stack(ranking_terms).mean() if ranking_terms else raw_mask.new_zeros(())
        raw_diversity = torch.stack(diversity_terms).mean() if diversity_terms else raw_mask.new_zeros(())
        score_scale = _linear_warmup_scale(data, int(getattr(lc, "geomaskformer_score_warmup_iters", 0)))
        ranking_scale = _linear_warmup_scale(data, int(getattr(lc, "geomaskformer_ranking_warmup_iters", 0)))
        diversity_scale = _linear_warmup_scale(data, int(getattr(lc, "geomaskformer_diversity_warmup_iters", 0)))
        out["raw_geomaskformer_bestofk_mask"] = raw_mask.detach()
        out["raw_geomaskformer_boundary"] = raw_boundary.detach()
        out["raw_geomaskformer_score"] = raw_score.detach()
        out["raw_geomaskformer_ranking"] = raw_ranking.detach()
        out["raw_geomaskformer_diversity"] = raw_diversity.detach()
        out["aux_geomaskformer_bestofk_mask"] = raw_mask * lc.lambda_geomaskformer_mask
        out["aux_geomaskformer_boundary"] = raw_boundary * lc.lambda_geomaskformer_boundary
        out["aux_geomaskformer_score"] = raw_score * lc.lambda_geomaskformer_score * score_scale
        out["aux_geomaskformer_ranking"] = raw_ranking * getattr(lc, "lambda_geomaskformer_ranking", 0.0) * ranking_scale
        out["aux_geomaskformer_diversity"] = (
            raw_diversity * getattr(lc, "lambda_geomaskformer_diversity", 0.0) * diversity_scale
        )
        out["geomaskformer/score_warmup_scale"] = raw_mask.new_tensor(score_scale)
        out["geomaskformer/ranking_warmup_scale"] = raw_mask.new_tensor(ranking_scale)
        out["geomaskformer/diversity_warmup_scale"] = raw_mask.new_tensor(diversity_scale)
        out["geomaskformer/proposal_selected_dice"] = torch.stack(top1_scores).mean()
        out["geomaskformer/proposal_oracle_best_dice"] = torch.stack(oracle_scores).mean()
        out["geomaskformer/proposal_oracle_topk_mean_dice"] = torch.stack(oracle_topk_mean_scores).mean()
        out["geomaskformer/proposal_oracle_top4_dice"] = torch.stack(oracle_top4_scores).mean()
        out["geomaskformer/proposal_oracle_top5_best_dice"] = torch.stack(oracle_top5_best_scores).mean()
        out["geomaskformer/proposal_oracle_top5_mean_dice"] = torch.stack(oracle_top5_mean_scores).mean()
        out["geomaskformer/proposal_top5_cover_rate_0p85"] = torch.stack(cover85).mean()
        out["geomaskformer/proposal_top5_cover_rate_0p90"] = torch.stack(cover90).mean()
        out["geomaskformer/proposal_oracle_top10_dice"] = torch.stack(oracle_top10_scores).mean()
        hist = torch.bincount(torch.stack(all_best_ids).long(), minlength=num_queries).float()
        probs = hist / hist.sum().clamp_min(1.0)
        out["geomaskformer/proposal_active_query_count"] = (hist > 0).float().sum()
        out["geomaskformer/proposal_query_usage_entropy"] = -(probs[probs > 0] * probs[probs > 0].log()).sum()
        quality_flat = torch.cat(all_score_quality)
        dice_flat = torch.cat(all_oracle_quality)
        q_rank = torch.argsort(torch.argsort(quality_flat)).float()
        d_rank = torch.argsort(torch.argsort(dice_flat)).float()
        q_rank = q_rank - q_rank.mean()
        d_rank = d_rank - d_rank.mean()
        out["geomaskformer/proposal_score_dice_rank_corr"] = (
            q_rank * d_rank
        ).mean() / (q_rank.std(unbiased=False) * d_rank.std(unbiased=False)).clamp_min(1.0e-6)
        out["geomaskformer/proposal_selection_gap"] = (
            out["geomaskformer/proposal_oracle_best_dice"] - out["geomaskformer/proposal_selected_dice"]
        )
    proposal_steps = data.get("proposal_logits_steps_lowres")
    if proposal_steps and mask_terms:
        refinement_terms = [
            _bestofk_mask_loss_for_logits(step_logits, gt_for_loss, supervised_mask, lc)
            for step_logits in proposal_steps
            if torch.is_tensor(step_logits)
        ]
        if refinement_terms:
            raw_refinement = torch.stack(refinement_terms).mean()
            refinement_scale = _linear_warmup_scale(data, int(getattr(lc, "geomaskformer_refinement_warmup_iters", 0)))
            out["raw_geomaskformer_refinement"] = raw_refinement.detach()
            out["aux_geomaskformer_refinement"] = (
                raw_refinement * getattr(lc, "lambda_geomaskformer_refinement", 0.0) * refinement_scale
            )
            out["geomaskformer/refinement_warmup_scale"] = proposal_logits.new_tensor(refinement_scale)
    if visible_reconstruction_terms:
        raw_visible = torch.stack(visible_reconstruction_terms).mean()
        out["raw_geomaskformer_visible_reconstruction"] = raw_visible.detach()
        out["aux_geomaskformer_visible_reconstruction"] = raw_visible * visible_reconstruction_weight
    for data_key, loss_key in (
        (
            "geomaskformer_mask_prompt_pixel_corruption_ratio",
            "geomaskformer/mask_prompt_pixel_corruption_ratio",
        ),
        (
            "geomaskformer_mask_prompt_block_corruption_ratio",
            "geomaskformer/mask_prompt_block_corruption_ratio",
        ),
    ):
        value = data.get(data_key)
        if torch.is_tensor(value):
            out[loss_key] = value.detach()
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
