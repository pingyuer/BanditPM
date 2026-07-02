from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F

from utils.tensor_utils import aggregate, cls_to_one_hot


def _compute_cardia_losses(
    lc,
    data: Dict[str, torch.Tensor],
    supervised_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not lc.is_cardia or "cls_gt" not in data:
        return {}

    batch_size = data["rgb"].shape[0]
    device = data["rgb"].device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    current_iter = int(data.get("current_iter", 0))
    lambda_base = lc.lambda_cardia_base if current_iter < lc.cardia_base_after_iter else lc.cardia_base_after_weight
    lambda_oracle = (
        lc.lambda_cardia_proposal_oracle
        if current_iter < lc.cardia_oracle_decay_start
        else lc.cardia_oracle_after_weight
    )
    flow_smooth_scale = 1.0
    if lc.cardia_flow_smooth_warmup_iters > 0:
        flow_smooth_scale = min(max(float(current_iter) / float(lc.cardia_flow_smooth_warmup_iters), 0.0), 1.0)
    base_terms = []
    proposal_terms = []
    top1_terms = []
    mhf_terms = []
    selector_global_terms = []
    selector_spatial_terms = []
    selector_margin_global_terms = []
    selector_margin_spatial_terms = []
    smooth_terms = []
    boundary_terms = []
    memory_stage2_terms = []
    memory_stage3_terms = []
    reliability_write_terms = []

    for bi in range(batch_size):
        curr_num_obj = int(data["info"]["num_objects"][bi].item()) if "info" in data else 1
        frame_ids = lc._frame_ids_for_sample(supervised_mask, bi)
        for ti in frame_ids:
            memory_aux = data.get(f"memory_aux_{ti}")
            aux = memory_aux.get("cardia_aux") if isinstance(memory_aux, dict) else None
            if not isinstance(aux, dict):
                continue
            soft_gt = cls_to_one_hot(data["cls_gt"][bi, ti : ti + 1], curr_num_obj)

            base_obj = aux.get("base_object_logits")
            if torch.is_tensor(base_obj) and lambda_base > 0:
                base_obj = base_obj[bi : bi + 1, :curr_num_obj]
                base_logits = aggregate(torch.sigmoid(base_obj), dim=1)
                ce, dice = lc.mask_loss(base_logits, soft_gt)
                base_terms.append(ce + dice)

            proposals = aux.get("proposal_logits")
            head_losses = []
            if torch.is_tensor(proposals) and (
                lambda_oracle > 0
                or lc.lambda_cardia_selector > 0
                or lc.lambda_cardia_proposal_top1 > 0
                or lc.lambda_cardia_selector_margin > 0
            ):
                proposals = proposals[bi : bi + 1, :curr_num_obj]
                for ki in range(proposals.shape[2]):
                    prop_obj = proposals[:, :, ki]
                    prop_logits = aggregate(torch.sigmoid(prop_obj), dim=1)
                    ce, dice = lc.mask_loss(prop_logits, soft_gt)
                    head_losses.append(ce + dice)
                if head_losses and lambda_oracle > 0:
                    stacked = torch.stack(head_losses)
                    if lc.cardia_proposal_loss in {"softmin", "soft_oracle"}:
                        tau = max(lc.cardia_proposal_softmin_temperature, 1.0e-4)
                        weights = torch.softmax(-stacked.detach() / tau, dim=0)
                        proposal_terms.append((weights * stacked).sum())
                    else:
                        proposal_terms.append(stacked.min())
                if head_losses and lc.lambda_cardia_selector > 0:
                    stacked = torch.stack(head_losses).detach()
                    target = torch.softmax(-stacked / max(lc.cardia_selector_temperature, 1.0e-4), dim=0).unsqueeze(0)
                    oracle_idx = int(stacked.argmin().item())
                    for key, terms, margin_terms, lambda_sel, lambda_margin in (
                        ("global_selector_logits", selector_global_terms, selector_margin_global_terms, lc.lambda_cardia_selector_global, lc.lambda_cardia_selector_margin_global),
                        ("spatial_pooled_selector_logits", selector_spatial_terms, selector_margin_spatial_terms, lc.lambda_cardia_selector_spatial, lc.lambda_cardia_selector_margin_spatial),
                    ):
                        selector_logits = aux.get(key)
                        if not torch.is_tensor(selector_logits):
                            continue
                        logits = selector_logits[bi : bi + 1, 0]
                        if lambda_sel > 0:
                            terms.append(F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean"))
                        if lambda_margin > 0 and logits.shape[-1] > 1:
                            oracle_logit = logits[:, oracle_idx : oracle_idx + 1]
                            margin_raw = lc.cardia_selector_margin - oracle_logit + logits
                            keep = torch.ones_like(margin_raw, dtype=torch.bool)
                            keep[:, oracle_idx] = False
                            margin_terms.append(F.relu(margin_raw[keep]).mean())
                if head_losses and lc.lambda_cardia_proposal_top1 > 0:
                    top1_obj = aux.get("proposal_top1_logits")
                    if torch.is_tensor(top1_obj):
                        top1_obj = top1_obj[bi : bi + 1, :curr_num_obj]
                        top1_logits = aggregate(torch.sigmoid(top1_obj), dim=1)
                        ce, dice = lc.mask_loss(top1_logits, soft_gt)
                        top1_terms.append(ce + dice)

            if lc.lambda_cardia_multi_head_fused > 0:
                mhf_obj = aux.get("multi_head_fused_logits")
                if torch.is_tensor(mhf_obj):
                    mhf_obj = mhf_obj[bi : bi + 1, :curr_num_obj]
                    mhf_logits = aggregate(torch.sigmoid(mhf_obj), dim=1)
                    ce, dice = lc.mask_loss(mhf_logits, soft_gt)
                    mhf_terms.append(ce + dice)

            for src, weight in (
                ("stage2_flow_smooth", lc.lambda_cardia_stage2_flow_smooth * flow_smooth_scale),
                ("stage3_flow_smooth", lc.lambda_cardia_stage3_flow_smooth * flow_smooth_scale),
            ):
                value = aux.get(src)
                if torch.is_tensor(value):
                    item = value[bi : bi + 1] if value.dim() > 0 and value.shape[0] > bi else value
                    smooth_terms.append((src, item.float().mean(), weight))

            boundary_logits = aux.get("boundary_logits")
            if torch.is_tensor(boundary_logits) and lc.lambda_cardia_boundary_aux > 0:
                gt_fg = (data["cls_gt"][bi, ti : ti + 1].float() > 0).float()
                if gt_fg.dim() == 3:
                    gt_fg = gt_fg.unsqueeze(1)
                if gt_fg.shape[-2:] != boundary_logits.shape[-2:]:
                    gt_fg = F.interpolate(gt_fg, size=boundary_logits.shape[-2:], mode="nearest")
                k = lc.cardia_boundary_dilation_kernel
                p = k // 2
                dil = F.max_pool2d(gt_fg, kernel_size=k, stride=1, padding=p)
                ero = 1.0 - F.max_pool2d(1.0 - gt_fg, kernel_size=k, stride=1, padding=p)
                boundary = (dil - ero).clamp(0.0, 1.0)
                pred = boundary_logits[bi : bi + 1]
                target = boundary.to(device=pred.device, dtype=pred.dtype).float()
                global_loss = F.binary_cross_entropy_with_logits(pred.float(), target)
                band_mask = target > 0.5
                if band_mask.any():
                    band_loss = F.binary_cross_entropy_with_logits(pred.float()[band_mask], target[band_mask].float())
                else:
                    band_loss = global_loss
                area_ratio = gt_fg.mean().clamp_min(1.0e-4)
                small_area_weight = 1.0 + 0.5 * (1.0 - area_ratio.detach())
                boundary_terms.append(small_area_weight * band_loss + 0.2 * global_loss)

            for key, terms, weight in (
                ("stage2_memory_mask_prior_logits", memory_stage2_terms, lc.lambda_cardia_memory_readout),
                ("stage3_memory_mask_prior_logits", memory_stage3_terms, lc.lambda_cardia_memory_readout_stage3),
            ):
                if weight <= 0:
                    continue
                mem_logits = aux.get(key)
                if torch.is_tensor(mem_logits):
                    pred = mem_logits[bi : bi + 1, :1]
                    gt_fg = (data["cls_gt"][bi, ti : ti + 1].float() > 0).float()
                    if gt_fg.dim() == 3:
                        gt_fg = gt_fg.unsqueeze(1)
                    if gt_fg.shape[-2:] != pred.shape[-2:]:
                        gt_fg = F.interpolate(gt_fg, size=pred.shape[-2:], mode="nearest")
                    logits = torch.cat([-pred, pred], dim=1)
                    target = torch.cat([1.0 - gt_fg, gt_fg], dim=1)
                    ce, dice = lc.mask_loss(logits, target)
                    terms.append(ce + dice)

            if lc.lambda_cardia_reliability_write > 0:
                for prefix in ("stage2", "stage3"):
                    write = aux.get(f"{prefix}_memory_write_mean")
                    reliability = aux.get(f"{prefix}_memory_reliability")
                    if torch.is_tensor(write) and torch.is_tensor(reliability):
                        w = write[bi : bi + 1] if write.dim() > 0 and write.shape[0] > bi else write.reshape(1)
                        r = reliability[bi : bi + 1] if reliability.dim() > 0 and reliability.shape[0] > bi else reliability.reshape(1)
                        reliability_write_terms.append((w.float() * (1.0 - r.float()).detach()).mean())

    out: Dict[str, torch.Tensor] = {}
    if base_terms:
        raw = torch.stack(base_terms).mean()
        out["raw_cardia_base"] = raw.detach()
        out["aux_cardia_base"] = raw * lambda_base
    if proposal_terms:
        raw = torch.stack(proposal_terms).mean()
        out["raw_cardia_proposal_oracle"] = raw.detach()
        out["aux_cardia_proposal_oracle"] = raw * lambda_oracle
    if top1_terms:
        raw = torch.stack(top1_terms).mean()
        out["raw_cardia_proposal_top1"] = raw.detach()
        out["aux_cardia_proposal_top1"] = raw * lc.lambda_cardia_proposal_top1
    if mhf_terms:
        raw = torch.stack(mhf_terms).mean()
        out["raw_cardia_multi_head_fused"] = raw.detach()
        out["aux_cardia_multi_head_fused"] = raw * lc.lambda_cardia_multi_head_fused
    selector_weighted = []
    if selector_global_terms:
        raw = torch.stack(selector_global_terms).mean()
        out["raw_cardia_selector_global"] = raw.detach()
        selector_weighted.append(raw * lc.lambda_cardia_selector_global)
    if selector_spatial_terms:
        raw = torch.stack(selector_spatial_terms).mean()
        out["raw_cardia_selector_spatial"] = raw.detach()
        selector_weighted.append(raw * lc.lambda_cardia_selector_spatial)
    if selector_weighted:
        out["aux_cardia_selector"] = torch.stack(selector_weighted).sum()
    margin_weighted = []
    if selector_margin_global_terms:
        raw = torch.stack(selector_margin_global_terms).mean()
        out["raw_cardia_selector_margin_global"] = raw.detach()
        margin_weighted.append(raw * lc.lambda_cardia_selector_margin_global)
    if selector_margin_spatial_terms:
        raw = torch.stack(selector_margin_spatial_terms).mean()
        out["raw_cardia_selector_margin_spatial"] = raw.detach()
        margin_weighted.append(raw * lc.lambda_cardia_selector_margin_spatial)
    if margin_weighted:
        out["aux_cardia_selector_margin"] = torch.stack(margin_weighted).sum()
    if smooth_terms:
        by_stage = {}
        for src, raw_value, weight in smooth_terms:
            by_stage.setdefault(src, []).append((raw_value, weight))
        weighted_values = []
        raw_values = []
        for src, terms in by_stage.items():
            raw_stage = torch.stack([term[0] for term in terms]).mean()
            weight = float(terms[0][1])
            weighted_stage = raw_stage * weight
            stage_name = "stage2" if src.startswith("stage2") else "stage3"
            out[f"raw_cardia_{stage_name}_flow_smooth"] = raw_stage.detach()
            out[f"weighted_cardia_{stage_name}_flow_smooth"] = weighted_stage.detach()
            raw_values.append(raw_stage)
            weighted_values.append(weighted_stage)
        raw_total = torch.stack(raw_values).mean()
        weighted_total = torch.stack(weighted_values).mean()
        out["raw_cardia_flow_smooth"] = raw_total.detach()
        out["aux_cardia_flow_smooth"] = weighted_total
    if boundary_terms:
        raw = torch.stack(boundary_terms).mean()
        out["raw_cardia_boundary_aux"] = raw.detach()
        out["aux_cardia_boundary_aux"] = raw * lc.lambda_cardia_boundary_aux
    if memory_stage2_terms:
        raw = torch.stack(memory_stage2_terms).mean()
        out["raw_cardia_memory_readout_stage2"] = raw.detach()
        out["aux_cardia_memory_readout_stage2"] = raw * lc.lambda_cardia_memory_readout
    if memory_stage3_terms:
        raw = torch.stack(memory_stage3_terms).mean()
        out["raw_cardia_memory_readout_stage3"] = raw.detach()
        out["aux_cardia_memory_readout_stage3"] = raw * lc.lambda_cardia_memory_readout_stage3
    if reliability_write_terms:
        raw = torch.stack(reliability_write_terms).mean()
        out["raw_cardia_reliability_write"] = raw.detach()
        out["aux_cardia_reliability_write"] = raw * lc.lambda_cardia_reliability_write

    if lc.lambda_head_diversity > 0:
        diversity_terms = []
        num_frames = data["rgb"].shape[1]
        for ti_dv in range(num_frames):
            memory_aux = data.get(f"memory_aux_{ti_dv}")
            aux = memory_aux.get("cardia_aux") if isinstance(memory_aux, dict) else None
            if not isinstance(aux, dict):
                continue
            head_usage = aux.get("stage2_head_usage")
            if torch.is_tensor(head_usage) and head_usage.dim() >= 1:
                usage_mean = head_usage.float().mean(dim=0)
                usage_mean_clamped = usage_mean.clamp_min(1.0e-6)
                entropy = -(usage_mean_clamped * usage_mean_clamped.log()).sum()
                max_entropy = float(usage_mean_clamped.shape[0])
                normalized_entropy = entropy / max_entropy
                diversity_terms.append(-normalized_entropy)
        if diversity_terms:
            raw_diversity = torch.stack(diversity_terms).mean()
            out["raw_cardia_head_diversity"] = raw_diversity.detach()
            out["aux_cardia_head_diversity"] = raw_diversity * lc.lambda_head_diversity

    if not out:
        out["aux_cardia_zero"] = zero
    return out
