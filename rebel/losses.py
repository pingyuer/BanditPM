from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from losses.base import dice_loss
from utils.tensor_utils import cls_to_one_hot


def dice_ce(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    ce = -(soft_gt * logp).sum(dim=1).mean()
    return ce + dice_loss(logits.softmax(dim=1), soft_gt)


def weighted_ce(logits: torch.Tensor, soft_gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ce = -(soft_gt * F.log_softmax(logits, dim=1)).sum(dim=1, keepdim=True)
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    if weight.shape[-2:] != ce.shape[-2:]:
        weight = F.interpolate(weight, size=ce.shape[-2:], mode="bilinear", align_corners=False)
    weight = weight.clamp_min(0.0)
    return (ce * weight).sum() / (weight.sum() + 1.0e-6)


def smoothness(offset: torch.Tensor) -> torch.Tensor:
    dx = offset[..., :, 1:] - offset[..., :, :-1]
    dy = offset[..., 1:, :] - offset[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def foreground_dice_metric(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1:].flatten(start_dim=2)
    gt = soft_gt[:, 1:].float().flatten(start_dim=2)
    dice = (2.0 * (prob * gt).sum(-1) + 1.0) / (prob.sum(-1) + gt.sum(-1) + 1.0)
    return dice.mean().detach()


def compute_rebel_losses(loss_computer, data, supervised_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = loss_computer.rebel_loss_cfg
    weights = {
        "final": float(cfg.get("final", 1.0)),
        "base_aux": float(cfg.get("base_aux", 0.35)),
        "belief_prior": float(cfg.get("belief_prior", 0.15)),
        "obs_aux": float(cfg.get("obs_aux", 0.20)),
        "rebel_aux": float(cfg.get("rebel_aux", 0.10)),
        "corrected_aux": float(cfg.get("corrected_aux", 0.05)),
        "candidate_oracle": float(cfg.get("candidate_oracle", 0.15)),
        "arbitration": float(cfg.get("arbitration", 0.05)),
        "correction": float(cfg.get("correction", 0.05)),
        "temporal": float(cfg.get("temporal", 0.03)),
        "offset_smooth": float(cfg.get("offset_smooth", 0.005)),
        "write_reg": float(cfg.get("write_reg", 0.01)),
    }
    oracle_tau = float(cfg.get("candidate_oracle_temperature", 0.20))
    losses = defaultdict(lambda: torch.zeros((), device=data["rgb"].device))
    candidate_dice = defaultdict(list)
    candidate_names = ("base_logits", "obs_logits", "belief_logits", "rebel_logits", "corrected_logits")
    batch_size, _num_frames = data["rgb"].shape[:2]
    for bi in range(batch_size):
        t_range = loss_computer._frame_ids_for_sample(supervised_mask, bi)
        if not t_range:
            continue
        curr_num_obj = int(data.get("num_objects", [1] * batch_size)[bi]) if not torch.is_tensor(data.get("num_objects")) else int(data["num_objects"][bi])
        valid_slice = slice(None, curr_num_obj + 1)
        cls_gt = data["cls_gt"][bi, t_range]
        if cls_gt.dim() == 3:
            cls_gt = cls_gt.unsqueeze(1)
        soft_gt = cls_to_one_hot(cls_gt, curr_num_obj)
        frame_sets = {}
        for name in ("logits", "base_logits", "belief_logits", "obs_logits", "rebel_logits", "corrected_logits", "correction_logits"):
            frame_sets[name] = torch.stack([data[f"aux_{ti}"].get(name, data[f"logits_{ti}"])[bi, valid_slice] if name != "logits" else data[f"logits_{ti}"][bi, valid_slice] for ti in t_range], dim=0)
        losses["rebel_final"] += dice_ce(frame_sets["logits"], soft_gt) / batch_size * weights["final"]
        losses["rebel_base_aux"] += dice_ce(frame_sets["base_logits"], soft_gt) / batch_size * weights["base_aux"]
        losses["rebel_belief_prior"] += dice_ce(frame_sets["belief_logits"], soft_gt) / batch_size * weights["belief_prior"]
        losses["rebel_obs_aux"] += dice_ce(frame_sets["obs_logits"], soft_gt) / batch_size * weights["obs_aux"]
        losses["rebel_decoder_aux"] += dice_ce(frame_sets["rebel_logits"], soft_gt) / batch_size * weights["rebel_aux"]
        losses["rebel_corrected_aux"] += dice_ce(frame_sets["corrected_logits"], soft_gt) / batch_size * weights["corrected_aux"]
        candidate_losses = torch.stack([dice_ce(frame_sets[name], soft_gt) for name in candidate_names])
        soft_oracle = torch.softmax(-candidate_losses.detach() / max(oracle_tau, 1.0e-6), dim=0)
        losses["rebel_candidate_oracle"] += (candidate_losses * soft_oracle).sum() / batch_size * weights["candidate_oracle"]
        for name in candidate_names:
            candidate_dice[name].append(foreground_dice_metric(frame_sets[name], soft_gt))
        final_dice = foreground_dice_metric(frame_sets["logits"], soft_gt)
        candidate_dice["final"].append(final_dice)

        arbitration_logits = []
        for ti in t_range:
            aux_t = data[f"aux_{ti}"]
            if torch.is_tensor(aux_t.get("arbitration_logits")):
                arbitration_logits.append(aux_t["arbitration_logits"][bi : bi + 1])
        if arbitration_logits:
            pooled_logits = torch.cat(arbitration_logits, dim=0).mean(dim=(-2, -1))
            target = torch.softmax(-candidate_losses.detach() / max(oracle_tau, 1.0e-6), dim=0).unsqueeze(0).expand_as(pooled_logits)
            losses["rebel_arbitration"] += F.kl_div(
                F.log_softmax(pooled_logits, dim=1),
                target,
                reduction="batchmean",
            ) / batch_size * weights["arbitration"]
        disagreement_frames = []
        for ti in t_range:
            dis = data[f"aux_{ti}"].get("rebel/disagreement")
            if torch.is_tensor(dis):
                disagreement_frames.append(dis[bi : bi + 1])
        if disagreement_frames:
            weight = 1.0 + torch.cat(disagreement_frames, dim=0)
            losses["rebel_correction"] += weighted_ce(frame_sets["logits"], soft_gt, weight) / batch_size * weights["correction"]
        if len(t_range) > 1:
            probs = torch.softmax(frame_sets["logits"], dim=1)[:, 1:2]
            losses["rebel_temporal"] += (probs[1:] - probs[:-1]).abs().mean() / batch_size * weights["temporal"]
    offset_terms = []
    write_terms = []
    for key, aux in data.items():
        if not key.startswith("memory_aux_") or not isinstance(aux, dict):
            continue
        rebel_aux = aux.get("rebel_aux", {})
        for offset_key in ("offset_obs_px", "offset_mem_px"):
            if torch.is_tensor(rebel_aux.get(offset_key)):
                offset_terms.append(smoothness(rebel_aux[offset_key]))
        if torch.is_tensor(rebel_aux.get("write_slow")) and torch.is_tensor(rebel_aux.get("write_fast")):
            write_terms.append(F.relu(rebel_aux["write_slow"] - rebel_aux["write_fast"]).mean())
    if offset_terms:
        losses["rebel_offset_smooth"] += torch.stack(offset_terms).mean() * weights["offset_smooth"]
    if write_terms:
        losses["rebel_write_reg"] += torch.stack(write_terms).mean() * weights["write_reg"]
    for name, vals in candidate_dice.items():
        if vals:
            metric_name = name.replace("_logits", "")
            losses[f"rebel/{metric_name}_dice"] = torch.stack(vals).mean()
    if candidate_dice.get("final") and candidate_dice.get("base_logits"):
        losses["rebel/final_minus_base_dice"] = torch.stack(candidate_dice["final"]).mean() - torch.stack(candidate_dice["base_logits"]).mean()
    if candidate_dice.get("corrected_logits") and candidate_dice.get("rebel_logits"):
        losses["rebel/corrected_minus_rebel_dice"] = torch.stack(candidate_dice["corrected_logits"]).mean() - torch.stack(candidate_dice["rebel_logits"]).mean()
    return dict(losses)
