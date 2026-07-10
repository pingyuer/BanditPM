from __future__ import annotations

from collections import defaultdict

import torch

from dpfr.grid import flow_smoothness
from utils.tensor_utils import cls_to_one_hot


def foreground_dice(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1:].flatten(start_dim=2)
    gt = soft_gt[:, 1:].float().flatten(start_dim=2)
    return ((2.0 * (prob * gt).sum(-1) + 1.0) / (prob.sum(-1) + gt.sum(-1) + 1.0)).mean().detach()


def _stack_logits(data, key: str, sample_idx: int, frame_ids: list[int], valid_slice: slice) -> torch.Tensor:
    tensor = data.get(key)
    if torch.is_tensor(tensor):
        return tensor[sample_idx, frame_ids, valid_slice]
    return torch.stack([data[f"aux_{ti}"][key][sample_idx, valid_slice] for ti in frame_ids], dim=0)


def compute_dpfr_losses(loss_computer, data, supervised_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = loss_computer.dpfr_loss_cfg
    weights = {
        "final": float(cfg.get("lambda_final", cfg.get("final", 1.0))),
        "anchor": float(cfg.get("lambda_anchor", cfg.get("anchor", 0.3))),
        "prompt": float(cfg.get("lambda_prompt", cfg.get("prompt", 0.5))),
        "flow_seg": float(cfg.get("lambda_flow_seg", cfg.get("flow_seg", 0.2))),
        "flow_mag": float(cfg.get("lambda_flow_mag", cfg.get("flow_mag", 0.005))),
        "flow_smooth": float(cfg.get("lambda_flow_smooth", cfg.get("flow_smooth", 0.01))),
        "flow_temp": float(cfg.get("lambda_flow_temp", cfg.get("flow_temp", 0.01))),
    }
    losses = defaultdict(lambda: torch.zeros((), device=data["rgb"].device))
    batch_size = data["rgb"].shape[0]
    final_dice = []
    anchor_dice = []
    prompt_dice = []
    flow_dice = []

    for bi in range(batch_size):
        frame_ids = loss_computer._frame_ids_for_sample(supervised_mask, bi)
        if not frame_ids:
            continue
        num_obj = int(data.get("num_objects", [1] * batch_size)[bi]) if not torch.is_tensor(data.get("num_objects")) else int(data["num_objects"][bi])
        num_obj = max(num_obj, 1)
        valid_slice = slice(None, num_obj + 1)
        cls_gt = data["cls_gt"][bi, frame_ids]
        if cls_gt.dim() == 3:
            cls_gt = cls_gt.unsqueeze(1)
        soft_gt = cls_to_one_hot(cls_gt, num_obj)

        final = torch.stack([data[f"logits_{ti}"][bi, valid_slice] for ti in frame_ids], dim=0)
        anchor = _stack_logits(data, "anchor_logits", bi, frame_ids, valid_slice)
        prompt = _stack_logits(data, "prompt_logits", bi, frame_ids, valid_slice)
        flow_logits = _stack_logits(data, "flow_logits", bi, frame_ids, valid_slice)

        final_ce, final_dice_loss = loss_computer.frame_mask_loss(final, soft_gt)
        anchor_ce, anchor_dice_loss = loss_computer.frame_mask_loss(anchor, soft_gt)
        prompt_ce, prompt_dice_loss = loss_computer.frame_mask_loss(prompt, soft_gt)
        flow_ce, flow_dice_loss = loss_computer.frame_mask_loss(flow_logits, soft_gt)
        losses["dpfr_final"] += (final_ce.mean() + final_dice_loss.mean()) / batch_size * weights["final"]
        losses["dpfr_anchor"] += (anchor_ce.mean() + anchor_dice_loss.mean()) / batch_size * weights["anchor"]
        losses["dpfr_prompt"] += (prompt_ce.mean() + prompt_dice_loss.mean()) / batch_size * weights["prompt"]
        losses["dpfr_flow"] += (flow_ce.mean() + flow_dice_loss.mean()) / batch_size * weights["flow_seg"]
        final_dice.append(foreground_dice(final, soft_gt))
        anchor_dice.append(foreground_dice(anchor, soft_gt))
        prompt_dice.append(foreground_dice(prompt, soft_gt))
        flow_dice.append(foreground_dice(flow_logits, soft_gt))

    flow = data.get("flow_grid")
    if torch.is_tensor(flow):
        losses["dpfr_flow_mag"] += flow.abs().mean() * weights["flow_mag"]
        losses["dpfr_flow_smooth"] += flow_smoothness(flow.flatten(0, 1)) * weights["flow_smooth"]
        if flow.shape[1] > 1:
            losses["dpfr_flow_temp"] += (flow[:, 1:] - flow[:, :-1]).abs().mean() * weights["flow_temp"]
        else:
            losses["dpfr_flow_temp"] += flow.sum() * 0.0
    if final_dice:
        losses["dpfr/final_dice"] = torch.stack(final_dice).mean()
        losses["dpfr/anchor_dice"] = torch.stack(anchor_dice).mean()
        losses["dpfr/prompt_dice"] = torch.stack(prompt_dice).mean()
        losses["dpfr/flow_dice"] = torch.stack(flow_dice).mean()
        losses["dpfr/final_minus_anchor_dice"] = losses["dpfr/final_dice"] - losses["dpfr/anchor_dice"]
        losses["dpfr/prompt_minus_anchor_dice"] = losses["dpfr/prompt_dice"] - losses["dpfr/anchor_dice"]
        losses["dpfr/flow_minus_prompt_dice"] = losses["dpfr/flow_dice"] - losses["dpfr/prompt_dice"]
    return dict(losses)
