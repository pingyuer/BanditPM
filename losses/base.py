from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.jit.script
def ce_loss(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, soft_gt, reduction='none')
    return loss.sum(0).mean()


@torch.jit.script
def dice_loss(mask: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    mask = mask[:, 1:].flatten(start_dim=2).contiguous()
    gt = soft_gt[:, 1:].float().flatten(start_dim=2).contiguous()
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum(0).mean()
