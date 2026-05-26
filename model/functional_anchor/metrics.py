from __future__ import annotations

import torch


def pairwise_cosine_diversity(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    sim = torch.matmul(x, x.transpose(-1, -2))
    A = sim.shape[-1]
    if A <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    eye = torch.eye(A, device=x.device, dtype=torch.bool)
    off_diag = sim[..., ~eye].view(*sim.shape[:-2], A, A - 1)
    return (1.0 - off_diag).mean()


def dice_per_anchor(proposals: torch.Tensor, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(proposals)
    gt = gt.to(device=proposals.device, dtype=proposals.dtype)
    while gt.dim() < probs.dim():
        gt = gt.unsqueeze(2)
    inter = (probs * gt).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return (2.0 * inter + eps) / (denom + eps)
