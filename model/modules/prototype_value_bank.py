from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeValueBank(nn.Module):
    """
    Learnable codebook for prototype-guided value augmentation.

    Shapes:
        feat_vec: [B, D]
        codebook: [K, D]
        assign_prob: [B, K]
        proto_value: [B, D]
    """

    def __init__(
        self,
        num_proto: int,
        dim: int,
        temperature: float = 1.0,
        normalize: bool = True,
        init_mode: str = "learnable",
        topk: int = 0,
    ) -> None:
        super().__init__()
        self.num_proto = num_proto
        self.dim = dim
        self.temperature = max(float(temperature), 1e-6)
        self.normalize = normalize
        self.topk = max(int(topk), 0)

        self.codebook = nn.Parameter(torch.empty(num_proto, dim))
        self.reset_parameters(init_mode=init_mode)

    def reset_parameters(self, init_mode: str = "learnable") -> None:
        if init_mode in {"learnable", "xavier_uniform"}:
            nn.init.xavier_uniform_(self.codebook)
        elif init_mode == "normal":
            nn.init.normal_(self.codebook, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unsupported codebook init_mode={init_mode!r}")

    def _apply_topk(self, assign_prob: torch.Tensor) -> torch.Tensor:
        if self.topk <= 0 or self.topk >= self.num_proto:
            return assign_prob

        values, indices = torch.topk(assign_prob, k=self.topk, dim=-1)
        sparse_prob = torch.zeros_like(assign_prob)
        sparse_prob.scatter_(dim=-1, index=indices, src=values)
        return sparse_prob / sparse_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, feat_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if feat_vec.ndim != 2:
            raise ValueError(f"PrototypeValueBank expects [B, D], got {tuple(feat_vec.shape)}")

        bank = self.codebook
        feat_for_sim = feat_vec
        if self.normalize:
            feat_for_sim = F.normalize(feat_for_sim, dim=-1)
            bank = F.normalize(bank, dim=-1)

        logits = torch.matmul(feat_for_sim, bank.t()) / self.temperature
        assign_prob = torch.softmax(logits, dim=-1)
        assign_prob = self._apply_topk(assign_prob)
        proto_value = torch.matmul(assign_prob, self.codebook)

        aux = {
            "logits": logits,
            "similarity": logits,
            "assign_prob": assign_prob,
        }
        return assign_prob, proto_value, aux
