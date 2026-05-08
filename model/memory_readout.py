from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareMemoryReadout(nn.Module):
    """Small online mask-aware memory for UNeXt-DynaKey refinement.

    This is deliberately lighter than full GDKVM memory. It stores per-object
    key/value prototypes and a low-resolution mask prior, then reads them with
    cosine similarity for the current frame.
    """

    def __init__(
        self,
        value_dim: int,
        *,
        num_slots: int = 4,
        ema_momentum: float = 0.9,
        temperature: float = 0.1,
        mask_size: int = 16,
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.value_dim = value_dim
        self.num_slots = num_slots
        self.ema_momentum = ema_momentum
        self.temperature = temperature
        self.mask_size = mask_size
        self.confidence_threshold = confidence_threshold

        self.register_buffer("_keys", torch.empty(0), persistent=False)
        self.register_buffer("_values", torch.empty(0), persistent=False)
        self.register_buffer("_mask_proto", torch.empty(0), persistent=False)
        self.register_buffer("_valid", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("_usage", torch.empty(0), persistent=False)

    def reset_state(self, batch_size: int, num_objects: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (batch_size, num_objects, self.num_slots)
        self._keys = torch.zeros(*shape, self.value_dim, device=device, dtype=dtype)
        self._values = torch.zeros_like(self._keys)
        self._mask_proto = torch.zeros(*shape, self.mask_size, self.mask_size, device=device, dtype=dtype)
        self._valid = torch.zeros(*shape, device=device, dtype=torch.bool)
        self._usage = torch.zeros(*shape, device=device, dtype=dtype)

    @staticmethod
    def _masked_pool(value_BNCHW: torch.Tensor, mask_BNHW: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = value_BNCHW.shape
        mask = F.interpolate(mask_BNHW.flatten(0, 1).unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False)
        value = value_BNCHW.flatten(0, 1)
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (value * mask).sum(dim=(2, 3)) / denom
        fallback = value.mean(dim=(2, 3))
        empty = denom <= 1.0
        pooled = torch.where(empty, fallback, pooled)
        return pooled.view(B, N, C)

    def _next_slot(self) -> torch.Tensor:
        invalid_score = torch.where(self._valid, torch.full_like(self._usage, 1.0e6), self._usage)
        free_slot = invalid_score.argmin(dim=-1)
        least_used = self._usage.argmin(dim=-1)
        has_free = (~self._valid).any(dim=-1)
        return torch.where(has_free, free_slot, least_used)

    def read(self, value_BNCHW: torch.Tensor, mask_BNHW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self._keys.numel() == 0:
            self.reset_state(value_BNCHW.shape[0], value_BNCHW.shape[1], value_BNCHW.device, value_BNCHW.dtype)
        query = self._masked_pool(value_BNCHW.detach(), mask_BNHW.detach())
        query_n = F.normalize(query, dim=-1)
        key_n = F.normalize(self._keys, dim=-1)
        logits = (query_n.unsqueeze(2) * key_n).sum(dim=-1) / max(self.temperature, 1e-6)
        logits = logits.masked_fill(~self._valid, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(self._valid.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        readout = (weights.unsqueeze(-1) * self._values).sum(dim=2)
        mask_prior = (weights.unsqueeze(-1).unsqueeze(-1) * self._mask_proto).sum(dim=2)
        aux = {
            "mask_memory_valid_slots": self._valid.sum(dim=-1).detach(),
            "mask_memory_entropy": (-(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)).detach(),
        }
        return readout.unsqueeze(-1).unsqueeze(-1), mask_prior, aux

    def update(self, value_BNCHW: torch.Tensor, prob_BNHW: torch.Tensor) -> None:
        if self._keys.numel() == 0:
            self.reset_state(value_BNCHW.shape[0], value_BNCHW.shape[1], value_BNCHW.device, value_BNCHW.dtype)
        with torch.no_grad():
            pooled = self._masked_pool(value_BNCHW.detach(), prob_BNHW.detach())
            mask_proto = F.interpolate(
                prob_BNHW.flatten(0, 1).unsqueeze(1).float(),
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            ).view(prob_BNHW.shape[0], prob_BNHW.shape[1], self.mask_size, self.mask_size)
            prob = prob_BNHW.detach().float().clamp(1e-6, 1.0 - 1e-6)
            entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
            confidence = 1.0 - entropy.mean(dim=(-2, -1))
            enabled = confidence >= self.confidence_threshold
            slot = self._next_slot()
            B, N = slot.shape
            for b in range(B):
                for n in range(N):
                    if not bool(enabled[b, n]):
                        continue
                    s = int(slot[b, n].item())
                    if self._valid[b, n, s]:
                        m = self.ema_momentum
                        self._keys[b, n, s] = m * self._keys[b, n, s] + (1.0 - m) * pooled[b, n]
                        self._values[b, n, s] = m * self._values[b, n, s] + (1.0 - m) * pooled[b, n]
                        self._mask_proto[b, n, s] = m * self._mask_proto[b, n, s] + (1.0 - m) * mask_proto[b, n]
                    else:
                        self._keys[b, n, s] = pooled[b, n]
                        self._values[b, n, s] = pooled[b, n]
                        self._mask_proto[b, n, s] = mask_proto[b, n]
                        self._valid[b, n, s] = True
                    self._usage[b, n, s] += 1.0
