from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import _group_count


class MatrixRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt().to(dtype=x.dtype)
        return x / rms


class SelectiveLinearDeformationMemory(nn.Module):
    """Compact key-map memory: appearance key -> deformation context."""

    def __init__(
        self,
        channels: int,
        *,
        key_dim: int = 64,
        value_dim: int | None = None,
        token_dim: int = 2,
        reliability_token_dim: int = 0,
        runtime_token_dim: int = 32,
        forget_bias: float = 1.0,
        write_bias: float = -1.0,
        use_rmsnorm: bool = True,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim or channels)
        self.runtime_token_dim = int(runtime_token_dim)
        self.reliability_token_dim = int(reliability_token_dim)
        self.zero_init = bool(zero_init)
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.q_proj = nn.Conv2d(channels, self.key_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, self.key_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, self.value_dim, kernel_size=1)
        self.read_proj = nn.Conv2d(self.value_dim, channels, kernel_size=1) if self.value_dim != channels else nn.Identity()
        gate_in_dim = channels + self.value_dim + token_dim + self.reliability_token_dim + self.runtime_token_dim
        gate_hidden = max(32, min(256, channels))
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )
        token_in_dim = channels * 2 + token_dim + self.runtime_token_dim
        self.runtime_token_update = nn.Sequential(
            nn.Linear(token_in_dim, self.runtime_token_dim * 2),
            nn.LayerNorm(self.runtime_token_dim * 2),
            nn.GELU(),
            nn.Linear(self.runtime_token_dim * 2, self.runtime_token_dim * 2),
        )
        self.memory_norm = MatrixRMSNorm() if use_rmsnorm else nn.Identity()
        nn.init.constant_(self.gate_mlp[-1].bias[0], float(forget_bias))
        nn.init.constant_(self.gate_mlp[-1].bias[1], float(write_bias))

    def _normalize_key(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.flatten(2), dim=1, eps=1.0e-6)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        memory_state_prev: torch.Tensor | None,
        area_token: torch.Tensor,
        runtime_token_prev: torch.Tensor | None = None,
        reliability_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, _, H, W = anchor_feat_t.shape
        if runtime_token_prev is None:
            runtime_token_prev = anchor_feat_t.new_zeros(B, self.runtime_token_dim)
        if memory_state_prev is None:
            memory_state_prev = anchor_feat_t.new_zeros(B, self.value_dim, self.key_dim)
        anchor_norm = self.anchor_norm(anchor_feat_t)
        q = self._normalize_key(self.q_proj(anchor_norm))
        k = self._normalize_key(self.k_proj(anchor_norm))
        v = self.v_proj(anchor_norm).flatten(2)
        old_v = torch.bmm(memory_state_prev, k)
        delta = v - old_v
        update_map = torch.bmm(delta, k.transpose(1, 2)) / max(float(H * W), 1.0)

        pooled_anchor = anchor_norm.mean(dim=(2, 3))
        pooled_old = old_v.mean(dim=-1)
        area = area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype)
        pieces = [pooled_anchor, pooled_old, area]
        if self.reliability_token_dim > 0:
            if reliability_token is None:
                reliability = anchor_feat_t.new_zeros(B, self.reliability_token_dim)
            else:
                reliability = reliability_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype)
                if reliability.shape[1] < self.reliability_token_dim:
                    pad = self.reliability_token_dim - reliability.shape[1]
                    reliability = F.pad(reliability, (0, pad))
                reliability = reliability[:, : self.reliability_token_dim]
            pieces.append(reliability)
        pieces.append(runtime_token_prev.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype))
        gate_logits = self.gate_mlp(torch.cat(pieces, dim=1))
        forget = torch.sigmoid(gate_logits[:, 0]).view(B, 1, 1)
        write = torch.sigmoid(gate_logits[:, 1]).view(B, 1, 1)
        memory_state_t = self.memory_norm(forget * memory_state_prev + write * update_map)
        read = torch.bmm(memory_state_t, q).view(B, self.value_dim, H, W)
        memory_context_t = self.read_proj(read)

        token_pooled = torch.cat(
            [
                pooled_anchor,
                memory_context_t.mean(dim=(2, 3)),
                area,
                runtime_token_prev.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype),
            ],
            dim=1,
        )
        token_gate_raw, token_delta = self.runtime_token_update(token_pooled).chunk(2, dim=1)
        token_gate = torch.sigmoid(token_gate_raw)
        runtime_token_t = (1.0 - token_gate) * runtime_token_prev + token_gate * token_delta

        memory_flat = memory_state_t.detach().flatten(1).float()
        update_norm = update_map.detach().flatten(1).float().norm(dim=1)
        delta_abs = delta.detach().abs().mean(dim=(1, 2))
        read_abs = memory_context_t.detach().abs().mean(dim=(1, 2, 3))
        aux = {
            "sldm_forget_mean": forget.detach().flatten(),
            "sldm_write_mean": write.detach().flatten(),
            "sldm_update_norm_mean": update_norm,
            "sldm_memory_norm_mean": memory_flat.norm(dim=1),
            "sldm_memory_norm_p95": torch.quantile(memory_flat.abs(), 0.95, dim=1).to(memory_state_t.dtype),
            "sldm_read_abs_mean": read_abs,
            "sldm_delta_abs_mean": delta_abs,
            "runtime_update_mean": write.detach().flatten(),
            "runtime_reset_mean": forget.detach().flatten(),
            "runtime_state_norm": memory_flat.norm(dim=1),
            "runtime_state_abs_mean": memory_state_t.detach().abs().mean(dim=(1, 2)),
            "runtime_state_rms": memory_state_t.detach().pow(2).mean(dim=(1, 2)).sqrt(),
            "runtime_token_abs_mean": runtime_token_t.detach().abs().mean(dim=1),
            "runtime_token_rms": runtime_token_t.detach().pow(2).mean(dim=1).sqrt(),
            "runtime_token_update_mean": token_gate.detach().mean(dim=1),
        }
        return memory_context_t, memory_state_t, runtime_token_t, aux
