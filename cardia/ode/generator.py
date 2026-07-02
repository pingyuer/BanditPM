from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory.helpers import _get_activation, _group_count, RelationEncoder


class MemoryODEGenerator(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        max_offset_px: float,
        hidden_dim: int | None = None,
        write_gate_bias: float = -0.5,
        selector_logit_scale_init: float = 2.0,
        selector_logit_scale_max: float = 8.0,
        enable_decay_gate: bool = False,
        decay_gate_bias: float = 1.5,
        stage2_bias_eps: float = 0.0,
        head_scales: list[float] | tuple[float, ...] | None = None,
        token_dim: int = 2,
        runtime_token_dim: int = 32,
        context_token_dim: int = 32,
        context_gate_init: float = 0.5,
        activation: str = "GELU",
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_offset_px = float(max_offset_px)
        self.selector_logit_scale_max = float(selector_logit_scale_max)
        self.stage2_bias_eps = float(stage2_bias_eps)
        self.enable_decay_gate = bool(enable_decay_gate)
        if head_scales is None:
            head_scales = [1.0] * self.num_heads
        scales = list(float(x) for x in head_scales)
        if len(scales) < self.num_heads:
            scales.extend([scales[-1] if scales else 1.0] * (self.num_heads - len(scales)))
        self.head_scales = tuple(scales[: self.num_heads])
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.state_norm = nn.GroupNorm(_group_count(channels), channels)
        act_cls = _get_activation(activation).__class__
        self.relation = RelationEncoder(channels, hidden_dim, activation=activation)
        hidden = self.relation.out_dim
        self.token_proj = nn.Sequential(
            nn.Conv2d(token_dim, hidden, kernel_size=1),
            act_cls(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.runtime_token_proj = nn.Sequential(
            nn.Linear(int(runtime_token_dim), hidden),
            nn.LayerNorm(hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
        )
        self.context_token_proj = nn.Sequential(
            nn.Linear(int(context_token_dim), hidden),
            nn.LayerNorm(hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
        )
        init = min(max(float(context_gate_init), 1.0e-4), 1.0 - 1.0e-4)
        self.raw_context_gate = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))
        self.offset_head = nn.Conv2d(hidden, self.num_heads * 2, kernel_size=1)
        self.spatial_selector = nn.Conv2d(hidden, self.num_heads, kernel_size=1)
        self.global_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            act_cls(),
            nn.Conv2d(hidden, self.num_heads, kernel_size=1),
        ) if self.num_heads > 1 else None
        self.write_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.decay_head = nn.Conv2d(hidden, 1, kernel_size=1) if self.enable_decay_gate else None
        self.raw_selector_logit_scale = nn.Parameter(torch.tensor(math.log(max(float(selector_logit_scale_init), 1.0e-3))))

        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.constant_(self.write_head.bias, float(write_gate_bias))
        if self.decay_head is not None:
            nn.init.constant_(self.decay_head.bias, float(decay_gate_bias))

    def _selector_scale(self) -> torch.Tensor:
        return self.raw_selector_logit_scale.exp().clamp(0.05, self.selector_logit_scale_max)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        memory_context_t: torch.Tensor,
        area_token: torch.Tensor | None = None,
        runtime_token_t: torch.Tensor | None = None,
        context_token_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = anchor_feat_t.shape
        relation = self.relation(self.anchor_norm(anchor_feat_t), self.state_norm(memory_context_t))
        if area_token is not None:
            token = area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype).view(B, -1, 1, 1)
            token = token.expand(-1, -1, H, W)
            relation = relation + self.token_proj(token)
        if runtime_token_t is not None:
            token_mod = self.runtime_token_proj(runtime_token_t.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype))
            relation = relation + token_mod[:, :, None, None]
        context_gate = torch.sigmoid(self.raw_context_gate)
        if context_token_t is not None:
            context_mod = self.context_token_proj(context_token_t.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype))
            relation = relation + context_gate * context_mod[:, :, None, None]
        raw_offsets = self.offset_head(relation).view(B, self.num_heads, 2, H, W)
        head_scales = raw_offsets.new_tensor(self.head_scales).view(1, self.num_heads, 1, 1, 1)
        offset_px = torch.tanh(raw_offsets) * self.max_offset_px * head_scales
        if self.stage2_bias_eps > 0 and self.num_heads > 1:
            pattern = torch.tensor(
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)],
                device=offset_px.device,
                dtype=offset_px.dtype,
            )
            repeats = math.ceil(self.num_heads / pattern.shape[0])
            bias = pattern.repeat(repeats, 1)[: self.num_heads] * self.stage2_bias_eps
            offset_px = offset_px + bias.view(1, self.num_heads, 2, 1, 1)
        flow_x = offset_px[:, :, 0] * (2.0 / float(W))
        flow_y = offset_px[:, :, 1] * (2.0 / float(H))
        ode_flow_t = torch.stack([flow_x, flow_y], dim=2)
        spatial_logits = self.spatial_selector(relation)
        if self.global_selector is not None:
            global_logits = self.global_selector(relation)
        else:
            global_logits = torch.zeros_like(spatial_logits[:, :, :1, :1])
        selector_scale = self._selector_scale()
        selector_map_logits = (spatial_logits + global_logits) * selector_scale
        selector_weights = torch.softmax(selector_map_logits, dim=1)
        global_selector_logits = (global_logits.flatten(1) * selector_scale)
        spatial_pooled_weights = selector_weights.mean(dim=(2, 3))
        spatial_pooled_logits = spatial_pooled_weights.clamp_min(1.0e-6).log()
        selector_scores = global_selector_logits
        write = torch.sigmoid(self.write_head(relation))
        decay = torch.sigmoid(self.decay_head(relation)) if self.decay_head is not None else torch.zeros_like(write)

        flow_dx = offset_px[:, :, :, :, 1:] - offset_px[:, :, :, :, :-1]
        flow_dy = offset_px[:, :, :, 1:, :] - offset_px[:, :, :, :-1, :]
        flow_smooth = flow_dx.abs().mean(dim=(1, 2, 3, 4)) + flow_dy.abs().mean(dim=(1, 2, 3, 4))
        weights_flat = selector_weights.flatten(2)
        entropy = -(weights_flat * weights_flat.clamp_min(1.0e-6).log()).sum(dim=1).mean(dim=-1)
        entropy = entropy / max(math.log(float(self.num_heads)), 1.0)
        usage = F.one_hot(selector_weights.argmax(dim=1), num_classes=self.num_heads).permute(0, 3, 1, 2).float().mean(dim=(2, 3))
        usage_entropy = -(usage * usage.clamp_min(1.0e-6).log()).sum(dim=1) / max(math.log(float(self.num_heads)), 1.0)
        global_top1 = torch.softmax(global_logits.flatten(1) * selector_scale, dim=1).argmax(dim=1)
        global_usage = F.one_hot(global_top1, num_classes=self.num_heads).float()
        spatial_top1 = selector_weights.argmax(dim=1)
        agreement = (spatial_top1 == global_top1[:, None, None]).float().mean(dim=(1, 2))
        global_weights = torch.softmax(global_logits.flatten(1) * selector_scale, dim=1)
        global_entropy = -(global_weights * global_weights.clamp_min(1.0e-6).log()).sum(dim=1)
        global_entropy = global_entropy / max(math.log(float(self.num_heads)), 1.0)
        offset_abs = offset_px.detach().abs()
        return {
            "relation": relation,
            "ode_flow_t": ode_flow_t,
            "offset_px": offset_px,
            "selector_weights": selector_weights,
            "global_selector_logits": global_selector_logits,
            "spatial_pooled_selector_logits": spatial_pooled_logits,
            "spatial_pooled_selector_weights": spatial_pooled_weights,
            "selector_scores": selector_scores,
            "selector_logits": global_selector_logits,
            "selector_logit_scale": selector_scale.detach().reshape(1),
            "context_gate": context_gate.detach().reshape(1),
            "write": write,
            "decay": decay,
            "flow_smooth": flow_smooth,
            "offset_px_mean": offset_abs.mean(dim=(1, 2, 3, 4)),
            "offset_px_p95": torch.quantile(offset_abs.flatten(1).float(), 0.95, dim=1).to(offset_abs.dtype),
            "write_mean": write.detach().mean(dim=(1, 2, 3)),
            "decay_mean": decay.detach().mean(dim=(1, 2, 3)),
            "global_selector_entropy": global_entropy.detach(),
            "head_entropy": entropy.detach(),
            "head_usage": usage.detach(),
            "global_head_usage": global_usage.detach(),
            "spatial_head_usage": usage.detach(),
            "global_spatial_agreement": agreement.detach(),
            "head_usage_entropy": usage_entropy.detach(),
        }
