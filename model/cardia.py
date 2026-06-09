from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _group_count(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-6)
    return math.log(math.exp(value) - 1.0)


class RelationEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.out_dim = hidden

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([lhs, rhs, lhs - rhs, lhs * rhs], dim=1))


class RuntimeMemory(nn.Module):
    """Gated runtime cardiac state, intentionally separate from anchor features."""

    def __init__(self, channels: int, hidden_dim: int | None = None, token_dim: int = 2) -> None:
        super().__init__()
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.state_norm = nn.GroupNorm(_group_count(channels), channels)
        self.relation = RelationEncoder(channels, hidden_dim)
        hidden = self.relation.out_dim
        self.token_proj = nn.Sequential(
            nn.Conv2d(token_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.reset_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.candidate = nn.Sequential(
            nn.Conv2d(hidden + channels, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        nn.init.constant_(self.update_gate.bias, -0.5)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        runtime_state_prev: torch.Tensor | None,
        area_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if runtime_state_prev is None:
            runtime_state_prev = anchor_feat_t.detach()
        if runtime_state_prev.shape[-2:] != anchor_feat_t.shape[-2:]:
            runtime_state_prev = F.interpolate(runtime_state_prev, size=anchor_feat_t.shape[-2:], mode="bilinear", align_corners=False)
        token = area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype).view(anchor_feat_t.shape[0], -1, 1, 1)
        token = token.expand(-1, -1, anchor_feat_t.shape[-2], anchor_feat_t.shape[-1])
        anchor_norm = self.anchor_norm(anchor_feat_t)
        state_norm = self.state_norm(runtime_state_prev)
        relation = self.relation(anchor_norm, state_norm) + self.token_proj(token)
        reset = torch.sigmoid(self.reset_gate(relation))
        update = torch.sigmoid(self.update_gate(relation))
        candidate = self.candidate(torch.cat([relation, reset * runtime_state_prev], dim=1))
        runtime_state_t = (1.0 - update) * runtime_state_prev + update * candidate
        aux = {
            "runtime_update_mean": update.detach().mean(dim=(1, 2, 3)),
            "runtime_reset_mean": reset.detach().mean(dim=(1, 2, 3)),
            "runtime_state_norm": runtime_state_t.detach().flatten(1).norm(dim=1),
            "runtime_state_abs_mean": runtime_state_t.detach().abs().mean(dim=(1, 2, 3)),
            "runtime_state_rms": runtime_state_t.detach().pow(2).mean(dim=(1, 2, 3)).sqrt(),
        }
        return runtime_state_t, aux


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
        token_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_offset_px = float(max_offset_px)
        self.selector_logit_scale_max = float(selector_logit_scale_max)
        self.stage2_bias_eps = float(stage2_bias_eps)
        self.enable_decay_gate = bool(enable_decay_gate)
        self.anchor_norm = nn.GroupNorm(_group_count(channels), channels)
        self.state_norm = nn.GroupNorm(_group_count(channels), channels)
        self.relation = RelationEncoder(channels, hidden_dim)
        hidden = self.relation.out_dim
        self.token_proj = nn.Sequential(
            nn.Conv2d(token_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.offset_head = nn.Conv2d(hidden, self.num_heads * 2, kernel_size=1)
        self.spatial_selector = nn.Conv2d(hidden, self.num_heads, kernel_size=1)
        self.global_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
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
        runtime_state_t: torch.Tensor,
        area_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, _, H, W = anchor_feat_t.shape
        relation = self.relation(self.anchor_norm(anchor_feat_t), self.state_norm(runtime_state_t))
        if area_token is not None:
            token = area_token.to(device=anchor_feat_t.device, dtype=anchor_feat_t.dtype).view(B, -1, 1, 1)
            token = token.expand(-1, -1, H, W)
            relation = relation + self.token_proj(token)
        raw_offsets = self.offset_head(relation).view(B, self.num_heads, 2, H, W)
        offset_px = torch.tanh(raw_offsets) * self.max_offset_px
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
            "selector_scores": selector_scores,
            "selector_logits": global_selector_logits,
            "selector_logit_scale": selector_scale.detach().reshape(1),
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
            "head_usage_entropy": usage_entropy.detach(),
        }


class GridODESolver(nn.Module):
    def __init__(self, padding_mode: str = "border", align_corners: bool = False) -> None:
        super().__init__()
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

    def _identity_grid(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        if self.align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, height, width, 2)

    def forward(self, anchor_feat_t: torch.Tensor, ode_flow_t: torch.Tensor, selector_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = anchor_feat_t.shape
        K = ode_flow_t.shape[1]
        base_grid = self._identity_grid(B, H, W, anchor_feat_t.device, anchor_feat_t.dtype)
        grid = base_grid[:, None] + ode_flow_t.permute(0, 1, 3, 4, 2)
        solved = F.grid_sample(
            anchor_feat_t[:, None].expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W),
            grid.reshape(B * K, H, W, 2),
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).view(B, K, C, H, W)
        dynamic_anchor_t = (solved * selector_weights[:, :, None]).sum(dim=1)
        return dynamic_anchor_t, solved


class DynamicAnchorFusion(nn.Module):
    def __init__(self, channels: int, gamma_init: float = 0.05) -> None:
        super().__init__()
        self.delta_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.delta_proj.bias)

    def forward(
        self,
        anchor_feat_t: torch.Tensor,
        dynamic_anchor_t: torch.Tensor,
        runtime_state_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta = self.delta_proj(dynamic_anchor_t - anchor_feat_t)
        gate = torch.sigmoid(self.gate(torch.cat([anchor_feat_t, dynamic_anchor_t, runtime_state_t], dim=1)))
        gamma = F.softplus(self.raw_gamma)
        final_feature_t = anchor_feat_t + gamma * gate * delta
        return final_feature_t, {
            "gamma": gamma.detach().reshape(1),
            "fusion_gate_mean": gate.detach().mean(dim=(1, 2, 3)),
            "delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
            "dynamic_anchor_minus_anchor_abs_mean": (dynamic_anchor_t - anchor_feat_t).detach().abs().mean(dim=(1, 2, 3)),
            "fused_minus_anchor_abs_mean": (final_feature_t - anchor_feat_t).detach().abs().mean(dim=(1, 2, 3)),
        }


class ShapeBoundaryFusion(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        skip_channels: int,
        context_channels: int,
        gamma_init: float = 0.03,
        edge_gate_floor: float = 0.05,
        edge_gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.edge_gate_floor = float(edge_gate_floor)
        self.boundary = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, groups=skip_channels),
            nn.GroupNorm(_group_count(skip_channels), skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, feature_channels, kernel_size=1),
            nn.GroupNorm(_group_count(feature_channels), feature_channels),
            nn.GELU(),
        )
        self.context_proj = nn.Conv2d(context_channels, feature_channels, kernel_size=1)
        self.delta_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.edge_gate_head = nn.Conv2d(feature_channels * 4, 1, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(gamma_init)))
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.constant_(self.edge_gate_head.bias, float(edge_gate_bias))

    def forward(
        self,
        decoder_feature_t: torch.Tensor,
        high_res_anchor_t: torch.Tensor,
        runtime_context_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary = self.boundary(high_res_anchor_t)
        context = self.context_proj(runtime_context_t)
        if boundary.shape[-2:] != decoder_feature_t.shape[-2:]:
            boundary = F.interpolate(boundary, size=decoder_feature_t.shape[-2:], mode="bilinear", align_corners=False)
        if context.shape[-2:] != decoder_feature_t.shape[-2:]:
            context = F.interpolate(context, size=decoder_feature_t.shape[-2:], mode="bilinear", align_corners=False)
        raw_delta = self.delta_proj(boundary + context - decoder_feature_t)
        delta = 0.5 * torch.tanh(raw_delta)
        gate_input = torch.cat([decoder_feature_t, boundary, context, delta], dim=1)
        edge_logit = self.edge_gate_head(gate_input)
        edge_gate = torch.sigmoid(edge_logit)
        edge_effective = self.edge_gate_floor + (1.0 - self.edge_gate_floor) * edge_gate
        channel_gate = self.channel_gate(gate_input)
        gamma = F.softplus(self.raw_gamma)
        out = decoder_feature_t + gamma * edge_effective * channel_gate * delta
        edge_flat = edge_gate.detach().flatten(1).float()
        return out, {
            "boundary_logits": edge_logit,
            "boundary_edge_gate": edge_gate,
            "boundary_edge_effective": edge_effective,
            "boundary_gamma": gamma.detach().reshape(1),
            "boundary_edge_gate_mean": edge_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_effective_mean": edge_effective.detach().mean(dim=(1, 2, 3)),
            "boundary_edge_gate_p05": torch.quantile(edge_flat, 0.05, dim=1).to(edge_gate.dtype),
            "boundary_edge_gate_p95": torch.quantile(edge_flat, 0.95, dim=1).to(edge_gate.dtype),
            "boundary_channel_gate_mean": channel_gate.detach().mean(dim=(1, 2, 3)),
            "boundary_delta_abs_mean": delta.detach().abs().mean(dim=(1, 2, 3)),
        }


class CARDIA(nn.Module):
    """Cardiac Anchor-guided Runtime Deformation Integration Architecture."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "cardia", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 120))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 256))
        self.stage3_num_heads = int(_cfg_get(method_cfg, "stage3_num_heads", 1))
        self.stage2_num_heads = int(_cfg_get(method_cfg, "stage2_num_heads", 3))
        self.stage3_max_offset_px = float(_cfg_get(method_cfg, "stage3_max_offset_px", 1.5))
        self.stage2_max_offset_px = float(_cfg_get(method_cfg, "stage2_max_offset_px", 3.0))
        self.padding_mode = str(_cfg_get(method_cfg, "padding_mode", "border"))
        self.align_corners = bool(_cfg_get(method_cfg, "align_corners", False))
        self.detach_runtime_state = bool(_cfg_get(method_cfg, "detach_runtime_state", True))
        hidden_dim = _cfg_get(method_cfg, "hidden_dim", None)
        hidden_dim = None if hidden_dim in (None, "null") else int(hidden_dim)
        write_gate_bias = float(_cfg_get(method_cfg, "write_gate_bias", -0.5))
        stage3_write_gate_bias = float(_cfg_get(method_cfg, "stage3_write_gate_bias", write_gate_bias))
        stage2_write_gate_bias = float(_cfg_get(method_cfg, "stage2_write_gate_bias", write_gate_bias))
        selector_logit_scale_init = float(_cfg_get(method_cfg, "selector_logit_scale_init", 2.0))
        selector_logit_scale_max = float(_cfg_get(method_cfg, "selector_logit_scale_max", 8.0))
        self.stage3_injection_scale = float(_cfg_get(method_cfg, "stage3_injection_scale", 0.5))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        self._load_pretrained_anchor_if_requested(method_cfg)
        self.runtime_memory3 = RuntimeMemory(self.base_dim * 4, hidden_dim)
        self.runtime_memory2 = RuntimeMemory(self.base_dim * 2, hidden_dim)
        self.ode_gen3 = MemoryODEGenerator(
            self.base_dim * 4,
            num_heads=self.stage3_num_heads,
            max_offset_px=self.stage3_max_offset_px,
            hidden_dim=hidden_dim,
            write_gate_bias=stage3_write_gate_bias,
            selector_logit_scale_init=selector_logit_scale_init,
            selector_logit_scale_max=selector_logit_scale_max,
            enable_decay_gate=bool(_cfg_get(method_cfg, "stage3_decay_gate", True)),
            decay_gate_bias=float(_cfg_get(method_cfg, "stage3_decay_gate_bias", 1.5)),
        )
        self.ode_gen2 = MemoryODEGenerator(
            self.base_dim * 2,
            num_heads=self.stage2_num_heads,
            max_offset_px=self.stage2_max_offset_px,
            hidden_dim=hidden_dim,
            write_gate_bias=stage2_write_gate_bias,
            selector_logit_scale_init=selector_logit_scale_init,
            selector_logit_scale_max=selector_logit_scale_max,
            stage2_bias_eps=float(_cfg_get(method_cfg, "stage2_head_bias_eps", 1.0e-3)),
        )
        self.grid_solver = GridODESolver(self.padding_mode, self.align_corners)
        self.fuse3 = DynamicAnchorFusion(self.base_dim * 4, gamma_init=float(_cfg_get(method_cfg, "stage3_gamma_init", 0.03)))
        self.fuse2 = DynamicAnchorFusion(self.base_dim * 2, gamma_init=float(_cfg_get(method_cfg, "stage2_gamma_init", 0.05)))
        self.proposal_head = nn.Conv2d(self.base_dim * 2, 1, kernel_size=1)
        self.boundary_fusion = ShapeBoundaryFusion(
            self.base_dim,
            self.base_dim,
            self.base_dim * 2,
            gamma_init=float(_cfg_get(method_cfg, "boundary_gamma_init", 0.03)),
            edge_gate_floor=float(_cfg_get(method_cfg, "boundary_edge_gate_floor", 0.05)),
            edge_gate_bias=float(_cfg_get(method_cfg, "boundary_edge_gate_bias", -1.0)),
        )

    def _load_pretrained_anchor_if_requested(self, method_cfg) -> None:
        path_value = _cfg_get(method_cfg, "pretrained_unext_path", None)
        require = bool(_cfg_get(method_cfg, "require_pretrained_unext", False))
        strict = bool(_cfg_get(method_cfg, "pretrained_unext_strict_backbone", False))
        if path_value in (None, "", "null"):
            if require:
                raise FileNotFoundError("model.cardia.require_pretrained_unext=true but pretrained_unext_path is empty.")
            return
        path = Path(str(path_value)).expanduser()
        if not path.exists():
            if require:
                raise FileNotFoundError(f"Pretrained UNeXt checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        target = self.backbone.state_dict()
        backbone_state = {}
        for key, value in state.items():
            clean = key[7:] if key.startswith("module.") else key
            if clean.startswith("backbone."):
                clean = clean[len("backbone."):]
            elif clean.startswith("model.backbone."):
                clean = clean[len("model.backbone."):]
            else:
                continue
            if clean in target and tuple(target[clean].shape) == tuple(value.shape):
                backbone_state[clean] = value
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict UNeXt checkpoint load failed: missing={missing}, unexpected={unexpected}")

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def _proposal_logits(self, solved_stage2: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        B, K, C, H, W = solved_stage2.shape
        logits = self.proposal_head(solved_stage2.reshape(B * K, C, H, W))
        logits = F.interpolate(logits, size=output_hw, mode="bilinear", align_corners=False)
        return logits.view(B, K, *output_hw)

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        runtime_state3 = None
        runtime_state2 = None
        prev_area = torch.zeros(B, 1, device=images.device, dtype=images.dtype)
        out: Dict = {"num_objects": num_objects}

        for ti in range(T):
            image = self._normalize(images[:, ti])
            encoded = self.backbone.encode(image)
            anchor_feat_t3 = encoded["high"]
            base_anchor_feat_t2 = self.backbone.up1(anchor_feat_t3, encoded["mid"])
            anchor_feat_t1 = encoded["low"]

            base_dec_low = self.backbone.up2(base_anchor_feat_t2, anchor_feat_t1)
            base_dec = F.interpolate(base_dec_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
            base_dec = self.backbone.full_res(base_dec)
            base_object_logits = self.backbone.logits_from_decoder_feature(base_dec)[:, 1:2].expand(-1, max_num_objects, -1, -1)
            area = torch.sigmoid(base_object_logits[:, :1]).mean(dim=(2, 3))
            area_token = torch.cat([area, area - prev_area], dim=1)
            prev_area = area.detach()

            runtime_state_t3, mem_aux3 = self.runtime_memory3(anchor_feat_t3, runtime_state3, area_token)
            ode3 = self.ode_gen3(anchor_feat_t3, runtime_state_t3, area_token)
            dynamic_anchor_t3, solved3 = self.grid_solver(anchor_feat_t3, ode3["ode_flow_t"], ode3["selector_weights"])
            final_feature_t3, fuse_aux3 = self.fuse3(anchor_feat_t3, dynamic_anchor_t3, runtime_state_t3)

            stage3_anchor_feat_t2 = self.backbone.up1(final_feature_t3, encoded["mid"])
            anchor_feat_t2 = base_anchor_feat_t2 + self.stage3_injection_scale * (stage3_anchor_feat_t2 - base_anchor_feat_t2)
            runtime_state_t2, mem_aux2 = self.runtime_memory2(anchor_feat_t2, runtime_state2, area_token)
            ode2 = self.ode_gen2(anchor_feat_t2, runtime_state_t2, area_token)
            dynamic_anchor_t2, solved2 = self.grid_solver(anchor_feat_t2, ode2["ode_flow_t"], ode2["selector_weights"])
            final_feature_t2, fuse_aux2 = self.fuse2(anchor_feat_t2, dynamic_anchor_t2, runtime_state_t2)

            dec_low = self.backbone.up2(final_feature_t2, anchor_feat_t1)
            dec = F.interpolate(dec_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
            dec = self.backbone.full_res(dec)
            dec, boundary_aux = self.boundary_fusion(dec, anchor_feat_t1, final_feature_t2)
            final_object_logits = self.backbone.logits_from_decoder_feature(dec)[:, 1:2].expand(-1, max_num_objects, -1, -1)
            logits, masks = self._object_logits_to_full(final_object_logits)

            proposal_logits = self._proposal_logits(solved2, image.shape[-2:])
            proposal_logits = proposal_logits[:, None].expand(-1, max_num_objects, -1, -1, -1)
            head_weights = torch.softmax(ode2["global_selector_logits"], dim=-1)[:, None].expand(-1, max_num_objects, -1)
            top_idx = head_weights[:, :1].argmax(dim=-1)
            top1 = proposal_logits[:, :1].gather(
                2,
                top_idx[:, :, None, None, None].expand(-1, -1, 1, proposal_logits.shape[-2], proposal_logits.shape[-1]),
            ).squeeze(2)

            if self.detach_runtime_state:
                runtime_state3 = runtime_state_t3.detach()
                runtime_state2 = runtime_state_t2.detach()
            else:
                runtime_state3 = runtime_state_t3
                runtime_state2 = runtime_state_t2

            cardia_aux = {
                "base_object_logits": base_object_logits,
                "final_object_logits": final_object_logits,
                "proposal_logits": proposal_logits,
                "proposal_top1_logits": top1.expand(-1, max_num_objects, -1, -1),
                "head_weights": head_weights,
                "selector_logits": ode2["global_selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "global_selector_logits": ode2["global_selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "selector_scores": ode2["selector_scores"][:, None].expand(-1, max_num_objects, -1),
                "boundary_logits": boundary_aux["boundary_logits"],
                "boundary_edge_gate": boundary_aux["boundary_edge_gate"],
                "runtime_state_detached": torch.tensor(float(self.detach_runtime_state), device=images.device, dtype=images.dtype),
                "stage3_flow_smooth": ode3["flow_smooth"],
                "stage3_offset_px_mean": ode3["offset_px_mean"],
                "stage3_offset_px_p95": ode3["offset_px_p95"],
                "stage3_write_mean": ode3["write_mean"],
                "stage3_decay_mean": ode3["decay_mean"],
                "stage3_gamma": fuse_aux3["gamma"],
                "stage3_fusion_gate_mean": fuse_aux3["fusion_gate_mean"],
                "stage3_dynamic_anchor_minus_anchor_abs_mean": fuse_aux3["dynamic_anchor_minus_anchor_abs_mean"],
                "stage3_fused_minus_anchor_abs_mean": fuse_aux3["fused_minus_anchor_abs_mean"],
                "stage3_injected_minus_base_abs_mean": (anchor_feat_t2 - base_anchor_feat_t2).detach().abs().mean(dim=(1, 2, 3)),
                "stage3_runtime_update_mean": mem_aux3["runtime_update_mean"],
                "stage3_runtime_reset_mean": mem_aux3["runtime_reset_mean"],
                "stage3_runtime_state_norm": mem_aux3["runtime_state_norm"],
                "stage3_runtime_state_abs_mean": mem_aux3["runtime_state_abs_mean"],
                "stage3_runtime_state_rms": mem_aux3["runtime_state_rms"],
                "stage3_head_usage": ode3["head_usage"],
                "stage3_global_selector_entropy": ode3["global_selector_entropy"],
                "stage2_flow_smooth": ode2["flow_smooth"],
                "stage2_offset_px_mean": ode2["offset_px_mean"],
                "stage2_offset_px_p95": ode2["offset_px_p95"],
                "stage2_write_mean": ode2["write_mean"],
                "stage2_decay_mean": ode2["decay_mean"],
                "stage2_gamma": fuse_aux2["gamma"],
                "stage2_dynamic_anchor_minus_anchor_abs_mean": fuse_aux2["dynamic_anchor_minus_anchor_abs_mean"],
                "stage2_fused_minus_anchor_abs_mean": fuse_aux2["fused_minus_anchor_abs_mean"],
                "stage2_selector_logit_scale": ode2["selector_logit_scale"],
                "stage2_global_selector_entropy": ode2["global_selector_entropy"],
                "stage2_head_entropy": ode2["head_entropy"],
                "stage2_head_usage": ode2["head_usage"],
                "stage2_head_usage_entropy": ode2["head_usage_entropy"],
                "stage2_runtime_update_mean": mem_aux2["runtime_update_mean"],
                "stage2_runtime_reset_mean": mem_aux2["runtime_reset_mean"],
                "stage2_runtime_state_norm": mem_aux2["runtime_state_norm"],
                "stage2_runtime_state_abs_mean": mem_aux2["runtime_state_abs_mean"],
                "stage2_runtime_state_rms": mem_aux2["runtime_state_rms"],
                "boundary_gamma": boundary_aux["boundary_gamma"],
                "boundary_edge_gate_mean": boundary_aux["boundary_edge_gate_mean"],
                "boundary_edge_effective_mean": boundary_aux["boundary_edge_effective_mean"],
                "boundary_edge_gate_p05": boundary_aux["boundary_edge_gate_p05"],
                "boundary_edge_gate_p95": boundary_aux["boundary_edge_gate_p95"],
                "boundary_channel_gate_mean": boundary_aux["boundary_channel_gate_mean"],
                "boundary_delta_abs_mean": boundary_aux["boundary_delta_abs_mean"],
                "final_minus_base_logit_abs_mean": (final_object_logits - base_object_logits).detach().abs().mean(dim=(1, 2, 3)),
            }
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": final_object_logits.detach(),
                "proposal_top1_logits": cardia_aux["proposal_top1_logits"].detach(),
            }
            out[f"memory_aux_{ti}"] = {"cardia_aux": cardia_aux}
        return out
