from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpfr.grid import flow_smoothness, grid_sample_logits, out_of_bound_ratio
from model.modules.unext import UNeXtBackbone, UNeXtOfficialBackbone


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else default


def _groups(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


def _init_near_zero(module: nn.Module, std: float) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        if std <= 0.0:
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


class DPFRTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_ratio: float, dropout: float, init_std: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        _init_near_zero(self.attn.out_proj, init_std)
        _init_near_zero(self.ffn[3], init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn_norm(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.ffn(self.ffn_norm(x))


class DPFRDualPromptEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        *,
        d_model: int,
        image_pool_hw: tuple[int, int],
        mask_pool_hw: tuple[int, int],
        prompt_scales: tuple[str, ...],
        max_time: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        init_std: float,
    ) -> None:
        super().__init__()
        self.prompt_scales = prompt_scales
        self.image_pool_hw = image_pool_hw
        self.mask_pool_hw = mask_pool_hw
        self.image_tokenizers = nn.ModuleDict(
            {
                scale: nn.Sequential(
                    nn.Conv2d(feature_dims[scale], d_model, 1, bias=False),
                    nn.GroupNorm(_groups(d_model), d_model),
                    nn.GELU(),
                    nn.Conv2d(d_model, d_model, 1),
                )
                for scale in prompt_scales
            }
        )
        self.mask_tokenizer = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(d_model), d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(d_model), d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.mask_token_value = nn.Parameter(torch.tensor(0.0))
        self.segment_embed = nn.Parameter(torch.zeros(1, 1, 3, d_model))
        self.time_embed = nn.Parameter(torch.zeros(1, max_time, 1, d_model))
        self.scale_embed = nn.Parameter(torch.zeros(1, 1, max(len(prompt_scales), 1), d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                DPFRTransformerBlock(d_model, heads, mlp_ratio, dropout, init_std)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.segment_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

    def _image_tokens(self, feats: dict[str, torch.Tensor], batch: int, time: int) -> torch.Tensor:
        tokens = []
        for idx, scale in enumerate(self.prompt_scales):
            z = self.image_tokenizers[scale](feats[scale].flatten(0, 1))
            z = F.adaptive_avg_pool2d(z, self.image_pool_hw).flatten(2).transpose(1, 2)
            z = z.view(batch, time, z.shape[1], z.shape[2])
            z = z + self.segment_embed[:, :, 1:2] + self.scale_embed[:, :, idx : idx + 1]
            tokens.append(z)
        return torch.cat(tokens, dim=2)

    def _mask_tokens(self, mask_prompt: torch.Tensor) -> torch.Tensor:
        batch, time = mask_prompt.shape[:2]
        z = self.mask_tokenizer(mask_prompt.flatten(0, 1))
        z = F.adaptive_avg_pool2d(z, self.mask_pool_hw).flatten(2).transpose(1, 2)
        z = z.view(batch, time, z.shape[1], z.shape[2])
        return z + self.segment_embed[:, :, 2:3]

    def forward(self, feats: dict[str, torch.Tensor], mask_prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time = mask_prompt.shape[:2]
        cls = self.cls_token.expand(batch, time, 1, -1) + self.segment_embed[:, :, 0:1]
        tokens = torch.cat([cls, self._image_tokens(feats, batch, time), self._mask_tokens(mask_prompt)], dim=2)
        tokens = tokens + self.time_embed[:, :time]
        tokens = self.dropout(self.input_norm(tokens))
        tokens = tokens.flatten(1, 2)
        for block in self.transformer:
            tokens = block(tokens)
        tokens = self.final_norm(tokens).view(batch, time, -1, tokens.shape[-1])
        return tokens[:, :, 0], tokens


class DPFRPromptModulator(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, int],
        *,
        d_model: int,
        prompt_injection: str,
        init_std: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.prompt_injection = prompt_injection
        self.heads = nn.ModuleDict()
        for scale, channels in feature_dims.items():
            out_dim = channels * 2 if prompt_injection == "film" else channels
            head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))
            _init_near_zero(head[-1], init_std)
            self.heads[scale] = head
        self.raw_gate = nn.ParameterDict(
            {scale: nn.Parameter(torch.tensor(float(gate_init))) for scale in feature_dims}
        )

    def forward(self, context: torch.Tensor, shapes: dict[str, tuple[int, int]]) -> tuple[dict, torch.Tensor]:
        batch, time = context.shape[:2]
        modulation = {}
        magnitudes = []
        for scale, head in self.heads.items():
            raw = head(context).flatten(0, 1)
            if self.prompt_injection == "film":
                scale_raw, shift = raw.chunk(2, dim=-1)
                scale_map = scale_raw[:, :, None, None].expand(-1, -1, *shapes[scale])
                shift_map = shift[:, :, None, None].expand(-1, -1, *shapes[scale])
                modulation[scale] = {"scale": scale_map, "shift": shift_map}
                magnitudes.append(scale_map.detach().abs().mean() + shift_map.detach().abs().mean())
            else:
                shift = raw
                if self.prompt_injection == "gated_add":
                    shift = shift * torch.sigmoid(self.raw_gate[scale])
                shift_map = shift[:, :, None, None].expand(-1, -1, *shapes[scale])
                modulation[scale] = shift_map
                magnitudes.append(shift_map.detach().abs().mean())
        mean_mag = torch.stack(magnitudes).mean() if magnitudes else context.new_tensor(0.0)
        return modulation, mean_mag


class DPFRFlowHead(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        context_channels: int,
        hidden_channels: int,
        max_disp: float,
        init_std: float,
    ) -> None:
        super().__init__()
        self.max_disp = float(max_disp)
        self.context_proj = nn.Linear(d_model, context_channels)
        self.head = nn.Sequential(
            nn.Conv2d(2 + context_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        _init_near_zero(self.head[-1], init_std)

    def forward(self, logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_time, _, height, width = logits.shape
        ctx = self.context_proj(context).view(batch_time, -1, 1, 1).expand(-1, -1, height, width)
        raw = self.head(torch.cat([logits, ctx], dim=1))
        return torch.tanh(raw) * self.max_disp


class DPFRResidualFusion(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        init_std: float,
        gate_init: float,
        max_prompt_scale: float,
        max_flow_scale: float,
    ) -> None:
        super().__init__()
        self.max_prompt_scale = float(max_prompt_scale)
        self.max_flow_scale = float(max_flow_scale)
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        _init_near_zero(self.gate[-1], init_std)
        nn.init.constant_(self.gate[-1].bias, float(gate_init))

    def forward(
        self,
        anchor_logits: torch.Tensor,
        prompt_logits: torch.Tensor,
        flow_logits: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_gate = self.gate(context).view(*context.shape[:2], 2, 1, 1, 1)
        gates = torch.sigmoid(raw_gate)
        prompt_gate = gates[:, :, 0] * self.max_prompt_scale
        flow_gate = gates[:, :, 1] * self.max_flow_scale
        prompt_delta = prompt_logits - anchor_logits
        flow_delta = flow_logits - prompt_logits
        final_logits = anchor_logits + prompt_gate * prompt_delta + flow_gate * flow_delta
        stats = {
            "prompt_gate_mean": prompt_gate.detach().mean(),
            "flow_gate_mean": flow_gate.detach().mean(),
            "final_anchor_delta_abs_mean": (final_logits - anchor_logits).detach().abs().mean(),
            "prompt_anchor_delta_abs_mean": prompt_delta.detach().abs().mean(),
            "flow_prompt_delta_abs_mean": flow_delta.detach().abs().mean(),
        }
        return final_logits, stats


class DPFRSegmenter(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "dpfr", cfg)
        backbone_cfg = _cfg_get(model_cfg, "backbone", {})
        self.in_channels = int(_cfg_get(model_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(model_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(backbone_cfg, "base_dim", _cfg_get(model_cfg, "base_dim", 96)))
        self.d_model = int(_cfg_get(model_cfg, "d_model", 192))
        self.prompt_injection = str(_cfg_get(model_cfg, "prompt_injection", "gated_add")).lower()
        if self.prompt_injection not in {"gated_add", "film", "add"}:
            raise ValueError(f"Unsupported DPFR prompt_injection: {self.prompt_injection}")
        prompt_scales = tuple(str(x).lower() for x in _cfg_get(model_cfg, "prompt_scales", ["low", "mid", "high"]))
        valid_scales = {"low", "mid", "high"}
        if not prompt_scales or any(scale not in valid_scales for scale in prompt_scales):
            raise ValueError(f"Unsupported DPFR prompt_scales: {prompt_scales}")
        self.prompt_scales = prompt_scales
        self.flow_steps = int(_cfg_get(model_cfg, "flow_steps", 1))
        self.padding_mode = str(_cfg_get(model_cfg, "padding_mode", "border"))
        self.align_corners = bool(_cfg_get(model_cfg, "align_corners", True))
        self.detach_mask_prompt = bool(_cfg_get(model_cfg, "detach_mask_prompt", True))
        self.mask_prompt_train_cfg = _cfg_get(model_cfg, "mask_prompt_train", {})
        self.mask_prompt_eval_cfg = _cfg_get(model_cfg, "mask_prompt_eval", {})

        name = str(_cfg_get(backbone_cfg, "name", "official")).lower()
        if name == "official":
            self.backbone = UNeXtOfficialBackbone(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                base_dim=self.base_dim,
                value_dim=self.d_model,
                mlp_expansion=float(_cfg_get(backbone_cfg, "mlp_expansion", 2.0)),
                latent_blocks=int(_cfg_get(backbone_cfg, "latent_blocks", 2)),
                decoder_mlp_blocks=int(_cfg_get(backbone_cfg, "decoder_mlp_blocks", 1)),
            )
        else:
            self.backbone = UNeXtBackbone(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                base_dim=self.base_dim,
                value_dim=self.d_model,
            )

        feature_dims = {"low": self.base_dim, "mid": self.base_dim * 2, "high": self.base_dim * 4}
        prompt_dims = {scale: feature_dims[scale] for scale in self.prompt_scales}
        init_std = float(_cfg_get(model_cfg, "output_init_std", 1.0e-5))
        image_pool = _cfg_get(model_cfg, "image_pool_hw", [2, 4])
        mask_pool = _cfg_get(model_cfg, "mask_pool_hw", [2, 2])
        self.prompt_encoder = DPFRDualPromptEncoder(
            feature_dims,
            d_model=self.d_model,
            image_pool_hw=(int(image_pool[0]), int(image_pool[1])),
            mask_pool_hw=(int(mask_pool[0]), int(mask_pool[1])),
            prompt_scales=self.prompt_scales,
            max_time=int(_cfg_get(model_cfg, "max_time", 32)),
            layers=int(_cfg_get(model_cfg, "temporal_layers", 4)),
            heads=int(_cfg_get(model_cfg, "temporal_heads", 6)),
            mlp_ratio=float(_cfg_get(model_cfg, "mlp_ratio", 4.0)),
            dropout=float(_cfg_get(model_cfg, "dropout", 0.1)),
            init_std=init_std,
        )
        self.prompt_heads = DPFRPromptModulator(
            prompt_dims,
            d_model=self.d_model,
            prompt_injection=self.prompt_injection,
            init_std=init_std,
            gate_init=float(_cfg_get(model_cfg, "gate_init", -2.0)),
        )
        self.flow_head = DPFRFlowHead(
            d_model=self.d_model,
            context_channels=int(_cfg_get(model_cfg, "flow_context_channels", 16)),
            hidden_channels=int(_cfg_get(model_cfg, "flow_hidden_channels", 64)),
            max_disp=float(_cfg_get(model_cfg, "max_disp", 0.05)),
            init_std=init_std,
        )
        fusion_cfg = _cfg_get(model_cfg, "final_fusion", {})
        self.final_fusion = DPFRResidualFusion(
            d_model=self.d_model,
            init_std=init_std,
            gate_init=float(_cfg_get(fusion_cfg, "gate_init", _cfg_get(model_cfg, "final_gate_init", -2.0))),
            max_prompt_scale=float(
                _cfg_get(fusion_cfg, "max_prompt_scale", _cfg_get(model_cfg, "max_prompt_residual_scale", 0.5))
            ),
            max_flow_scale=float(
                _cfg_get(fusion_cfg, "max_flow_scale", _cfg_get(model_cfg, "max_flow_residual_scale", 0.5))
            ),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _num_objects(self, data: Dict, batch: int) -> list[int]:
        info = data.get("info", {})
        values = info.get("num_objects") if isinstance(info, dict) else None
        if torch.is_tensor(values):
            return [max(int(v.item()), 1) for v in values]
        if isinstance(values, (list, tuple)):
            return [max(int(v), 1) for v in values]
        return [1] * batch

    def _gt_prob(self, data: Dict) -> float:
        cfg = self.mask_prompt_train_cfg
        start = float(_cfg_get(cfg, "gt_prob_start", 1.0))
        end = float(_cfg_get(cfg, "gt_prob_end", 0.5))
        steps = max(int(_cfg_get(cfg, "schedule_iters", 1500)), 1)
        it = int(data.get("global_step", data.get("current_iter", 0)) or 0)
        ratio = min(max(it / steps, 0.0), 1.0)
        return start + (end - start) * ratio

    def _label_valid(self, data: Dict, batch: int, time: int, device: torch.device) -> torch.Tensor:
        source = data.get("label_valid")
        if torch.is_tensor(source):
            return source.to(device=device).bool().view(batch, time)
        return torch.zeros(batch, time, device=device, dtype=torch.bool)

    def _gt_masks(self, data: Dict, batch: int, time: int, size: tuple[int, int], device: torch.device) -> torch.Tensor:
        gt = data.get("cls_gt")
        if not torch.is_tensor(gt):
            return torch.zeros(batch, time, 1, *size, device=device)
        gt = gt.to(device=device)
        if gt.dim() == 4:
            gt = gt.unsqueeze(2)
        return (gt[:, :time, :1].float() > 0).float()

    def _build_mask_prompt(self, data: Dict, anchor_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, _, height, width = anchor_logits.shape
        anchor = torch.softmax(anchor_logits, dim=2)[:, :, 1:2]
        if self.detach_mask_prompt:
            anchor = anchor.detach()
        mask_prompt = anchor
        label_valid = self._label_valid(data, batch, time, anchor_logits.device)
        use_gt = torch.zeros(batch, time, device=anchor_logits.device, dtype=torch.bool)
        use_mask = torch.zeros_like(use_gt)
        if self.training and bool(_cfg_get(self.mask_prompt_train_cfg, "use_gt", True)):
            gt_prob = self._gt_prob(data)
            mask_prob = float(_cfg_get(self.mask_prompt_train_cfg, "mask_prob", 0.1))
            rand = torch.rand(batch, time, device=anchor_logits.device)
            use_gt = label_valid & (rand < gt_prob)
            use_mask = (~use_gt) & (torch.rand(batch, time, device=anchor_logits.device) < mask_prob)
            gt_masks = self._gt_masks(data, batch, time, (height, width), anchor_logits.device)
            learned_mask = torch.sigmoid(self.prompt_encoder.mask_token_value).view(1, 1, 1, 1, 1)
            mask_prompt = torch.where(use_gt[:, :, None, None, None], gt_masks, anchor)
            mask_prompt = torch.where(use_mask[:, :, None, None, None], learned_mask.expand_as(mask_prompt), mask_prompt)
        else:
            source = str(_cfg_get(self.mask_prompt_eval_cfg, "source", "anchor_or_mask")).lower()
            if source == "mask":
                learned_mask = torch.sigmoid(self.prompt_encoder.mask_token_value).view(1, 1, 1, 1, 1)
                mask_prompt = learned_mask.expand_as(anchor)
                use_mask = torch.ones_like(use_mask)
        total = float(max(batch * time, 1))
        stats = {
            "gt_ratio": anchor_logits.new_tensor(use_gt.float().sum().item() / total),
            "mask_ratio": anchor_logits.new_tensor(use_mask.float().sum().item() / total),
            "anchor_ratio": anchor_logits.new_tensor(1.0 - (use_gt.float().sum().item() + use_mask.float().sum().item()) / total),
        }
        return mask_prompt, stats

    def _flow_refine(self, prompt_logits: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _, height, width = prompt_logits.shape
        current = prompt_logits.flatten(0, 1)
        ctx = context.flatten(0, 1)
        flow_steps = []
        for _ in range(max(self.flow_steps, 0)):
            flow = self.flow_head(current, ctx)
            current = grid_sample_logits(
                current,
                flow,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            flow_steps.append(flow.view(batch, time, 2, height, width))
        if flow_steps:
            stacked = torch.stack(flow_steps, dim=2)
            total = stacked.sum(dim=2)
        else:
            stacked = prompt_logits.new_zeros(batch, time, 0, 2, height, width)
            total = prompt_logits.new_zeros(batch, time, 2, height, width)
        return current.view(batch, time, self.num_classes, height, width), total, stacked

    def forward(self, data: Dict) -> Dict:
        video = data["rgb"]
        if video.dim() != 5:
            raise ValueError("DPFR expects batch['rgb'] with shape B,T,C,H,W.")
        batch, time, channels, height, width = video.shape
        flat = self._normalize(video.reshape(batch * time, channels, height, width))
        encoded_flat = self.backbone.encode(flat)
        decoded_anchor = self.backbone.decode(
            encoded_flat["low"], encoded_flat["mid"], encoded_flat["high"], (height, width)
        )
        anchor_logits = decoded_anchor["logits"].view(batch, time, self.num_classes, height, width)
        feats = {
            "low": encoded_flat["low"].view(batch, time, self.base_dim, *encoded_flat["low"].shape[-2:]),
            "mid": encoded_flat["mid"].view(batch, time, self.base_dim * 2, *encoded_flat["mid"].shape[-2:]),
            "high": encoded_flat["high"].view(batch, time, self.base_dim * 4, *encoded_flat["high"].shape[-2:]),
        }
        mask_prompt, mask_stats = self._build_mask_prompt(data, anchor_logits)
        context, _ = self.prompt_encoder(feats, mask_prompt)
        shapes = {scale: feats[scale].shape[-2:] for scale in self.prompt_scales}
        modulation, modulation_abs = self.prompt_heads(context, shapes)
        decoded_prompt = self.backbone.decode(
            encoded_flat["low"], encoded_flat["mid"], encoded_flat["high"], (height, width), modulation=modulation
        )
        prompt_logits = decoded_prompt["logits"].view(batch, time, self.num_classes, height, width)
        flow_logits, flow_grid, flow_steps = self._flow_refine(prompt_logits, context)
        final_logits, fusion_stats = self.final_fusion(anchor_logits, prompt_logits, flow_logits, context)

        flat_flow = flow_grid.flatten(0, 1)
        aux_summary = {
            "dpfr/flow/abs_mean": flow_grid.detach().abs().mean(),
            "dpfr/flow/abs_max": flow_grid.detach().abs().amax(),
            "dpfr/flow/smoothness": flow_smoothness(flat_flow).detach(),
            "dpfr/flow/out_of_bound_ratio": out_of_bound_ratio(flat_flow, align_corners=self.align_corners).detach(),
            "dpfr/prompt/modulation_abs_mean": modulation_abs.detach(),
            "dpfr/mask_prompt/gt_ratio": mask_stats["gt_ratio"].detach(),
            "dpfr/mask_prompt/mask_ratio": mask_stats["mask_ratio"].detach(),
            "dpfr/mask_prompt/anchor_ratio": mask_stats["anchor_ratio"].detach(),
            "dpfr/fusion/prompt_gate_mean": fusion_stats["prompt_gate_mean"],
            "dpfr/fusion/flow_gate_mean": fusion_stats["flow_gate_mean"],
            "dpfr/fusion/final_anchor_delta_abs_mean": fusion_stats["final_anchor_delta_abs_mean"],
            "dpfr/fusion/prompt_anchor_delta_abs_mean": fusion_stats["prompt_anchor_delta_abs_mean"],
            "dpfr/fusion/flow_prompt_delta_abs_mean": fusion_stats["flow_prompt_delta_abs_mean"],
        }
        out: Dict[str, torch.Tensor | dict | list[int]] = {
            "num_objects": self._num_objects(data, batch),
            "logits": final_logits,
            "final_logits": final_logits,
            "anchor_logits": anchor_logits,
            "prompt_logits": prompt_logits,
            "flow_logits": flow_logits,
            "flow_grid": flow_grid,
            "flow_steps": flow_steps,
            "aux": aux_summary,
        }
        for ti in range(time):
            out[f"logits_{ti}"] = final_logits[:, ti]
            out[f"masks_{ti}"] = torch.softmax(final_logits[:, ti], dim=1)[:, 1:]
            out[f"aux_{ti}"] = {
                "final_logits": final_logits[:, ti],
                "anchor_logits": anchor_logits[:, ti],
                "prompt_logits": prompt_logits[:, ti],
                "flow_logits": flow_logits[:, ti],
                "dpfr_aux": aux_summary,
            }
            out[f"memory_aux_{ti}"] = {"dpfr_aux": aux_summary}
        return out
