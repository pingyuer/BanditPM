from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.unext import UNeXtBackbone
from .backbone.unext_official import UNeXtOfficialBackbone
from .memory.helpers import _cfg_get, _cfg_section
from .memory.cardiac_context import CardiacContextEncoder
from .memory.kv_memory import CardiacKVMemory
from .memory.runtime_memory import RuntimeMemory
from .memory.sldm import SelectiveLinearDeformationMemory
from .ode.generator import MemoryODEGenerator
from .ode.solver import GridODESolver
from .fusion.dynamic_anchor import DynamicAnchorFusion
from .fusion.shape_boundary import ShapeBoundaryFusion
from .fusion.cross_attention import Stage3Stage2CrossAttention
from .fusion.logit_fusion import RuntimeLogitFusion
from .memory.helpers import _get_activation
from utils.tensor_utils import aggregate


def _apply_backbone_init(module: nn.Module, strategy: str) -> None:
    strategy = str(strategy).lower()
    if strategy in ("default", ""):
        return
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if strategy == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif strategy == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif strategy == "trunc_normal":
                nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


def _detach_state(state):
    if isinstance(state, dict):
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in state.items()}
    if torch.is_tensor(state):
        return state.detach()
    return state


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
        self.memory_type = str(_cfg_get(method_cfg, "memory_type", "runtime")).lower()
        sldm_cfg = _cfg_get(method_cfg, "sldm", {})
        hidden_dim = _cfg_get(method_cfg, "hidden_dim", None)
        hidden_dim = None if hidden_dim in (None, "null") else int(hidden_dim)
        write_gate_bias = float(_cfg_get(method_cfg, "write_gate_bias", -0.5))
        stage3_write_gate_bias = float(_cfg_get(method_cfg, "stage3_write_gate_bias", write_gate_bias))
        stage2_write_gate_bias = float(_cfg_get(method_cfg, "stage2_write_gate_bias", write_gate_bias))
        selector_logit_scale_init = float(_cfg_get(method_cfg, "selector_logit_scale_init", 2.0))
        selector_logit_scale_max = float(_cfg_get(method_cfg, "selector_logit_scale_max", 8.0))
        self.stage3_injection_scale = float(_cfg_get(method_cfg, "stage3_injection_scale", 0.5))
        self.stage3_injection_learnable = bool(_cfg_get(method_cfg, "stage3_injection_learnable", True))
        self.stage3_injection_max = float(_cfg_get(method_cfg, "stage3_injection_max", 1.0))
        if self.stage3_injection_learnable:
            init = min(max(self.stage3_injection_scale / max(self.stage3_injection_max, 1.0e-6), 1.0e-4), 1.0 - 1.0e-4)
            self.raw_stage3_injection_scale = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))
        else:
            self.register_buffer("raw_stage3_injection_scale", torch.tensor(0.0), persistent=False)
        runtime_token_dim = int(_cfg_get(method_cfg, "runtime_token_dim", 32))
        self.use_cardiac_context = bool(_cfg_get(method_cfg, "use_cardiac_context", True))
        cardiac_context_detach = bool(_cfg_get(method_cfg, "cardiac_context_detach_observation", True))
        cardiac_context_hidden_dim = int(_cfg_get(method_cfg, "cardiac_context_hidden_dim", 64))
        cardiac_context_gate_init = float(_cfg_get(method_cfg, "cardiac_context_gate_init", 0.35))
        self.dynamic_context_trust_floor = float(_cfg_get(method_cfg, "dynamic_context_trust_floor", 0.35))
        runtime_logit_cfg = _cfg_get(method_cfg, "runtime_logit_fusion", {})
        self.use_runtime_logit_fusion = bool(_cfg_get(runtime_logit_cfg, "enabled", _cfg_get(method_cfg, "use_runtime_logit_fusion", True)))
        sldm_key_dim = int(_cfg_get(sldm_cfg, "key_dim", _cfg_get(method_cfg, "sldm_key_dim", 64)))
        sldm_value_dim = int(_cfg_get(sldm_cfg, "value_dim", _cfg_get(method_cfg, "sldm_value_dim", 128)))
        sldm_use_rmsnorm = bool(_cfg_get(sldm_cfg, "use_rmsnorm", True))
        sldm_zero_init = bool(_cfg_get(sldm_cfg, "zero_init", True))
        sldm_forget_bias = float(_cfg_get(sldm_cfg, "forget_bias", 1.0))
        sldm_write_bias = float(_cfg_get(sldm_cfg, "write_bias", -1.0))
        stage2_head_scales = _cfg_get(method_cfg, "stage2_head_scales", _cfg_get(method_cfg, "head_scales", [0.5, 1.0, 1.5]))
        stage3_head_scales = _cfg_get(method_cfg, "stage3_head_scales", [1.0])
        activation = str(_cfg_get(method_cfg, "activation", "GELU"))
        backbone_init = str(_cfg_get(method_cfg, "backbone_init", "default"))

        backbone_cfg = _cfg_section(method_cfg, "backbone")
        self.backbone_name = str(_cfg_get(backbone_cfg, "name", _cfg_get(method_cfg, "backbone_name", "current"))).lower()
        if self.backbone_name in {"official", "official_unext", "unext_official"}:
            official_cfg = _cfg_section(backbone_cfg, "official")
            self.backbone = UNeXtOfficialBackbone(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                base_dim=self.base_dim,
                value_dim=self.value_dim,
                mlp_expansion=float(_cfg_get(official_cfg, "mlp_expansion", 3.0)),
                latent_blocks=int(_cfg_get(official_cfg, "latent_blocks", 2)),
                decoder_mlp_blocks=int(_cfg_get(official_cfg, "decoder_mlp_blocks", 1)),
            )
        elif self.backbone_name in {"current", "lightweight", "legacy", "unext"}:
            self.backbone = UNeXtBackbone(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                base_dim=self.base_dim,
                value_dim=self.value_dim,
                activation=activation,
            )
        else:
            raise ValueError(f"Unsupported CARDIA backbone.name={self.backbone_name!r}")
        _apply_backbone_init(self.backbone, backbone_init)
        self._load_pretrained_anchor_if_requested(method_cfg)
        kv_cfg = _cfg_get(method_cfg, "kv_memory", {})
        self.deformation_source = str(_cfg_get(method_cfg, "deformation_source", "memory")).lower()
        if self.memory_type == "kv":
            self.runtime_memory3 = None
            self.runtime_memory2 = None
            self.sldm3 = None
            self.sldm2 = None
            self.kv_memory3 = CardiacKVMemory(
                self.base_dim * 4,
                key_dim=int(_cfg_get(kv_cfg, "stage3_key_dim", _cfg_get(kv_cfg, "key_dim", 64))),
                value_dim=int(_cfg_get(kv_cfg, "stage3_value_dim", _cfg_get(kv_cfg, "value_dim", self.base_dim * 2))),
                runtime_token_dim=runtime_token_dim,
                hidden_dim=int(_cfg_get(kv_cfg, "hidden_dim", 128)),
                write_bias=float(_cfg_get(kv_cfg, "write_bias", -1.0)),
                decay_bias=float(_cfg_get(kv_cfg, "decay_bias", 1.0)),
                reliability_floor=float(_cfg_get(kv_cfg, "reliability_floor", 0.05)),
                activation=activation,
            )
            self.kv_memory2 = CardiacKVMemory(
                self.base_dim * 2,
                key_dim=int(_cfg_get(kv_cfg, "stage2_key_dim", _cfg_get(kv_cfg, "key_dim", 64))),
                value_dim=int(_cfg_get(kv_cfg, "stage2_value_dim", _cfg_get(kv_cfg, "value_dim", self.base_dim))),
                runtime_token_dim=runtime_token_dim,
                hidden_dim=int(_cfg_get(kv_cfg, "hidden_dim", 128)),
                write_bias=float(_cfg_get(kv_cfg, "write_bias", -1.0)),
                decay_bias=float(_cfg_get(kv_cfg, "decay_bias", 1.0)),
                reliability_floor=float(_cfg_get(kv_cfg, "reliability_floor", 0.05)),
                activation=activation,
            )
        elif self.memory_type == "sldm":
            self.runtime_memory3 = None
            self.runtime_memory2 = None
            self.kv_memory3 = None
            self.kv_memory2 = None
            self.sldm3 = SelectiveLinearDeformationMemory(
                self.base_dim * 4,
                key_dim=sldm_key_dim,
                value_dim=sldm_value_dim,
                runtime_token_dim=runtime_token_dim,
                forget_bias=sldm_forget_bias,
                write_bias=sldm_write_bias,
                use_rmsnorm=sldm_use_rmsnorm,
                zero_init=sldm_zero_init,
            )
            self.sldm2 = SelectiveLinearDeformationMemory(
                self.base_dim * 2,
                key_dim=sldm_key_dim,
                value_dim=sldm_value_dim,
                runtime_token_dim=runtime_token_dim,
                forget_bias=sldm_forget_bias,
                write_bias=sldm_write_bias,
                use_rmsnorm=sldm_use_rmsnorm,
                zero_init=sldm_zero_init,
            )
        else:
            self.kv_memory3 = None
            self.kv_memory2 = None
            self.runtime_memory3 = RuntimeMemory(self.base_dim * 4, hidden_dim, runtime_token_dim=runtime_token_dim, activation=activation)
            self.runtime_memory2 = RuntimeMemory(self.base_dim * 2, hidden_dim, runtime_token_dim=runtime_token_dim, activation=activation)
            self.sldm3 = None
            self.sldm2 = None
        self.cardiac_context = CardiacContextEncoder(
            runtime_token_dim,
            hidden_dim=cardiac_context_hidden_dim,
            detach_observation=cardiac_context_detach,
            activation=activation,
        ) if self.use_cardiac_context else None
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
            runtime_token_dim=runtime_token_dim,
            context_token_dim=runtime_token_dim,
            context_gate_init=cardiac_context_gate_init,
            head_scales=stage3_head_scales,
            activation=activation,
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
            runtime_token_dim=runtime_token_dim,
            context_token_dim=runtime_token_dim,
            context_gate_init=cardiac_context_gate_init,
            head_scales=stage2_head_scales,
            activation=activation,
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
            activation=activation,
        )
        self.logit_fusion = RuntimeLogitFusion(
            self.base_dim,
            hidden_dim=int(_cfg_get(runtime_logit_cfg, "hidden_dim", max(self.base_dim // 2, 16))),
            init_biases=list(_cfg_get(runtime_logit_cfg, "init_biases", [1.0, 0.8, -0.2, -0.2, -0.6])),
            temperature_init=float(_cfg_get(runtime_logit_cfg, "temperature_init", 1.0)),
            temperature_min=float(_cfg_get(runtime_logit_cfg, "temperature_min", 0.35)),
            temperature_max=float(_cfg_get(runtime_logit_cfg, "temperature_max", 4.0)),
            activation=activation,
        )
        self.use_multi_head_fusion = bool(_cfg_get(method_cfg, "use_multi_head_fusion", False))
        self.use_cross_attention = bool(_cfg_get(method_cfg, "use_cross_attention", False))
        if self.use_cross_attention:
            self.cross_attn_s3s2 = Stage3Stage2CrossAttention(
                self.base_dim * 2,
                num_heads=int(_cfg_get(method_cfg, "cross_attn_num_heads", 4)),
                gamma_init=float(_cfg_get(method_cfg, "cross_attn_gamma_init", 0.1)),
                dropout=float(_cfg_get(method_cfg, "cross_attn_dropout", 0.1)),
            )

    def _stage3_injection_scale(self) -> torch.Tensor:
        if self.stage3_injection_learnable:
            return torch.sigmoid(self.raw_stage3_injection_scale) * self.stage3_injection_max
        return self.raw_stage3_injection_scale.new_tensor(self.stage3_injection_scale)

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
        memory_state3 = None
        memory_state2 = None
        runtime_token3 = None
        runtime_token2 = None
        cardiac_context_token = None
        cardiac_context_observation = None
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
            if self.cardiac_context is not None:
                cardiac_context_token_t, cardiac_context_observation_t, context_aux = self.cardiac_context(
                    base_object_logits[:, :1],
                    cardiac_context_token,
                    cardiac_context_observation,
                )
                uncertainty = context_aux["context_uncertainty"].to(device=images.device, dtype=images.dtype)
                dynamic_context_trust = self.dynamic_context_trust_floor + (1.0 - self.dynamic_context_trust_floor) * (1.0 - 4.0 * uncertainty).clamp(0.0, 1.0)
            else:
                cardiac_context_token_t = None
                cardiac_context_observation_t = None
                context_aux = {}
                dynamic_context_trust = torch.ones(B, device=images.device, dtype=images.dtype)

            if self.memory_type == "sldm":
                memory_context_t3, memory_state_t3, runtime_token_t3, mem_aux3 = self.sldm3(anchor_feat_t3, memory_state3, area_token, runtime_token3)
            elif self.memory_type == "kv":
                memory_context_t3, memory_state_t3, runtime_token_t3, mem_aux3 = self.kv_memory3(
                    anchor_feat_t3,
                    memory_state3,
                    base_object_logits[:, :1],
                    area_token,
                    runtime_token3,
                )
            else:
                memory_context_t3, runtime_token_t3, mem_aux3 = self.runtime_memory3(anchor_feat_t3, memory_state3, area_token, runtime_token3)
                memory_state_t3 = memory_context_t3
            ode3 = self.ode_gen3(anchor_feat_t3, memory_context_t3, area_token, runtime_token_t3, cardiac_context_token_t)
            source_feat_t3 = memory_context_t3 if self.deformation_source == "memory" else anchor_feat_t3
            dynamic_anchor_t3, solved3, solver_aux3 = self.grid_solver(source_feat_t3, ode3["ode_flow_t"], ode3["selector_weights"])
            stage3_trust = mem_aux3.get("memory_reliability", dynamic_context_trust)
            final_feature_t3, fuse_aux3 = self.fuse3(anchor_feat_t3, dynamic_anchor_t3, memory_context_t3, stage3_trust)

            stage3_anchor_feat_t2 = self.backbone.up1(final_feature_t3, encoded["mid"])
            stage3_injection_scale = self._stage3_injection_scale().to(device=base_anchor_feat_t2.device, dtype=base_anchor_feat_t2.dtype)
            anchor_feat_t2 = base_anchor_feat_t2 + stage3_injection_scale * (stage3_anchor_feat_t2 - base_anchor_feat_t2)
            if self.use_cross_attention:
                cross_residual, cross_aux = self.cross_attn_s3s2(base_anchor_feat_t2, stage3_anchor_feat_t2)
                anchor_feat_t2 = anchor_feat_t2 + cross_residual
            else:
                cross_aux = None
            if self.memory_type == "sldm":
                memory_context_t2, memory_state_t2, runtime_token_t2, mem_aux2 = self.sldm2(anchor_feat_t2, memory_state2, area_token, runtime_token2)
            elif self.memory_type == "kv":
                memory_context_t2, memory_state_t2, runtime_token_t2, mem_aux2 = self.kv_memory2(
                    anchor_feat_t2,
                    memory_state2,
                    base_object_logits[:, :1],
                    area_token,
                    runtime_token2,
                )
            else:
                memory_context_t2, runtime_token_t2, mem_aux2 = self.runtime_memory2(anchor_feat_t2, memory_state2, area_token, runtime_token2)
                memory_state_t2 = memory_context_t2
            ode2 = self.ode_gen2(anchor_feat_t2, memory_context_t2, area_token, runtime_token_t2, cardiac_context_token_t)
            source_feat_t2 = memory_context_t2 if self.deformation_source == "memory" else anchor_feat_t2
            dynamic_anchor_t2, solved2, solver_aux2 = self.grid_solver(source_feat_t2, ode2["ode_flow_t"], ode2["selector_weights"])
            stage2_trust = mem_aux2.get("memory_reliability", dynamic_context_trust)
            final_feature_t2, fuse_aux2 = self.fuse2(anchor_feat_t2, dynamic_anchor_t2, memory_context_t2, stage2_trust)

            dec_low = self.backbone.up2(final_feature_t2, anchor_feat_t1)
            dec = F.interpolate(dec_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
            dec = self.backbone.full_res(dec)
            proposal_logits = self._proposal_logits(solved2, image.shape[-2:])
            proposal_logits = proposal_logits[:, None].expand(-1, max_num_objects, -1, -1, -1)
            head_weights = torch.softmax(ode2["global_selector_logits"], dim=-1)[:, None].expand(-1, max_num_objects, -1)
            top_idx = head_weights[:, :1].argmax(dim=-1)
            top1 = proposal_logits[:, :1].gather(
                2,
                top_idx[:, :, None, None, None].expand(-1, -1, 1, proposal_logits.shape[-2], proposal_logits.shape[-1]),
            ).squeeze(2)
            spatial_head_weights = ode2["spatial_pooled_selector_weights"][:, None].expand(-1, max_num_objects, -1)
            spatial_top_idx = spatial_head_weights[:, :1].argmax(dim=-1)
            spatial_top1 = proposal_logits[:, :1].gather(
                2,
                spatial_top_idx[:, :, None, None, None].expand(-1, -1, 1, proposal_logits.shape[-2], proposal_logits.shape[-1]),
            ).squeeze(2)
            mixture_proxy = (proposal_logits[:, :1] * spatial_head_weights[:, :1, :, None, None]).sum(dim=2)
            if self.use_multi_head_fusion:
                mhf_weights = head_weights[:, :, :, None, None]
                multi_head_fused = (proposal_logits * mhf_weights).sum(dim=2)
            dec, boundary_aux = self.boundary_fusion(dec, anchor_feat_t1, final_feature_t2)
            dynamic_decoder_logits = self.backbone.logits_from_decoder_feature(dec)[:, 1:2].expand(-1, max_num_objects, -1, -1)
            memory_prior = mem_aux2.get("memory_mask_prior_logits")
            if torch.is_tensor(memory_prior):
                memory_prior = F.interpolate(memory_prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
            else:
                memory_prior = torch.zeros(B, 1, *image.shape[-2:], device=images.device, dtype=images.dtype)
            memory_prior = memory_prior.expand(-1, max_num_objects, -1, -1)
            if self.use_runtime_logit_fusion:
                final_object_logits, logit_fusion_aux = self.logit_fusion(
                    dec,
                    dynamic_decoder_logits,
                    base_object_logits,
                    top1.expand(-1, max_num_objects, -1, -1),
                    mixture_proxy.expand(-1, max_num_objects, -1, -1),
                    memory_prior,
                )
            else:
                final_object_logits = dynamic_decoder_logits
                zeros_b = torch.zeros(B, device=images.device, dtype=images.dtype)
                logit_fusion_aux = {
                    "logit_fusion_temperature": torch.ones(1, device=images.device, dtype=images.dtype),
                    "logit_fusion_entropy": zeros_b,
                    "logit_fusion_fused_minus_base_abs_mean": (final_object_logits - base_object_logits).detach().abs().mean(dim=(1, 2, 3)),
                    "logit_fusion_fused_minus_dynamic_abs_mean": zeros_b,
                    "logit_fusion_weight_dynamic": torch.ones(B, device=images.device, dtype=images.dtype),
                    "logit_fusion_weight_base": zeros_b,
                    "logit_fusion_weight_proposal_top1": zeros_b,
                    "logit_fusion_weight_proposal_mixture": zeros_b,
                    "logit_fusion_weight_memory_prior": zeros_b,
                }
            logits, masks = self._object_logits_to_full(final_object_logits)

            if self.detach_runtime_state:
                memory_state3 = _detach_state(memory_state_t3)
                memory_state2 = _detach_state(memory_state_t2)
                runtime_token3 = runtime_token_t3.detach()
                runtime_token2 = runtime_token_t2.detach()
                cardiac_context_token = cardiac_context_token_t.detach() if cardiac_context_token_t is not None else None
                cardiac_context_observation = cardiac_context_observation_t.detach() if cardiac_context_observation_t is not None else None
            else:
                memory_state3 = memory_state_t3
                memory_state2 = memory_state_t2
                runtime_token3 = runtime_token_t3
                runtime_token2 = runtime_token_t2
                cardiac_context_token = cardiac_context_token_t
                cardiac_context_observation = cardiac_context_observation_t

            cardia_aux = {
                "base_object_logits": base_object_logits,
                "dynamic_decoder_logits": dynamic_decoder_logits,
                "final_object_logits": final_object_logits,
                "proposal_logits": proposal_logits,
                "proposal_top1_logits": top1.expand(-1, max_num_objects, -1, -1),
                "proposal_spatial_top1_logits": spatial_top1.expand(-1, max_num_objects, -1, -1),
                "proposal_mixture_proxy_logits": mixture_proxy.expand(-1, max_num_objects, -1, -1),
                "head_weights": head_weights,
                "spatial_head_weights": spatial_head_weights,
                "selector_logits": ode2["global_selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "global_selector_logits": ode2["global_selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "spatial_pooled_selector_logits": ode2["spatial_pooled_selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "spatial_pooled_selector_weights": spatial_head_weights,
                "selector_scores": ode2["selector_scores"][:, None].expand(-1, max_num_objects, -1),
                "boundary_logits": boundary_aux["boundary_logits"],
                "boundary_edge_gate": boundary_aux["boundary_edge_gate"],
                "boundary_edge_effective": boundary_aux["boundary_edge_effective"],
                "boundary_delta_map": boundary_aux["boundary_delta_map"],
                "multi_head_fused_logits": multi_head_fused.expand(-1, max_num_objects, -1, -1) if self.use_multi_head_fusion else mixture_proxy.expand(-1, max_num_objects, -1, -1),
                "memory_prior_logits": memory_prior,
                "runtime_logit_fusion_enabled": torch.tensor(float(self.use_runtime_logit_fusion), device=images.device, dtype=images.dtype),
                **logit_fusion_aux,
                "use_multi_head_fusion": torch.tensor(float(self.use_multi_head_fusion), device=images.device, dtype=images.dtype),
                "cross_attn_entropy": cross_aux["cross_attn_entropy"] if self.use_cross_attention else torch.zeros(1, device=images.device, dtype=images.dtype),
                "cross_attn_gamma": cross_aux["cross_attn_gamma"] if self.use_cross_attention else torch.zeros(1, device=images.device, dtype=images.dtype),
                "cross_attn_weight_std": cross_aux["cross_attn_weight_std"] if self.use_cross_attention else torch.zeros(1, device=images.device, dtype=images.dtype),
                "cross_attn_residual_abs_mean": cross_aux["cross_attn_residual_abs_mean"] if self.use_cross_attention else torch.zeros(B, device=images.device, dtype=images.dtype),
                "runtime_state_detached": torch.tensor(float(self.detach_runtime_state), device=images.device, dtype=images.dtype),
                "memory_type_kv": torch.tensor(float(self.memory_type == "kv"), device=images.device, dtype=images.dtype),
                "deformation_source_memory": torch.tensor(float(self.deformation_source == "memory"), device=images.device, dtype=images.dtype),
                "context_enabled": torch.tensor(float(self.cardiac_context is not None), device=images.device, dtype=images.dtype),
                "context_area": context_aux.get("context_area", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_delta_area": context_aux.get("context_delta_area", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_centroid_x": context_aux.get("context_centroid_x", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_centroid_y": context_aux.get("context_centroid_y", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_delta_centroid_abs": context_aux.get("context_delta_centroid_abs", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_scale_x": context_aux.get("context_scale_x", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_scale_y": context_aux.get("context_scale_y", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_boundary_energy": context_aux.get("context_boundary_energy", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_uncertainty": context_aux.get("context_uncertainty", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_token_rms": context_aux.get("context_token_rms", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "context_update_mean": context_aux.get("context_update_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "dynamic_context_trust": dynamic_context_trust.detach(),
                "stage3_flow_smooth": ode3["flow_smooth"],
                "stage3_offset_px_mean": ode3["offset_px_mean"],
                "stage3_offset_px_p95": ode3["offset_px_p95"],
                "stage3_grid_oob_ratio": solver_aux3["grid_oob_ratio"],
                "stage3_write_mean": ode3["write_mean"],
                "stage3_decay_mean": ode3["decay_mean"],
                "stage3_context_gate": ode3["context_gate"],
                "stage3_gamma": fuse_aux3["gamma"],
                "stage3_dynamic_trust_mean": fuse_aux3["dynamic_trust_mean"],
                "stage3_fusion_gate_mean": fuse_aux3["fusion_gate_mean"],
                "stage3_fusion_gate_p05": fuse_aux3["fusion_gate_p05"],
                "stage3_fusion_gate_p95": fuse_aux3["fusion_gate_p95"],
                "stage3_dynamic_anchor_minus_anchor_abs_mean": fuse_aux3["dynamic_anchor_minus_anchor_abs_mean"],
                "stage3_fused_minus_anchor_abs_mean": fuse_aux3["fused_minus_anchor_abs_mean"],
                "stage3_injected_minus_base_abs_mean": (anchor_feat_t2 - base_anchor_feat_t2).detach().abs().mean(dim=(1, 2, 3)),
                "stage3_injection_scale": stage3_injection_scale.detach().reshape(1),
                "stage3_runtime_update_mean": mem_aux3["runtime_update_mean"],
                "stage3_runtime_reset_mean": mem_aux3["runtime_reset_mean"],
                "stage3_runtime_state_norm": mem_aux3["runtime_state_norm"],
                "stage3_runtime_state_abs_mean": mem_aux3["runtime_state_abs_mean"],
                "stage3_runtime_state_rms": mem_aux3["runtime_state_rms"],
                "stage3_runtime_token_abs_mean": mem_aux3["runtime_token_abs_mean"],
                "stage3_runtime_token_rms": mem_aux3["runtime_token_rms"],
                "stage3_runtime_token_update_mean": mem_aux3["runtime_token_update_mean"],
                "stage3_memory_reliability": mem_aux3.get("memory_reliability", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_write_mean": mem_aux3.get("memory_write_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_decay_mean": mem_aux3.get("memory_decay_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_read_gate_mean": mem_aux3.get("memory_read_gate_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_current_agreement": mem_aux3.get("memory_current_agreement", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_boundary_quality": mem_aux3.get("memory_boundary_quality", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_area_ok": mem_aux3.get("memory_area_ok", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_readout_abs_mean": mem_aux3.get("memory_readout_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_memory_mask_prior_logits": mem_aux3.get("memory_mask_prior_logits", torch.zeros(B, 1, *anchor_feat_t3.shape[-2:], device=images.device, dtype=images.dtype)),
                "stage3_delta_proj_abs_mean": fuse_aux3["delta_abs_mean"],
                "stage3_sldm_memory_norm_mean": mem_aux3.get("sldm_memory_norm_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_memory_norm_p95": mem_aux3.get("sldm_memory_norm_p95", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_update_norm_mean": mem_aux3.get("sldm_update_norm_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_forget_mean": mem_aux3.get("sldm_forget_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_write_mean": mem_aux3.get("sldm_write_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_read_abs_mean": mem_aux3.get("sldm_read_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_sldm_delta_abs_mean": mem_aux3.get("sldm_delta_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage3_head_usage": ode3["head_usage"],
                "stage3_global_selector_entropy": ode3["global_selector_entropy"],
                "stage3_global_head_usage": ode3["global_head_usage"],
                "stage3_spatial_head_usage": ode3["spatial_head_usage"],
                "stage3_global_spatial_agreement": ode3["global_spatial_agreement"],
                "stage2_flow_smooth": ode2["flow_smooth"],
                "stage2_offset_px_mean": ode2["offset_px_mean"],
                "stage2_offset_px_p95": ode2["offset_px_p95"],
                "stage2_grid_oob_ratio": solver_aux2["grid_oob_ratio"],
                "stage2_write_mean": ode2["write_mean"],
                "stage2_decay_mean": ode2["decay_mean"],
                "stage2_context_gate": ode2["context_gate"],
                "stage2_gamma": fuse_aux2["gamma"],
                "stage2_dynamic_trust_mean": fuse_aux2["dynamic_trust_mean"],
                "stage2_fusion_gate_mean": fuse_aux2["fusion_gate_mean"],
                "stage2_fusion_gate_p05": fuse_aux2["fusion_gate_p05"],
                "stage2_fusion_gate_p95": fuse_aux2["fusion_gate_p95"],
                "stage2_dynamic_anchor_minus_anchor_abs_mean": fuse_aux2["dynamic_anchor_minus_anchor_abs_mean"],
                "stage2_fused_minus_anchor_abs_mean": fuse_aux2["fused_minus_anchor_abs_mean"],
                "stage2_delta_proj_abs_mean": fuse_aux2["delta_abs_mean"],
                "stage2_selector_logit_scale": ode2["selector_logit_scale"],
                "stage2_global_selector_entropy": ode2["global_selector_entropy"],
                "stage2_head_entropy": ode2["head_entropy"],
                "stage2_head_usage": ode2["head_usage"],
                "stage2_global_head_usage": ode2["global_head_usage"],
                "stage2_spatial_head_usage": ode2["spatial_head_usage"],
                "stage2_global_spatial_agreement": ode2["global_spatial_agreement"],
                "stage2_head_usage_entropy": ode2["head_usage_entropy"],
                "stage2_runtime_update_mean": mem_aux2["runtime_update_mean"],
                "stage2_runtime_reset_mean": mem_aux2["runtime_reset_mean"],
                "stage2_runtime_state_norm": mem_aux2["runtime_state_norm"],
                "stage2_runtime_state_abs_mean": mem_aux2["runtime_state_abs_mean"],
                "stage2_runtime_state_rms": mem_aux2["runtime_state_rms"],
                "stage2_runtime_token_abs_mean": mem_aux2["runtime_token_abs_mean"],
                "stage2_runtime_token_rms": mem_aux2["runtime_token_rms"],
                "stage2_runtime_token_update_mean": mem_aux2["runtime_token_update_mean"],
                "stage2_memory_reliability": mem_aux2.get("memory_reliability", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_write_mean": mem_aux2.get("memory_write_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_decay_mean": mem_aux2.get("memory_decay_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_read_gate_mean": mem_aux2.get("memory_read_gate_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_current_agreement": mem_aux2.get("memory_current_agreement", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_boundary_quality": mem_aux2.get("memory_boundary_quality", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_area_ok": mem_aux2.get("memory_area_ok", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_readout_abs_mean": mem_aux2.get("memory_readout_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_memory_mask_prior_logits": mem_aux2.get("memory_mask_prior_logits", torch.zeros(B, 1, *anchor_feat_t2.shape[-2:], device=images.device, dtype=images.dtype)),
                "stage2_sldm_memory_norm_mean": mem_aux2.get("sldm_memory_norm_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_memory_norm_p95": mem_aux2.get("sldm_memory_norm_p95", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_update_norm_mean": mem_aux2.get("sldm_update_norm_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_forget_mean": mem_aux2.get("sldm_forget_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_write_mean": mem_aux2.get("sldm_write_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_read_abs_mean": mem_aux2.get("sldm_read_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
                "stage2_sldm_delta_abs_mean": mem_aux2.get("sldm_delta_abs_mean", torch.zeros(B, device=images.device, dtype=images.dtype)),
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
