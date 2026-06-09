from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.grid_anchor_router import BoundaryAwareFusion, GridAnchorRouter
from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


def _cfg_get(cfg, key: str, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class UNeXtGAR(nn.Module):
    """UNeXt decoder with GridAnchorRouter at stage3 and stage2."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "unext_gar", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.base_dim = int(_cfg_get(method_cfg, "base_dim", 32))
        self.value_dim = int(_cfg_get(method_cfg, "value_dim", 128))
        self.num_heads = int(_cfg_get(method_cfg, "num_heads", 4))
        self.stage3_num_heads = int(_cfg_get(method_cfg, "stage3_num_heads", _cfg_get(method_cfg, "stage3_heads", 1)))
        self.stage2_num_heads = int(_cfg_get(method_cfg, "stage2_num_heads", _cfg_get(method_cfg, "stage2_heads", self.num_heads)))
        self.stage3_max_offset_px = float(_cfg_get(method_cfg, "stage3_max_offset_px", _cfg_get(method_cfg, "max_offset_px", 2.0)))
        self.stage2_max_offset_px = float(_cfg_get(method_cfg, "stage2_max_offset_px", _cfg_get(method_cfg, "max_offset_px", 3.0)))
        self.padding_mode = str(_cfg_get(method_cfg, "padding_mode", "border"))
        self.align_corners = bool(_cfg_get(method_cfg, "align_corners", False))
        self.detach_state = bool(_cfg_get(method_cfg, "detach_state", True))
        self.require_pretrained_unext = bool(_cfg_get(method_cfg, "require_pretrained_unext", False))
        self.pretrained_unext_path = _cfg_get(method_cfg, "pretrained_unext_path", None)
        self.pretrained_unext_strict_backbone = bool(_cfg_get(method_cfg, "pretrained_unext_strict_backbone", False))
        hidden_dim = _cfg_get(method_cfg, "hidden_dim", None)
        hidden_dim = None if hidden_dim in (None, "null") else int(hidden_dim)
        write_gate_bias = float(_cfg_get(method_cfg, "write_gate_bias", -0.5))
        relation_norm = str(_cfg_get(method_cfg, "relation_norm", "group"))
        selector_logit_scale_init = float(_cfg_get(method_cfg, "selector_logit_scale_init", 2.0))
        selector_logit_scale_max = float(_cfg_get(method_cfg, "selector_logit_scale_max", 8.0))
        stage3_gamma_init = float(_cfg_get(method_cfg, "stage3_gamma_init", 0.03))
        stage2_gamma_init = float(_cfg_get(method_cfg, "stage2_gamma_init", 0.05))
        boundary_gamma_init = float(_cfg_get(method_cfg, "boundary_gamma_init", 0.03))
        stage3_decay_gate = bool(_cfg_get(method_cfg, "stage3_decay_gate", True))
        stage3_decay_gate_bias = float(_cfg_get(method_cfg, "stage3_decay_gate_bias", 1.5))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=self.base_dim,
            value_dim=self.value_dim,
        )
        self._load_pretrained_unext_if_requested()

        self.gar_stage3 = GridAnchorRouter(
            self.base_dim * 4,
            num_heads=self.stage3_num_heads,
            hidden_dim=hidden_dim,
            max_offset_px=self.stage3_max_offset_px,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            write_gate_bias=write_gate_bias,
            gamma_init=stage3_gamma_init,
            relation_norm=relation_norm,
            selector_logit_scale_init=selector_logit_scale_init,
            selector_logit_scale_max=selector_logit_scale_max,
            use_global_selector=False,
            enable_decay_gate=stage3_decay_gate,
            decay_gate_bias=stage3_decay_gate_bias,
        )
        self.gar_stage2 = GridAnchorRouter(
            self.base_dim * 2,
            num_heads=self.stage2_num_heads,
            hidden_dim=hidden_dim,
            max_offset_px=self.stage2_max_offset_px,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            write_gate_bias=write_gate_bias,
            gamma_init=stage2_gamma_init,
            relation_norm=relation_norm,
            selector_logit_scale_init=selector_logit_scale_init,
            selector_logit_scale_max=selector_logit_scale_max,
            use_global_selector=True,
            enable_decay_gate=False,
        )
        self.proposal_head = nn.Conv2d(self.base_dim * 2, 1, kernel_size=1)
        self.boundary_fusion = BoundaryAwareFusion(self.base_dim, self.base_dim, gamma_init=boundary_gamma_init)

    def _load_pretrained_unext_if_requested(self) -> None:
        path_value = self.pretrained_unext_path
        if path_value in (None, "", "null"):
            if self.require_pretrained_unext:
                raise FileNotFoundError(
                    "model.unext_gar.require_pretrained_unext=true but pretrained_unext_path is empty."
                )
            return
        path = Path(str(path_value)).expanduser()
        if not path.exists():
            if self.require_pretrained_unext:
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
        if self.pretrained_unext_strict_backbone and (missing or unexpected):
            raise RuntimeError(f"Strict UNeXt checkpoint load failed: missing={missing}, unexpected={unexpected}")
        if self.require_pretrained_unext and not backbone_state:
            raise RuntimeError(f"No compatible UNeXt backbone weights found in {path}")

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_logits_to_full(self, object_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits)
        logits = aggregate(masks, dim=1)
        return logits, torch.softmax(logits, dim=1)[:, 1:]

    def _proposal_logits(
        self,
        warped_stage2: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        B, K, C, H, W = warped_stage2.shape
        logits = self.proposal_head(warped_stage2.reshape(B * K, C, H, W))
        logits = F.interpolate(logits, size=output_hw, mode="bilinear", align_corners=False)
        return logits.view(B, K, *output_hw)

    def forward(self, data: Dict) -> Dict:
        images = data["rgb"]
        B, T = images.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(max(num_objects), 1)
        state_high = None
        state_mid = None
        out: Dict = {"num_objects": num_objects}

        for ti in range(T):
            image = self._normalize(images[:, ti])
            encoded = self.backbone.encode(image)

            base_decoded = self.backbone.decode(encoded["low"], encoded["mid"], encoded["high"], image.shape[-2:])
            base_object_logits = base_decoded["logits"][:, 1:2].expand(-1, max_num_objects, -1, -1)

            high, next_high, aux_high = self.gar_stage3(encoded["high"], state_high)
            dec_mid_pre = self.backbone.up1(high, encoded["mid"])
            dec_mid, next_mid, aux_mid = self.gar_stage2(dec_mid_pre, state_mid)
            dec_low = self.backbone.up2(dec_mid, encoded["low"])
            dec = F.interpolate(dec_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
            dec = self.backbone.full_res(dec)
            dec, aux_boundary = self.boundary_fusion(dec, encoded["low"])

            final_object_logits = self.backbone.logits_from_decoder_feature(dec)[:, 1:2].expand(-1, max_num_objects, -1, -1)
            logits, masks = self._object_logits_to_full(final_object_logits)

            proposal_logits = self._proposal_logits(aux_mid["warped_features"], image.shape[-2:])
            proposal_logits = proposal_logits[:, None].expand(-1, max_num_objects, -1, -1, -1)
            head_weights = torch.softmax(aux_mid["selector_logits"], dim=-1)[:, None].expand(-1, max_num_objects, -1)
            top_idx = head_weights[:, :1].argmax(dim=-1)
            top1 = proposal_logits[:, :1].gather(
                2,
                top_idx[:, :, None, None, None].expand(-1, -1, 1, proposal_logits.shape[-2], proposal_logits.shape[-1]),
            ).squeeze(2)

            if self.detach_state:
                state_high = next_high.detach()
                state_mid = next_mid.detach()
            else:
                state_high = next_high
                state_mid = next_mid

            gar_aux = {
                "base_object_logits": base_object_logits,
                "final_object_logits": final_object_logits,
                "proposal_logits": proposal_logits,
                "proposal_top1_logits": top1.expand(-1, max_num_objects, -1, -1),
                "head_weights": head_weights,
                "selector_logits": aux_mid["selector_logits"][:, None].expand(-1, max_num_objects, -1),
                "boundary_logits": aux_boundary["boundary_logits"],
                "boundary_edge_gate": aux_boundary["boundary_edge_gate"],
                "stage3_flow_smooth": aux_high["flow_smooth"],
                "stage3_offset_px_mean": aux_high["offset_px_mean"],
                "stage3_offset_px_p95": aux_high["offset_px_p95"],
                "stage3_write_mean": aux_high["write_mean"],
                "stage3_write_p05": aux_high["write_p05"],
                "stage3_write_p95": aux_high["write_p95"],
                "stage3_decay_mean": aux_high["decay_mean"],
                "stage3_gamma": aux_high["gamma"],
                "stage3_selector_logit_scale": aux_high["selector_logit_scale"],
                "stage3_head_entropy": aux_high["head_entropy"],
                "stage3_global_selector_entropy": aux_high["global_selector_entropy"],
                "stage3_head_usage": aux_high["head_usage"],
                "stage3_head_usage_entropy": aux_high["head_usage_entropy"],
                "stage3_head_usage_max": aux_high["head_usage_max"],
                "stage3_head_usage_min": aux_high["head_usage_min"],
                "stage3_head_max_weight": aux_high["head_max_weight"],
                "stage2_flow_smooth": aux_mid["flow_smooth"],
                "stage2_offset_px_mean": aux_mid["offset_px_mean"],
                "stage2_offset_px_p95": aux_mid["offset_px_p95"],
                "stage2_write_mean": aux_mid["write_mean"],
                "stage2_write_p05": aux_mid["write_p05"],
                "stage2_write_p95": aux_mid["write_p95"],
                "stage2_decay_mean": aux_mid["decay_mean"],
                "stage2_gamma": aux_mid["gamma"],
                "stage2_selector_logit_scale": aux_mid["selector_logit_scale"],
                "stage2_head_entropy": aux_mid["head_entropy"],
                "stage2_global_selector_entropy": aux_mid["global_selector_entropy"],
                "stage2_head_usage": aux_mid["head_usage"],
                "stage2_head_usage_entropy": aux_mid["head_usage_entropy"],
                "stage2_head_usage_max": aux_mid["head_usage_max"],
                "stage2_head_usage_min": aux_mid["head_usage_min"],
                "stage2_head_max_weight": aux_mid["head_max_weight"],
                "boundary_gamma": aux_boundary["boundary_gamma"],
                "boundary_gate_mean": aux_boundary["boundary_gate_mean"],
                "boundary_edge_gate_mean": aux_boundary["boundary_edge_gate_mean"],
                "boundary_edge_gate_p05": aux_boundary["boundary_edge_gate_p05"],
                "boundary_edge_gate_p95": aux_boundary["boundary_edge_gate_p95"],
                "boundary_channel_gate_mean": aux_boundary["boundary_channel_gate_mean"],
                "boundary_delta_abs_mean": aux_boundary["boundary_delta_abs_mean"],
                "boundary_raw_delta_abs_mean": aux_boundary["boundary_raw_delta_abs_mean"],
                "final_minus_base_logit_abs_mean": (final_object_logits - base_object_logits).detach().abs().mean(dim=(1, 2, 3)),
                "state_detached": torch.tensor(float(self.detach_state), device=images.device, dtype=images.dtype),
            }
            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks
            out[f"aux_{ti}"] = {
                "base_foreground_logits": base_object_logits.detach(),
                "object_logits": final_object_logits.detach(),
                "proposal_top1_logits": gar_aux["proposal_top1_logits"].detach(),
            }
            out[f"memory_aux_{ti}"] = {"gar_aux": gar_aux}
        return out
