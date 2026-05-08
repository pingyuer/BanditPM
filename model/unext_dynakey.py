from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.memory_core import MemoryCore
from model.modules.unext import UNeXtBackbone
from utils.tensor_utils import aggregate


class UNeXtDynaKeySegmenter(nn.Module):
    """Video segmenter using UNeXt per-frame features and DynaKey latent dynamics."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("unext_dynakey", {})
        self.use_first_frame_gt_init = bool(cfg.get("use_first_frame_gt_init", True))
        self.use_temporal_refine = bool(model_cfg.get("use_temporal_refine", True))
        self.value_dim = int(model_cfg.get("value_dim", 256))
        self.num_classes = int(model_cfg.get("num_classes", 2))
        self.in_channels = int(model_cfg.get("in_channels", 1))
        base_dim = int(model_cfg.get("base_dim", 32))

        self.backbone = UNeXtBackbone(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_dim=base_dim,
            value_dim=self.value_dim,
        )
        self.memory_core = MemoryCore(
            value_dim=self.value_dim,
            key_dim=self.value_dim,
            prototype_value_cfg=None,
            temporal_memory_cfg=cfg.get("temporal_memory", None),
            memory_core_cfg=cfg.get("memory_core", None),
        )
        dec_dim = self.backbone.decoder_dim
        self.temporal_delta_proj = nn.Conv2d(self.value_dim, dec_dim, kernel_size=1)
        self.temporal_refine_head = nn.Sequential(
            nn.Conv2d(dec_dim * 2 + 1, dec_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_dim, 1, kernel_size=1),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - 0.5) / 0.5

    def _object_value(self, value_BCHW: torch.Tensor, num_objects: int) -> torch.Tensor:
        return value_BCHW.unsqueeze(1).expand(-1, num_objects, -1, -1, -1).contiguous()

    def _refine_object_logits(
        self,
        base_logits_B2HW: torch.Tensor,
        decoder_BCHW: torch.Tensor,
        temporal_delta_BNCHW: torch.Tensor,
        mask_BNHW: torch.Tensor,
    ) -> torch.Tensor:
        B, N = mask_BNHW.shape[:2]
        fg_base = base_logits_B2HW[:, 1:2].expand(-1, N, -1, -1)
        if not self.use_temporal_refine:
            return fg_base

        temporal = self.temporal_delta_proj(temporal_delta_BNCHW.flatten(0, 1))
        temporal = F.interpolate(temporal, size=decoder_BCHW.shape[-2:], mode="bilinear", align_corners=False)
        decoder = decoder_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(0, 1)
        soft_mask = mask_BNHW.float()
        soft_mask = F.interpolate(soft_mask.flatten(0, 1).unsqueeze(1), size=decoder_BCHW.shape[-2:], mode="bilinear", align_corners=False)
        temporal = temporal * soft_mask
        refine_in = torch.cat([decoder, temporal, soft_mask], dim=1)
        residual = self.temporal_refine_head(refine_in).view(B, N, *decoder_BCHW.shape[-2:])
        return fg_base + residual

    def _aggregate_logits(self, object_logits_BNHW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.sigmoid(object_logits_BNHW)
        logits = aggregate(masks, dim=1)
        masks_all = torch.softmax(logits, dim=1)
        return logits, masks_all

    def forward(self, data: Dict) -> Dict:
        images_BTCHW = data["rgb"]
        B, T = images_BTCHW.shape[:2]
        num_objects = [int(x.item()) for x in data["info"]["num_objects"]]
        max_num_objects = max(num_objects)
        out: Dict = {"num_objects": num_objects}
        self.memory_core.reset_state(B, max_num_objects, images_BTCHW.device)
        policy_meta = {
            "current_iter": int(data.get("current_iter", 0)),
            "current_epoch": int(data.get("current_epoch", 0)),
            "training": bool(self.training),
        }

        init_mode = str(data.get("init_mode", "oracle_gt"))
        init_mask = data["ff_gt"][:, 0, :max_num_objects].float()
        if init_mode != "oracle_gt" or not self.use_first_frame_gt_init:
            init_mask = torch.zeros_like(init_mask)
        last_masks = init_mask

        for ti in range(T):
            frame = self._normalize(images_BTCHW[:, ti])
            feat = self.backbone(frame)
            value_BNCHW = self._object_value(feat["value"], max_num_objects)
            readout_BNCHW, memory_aux = self.memory_core(
                value_BNCHW=value_BNCHW,
                key_BCHW=feat["high_value"],
                pixfeat_BCHW=feat["value"],
                mask_BNHW=last_masks,
                policy_meta=policy_meta,
            )
            temporal_delta = readout_BNCHW - value_BNCHW
            object_logits = self._refine_object_logits(
                feat["logits"],
                feat["decoder_feature"],
                temporal_delta,
                last_masks,
            )
            logits, masks_all = self._aggregate_logits(object_logits)
            last_masks = masks_all[:, 1:].detach()

            out[f"logits_{ti}"] = logits
            out[f"masks_{ti}"] = masks_all[:, 1:]
            out[f"aux_{ti}"] = {}
            out[f"memory_aux_{ti}"] = memory_aux
            if "dynakey_aux" in memory_aux:
                memory_aux["dynakey_aux"]["temporal_delta_norm"] = temporal_delta.detach().pow(2).mean(dim=(2, 3, 4))
        return out
