from __future__ import annotations

import torch
import torch.nn as nn

from model.modules.unext import UNeXtBackbone, UNeXtOfficialBackbone
from rebel.ode_field import _groups


class ObservationEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        backbone_cfg = cfg.get("backbone", {})
        name = str(backbone_cfg.get("name", "official")).lower()
        in_channels = int(cfg.get("in_channels", 1))
        num_classes = int(cfg.get("num_classes", 2))
        base_dim = int(backbone_cfg.get("base_dim", cfg.get("base_dim", 120)))
        belief_dim = int(cfg.get("belief_dim", 256))
        if name == "official":
            official = backbone_cfg.get("official", backbone_cfg)
            self.backbone = UNeXtOfficialBackbone(
                in_channels=in_channels,
                num_classes=num_classes,
                base_dim=base_dim,
                value_dim=belief_dim,
                mlp_expansion=float(official.get("mlp_expansion", 2.0)),
                latent_blocks=int(official.get("latent_blocks", 2)),
                decoder_mlp_blocks=int(official.get("decoder_mlp_blocks", 1)),
            )
        else:
            self.backbone = UNeXtBackbone(in_channels=in_channels, num_classes=num_classes, base_dim=base_dim, value_dim=belief_dim)
        self.low_dim = base_dim
        self.mid_dim = base_dim * 2
        self.high_dim = base_dim * 4
        self.obs_project = nn.Sequential(
            nn.Conv2d(self.high_dim, belief_dim, 1, bias=False),
            nn.GroupNorm(_groups(belief_dim), belief_dim),
            nn.SiLU(),
        )

    def forward(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(frame)
        feats["obs"] = self.obs_project(feats["high"])
        feats["base_logits"] = feats["logits"]
        return feats
