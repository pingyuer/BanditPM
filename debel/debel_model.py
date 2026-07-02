from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from debel.factorized_transformer import FactorizedVideoTransformer
from debel.grid import flow_smoothness, grid_sample_logits, out_of_bound_ratio
from debel.query_memory import BoundedGridSolver, SolverQueryDecoder, WeakBoundaryResidual
from debel.tokenizer import DEBELSpatialTokenizer
from model.modules.unext import UNeXtBackbone, UNeXtOfficialBackbone


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else default


class DEBEL(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "debel", cfg)
        backbone_cfg = _cfg_get(model_cfg, "backbone", {})
        name = str(_cfg_get(backbone_cfg, "name", "official")).lower()
        in_channels = int(_cfg_get(model_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(model_cfg, "num_classes", 2))
        base_dim = int(_cfg_get(backbone_cfg, "base_dim", _cfg_get(model_cfg, "base_dim", 96)))
        self.d_model = int(_cfg_get(model_cfg, "d_model", 192))
        if name == "official":
            self.frame_net = UNeXtOfficialBackbone(
                in_channels=in_channels,
                num_classes=self.num_classes,
                base_dim=base_dim,
                value_dim=self.d_model,
                mlp_expansion=float(_cfg_get(backbone_cfg, "mlp_expansion", 2.0)),
                latent_blocks=int(_cfg_get(backbone_cfg, "latent_blocks", 2)),
                decoder_mlp_blocks=int(_cfg_get(backbone_cfg, "decoder_mlp_blocks", 1)),
            )
        else:
            self.frame_net = UNeXtBackbone(in_channels=in_channels, num_classes=self.num_classes, base_dim=base_dim, value_dim=self.d_model)
        self.backbone = self.frame_net
        self.backbone_name = name
        self.base_dim = base_dim
        self.value_dim = self.d_model
        self.enable_temporal = bool(_cfg_get(model_cfg, "enable_temporal", True))
        self.solver_steps = int(_cfg_get(model_cfg, "solver_steps", 2))
        self.padding_mode = str(_cfg_get(model_cfg, "padding_mode", "border"))
        self.align_corners = bool(_cfg_get(model_cfg, "align_corners", True))
        self.use_residual = bool(_cfg_get(model_cfg, "use_residual", True))
        summary_tokens = int(_cfg_get(model_cfg, "summary_tokens", 4))
        token_hw = int(_cfg_get(model_cfg, "spatial_token_hw", 8))
        heads = int(_cfg_get(model_cfg, "temporal_heads", 6))
        self.tokenizer = DEBELSpatialTokenizer(self.d_model, self.d_model, token_hw, summary_tokens)
        self.video_encoder = FactorizedVideoTransformer(
            self.d_model,
            heads=heads,
            layers=int(_cfg_get(model_cfg, "temporal_layers", 4)),
            mlp_ratio=float(_cfg_get(model_cfg, "mlp_ratio", 4.0)),
            dropout=float(_cfg_get(model_cfg, "dropout", 0.1)),
            max_time=int(_cfg_get(model_cfg, "max_time", 32)),
            max_tokens=summary_tokens + token_hw * token_hw,
        )
        self.query_decoder = SolverQueryDecoder(
            self.d_model,
            d_model=self.d_model,
            solver_queries=int(_cfg_get(model_cfg, "solver_queries", 8)),
            heads=heads,
            dropout=float(_cfg_get(model_cfg, "dropout", 0.1)),
        )
        self.grid_solver = BoundedGridSolver(
            self.d_model,
            d_model=self.d_model,
            grid_head_channels=int(_cfg_get(model_cfg, "grid_head_channels", 64)),
            max_disp=float(_cfg_get(model_cfg, "max_disp", 0.05)),
        )
        self.boundary_residual = WeakBoundaryResidual(
            self.frame_net.decoder_dim,
            num_classes=self.num_classes,
            alpha_max=float(_cfg_get(model_cfg, "residual_alpha_max", 0.2)),
        )

    def forward(self, data: Dict) -> Dict:
        video = data["rgb"]
        if video.dim() != 5:
            raise ValueError("DEBEL expects batch['rgb'] with shape B,T,C,H,W.")
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        frame = self.frame_net(flat)
        anchor = frame["logits"].view(b, t, self.num_classes, h, w)
        feat = frame["high_value"].view(b, t, self.d_model, *frame["high_value"].shape[-2:])
        dec_feat = frame["decoder_feature"].view(b, t, self.frame_net.decoder_dim, h, w)
        current = anchor
        delta_steps = []
        attn_steps = []
        token_map = None
        memory = None
        if self.enable_temporal and self.solver_steps > 0:
            tokens, token_map = self.tokenizer(feat, anchor)
            memory = self.video_encoder(tokens)
            for _ in range(self.solver_steps):
                solver_state, attn = self.query_decoder(current, token_map, memory)
                delta = self.grid_solver(solver_state, token_map, current, (h, w))
                current = grid_sample_logits(
                    current.flatten(0, 1),
                    delta.flatten(0, 1),
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                ).view(b, t, self.num_classes, h, w)
                delta_steps.append(delta)
                attn_steps.append(attn)
        warped = current
        residual, alpha = self.boundary_residual(dec_feat.flatten(0, 1))
        residual = residual.view(b, t, self.num_classes, h, w)
        if self.use_residual:
            final = warped + alpha * residual
        else:
            final = warped
        out: Dict[str, torch.Tensor | dict] = {
            "logits": final,
            "anchor_logits": anchor,
            "warped_logits": warped,
            "residual_logits": residual,
            "delta_grids": torch.stack(delta_steps, dim=2) if delta_steps else anchor.new_zeros(b, t, 0, 2, h, w),
        }
        delta_total = out["delta_grids"].sum(dim=2) if delta_steps else anchor.new_zeros(b, t, 2, h, w)
        if attn_steps:
            attn_all = torch.stack(attn_steps, dim=2)
            attn_mean = attn_all.mean(dim=2)
            attn_entropy = -(attn_mean.clamp_min(1.0e-6) * attn_mean.clamp_min(1.0e-6).log()).sum(dim=-1).mean()
            attn_max = attn_mean.max(dim=-1).values.mean()
            token_usage_std = attn_mean.mean(dim=(0, 1, 2)).std(unbiased=False)
        else:
            attn_entropy = anchor.new_tensor(0.0)
            attn_max = anchor.new_tensor(0.0)
            token_usage_std = anchor.new_tensor(0.0)
        aux_summary = {
            "debel/grid/delta_abs_mean": delta_total.detach().abs().mean(),
            "debel/grid/delta_abs_max": delta_total.detach().abs().amax(),
            "debel/grid/smoothness": flow_smoothness(delta_total.flatten(0, 1)).detach(),
            "debel/grid/out_of_bound_ratio": out_of_bound_ratio(delta_total.flatten(0, 1), align_corners=self.align_corners).detach(),
            "debel/warped/warped_minus_anchor_abs_mean": (warped - anchor).detach().abs().mean(),
            "debel/residual/residual_abs_mean": residual.detach().abs().mean(),
            "debel/residual/final_minus_warped_abs_mean": (final - warped).detach().abs().mean(),
            "debel/anchor/final_minus_anchor_abs_mean": (final - anchor).detach().abs().mean(),
            "debel/query/attn_entropy": attn_entropy.detach(),
            "debel/query/attn_max": attn_max.detach(),
            "debel/query/token_usage_std": token_usage_std.detach(),
            "debel/query/query_norm": self.query_decoder.queries.detach().norm(dim=-1).mean(),
            "debel/memory/memory_norm": memory.detach().norm(dim=-1).mean() if torch.is_tensor(memory) else anchor.new_tensor(0.0),
            "debel/residual/alpha": alpha.detach(),
        }
        out["aux"] = aux_summary
        for ti in range(t):
            out[f"logits_{ti}"] = final[:, ti]
            out[f"masks_{ti}"] = torch.softmax(final[:, ti], dim=1)[:, 1:]
            out[f"aux_{ti}"] = {
                "anchor_logits": anchor[:, ti],
                "warped_logits": warped[:, ti],
                "residual_logits": residual[:, ti],
                "delta_grid": delta_total[:, ti],
                "debel_aux": aux_summary,
            }
            out[f"memory_aux_{ti}"] = {"debel_aux": aux_summary}
        return out
