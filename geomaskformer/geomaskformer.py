from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_section(cfg: Any, key: str) -> Any:
    return _cfg_get(cfg, key, {}) or {}


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageTokenizer(nn.Module):
    """Boundary-aware image tokenizer for echocardiography frames."""

    def __init__(self, in_channels: int = 1, dim: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        self.dim = int(dim)
        c2 = int(base_channels)
        c4 = int(base_channels) * 2
        c8 = int(base_channels) * 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.down2 = DepthwiseSeparableConv(c2, c2, stride=2)
        self.local_contrast = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.high_freq = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        self.down4 = DepthwiseSeparableConv(c2, c4, stride=2)
        self.down8 = DepthwiseSeparableConv(c4, c8, stride=2)
        self.patch_proj = nn.Conv2d(c8, dim, 1)
        self.out_channels = {"f2": c2, "f4": c4, "f8": c8, "tokens": dim}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        f1 = self.conv1(x)
        f2_base = self.down2(f1)
        contrast = f2_base - F.avg_pool2d(f2_base, kernel_size=5, stride=1, padding=2)
        high = self.high_freq(f2_base) - F.avg_pool2d(self.high_freq(f2_base), kernel_size=3, stride=1, padding=1)
        f2 = f2_base + 0.25 * self.local_contrast(contrast) + 0.25 * high
        f4 = self.down4(f2)
        f8 = self.down8(f4)
        token_map = self.patch_proj(f8)
        _, _, hp, wp = token_map.shape
        tokens = token_map.flatten(2).transpose(1, 2).reshape(b, t, hp * wp, self.dim)
        return {"tokens": tokens, "token_map": token_map, "f2": f2, "f4": f4, "f8": f8, "grid_size": (hp, wp)}


def _soft_boundary(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def _local_geometry_prior(mask: torch.Tensor) -> torch.Tensor:
    inside = mask
    outside = 1.0 - mask
    smooth_inside = F.avg_pool2d(inside, 9, stride=1, padding=4)
    smooth_outside = F.avg_pool2d(outside, 9, stride=1, padding=4)
    signed = smooth_inside - smooth_outside
    return signed.clamp(-1.0, 1.0)


class MaskTokenizer(nn.Module):
    """Geometry-aware mask tokenizer with a learnable [MASK] token.

    The middle geometry channel is a bounded local inside/outside prior, not a
    true Euclidean signed distance transform.
    """

    def __init__(self, dim: int = 128, max_frames: int = 16, max_grid: int = 64, use_geometry: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_frames = int(max_frames)
        self.max_grid = int(max_grid)
        self.use_geometry = bool(use_geometry)
        in_channels = 3 if self.use_geometry else 1
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        self.modality_embed = nn.Embedding(2, dim)
        self.visibility_embed = nn.Embedding(4, dim)
        self.temporal_embed = nn.Embedding(max_frames, dim)
        self.row_embed = nn.Embedding(max_grid, dim)
        self.col_embed = nn.Embedding(max_grid, dim)
        self.cond_prev_embed = nn.Embedding(max_frames + 1, dim)
        self.cond_next_embed = nn.Embedding(max_frames + 1, dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def geometry(self, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.float().clamp(0.0, 1.0)
        if not self.use_geometry:
            return masks
        signed = _local_geometry_prior(masks)
        boundary = _soft_boundary(masks)
        return torch.cat([masks, signed, boundary], dim=1)

    def _positional(self, b: int, t: int, hp: int, wp: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(hp, device=device).clamp_max(self.max_grid - 1)
        cols = torch.arange(wp, device=device).clamp_max(self.max_grid - 1)
        spatial = self.row_embed(rows)[:, None] + self.col_embed(cols)[None, :]
        spatial = spatial.reshape(1, 1, hp * wp, self.dim)
        time = torch.arange(t, device=device).clamp_max(self.max_frames - 1)
        temporal = self.temporal_embed(time).reshape(1, t, 1, self.dim)
        modality = self.modality_embed(torch.ones((), dtype=torch.long, device=device)).reshape(1, 1, 1, self.dim)
        return spatial + temporal + modality

    def _condition_distance(self, visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = visibility.shape
        visible = visibility > 0
        no_cond = t
        prev = torch.full((b, t), no_cond, dtype=torch.long, device=visibility.device)
        nxt = torch.full((b, t), no_cond, dtype=torch.long, device=visibility.device)
        for bi in range(b):
            last = None
            for ti in range(t):
                if bool(visible[bi, ti]):
                    last = ti
                    prev[bi, ti] = 0
                elif last is not None:
                    prev[bi, ti] = ti - last
            future = None
            for ti in range(t - 1, -1, -1):
                if bool(visible[bi, ti]):
                    future = ti
                    nxt[bi, ti] = 0
                elif future is not None:
                    nxt[bi, ti] = future - ti
        max_idx = self.max_frames
        return prev.clamp_max(max_idx), nxt.clamp_max(max_idx)

    def forward(self, masks: torch.Tensor, visibility: torch.Tensor, grid_size: tuple[int, int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, _, h, w = masks.shape
        hp, wp = grid_size
        flat = masks.reshape(b * t, 1, h, w).float()
        geom = self.geometry(flat)
        visible_tokens = self.patch(geom)
        visible_tokens = F.interpolate(visible_tokens, size=(hp, wp), mode="bilinear", align_corners=False)
        visible_tokens = visible_tokens.flatten(2).transpose(1, 2).reshape(b, t, hp * wp, self.dim)
        mask_tokens = self.mask_token.expand(b, t * hp * wp, self.dim).reshape(b, t, hp * wp, self.dim)
        is_visible = visibility.reshape(b, t, 1, 1) > 0
        tokens = torch.where(is_visible, visible_tokens, mask_tokens)
        vis = visibility.long().clamp(0, 3)
        tokens = tokens + self._positional(b, t, hp, wp, masks.device)
        tokens = tokens + self.visibility_embed(vis).reshape(b, t, 1, self.dim)
        prev, nxt = self._condition_distance(vis)
        cond = self.cond_prev_embed(prev) + self.cond_next_embed(nxt)
        tokens = tokens + cond.reshape(b, t, 1, self.dim)
        aux = {
            "geometry": geom.detach(),
            "geometry_channel_names": ("binary", "local_geometry_prior", "boundary_band"),
            "condition_prev": prev,
            "condition_next": nxt,
            "visible_frame_count": (visibility > 0).float().sum(dim=1),
            "masked_frame_count": (visibility == 0).float().sum(dim=1),
        }
        return tokens, aux


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualStreamBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, image_to_mask_gate: bool = True) -> None:
        super().__init__()
        self.x_spatial = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.m_spatial = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.x_temporal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.m_temporal = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.m_from_x = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.x_from_m = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.image_to_mask_gate = bool(image_to_mask_gate)
        self.raw_x_from_m_gate = nn.Parameter(torch.tensor(-12.0))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(8)])
        self.x_mlp = FeedForward(dim, dropout=dropout)
        self.m_mlp = FeedForward(dim, dropout=dropout)

    @property
    def x_from_m_gate(self) -> torch.Tensor:
        if not self.image_to_mask_gate:
            return torch.zeros((), device=self.raw_x_from_m_gate.device, dtype=self.raw_x_from_m_gate.dtype)
        return torch.sigmoid(self.raw_x_from_m_gate)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, t, n, d = x.shape
        xs = self.norms[0](x).reshape(b * t, n, d)
        ms = self.norms[1](m).reshape(b * t, n, d)
        x = x + self.x_spatial(xs, xs, xs, need_weights=False)[0].reshape(b, t, n, d)
        m = m + self.m_spatial(ms, ms, ms, need_weights=False)[0].reshape(b, t, n, d)

        xt = self.norms[2](x).permute(0, 2, 1, 3).reshape(b * n, t, d)
        mt = self.norms[3](m).permute(0, 2, 1, 3).reshape(b * n, t, d)
        x = x + self.x_temporal(xt, xt, xt, need_weights=False)[0].reshape(b, n, t, d).permute(0, 2, 1, 3)
        m = m + self.m_temporal(mt, mt, mt, need_weights=False)[0].reshape(b, n, t, d).permute(0, 2, 1, 3)

        xf = self.norms[4](x).reshape(b, t * n, d)
        mf = self.norms[5](m).reshape(b, t * n, d)
        m_from_x = self.m_from_x(mf, xf, xf, need_weights=False)[0]
        m = m + m_from_x.reshape(b, t, n, d)
        gate = self.x_from_m_gate
        x_from_m = self.x_from_m(xf, mf, mf, need_weights=False)[0]
        x = x + gate * x_from_m.reshape(b, t, n, d)
        x = x + self.x_mlp(self.norms[6](x))
        m = m + self.m_mlp(self.norms[7](m))
        return x, m, {
            "image_from_mask_gate": gate.detach(),
            "image_to_mask_cross_attention_norm": m_from_x.detach().norm(dim=-1).mean(),
            "mask_to_image_cross_attention_norm": (gate.detach() * x_from_m.detach()).norm(dim=-1).mean(),
        }


class DualStreamFactorizedTransformer(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 3, heads: int = 4, dropout: float = 0.0, image_to_mask_gate: bool = True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DualStreamBlock(dim, heads=heads, dropout=dropout, image_to_mask_gate=image_to_mask_gate) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.blocks):
            x, m, block_aux = block(x, m)
            for key, value in block_aux.items():
                aux[f"block{idx}_{key}"] = value
        return x, m, aux


class PixelDecoder(nn.Module):
    def __init__(self, dim: int, f2_channels: int, f4_channels: int, f8_channels: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = int(out_dim or dim)
        self.f8_proj = nn.Conv2d(f8_channels + dim, dim, 1)
        self.f4_proj = nn.Conv2d(f4_channels, dim, 1)
        self.f2_proj = nn.Conv2d(f2_channels, dim, 1)
        self.refine4 = DepthwiseSeparableConv(dim, dim)
        self.refine2 = DepthwiseSeparableConv(dim, out_dim)
        self.out_dim = out_dim

    def forward(self, image_tokens: torch.Tensor, features: dict[str, torch.Tensor]) -> torch.Tensor:
        b, t, n, d = image_tokens.shape
        hp, wp = features["grid_size"]
        z8 = image_tokens.reshape(b * t, hp, wp, d).permute(0, 3, 1, 2).contiguous()
        z = self.f8_proj(torch.cat([z8, features["f8"]], dim=1))
        z = F.interpolate(z, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False)
        z = self.refine4(z + self.f4_proj(features["f4"]))
        z = F.interpolate(z, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False)
        z = self.refine2(z + self.f2_proj(features["f2"]))
        return z.reshape(b, t, self.out_dim, z.shape[-2], z.shape[-1])


class ProposalDecoder(nn.Module):
    def __init__(self, dim: int = 128, pixel_dim: int = 128, num_queries: int = 20, num_layers: int = 2, heads: int = 4, max_frames: int = 16) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_embed = nn.Parameter(torch.empty(num_queries, dim))
        self.frame_cond = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.temporal_embed = nn.Embedding(max_frames, dim)
        self.cond_prev_embed = nn.Embedding(max_frames + 1, dim)
        self.cond_next_embed = nn.Embedding(max_frames + 1, dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "cross": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "ln1": nn.LayerNorm(dim),
                        "ln2": nn.LayerNorm(dim),
                        "ln3": nn.LayerNorm(dim),
                        "mlp": FeedForward(dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.mask_embed = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, pixel_dim))
        self.quality = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.constant_(self.quality[-1].bias, -2.0)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        pixel_embedding: torch.Tensor,
        condition_prev: torch.Tensor | None = None,
        condition_next: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t, n, d = x.shape
        context = torch.cat([x, m], dim=2).reshape(b, t * n * 2, d)
        pooled = x.mean(dim=2)
        q = self.query_embed.reshape(1, 1, self.num_queries, d) + self.frame_cond(pooled).unsqueeze(2)
        time = torch.arange(t, device=x.device).clamp_max(self.temporal_embed.num_embeddings - 1)
        q = q + self.temporal_embed(time).reshape(1, t, 1, d)
        if condition_prev is not None and condition_next is not None:
            max_idx = self.cond_prev_embed.num_embeddings - 1
            cp = condition_prev.long().clamp(0, max_idx)
            cn = condition_next.long().clamp(0, max_idx)
            q = q + self.cond_prev_embed(cp).unsqueeze(2) + self.cond_next_embed(cn).unsqueeze(2)
        q = q.reshape(b * t, self.num_queries, d)
        ctx = context.repeat_interleave(t, dim=0)
        for layer in self.layers:
            q = q + layer["self"](layer["ln1"](q), layer["ln1"](q), layer["ln1"](q), need_weights=False)[0]
            q = q + layer["cross"](layer["ln2"](q), ctx, ctx, need_weights=False)[0]
            q = q + layer["mlp"](layer["ln3"](q))
        q = q.reshape(b, t, self.num_queries, d)
        mask_embed = self.mask_embed(q)
        scores = self.quality(q).squeeze(-1)
        pe = pixel_embedding
        logits = torch.einsum("btkd,btdhw->btkhw", mask_embed, pe) / math.sqrt(pe.shape[2])
        return {"proposal_logits": logits, "quality_scores": scores, "query_features": q}


class GeoMaskFormer(nn.Module):
    """Masked Geometry Token MaskFormer for ultrasound video segmentation."""

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {}
        method_cfg = _cfg_get(cfg, "geomaskformer", cfg)
        self.in_channels = int(_cfg_get(method_cfg, "in_channels", 1))
        self.num_classes = int(_cfg_get(method_cfg, "num_classes", 2))
        self.dim = int(_cfg_get(method_cfg, "dim", 128))
        self.base_channels = int(_cfg_get(method_cfg, "base_channels", 32))
        self.depth = int(_cfg_get(method_cfg, "depth", 3))
        self.heads = int(_cfg_get(method_cfg, "heads", 4))
        self.num_queries = int(_cfg_get(method_cfg, "num_queries", 20))
        self.decoder_layers = int(_cfg_get(method_cfg, "decoder_layers", 2))
        self.max_frames = int(_cfg_get(method_cfg, "max_frames", 16))
        self.stage = str(_cfg_get(method_cfg, "training_stage", "stage2")).lower()
        self.protocol_cfg = _cfg_section(method_cfg, "protocol")
        self.topk_inference = str(_cfg_get(method_cfg, "inference", "top1")).lower()
        self.image_token_dropout = float(_cfg_get(method_cfg, "image_token_dropout", 0.05))
        self.mask_token_dropout = float(_cfg_get(method_cfg, "mask_token_dropout", 0.10))
        self.condition_dropout = float(_cfg_get(method_cfg, "condition_dropout", 0.10))
        self.visible_reconstruction_weight = float(
            _cfg_get(_cfg_section(method_cfg, "loss"), "visible_reconstruction", 0.0)
        )
        self.image_tokenizer = ImageTokenizer(self.in_channels, self.dim, self.base_channels)
        self.mask_tokenizer = MaskTokenizer(
            self.dim,
            max_frames=self.max_frames,
            use_geometry=bool(_cfg_get(method_cfg, "use_geometry_tokenizer", True)),
        )
        self.transformer = DualStreamFactorizedTransformer(
            self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=float(_cfg_get(method_cfg, "dropout", 0.0)),
            image_to_mask_gate=bool(_cfg_get(method_cfg, "gated_image_from_mask", True)),
        )
        c = self.image_tokenizer.out_channels
        self.pixel_decoder = PixelDecoder(self.dim, c["f2"], c["f4"], c["f8"], out_dim=self.dim)
        self.proposal_decoder = ProposalDecoder(
            self.dim,
            pixel_dim=self.dim,
            num_queries=self.num_queries,
            num_layers=self.decoder_layers,
            heads=self.heads,
            max_frames=self.max_frames,
        )
        self.backbone_name = "geomaskformer"
        self.base_dim = self.base_channels
        self.value_dim = self.dim
        self.query_dim = self.dim

    def _resolve_visibility(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = data["rgb"]
        b, t = rgb.shape[:2]
        device = rgb.device
        if "geomaskformer_mask_visibility" in data:
            mask_visibility = data["geomaskformer_mask_visibility"].to(device=device).long()
        else:
            valid = data.get("label_valid")
            if torch.is_tensor(valid):
                valid = valid.to(device=device).bool()
            else:
                valid = torch.zeros(b, t, dtype=torch.bool, device=device)
                valid[:, 0] = True
            mask_visibility = torch.zeros(b, t, dtype=torch.long, device=device)
            if self.training and self.stage in {"stage2", "stage3"}:
                probs = torch.tensor(
                    [
                        float(_cfg_get(self.protocol_cfg, "no_mask", 0.50)),
                        float(_cfg_get(self.protocol_cfg, "first_frame", 0.25)),
                        float(_cfg_get(self.protocol_cfg, "random_keyframe", 0.25)),
                    ],
                    device=device,
                )
                probs = probs / probs.sum().clamp_min(1.0e-6)
                choices = torch.multinomial(probs, b, replacement=True)
                for bi in range(b):
                    if int(choices[bi]) == 1:
                        if bool(valid[bi, 0]):
                            mask_visibility[bi, 0] = 1
                        else:
                            idx = torch.nonzero(valid[bi], as_tuple=False).flatten()
                            if idx.numel() > 0:
                                mask_visibility[bi, idx[0]] = 1
                    elif int(choices[bi]) == 2:
                        idx = torch.nonzero(valid[bi], as_tuple=False).flatten()
                        if idx.numel() > 0:
                            count = min(2, idx.numel())
                            chosen = idx[torch.randperm(idx.numel(), device=device)[:count]]
                            mask_visibility[bi, chosen] = 1
            elif self.stage == "stage1":
                mask_visibility.zero_()
            else:
                mask_visibility = valid.long()
        if self.training and self.condition_dropout > 0:
            keep = torch.rand_like(mask_visibility.float()) > self.condition_dropout
            mask_visibility = mask_visibility * keep.long()
        explicit_loss_visibility = "geomaskformer_loss_visibility" in data
        if explicit_loss_visibility:
            loss_visibility = data["geomaskformer_loss_visibility"].to(device=device).bool()
        elif torch.is_tensor(data.get("label_valid")):
            loss_visibility = data["label_valid"].to(device=device).bool()
        else:
            loss_visibility = torch.ones(b, t, dtype=torch.bool, device=device)
        if self.training and not explicit_loss_visibility and self.stage in {"stage2", "stage3"}:
            main_loss_visibility = loss_visibility & (mask_visibility == 0)
            empty = ~main_loss_visibility.any(dim=1)
            if empty.any():
                main_loss_visibility[empty] = loss_visibility[empty]
            loss_visibility = main_loss_visibility
        return mask_visibility, loss_visibility

    def _apply_token_dropout(self, x: torch.Tensor, m: torch.Tensor, mask_visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.training and self.image_token_dropout > 0:
            drop = torch.rand(x.shape[:3], device=x.device).unsqueeze(-1) < self.image_token_dropout
            x = x.masked_fill(drop, 0.0)
            aux["image_dropout_ratio"] = drop.float().mean().detach()
        else:
            aux["image_dropout_ratio"] = torch.zeros((), device=x.device)
        if self.training and self.mask_token_dropout > 0:
            visible = mask_visibility.reshape(mask_visibility.shape[0], mask_visibility.shape[1], 1, 1) > 0
            drop = (torch.rand(m.shape[:3], device=m.device).unsqueeze(-1) < self.mask_token_dropout) & visible
            replacement = self.mask_tokenizer.mask_token.reshape(1, 1, 1, self.dim)
            m = torch.where(drop, replacement, m)
            aux["mask_dropout_ratio"] = drop.float().mean().detach()
        else:
            aux["mask_dropout_ratio"] = torch.zeros((), device=m.device)
        return x, m, aux

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = data["rgb"].float()
        masks = data.get("cls_gt")
        if not torch.is_tensor(masks):
            masks = torch.zeros(images.shape[0], images.shape[1], 1, images.shape[-2], images.shape[-1], device=images.device, dtype=images.dtype)
        masks = (masks.float() > 0).float()
        mask_visibility, loss_visibility = self._resolve_visibility(data)
        img = self.image_tokenizer(images)
        x = img["tokens"]
        hp, wp = img["grid_size"]
        rows = torch.arange(hp, device=images.device).clamp_max(self.mask_tokenizer.max_grid - 1)
        cols = torch.arange(wp, device=images.device).clamp_max(self.mask_tokenizer.max_grid - 1)
        spatial = self.mask_tokenizer.row_embed(rows)[:, None] + self.mask_tokenizer.col_embed(cols)[None, :]
        spatial = spatial.reshape(1, 1, hp * wp, self.dim)
        time = torch.arange(images.shape[1], device=images.device).clamp_max(self.max_frames - 1)
        x = x + spatial + self.mask_tokenizer.temporal_embed(time).reshape(1, images.shape[1], 1, self.dim)
        x = x + self.mask_tokenizer.modality_embed(torch.zeros((), dtype=torch.long, device=images.device)).reshape(1, 1, 1, self.dim)
        m, mask_aux = self.mask_tokenizer(masks, mask_visibility, img["grid_size"])
        x, m, drop_aux = self._apply_token_dropout(x, m, mask_visibility)
        x, m, former_aux = self.transformer(x, m)
        pixel = self.pixel_decoder(x, img)
        proposal = self.proposal_decoder(
            x,
            m,
            pixel,
            condition_prev=mask_aux.get("condition_prev"),
            condition_next=mask_aux.get("condition_next"),
        )
        prop_logits = proposal["proposal_logits"]
        scores = proposal["quality_scores"]
        if self.topk_inference == "top3":
            topk = min(3, self.num_queries)
            top_scores, top_idx = torch.topk(scores, k=topk, dim=2)
            gather_idx = top_idx[..., None, None].expand(-1, -1, -1, prop_logits.shape[-2], prop_logits.shape[-1])
            top_logits = torch.gather(prop_logits, 2, gather_idx)
            weights = torch.softmax(top_scores, dim=2)[..., None, None]
            fg_low = torch.logit((torch.sigmoid(top_logits) * weights).sum(dim=2).clamp(1.0e-4, 1.0 - 1.0e-4))
            best_idx = top_idx[..., 0]
        else:
            best_idx = scores.argmax(dim=2)
            gather_idx = best_idx[..., None, None, None].expand(-1, -1, 1, prop_logits.shape[-2], prop_logits.shape[-1])
            fg_low = torch.gather(prop_logits, 2, gather_idx).squeeze(2)
        fg = F.interpolate(
            fg_low.reshape(images.shape[0] * images.shape[1], 1, fg_low.shape[-2], fg_low.shape[-1]),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(images.shape[0], images.shape[1], images.shape[-2], images.shape[-1])
        logits = torch.stack([-fg, fg], dim=2)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "proposal_logits": F.interpolate(
                prop_logits.reshape(images.shape[0] * images.shape[1] * self.num_queries, 1, prop_logits.shape[-2], prop_logits.shape[-1]),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(images.shape[0], images.shape[1], self.num_queries, images.shape[-2], images.shape[-1]),
            "proposal_logits_lowres": prop_logits,
            "quality_scores": scores,
            "mask_visibility": mask_visibility,
            "loss_visibility": loss_visibility,
            "geomaskformer_aux": {
                **mask_aux,
                **drop_aux,
                **former_aux,
                "best_query": best_idx.detach(),
                "quality_prob_mean": torch.sigmoid(scores).mean().detach(),
                "active_query_count": best_idx.flatten().unique().numel(),
                "masked_token_norm": m[mask_visibility == 0].detach().norm(dim=-1).mean()
                if (mask_visibility == 0).any()
                else torch.zeros((), device=images.device),
                "visible_token_norm": m[mask_visibility > 0].detach().norm(dim=-1).mean()
                if (mask_visibility > 0).any()
                else torch.zeros((), device=images.device),
            },
            "num_objects": [1] * images.shape[0],
        }
        for ti in range(images.shape[1]):
            out[f"logits_{ti}"] = logits[:, ti]
            out[f"masks_{ti}"] = torch.sigmoid(fg[:, ti : ti + 1])
            aux = {
                "geomaskformer_proposal_logits": out["proposal_logits"][:, ti],
                "geomaskformer_quality_scores": scores[:, ti],
                "geomaskformer_best_query": best_idx[:, ti],
            }
            out[f"aux_{ti}"] = aux
            out[f"memory_aux_{ti}"] = {"geomaskformer_aux": aux}
        return out
