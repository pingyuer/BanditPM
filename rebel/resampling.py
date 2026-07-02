from __future__ import annotations

import torch
import torch.nn.functional as F


def make_identity_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    align_corners: bool = False,
) -> torch.Tensor:
    if align_corners:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    else:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch_size, height, width, 2)


def offset_px_to_normalized(
    offset_px: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool = False,
) -> torch.Tensor:
    denom_x = max(width - 1, 1) if align_corners else max(width, 1)
    denom_y = max(height - 1, 1) if align_corners else max(height, 1)
    out = torch.empty_like(offset_px)
    out[:, 0] = offset_px[:, 0] * (2.0 / denom_x)
    out[:, 1] = offset_px[:, 1] * (2.0 / denom_y)
    return out


def sample_feature(
    feature: torch.Tensor,
    offset_px: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    b, _, h, w = feature.shape
    if offset_px.shape[-2:] != (h, w):
        offset_px = F.interpolate(offset_px, size=(h, w), mode="bilinear", align_corners=False)
    grid = make_identity_grid(b, h, w, feature.device, feature.dtype, align_corners=align_corners)
    norm_offset = offset_px_to_normalized(offset_px.to(dtype=feature.dtype), h, w, align_corners=align_corners)
    grid = grid + norm_offset.permute(0, 2, 3, 1)
    return F.grid_sample(feature, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
