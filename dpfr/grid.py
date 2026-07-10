from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    if align_corners:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    else:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, height, width, 2)


def grid_sample_logits(
    logits: torch.Tensor,
    delta_grid: torch.Tensor,
    *,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    b, _, h, w = logits.shape
    if delta_grid.shape[-2:] != (h, w):
        delta_grid = F.interpolate(delta_grid, size=(h, w), mode="bilinear", align_corners=False)
    grid = identity_grid(b, h, w, logits.device, logits.dtype, align_corners=align_corners)
    grid = grid + delta_grid.to(dtype=logits.dtype).permute(0, 2, 3, 1)
    return F.grid_sample(logits, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)


def flow_smoothness(delta_grid: torch.Tensor) -> torch.Tensor:
    dx = delta_grid[..., :, 1:] - delta_grid[..., :, :-1]
    dy = delta_grid[..., 1:, :] - delta_grid[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def out_of_bound_ratio(delta_grid: torch.Tensor, *, align_corners: bool = True) -> torch.Tensor:
    b, _, h, w = delta_grid.shape
    grid = identity_grid(b, h, w, delta_grid.device, delta_grid.dtype, align_corners=align_corners)
    grid = grid + delta_grid.permute(0, 2, 3, 1)
    invalid = (grid < -1.0) | (grid > 1.0)
    return invalid.any(dim=-1).float().mean()
