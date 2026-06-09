from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / float(height)) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / float(width)) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def _compose_displacement(disp: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    b, _, h, w = disp.shape
    grid = _identity_grid(b, h, w, disp.device, disp.dtype) + update.permute(0, 2, 3, 1)
    sampled = F.grid_sample(disp, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return update + sampled


def _integrate_stationary_velocity(velocity: torch.Tensor, steps: int) -> torch.Tensor:
    steps = max(int(steps), 0)
    disp = velocity / float(2**steps)
    for _ in range(steps):
        disp = _compose_displacement(disp, disp)
    return disp


class DenseMomentumWarp(nn.Module):
    """Low-resolution stationary velocity warp for anchor/proposal logits.

    The field is predicted as a backward sampling displacement in normalized
    grid coordinates and integrated with a small scaling-and-squaring loop
    before resampling logits. Positive x samples from the right, so the visible
    content moves left. The final convolution is zero-initialized so the module
    starts as an identity warp.
    """

    def __init__(
        self,
        *,
        decoder_dim: int,
        hidden_dim: int,
        flow_size: int = 16,
        max_displacement: float = 0.08,
        integration_steps: int = 4,
    ) -> None:
        super().__init__()
        self.flow_size = int(flow_size)
        self.max_displacement = float(max_displacement)
        self.integration_steps = int(integration_steps)
        self.net = nn.Sequential(
            nn.Conv2d(decoder_dim + 4, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _smoothness(flow: torch.Tensor) -> torch.Tensor:
        dx = flow[..., :, 1:] - flow[..., :, :-1]
        dy = flow[..., 1:, :] - flow[..., :-1, :]
        return dx.float().pow(2).mean() + dy.float().pow(2).mean()

    def forward(
        self,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
        proposal_logits: torch.Tensor,
        uncertainty_map: torch.Tensor,
        boundary_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, n, h, w = proposal_logits.shape
        target = (self.flow_size, self.flow_size)
        dec = F.interpolate(decoder_feature, size=target, mode="bilinear", align_corners=False)
        dec = dec.unsqueeze(1).expand(-1, n, -1, -1, -1).flatten(0, 1)
        base = F.interpolate(base_logits.flatten(0, 1).unsqueeze(1), size=target, mode="bilinear", align_corners=False)
        proposal = F.interpolate(proposal_logits.flatten(0, 1).unsqueeze(1), size=target, mode="bilinear", align_corners=False)
        uncertainty = F.interpolate(uncertainty_map.expand(-1, n, -1, -1).flatten(0, 1).unsqueeze(1), size=target, mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary_map.expand(-1, n, -1, -1).flatten(0, 1).unsqueeze(1), size=target, mode="bilinear", align_corners=False)
        flow_in = torch.cat([dec, base, proposal, uncertainty, boundary], dim=1)
        velocity = torch.tanh(self.net(flow_in)) * self.max_displacement
        disp = _integrate_stationary_velocity(velocity, self.integration_steps)
        disp_full = F.interpolate(disp, size=(h, w), mode="bilinear", align_corners=False)
        grid = _identity_grid(b * n, h, w, proposal_logits.device, proposal_logits.dtype)
        grid = grid + disp_full.permute(0, 2, 3, 1)
        valid_grid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        warped = F.grid_sample(
            proposal_logits.flatten(0, 1).unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(b, n, h, w)
        aux = {
            "dense_velocity": velocity.view(b, n, 2, *target),
            "dense_displacement": disp.view(b, n, 2, *target),
            "dense_backward_displacement": disp.view(b, n, 2, *target),
            "dense_displacement_full": disp_full.view(b, n, 2, h, w),
            "dense_backward_displacement_full": disp_full.view(b, n, 2, h, w),
            "dense_content_motion_full": (-disp_full).view(b, n, 2, h, w),
            "dense_flow_abs_mean": disp_full.detach().abs().mean(),
            "dense_flow_abs_max": disp_full.detach().abs().amax(),
            "dense_flow_pixel_mean": (disp_full.detach().abs().mean() * ((h + w) / 4.0)).to(dtype=proposal_logits.dtype),
            "dense_valid_ratio": valid_grid.detach().float().mean().to(dtype=proposal_logits.dtype),
            "dense_oob_ratio": (~valid_grid).detach().float().mean().to(dtype=proposal_logits.dtype),
            "dense_flow_smoothness": self._smoothness(disp_full),
            "dense_warp_delta_abs_mean": (warped.detach() - proposal_logits.detach()).abs().mean(),
        }
        return warped, aux
