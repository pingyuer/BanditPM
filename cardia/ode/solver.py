from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridODESolver(nn.Module):
    def __init__(self, padding_mode: str = "border", align_corners: bool = False) -> None:
        super().__init__()
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

    def _identity_grid(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        if self.align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, height, width, 2)

    def forward(self, anchor_feat_t: torch.Tensor, ode_flow_t: torch.Tensor, selector_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, C, H, W = anchor_feat_t.shape
        K = ode_flow_t.shape[1]
        base_grid = self._identity_grid(B, H, W, anchor_feat_t.device, anchor_feat_t.dtype)
        grid = base_grid[:, None] + ode_flow_t.permute(0, 1, 3, 4, 2)
        oob = (grid.abs() > 1.0).any(dim=-1).float().mean(dim=(1, 2, 3))
        solved = F.grid_sample(
            anchor_feat_t[:, None].expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W),
            grid.reshape(B * K, H, W, 2),
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).view(B, K, C, H, W)
        dynamic_anchor_t = (solved * selector_weights[:, :, None]).sum(dim=1)
        return dynamic_anchor_t, solved, {"grid_oob_ratio": oob.detach()}
