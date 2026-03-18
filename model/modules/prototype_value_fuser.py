from __future__ import annotations

import torch
import torch.nn as nn


class PrototypeValueFuser(nn.Module):
    """
    Fuse prototype-guided value with original value.

    Shapes:
        v_orig: [B, N, C, H, W]
        v_proto: [B, C]
        output: [B, N, C, H, W]
    """

    def __init__(
        self,
        mode: str = "add",
        value_dim: int | None = None,
        proto_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.value_dim = value_dim
        self.proto_dim = proto_dim or value_dim
        self.hidden_dim = hidden_dim or value_dim

        if value_dim is None:
            raise ValueError("PrototypeValueFuser requires value_dim")

        if mode == "add":
            self.fuse_net = None
        elif mode == "concat":
            self.fuse_net = nn.Sequential(
                nn.Conv2d(value_dim + self.proto_dim, self.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, value_dim, kernel_size=1),
            )
        elif mode == "gated":
            self.fuse_net = nn.Sequential(
                nn.Conv2d(value_dim + self.proto_dim, self.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, value_dim, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported prototype fuse mode={mode!r}")

    def _expand_proto(self, v_orig: torch.Tensor, v_proto: torch.Tensor) -> torch.Tensor:
        B, N, _, H, W = v_orig.shape
        return v_proto[:, None, :, None, None].expand(B, N, v_proto.shape[-1], H, W)

    def forward(self, v_orig: torch.Tensor, v_proto: torch.Tensor) -> torch.Tensor:
        proto_map = self._expand_proto(v_orig, v_proto)

        if self.mode == "add":
            return v_orig + proto_map

        B, N, _, H, W = v_orig.shape
        cat = torch.cat([v_orig, proto_map], dim=2).reshape(B * N, -1, H, W)

        if self.mode == "concat":
            fused = self.fuse_net(cat)
            return fused.view(B, N, self.value_dim, H, W)

        gate = self.fuse_net(cat).view(B, N, self.value_dim, H, W)
        return gate * v_orig + (1.0 - gate) * proto_map
