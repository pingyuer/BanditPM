from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebel.ode_field import ConvNeXtLiteBlock, _groups


class BeliefSkipAdapter(nn.Module):
    def __init__(self, skip_dim: int, belief_dim: int, out_dim: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(skip_dim, out_dim, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_dim + belief_dim + 2, out_dim, 1, bias=False),
            nn.GroupNorm(_groups(out_dim), out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, belief_up: torch.Tensor, disagreement: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        if belief_up.shape[-2:] != skip.shape[-2:]:
            belief_up = F.interpolate(belief_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if disagreement.shape[-2:] != skip.shape[-2:]:
            disagreement = F.interpolate(disagreement, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if reliability.shape[-2:] != skip.shape[-2:]:
            reliability = F.interpolate(reliability, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([skip, belief_up, disagreement, reliability], dim=1))
        return gate * self.project(skip)


class BeliefUpBlock(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim + skip_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_dim), out_dim),
            nn.SiLU(),
            ConvNeXtLiteBlock(out_dim),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, skip], dim=1))


class BeliefDecoder(nn.Module):
    def __init__(
        self,
        belief_dim: int,
        low_dim: int,
        mid_dim: int,
        high_dim: int,
        decoder_dim: int = 192,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.skip_high = BeliefSkipAdapter(high_dim, belief_dim, decoder_dim)
        self.skip_mid = BeliefSkipAdapter(mid_dim, decoder_dim, decoder_dim)
        self.skip_low = BeliefSkipAdapter(low_dim, decoder_dim, decoder_dim)
        self.up_high = BeliefUpBlock(belief_dim, decoder_dim, decoder_dim)
        self.up_mid = BeliefUpBlock(decoder_dim, decoder_dim, decoder_dim)
        self.up_low = BeliefUpBlock(decoder_dim, decoder_dim, decoder_dim)
        self.full = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(decoder_dim), decoder_dim),
            nn.SiLU(),
            ConvNeXtLiteBlock(decoder_dim),
        )
        self.mask_head = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(
        self,
        belief: torch.Tensor,
        skips: dict[str, torch.Tensor],
        disagreement: torch.Tensor,
        reliability: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high_skip = self.skip_high(skips["high"], belief, disagreement, reliability)
        x = self.up_high(belief, high_skip)
        mid_skip = self.skip_mid(skips["mid"], x, disagreement, reliability)
        x = self.up_mid(x, mid_skip)
        low_skip = self.skip_low(skips["low"], x, disagreement, reliability)
        x = self.up_low(x, low_skip)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        feat = self.full(x)
        return self.mask_head(feat), feat
