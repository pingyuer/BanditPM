from __future__ import annotations

from typing import Any

import torch


def build_default_endpoint_mask(
    batch_size: int,
    total_frames: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, total_frames), device=device, dtype=torch.bool)
    mask[:, 0] = True
    if total_frames > 1:
        mask[:, total_frames - 1] = True
    return mask


def normalize_frame_validity_mask(
    source: torch.Tensor | None,
    *,
    batch_size: int,
    total_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if source is None:
        return build_default_endpoint_mask(batch_size, total_frames, device=device)

    source = source.to(device=device, dtype=torch.bool)
    if source.dim() == 1:
        source = source.unsqueeze(0)
    if source.dim() != 2:
        raise ValueError(f"Expected 1D/2D frame-valid mask, got shape={tuple(source.shape)}")
    if source.shape[1] != total_frames:
        raise ValueError(f"Frame-valid mask has T={source.shape[1]}, expected {total_frames}")
    if source.shape[0] == 1 and batch_size > 1:
        source = source.expand(batch_size, -1)
    if source.shape[0] != batch_size:
        raise ValueError(f"Frame-valid mask has batch={source.shape[0]}, expected {batch_size}")
    return source


def mask_to_frame_ids(frame_mask: torch.Tensor) -> list[int]:
    return torch.nonzero(frame_mask, as_tuple=False).flatten().tolist()


def summarize_frame_mask(frame_mask: torch.Tensor, max_samples: int = 3) -> list[Any] | list[int]:
    frame_mask = frame_mask.detach().to("cpu", dtype=torch.bool)
    if frame_mask.dim() == 1:
        return mask_to_frame_ids(frame_mask)

    summaries: list[Any] = []
    for sample_idx in range(min(frame_mask.shape[0], max_samples)):
        summaries.append(mask_to_frame_ids(frame_mask[sample_idx]))
    if frame_mask.shape[0] > max_samples:
        summaries.append("...")
    return summaries
