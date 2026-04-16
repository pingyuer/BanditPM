from __future__ import annotations

import torch


def make_frame_valid_mask(*rows: list[bool]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def make_video_batch(
    batch_size: int,
    num_frames: int,
    *,
    channels: int = 1,
    height: int = 8,
    width: int = 8,
) -> torch.Tensor:
    return torch.zeros(batch_size, num_frames, channels, height, width)


def make_cls_gt_from_frame_labels(frame_labels: list[list[int]], *, height: int = 8, width: int = 8) -> torch.Tensor:
    batch = []
    for sample_labels in frame_labels:
        sample_frames = []
        for label in sample_labels:
            sample_frames.append([[[label] * width for _ in range(height)]])
        batch.append(sample_frames)
    return torch.tensor(batch, dtype=torch.long)
