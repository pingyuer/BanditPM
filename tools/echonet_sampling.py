#!/usr/bin/env python3
"""Shared sampling utilities for EchoNet-style sparse video segmentation preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplePlan:
    """Sampling plan for a single video clip."""

    indices: list[int]
    label_indices: list[int]
    used_repeat: bool
    window_start: int
    window_end: int
    mode: str


def sample_linear_window(start_idx: int, end_idx: int, num_frames: int) -> tuple[list[int], bool]:
    """Uniformly sample integer frame indices inside an inclusive temporal window."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    indices = np.linspace(start_idx, end_idx, num_frames)
    indices = np.clip(np.round(indices).astype(int), start_idx, end_idx).tolist()
    return indices, len(set(indices)) < num_frames


def sample_two_segment_cycle(
    start_idx: int,
    mid_idx: int,
    end_idx: int,
    num_frames: int,
) -> tuple[list[int], bool, int]:
    """Sample a clip with a fixed middle anchor for the second traced frame.

    The traced pair is mapped to deterministic clip positions:
    - position 0   -> first traced frame (typically ED)
    - position mid -> second traced frame (typically ES)
    - position -1  -> estimated cycle end
    """
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2 for cycle sampling")
    if not (start_idx <= mid_idx <= end_idx):
        raise ValueError(
            f"expected start <= mid <= end, got start={start_idx}, mid={mid_idx}, end={end_idx}"
        )

    anchor_pos = num_frames // 2
    first_len = anchor_pos + 1
    second_len = num_frames - anchor_pos

    first_indices, first_repeat = sample_linear_window(start_idx, mid_idx, first_len)
    second_indices, second_repeat = sample_linear_window(mid_idx, end_idx, second_len)
    indices = first_indices + second_indices[1:]
    return indices, (first_repeat or second_repeat or len(set(indices)) < num_frames), anchor_pos


def nearest_index_position(indices: list[int], target_idx: int) -> int:
    """Return the sampled-position index closest to a target frame index."""
    if not indices:
        raise ValueError("indices must not be empty")
    best_pos = 0
    best_dist = abs(indices[0] - target_idx)
    for pos, frame_idx in enumerate(indices[1:], start=1):
        dist = abs(frame_idx - target_idx)
        if dist < best_dist:
            best_pos = pos
            best_dist = dist
    return best_pos


def build_sample_plan(
    annotated_frames: list[int],
    *,
    frame_count: int,
    num_frames: int,
    mode: str,
) -> SamplePlan:
    """Build a temporal sampling plan for endpoint or full-cycle supervision."""
    if len(annotated_frames) != 2:
        raise ValueError(f"expected 2 annotated frames, found {len(annotated_frames)}")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    start_frame, end_frame = sorted(annotated_frames)
    if start_frame < 0 or end_frame >= frame_count:
        raise ValueError(
            f"annotated frames out of range: annotated={annotated_frames}, frame_count={frame_count}"
        )

    normalized_mode = mode.strip().lower()
    if normalized_mode == "ed_to_es":
        sampled_indices, used_repeat = sample_linear_window(start_frame, end_frame, num_frames)
        label_indices = [0, num_frames - 1]
        window_start = start_frame
        window_end = end_frame
    elif normalized_mode == "full_cycle":
        gap = max(1, end_frame - start_frame)
        estimated_cycle_end = min(frame_count - 1, start_frame + 2 * gap)
        sampled_indices, used_repeat, anchor_pos = sample_two_segment_cycle(
            start_frame,
            end_frame,
            estimated_cycle_end,
            num_frames,
        )
        label_indices = [0, anchor_pos]
        window_start = start_frame
        window_end = estimated_cycle_end
    else:
        raise ValueError(f"unsupported sampling mode: {mode}")

    return SamplePlan(
        indices=sampled_indices,
        label_indices=label_indices,
        used_repeat=used_repeat,
        window_start=window_start,
        window_end=window_end,
        mode=normalized_mode,
    )
