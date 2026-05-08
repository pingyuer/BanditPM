from __future__ import annotations

import logging
import os
import re
from typing import Any, Mapping


LOGGER = logging.getLogger(__name__)


def _metadata_get(metadata: Mapping[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if not metadata:
        return None
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _coerce_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    if isinstance(value, (list, tuple)) and value:
        return _coerce_index(value[0])
    if isinstance(value, Mapping):
        for key in ("frame", "frame_index", "index", "idx"):
            idx = _coerce_index(value.get(key))
            if idx is not None:
                return idx
    return None


def _parse_ed_es(label: str, metadata: Mapping[str, Any] | None) -> int | None:
    label_upper = label.upper()
    if label_upper not in {"ED", "ES"}:
        return None

    direct_keys = (
        f"{label_upper}_frame",
        f"{label_upper}_index",
        f"{label_upper}_frame_index",
        label_upper,
        label_upper.lower(),
        f"{label_upper.lower()}_frame",
        f"{label_upper.lower()}_index",
    )
    idx = _coerce_index(_metadata_get(metadata, direct_keys))
    if idx is not None:
        return idx

    for container_key in ("keyframes", "frame_labels", "labels", "events", "phase_frames"):
        container = _metadata_get(metadata, (container_key,))
        if isinstance(container, Mapping):
            idx = _coerce_index(_metadata_get(container, direct_keys))
            if idx is not None:
                return idx
    return None


def parse_frame_index(filename: str, metadata: Mapping[str, Any] | None = None) -> int | None:
    """Parse a frame index from common sparse echocardiography mask names.

    Supported examples include ``000.png``, ``000001_mask.png``,
    ``frame_000.png``, ``frame001.png``, and ``ED/ES`` when metadata maps those
    phase names to concrete frame indices.
    """

    stem = os.path.splitext(os.path.basename(filename))[0]
    clean = stem.strip()
    if not clean:
        return None

    phase_idx = _parse_ed_es(clean, metadata)
    if phase_idx is not None:
        return phase_idx

    if clean.isdigit():
        return int(clean)

    patterns = (
        r"^frame[_-]?(\d+)$",
        r"^img[_-]?(\d+)$",
        r"^(\d+)[_-]?(?:mask|label|gt|seg|lv)$",
        r"^(?:mask|label|gt|seg|lv)[_-]?(\d+)$",
    )
    lowered = clean.lower()
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            return int(match.group(1))

    digit_groups = re.findall(r"\d+", lowered)
    if len(digit_groups) == 1:
        return int(digit_groups[0])
    return None


def build_label_map(
    label_files: list[str],
    metadata: Mapping[str, Any] | None = None,
    *,
    sample_name: str = "",
    logger: logging.Logger | None = None,
) -> dict[int, str]:
    logger = logger or LOGGER
    label_map: dict[int, str] = {}
    for label_name in label_files:
        frame_idx = parse_frame_index(label_name, metadata)
        if frame_idx is None:
            logger.warning("Could not parse frame index from label '%s' in sample '%s'", label_name, sample_name)
            continue
        if frame_idx in label_map:
            logger.warning(
                "Duplicate label frame index %s in sample '%s': keeping '%s', ignoring '%s'",
                frame_idx,
                sample_name,
                label_map[frame_idx],
                label_name,
            )
            continue
        label_map[frame_idx] = label_name
    return label_map
