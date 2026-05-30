from __future__ import annotations

import numpy as np


def binarize_lv_mask(mask: np.ndarray, lv_class_id: int | None = None) -> np.ndarray:
    """Return a uint8 LV foreground mask using a stable dataset protocol.

    Processed datasets usually store binary masks as 0/1, but this helper also
    accepts 0/255 masks. For multi-class masks, callers can pass lv_class_id to
    select the LV class explicitly.
    """

    if mask is None:
        raise ValueError("mask must not be None")
    if lv_class_id is not None:
        return (mask == int(lv_class_id)).astype(np.uint8)
    return (mask > 0).astype(np.uint8)
