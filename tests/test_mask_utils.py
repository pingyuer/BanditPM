import numpy as np

from dataset.mask_utils import binarize_lv_mask


def test_binarize_lv_mask_accepts_binary_0255():
    mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)

    out = binarize_lv_mask(mask)

    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [0, 1]]


def test_binarize_lv_mask_selects_configured_class_id():
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    out = binarize_lv_mask(mask, lv_class_id=2)

    assert out.tolist() == [[0, 0, 1], [1, 0, 0]]
