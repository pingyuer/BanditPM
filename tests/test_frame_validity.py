import unittest

import torch

from utils.frame_validity import (
    build_default_endpoint_mask,
    mask_to_frame_ids,
    normalize_frame_validity_mask,
    summarize_frame_mask,
)


class FrameValidityTests(unittest.TestCase):
    def test_default_endpoint_mask_marks_first_and_last(self):
        mask = build_default_endpoint_mask(batch_size=2, total_frames=5, device=torch.device("cpu"))
        expected = torch.tensor(
            [
                [True, False, False, False, True],
                [True, False, False, False, True],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_normalize_expands_single_mask_to_batch(self):
        source = torch.tensor([True, False, True, False], dtype=torch.bool)
        mask = normalize_frame_validity_mask(
            source,
            batch_size=3,
            total_frames=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(mask.shape, (3, 4))
        self.assertTrue(mask[:, 0].all())
        self.assertTrue(mask[:, 2].all())

    def test_summarize_frame_mask_formats_sample_level_ids(self):
        mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, True, False],
            ],
            dtype=torch.bool,
        )
        self.assertEqual(mask_to_frame_ids(mask[0]), [0, 3])
        self.assertEqual(summarize_frame_mask(mask), [[0, 3], [1, 2]])


if __name__ == "__main__":
    unittest.main()
