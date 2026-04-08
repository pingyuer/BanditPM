import os
import tempfile
import unittest

import cv2
import numpy as np

from dataset.echo import EchoDataset


class EchoDatasetSparseLabelTests(unittest.TestCase):
    def _write_png(self, path: str, value: int) -> None:
        image = np.full((8, 8), value, dtype=np.uint8)
        cv2.imwrite(path, image)

    def test_sparse_label_indices_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "train", "img", "case001")
            label_dir = os.path.join(tmpdir, "train", "label", "case001")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            for idx in range(10):
                self._write_png(os.path.join(img_dir, f"{idx:04d}.png"), 64)
            self._write_png(os.path.join(label_dir, "0000.png"), 1)
            self._write_png(os.path.join(label_dir, "0004.png"), 1)

            dataset = EchoDataset(tmpdir, mode="train", seq_length=10, size=8)
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertTrue(sample["label_valid"][0].item())
            self.assertTrue(sample["label_valid"][4].item())
            self.assertFalse(sample["label_valid"][9].item())


if __name__ == "__main__":
    unittest.main()
