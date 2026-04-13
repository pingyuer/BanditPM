import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from dataset.echo import EchoDataset
from tools.preprocess_cardiacuda import preprocess_dataset
from train import resolve_dataset_class


class CardiacUDAPreprocessTests(unittest.TestCase):
    def _write_volume(self, path: Path, array: np.ndarray) -> None:
        image = sitk.GetImageFromArray(array)
        sitk.WriteImage(image, str(path))

    def _build_sparse_case(self, site_dir: Path, case_name: str) -> None:
        frames, height, width = 24, 24, 32
        image = np.stack(
            [np.full((height, width), fill_value=frame_idx * 10, dtype=np.uint8) for frame_idx in range(frames)],
            axis=0,
        )
        label = np.zeros((frames, height, width), dtype=np.uint8)
        labelled_frames = [1, 4, 8, 12, 16, 20]
        for frame_idx in labelled_frames:
            label[frame_idx, 6:14, 7:15] = 1

        self._write_volume(site_dir / f"{case_name}_image.nii.gz", image)
        self._write_volume(site_dir / f"{case_name}_label.nii.gz", label)

    def test_preprocess_generates_sparse_gdkvm_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_root = tmp_path / "cardiacUDC_dataset"
            for site in ["Site_G_100", "Site_G_20", "Site_G_29"]:
                (raw_root / site).mkdir(parents=True, exist_ok=True)

            self._build_sparse_case(raw_root / "Site_G_100", "patient-1-4")
            self._build_sparse_case(raw_root / "Site_G_20", "patient-2-4")
            self._build_sparse_case(raw_root / "Site_G_29", "patient-3-4")

            args = Namespace(
                input_root=tmp_path,
                output_root=tmp_path / "processed",
                num_frames=10,
                image_size=32,
                target_label=1,
                train_sites="Site_G_100",
                val_sites="Site_G_20",
                test_sites="Site_G_29",
                overwrite=False,
            )
            output_dir = preprocess_dataset(args)

            dataset = EchoDataset(str(output_dir), mode="train", seq_length=10, size=32)
            sample = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["protocol_name"], "cardiacuda_a4c_lv_sparse")
            self.assertEqual(sample["label_valid"].sum().item(), 6)

            meta_path = output_dir / "train" / "metadata" / "site_g_100__patient-1-4.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["label_indices"][0], 0)
            self.assertEqual(len(meta["label_indices"]), 6)
            self.assertEqual(meta["target_name"], "lv")

    def test_resolve_dataset_class_supports_cardiacuda(self):
        cfg = {
            "dataset_name": "cardiacuda",
            "data_path": "/tmp/cardiacuda_a4c_lv_png128_10f",
        }
        dataset_name, dataset_cls = resolve_dataset_class(cfg)
        self.assertEqual(dataset_name, "cardiacuda")
        self.assertIs(dataset_cls, EchoDataset)


if __name__ == "__main__":
    unittest.main()
