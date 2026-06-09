from pathlib import Path
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.echo import EchoDataset
from dataset.vos_dataset import TenCamusDataset
from model.delay_ode import DelayODEKeyMapSegmenter
from model.gdkvm01 import GDKVM
from training import build_model_from_cfg
from model.unext_dynakey import UNeXtDynaKeySegmenter
from train import resolve_dataset_class


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


CANONICAL_CONFIGS = [
    "gdkvm_echo",
    "gdkvm_camus",
    "gdkvm_domain",
    "kpff_echo",
    "kpff_camus",
    "kpff_domain",
    "unext_fusion_echo",
    "unext_fusion_camus",
    "unext_fusion_domain",
    "delay_ode_echo",
    "delay_ode_camus",
    "delay_ode_domain",
]

BASE_CONFIGS = [
    "_base_/datasets/echo",
    "_base_/datasets/camus",
    "_base_/datasets/domain",
    "_base_/models/gdkvm",
    "_base_/models/kpff",
    "_base_/models/unext_fusion",
    "_base_/models/delay_ode",
    "_base_/runtime/default_runtime",
    "_base_/schedules/default_3k",
]


class FrameworkRefactorTests(unittest.TestCase):
    def _compose(self, name: str):
        with initialize_config_dir(version_base="1.3.2", config_dir=str(CONFIG_DIR), job_name=f"test_{name}"):
            return compose(config_name=name)

    def test_canonical_configs_compose(self):
        for name in CANONICAL_CONFIGS:
            with self.subTest(config=name):
                cfg = self._compose(name)
                self.assertEqual(str(cfg.exp_id), name)
                self.assertIn(str(cfg.model.name), {"gdkvm", "kpff", "unext_fusion", "delay_ode"})
                self.assertIn(str(cfg.dataset_name), {"echonet", "camus", "domain"})
                self.assertEqual(str(cfg.evaluation.init_mode), "pred_or_zero")
                self.assertTrue(bool(cfg.evaluation.exclude_init_frame))
                self.assertEqual(str(cfg.evaluation.protocol_version), "v3_canonical_no_leak")

    def test_gdkvm_configs_use_aligned_eval_window(self):
        expected = {
            "gdkvm_echo": {
                "batch_size": 8,
                "tta": False,
                "tta_modes": ["identity"],
                "postprocess_min_size": 16,
                "binary_closing": True,
                "frame_scope": "supervised_only",
            },
            "gdkvm_camus": {
                "batch_size": 4,
                "tta": True,
                "tta_modes": ["identity", "hflip"],
                "postprocess_min_size": 8,
                "binary_closing": False,
                "frame_scope": "all_available",
            },
        }
        for name, values in expected.items():
            with self.subTest(config=name):
                cfg = self._compose(name)
                self.assertEqual(str(cfg.evaluation.init_mode), "pred_or_zero")
                self.assertTrue(bool(cfg.evaluation.exclude_init_frame))
                self.assertEqual(str(cfg.evaluation.frame_scope), values["frame_scope"])
                self.assertFalse(bool(cfg.evaluation.threshold_search_during_training))
                self.assertEqual(float(cfg.evaluation.threshold_search_start), 0.30)
                self.assertEqual(float(cfg.evaluation.threshold_search_end), 0.75)
                self.assertEqual(float(cfg.evaluation.threshold_search_step), 0.01)
                self.assertEqual(bool(cfg.evaluation.tta.enabled), values["tta"])
                self.assertEqual(list(cfg.evaluation.tta.modes), values["tta_modes"])
                self.assertTrue(bool(cfg.evaluation.postprocess.enabled))
                self.assertEqual(int(cfg.evaluation.postprocess.min_size), values["postprocess_min_size"])
                self.assertEqual(bool(cfg.evaluation.postprocess.binary_closing), values["binary_closing"])
                self.assertFalse(bool(cfg.eval_stage.test_every_eval))
                self.assertEqual(int(cfg.eval_stage.eval_interval), 500)
                self.assertEqual(int(cfg.eval_stage.num_vis), 0)
                self.assertEqual(int(cfg.main_training.num_iterations), 4000)
                self.assertEqual(int(cfg.main_training.batch_size), values["batch_size"])
                self.assertEqual(int(cfg.main_training.num_workers), 4)

    def test_legacy_gdkvm_eval_windows_are_removed(self):
        forbidden = [
            "config_gdkvm_01.yaml",
            "config_gdkvm_bpm.yaml",
            "config_gdkvm_proto_fast.yaml",
            "config_gdkvm_proto_slow.yaml",
            "config_gdkvm_proto_ablate_fuse_concat.yaml",
            "config_gdkvm_proto_ablate_no_temporal.yaml",
            "config_gdkvm_proto_ablate_replace.yaml",
        ]
        for filename in forbidden:
            with self.subTest(filename=filename):
                self.assertFalse((CONFIG_DIR / filename).exists())

    def test_base_configs_compose(self):
        for name in BASE_CONFIGS:
            with self.subTest(config=name):
                cfg = self._compose(name)
                self.assertGreater(len(OmegaConf.to_container(cfg, resolve=False)), 0)

    def test_dataset_aliases_resolve(self):
        cases = {
            "echonet": EchoDataset,
            "camus": TenCamusDataset,
            "domain": EchoDataset,
            "cardiacuda": EchoDataset,
        }
        for dataset_name, expected_cls in cases.items():
            with self.subTest(dataset_name=dataset_name):
                resolved_name, dataset_cls = resolve_dataset_class(
                    {"dataset_name": dataset_name, "data_path": "/tmp/unused"}
                )
                self.assertEqual(resolved_name, dataset_name)
                self.assertIs(dataset_cls, expected_cls)

    def test_model_builder_canonical_names(self):
        base = {
            "model": {
                "name": "gdkvm",
                "allow_oracle_init_when_requested": False,
                "use_kpff": True,
                "memory_core": {"type": "original_gdr"},
                "temporal_memory": {"type": "none", "bpm": {}},
                "prototype_value": {"enable": False},
            }
        }
        gdkvm = build_model_from_cfg(OmegaConf.create(base), torch.device("cpu"))
        self.assertIsInstance(gdkvm, GDKVM)
        self.assertTrue(gdkvm.use_kpff)

        kpff_cfg = OmegaConf.create(base)
        kpff_cfg.model.name = "kpff"
        kpff = build_model_from_cfg(kpff_cfg, torch.device("cpu"))
        self.assertIsInstance(kpff, GDKVM)
        self.assertTrue(kpff.use_kpff)
        self.assertEqual(kpff.memory_core.memory_type, "none")

        unext_cfg = OmegaConf.create(
            {
                "model": {
                    "name": "unext_fusion",
                    "allow_oracle_init_when_requested": False,
                    "memory_core": {"type": "none", "dynakey": {}},
                    "temporal_memory": {"type": "none", "bpm": {}},
                    "unext_dynakey": {
                        "in_channels": 1,
                        "num_classes": 2,
                        "base_dim": 8,
                        "value_dim": 16,
                        "use_dynakey": False,
                        "use_temporal_refine": False,
                        "use_mask_memory": False,
                    },
                }
            }
        )
        unext = build_model_from_cfg(unext_cfg, torch.device("cpu"))
        self.assertIsInstance(unext, UNeXtDynaKeySegmenter)

        delay_cfg = OmegaConf.create(
            {
                "model": {
                    "name": "delay_ode",
                    "memory_core": {"type": "none", "dynakey": {}},
                    "temporal_memory": {"type": "none", "bpm": {}},
                    "delay_ode": {
                        "in_channels": 1,
                        "num_classes": 2,
                        "base_dim": 8,
                        "delay_ode_value_dim": 16,
                        "delay_ode_key_dim": 12,
                        "delay_ode_state_dim": 20,
                        "delay_ode_num_slots": 4,
                    },
                }
            }
        )
        delay_ode = build_model_from_cfg(delay_cfg, torch.device("cpu"))
        self.assertIsInstance(delay_ode, DelayODEKeyMapSegmenter)


if __name__ == "__main__":
    unittest.main()
