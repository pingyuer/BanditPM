import unittest

import torch
from omegaconf import OmegaConf

from dataset.echo import EchoDataset
from dataset.registry import DATASET_REGISTRY, resolve_dataset_class_from_cfg
from dataset.vos_dataset import TenCamusDataset
from model.anchor_ode import UNeXtAnchorODESegmenter
from model.delay_ode import DelayODEKeyMapSegmenter
from model.gdkvm01 import GDKVM
from model.registry import MODEL_REGISTRY
from model.unext_dynakey import UNeXtDynaKeySegmenter
from utils.registry import Registry


class RegistryBuilderTests(unittest.TestCase):
    def test_registry_register_duplicate_and_unknown(self):
        registry = Registry("toy")

        @registry.register("thing")
        def build_thing(cfg, **kwargs):
            return {"cfg": cfg, **kwargs}

        self.assertEqual(registry.build({"name": "thing"}, flag=True)["flag"], True)
        with self.assertRaises(KeyError):
            registry.register("thing", module=lambda cfg: cfg)
        with self.assertRaises(KeyError):
            registry.build({"name": "missing"})

    def test_dataset_registry_aliases(self):
        cases = {
            "echo": EchoDataset,
            "echonet": EchoDataset,
            "domain": EchoDataset,
            "cardiacuda": EchoDataset,
            "camus": TenCamusDataset,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertIs(DATASET_REGISTRY.get(name), expected)
                resolved_name, resolved_cls = resolve_dataset_class_from_cfg(
                    {"dataset_name": name, "data_path": "/tmp/unused"}
                )
                self.assertEqual(resolved_name, name)
                self.assertIs(resolved_cls, expected)

    def test_model_registry_builds_canonical_and_aliases(self):
        gdkvm_cfg = OmegaConf.create(
            {
                "model": {
                    "name": "gdkvm",
                    "allow_oracle_init_when_requested": False,
                    "use_kpff": True,
                    "memory_core": {"type": "original_gdr"},
                    "temporal_memory": {"type": "none", "bpm": {}},
                    "prototype_value": {"enable": False},
                }
            }
        )
        self.assertIsInstance(MODEL_REGISTRY.build(gdkvm_cfg, device=torch.device("cpu")), GDKVM)

        kpff_cfg = OmegaConf.create(gdkvm_cfg)
        kpff_cfg.model.name = "kpff"
        kpff = MODEL_REGISTRY.build(kpff_cfg, device=torch.device("cpu"))
        self.assertIsInstance(kpff, GDKVM)
        self.assertEqual(kpff.memory_core.memory_type, "none")

        for alias in ["unext_fusion", "unext_dynakey", "dynakey_unext", "unextdynakey"]:
            with self.subTest(alias=alias):
                cfg = OmegaConf.create(
                    {
                        "model": {
                            "name": alias,
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
                self.assertIsInstance(
                    MODEL_REGISTRY.build(cfg, device=torch.device("cpu")),
                    UNeXtDynaKeySegmenter,
                )

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
        self.assertIsInstance(MODEL_REGISTRY.build(delay_cfg, device=torch.device("cpu")), DelayODEKeyMapSegmenter)

        for alias in ["anchor_ode", "unext_anchor_ode"]:
            with self.subTest(alias=alias):
                cfg = OmegaConf.create(
                    {
                        "model": {
                            "name": alias,
                            "memory_core": {"type": "none", "dynakey": {}},
                            "temporal_memory": {"type": "none", "bpm": {}},
                            "anchor_ode": {
                                "in_channels": 1,
                                "num_classes": 2,
                                "base_dim": 8,
                                "value_dim": 16,
                                "num_slots": 4,
                                "state_dim": 24,
                                "hidden_dim": 24,
                                "anchor_size": 16,
                                "condition_dim": 12,
                            },
                        }
                    }
                )
                self.assertIsInstance(
                    MODEL_REGISTRY.build(cfg, device=torch.device("cpu")),
                    UNeXtAnchorODESegmenter,
                )


if __name__ == "__main__":
    unittest.main()
