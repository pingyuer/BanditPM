import unittest
from unittest import mock

from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

from model.gdkvm01 import GDKVM


class GDKVMTemporalModeTests(unittest.TestCase):
    @mock.patch("model.gdkvm01.resnet.resnet18")
    @mock.patch("model.gdkvm01.resnet.resnet50")
    def test_bpm_mode_freezes_gdr_only_parameters(self, mock_resnet50, mock_resnet18):
        class DummyBackbone:
            def __init__(self):
                import torch.nn as nn

                self.conv1 = nn.Conv2d(1, 64, kernel_size=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(1)
                self.layer1 = nn.Identity()
                self.layer2 = nn.Identity()
                self.layer3 = nn.Identity()

        mock_resnet50.return_value = DummyBackbone()
        mock_resnet18.return_value = DummyBackbone()

        with initialize(version_base="1.3.2", config_path="../config"):
            cfg = compose(config_name="gdkvm_echo")
        with open_dict(cfg):
            cfg.model.temporal_memory = OmegaConf.create(
                {
                    "type": "bpm",
                    "bpm": {
                        "ENABLE": True,
                        "USE_RULE_BASED_POLICY": True,
                        "USE_LEARNED_POLICY": True,
                        "EXEC_POLICY": "rule",
                        "ENABLE_POLICY_LOSS": True,
                        "ENABLE_POLICY_CE_LOSS": True,
                        "ENABLE_RL_LOSS": False,
                        "TRAIN_POLICY_ONLY": False,
                        "FREEZE_BACKBONE": False,
                        "BANK_SIZE": 4,
                        "PROTO_ALPHA": 0.1,
                        "REFINE_EMA_ALPHA": 0.2,
                        "POLICY_WARMUP_EPOCHS": 20,
                        "POLICY_LOSS_WEIGHT": 0.2,
                        "LAMBDA_POLICY_CE": 0.2,
                        "LAMBDA_RL": 0.05,
                        "LAMBDA_ENTROPY": 0.001,
                        "RL_BASELINE_MOMENTUM": 0.95,
                        "ADV_CLAMP": 1.0,
                        "EPSILON_RULE_MIX_INIT": 1.0,
                        "EPSILON_RULE_MIX_FINAL": 0.1,
                        "EPSILON_RULE_MIX_EPOCHS": 30,
                        "SPAWN_WITHOUT_EMPTY_SLOT": "replace_fallback",
                        "SIM_THRESHOLD_HIGH": 0.8,
                        "SIM_THRESHOLD_LOW": 0.5,
                        "FUSION_TYPE": "add",
                        "PROTO_POOLING": "mask",
                        "DEBUG_MODE": False,
                    },
                }
            )
        model = GDKVM(
            use_first_frame_gt_init=bool(cfg.model.get("use_first_frame_gt_init", True)),
            prototype_value_cfg=cfg.model.get("prototype_value", None),
            temporal_memory_cfg=cfg.model.get("temporal_memory", None),
            memory_core_cfg=cfg.model.get("memory_core", None),
            use_kpff=bool(cfg.model.get("use_kpff", True)),
        )

        self.assertFalse(model.memory_core.gdr_core.A_log.requires_grad)
        self.assertFalse(model.memory_core.gdr_core.dt_bias.requires_grad)
        self.assertFalse(model.memory_core.gdr_core.b_proj.weight.requires_grad)
        self.assertFalse(model.memory_core.gdr_core.a_proj.weight.requires_grad)
        self.assertIsNotNone(model.bpm_key_adapter)


if __name__ == "__main__":
    unittest.main()
