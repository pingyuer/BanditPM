import unittest
from unittest import mock

from omegaconf import OmegaConf

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

        cfg = OmegaConf.load("config/config_gdkvm_bpm.yaml")
        model = GDKVM(
            use_first_frame_gt_init=bool(cfg.model.get("use_first_frame_gt_init", True)),
            prototype_value_cfg=cfg.model.get("prototype_value", None),
            temporal_memory_cfg=cfg.model.get("temporal_memory", None),
        )

        self.assertFalse(model.A_log.requires_grad)
        self.assertFalse(model.dt_bias.requires_grad)
        self.assertFalse(model.b_proj.weight.requires_grad)
        self.assertFalse(model.a_proj.weight.requires_grad)
        self.assertIsNotNone(model.bpm_key_adapter)


if __name__ == "__main__":
    unittest.main()
