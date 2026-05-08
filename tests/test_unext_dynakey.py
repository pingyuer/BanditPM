import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.losses import LossComputer
from model.modules.memory_core import MemoryCore
from model.modules.unext.unext import UNeXtBackbone
from model.unext_dynakey import UNeXtDynaKeySegmenter


def _cfg(*, temporal_refine=True, ode_aux=False):
    return OmegaConf.create(
        {
            "model": {
                "aux_loss": {
                    "sensory": {"weight": 0.0},
                    "query": {"weight": 0.0},
                },
                "memory_core": {
                    "type": "dynakey",
                    "dynakey": {
                        "BANK_SIZE": 3,
                        "DT": 1.0,
                        "EMA_ALPHA": 0.3,
                        "HIDDEN_DIM": 32,
                        "GATE_INIT": 1.0,
                        "POLICY_MODE": "fixed_residual",
                        "ENABLE_Q_LOSS": False,
                    },
                },
                "temporal_memory": {"type": "dynakey", "bpm": {}},
                "unext_dynakey": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "use_temporal_refine": temporal_refine,
                    "use_ode_aux_loss": ode_aux,
                },
            }
        }
    )


def _batch(batch_size=2, frames=3, height=32, width=32):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.randint(0, 2, (batch_size, frames, 1, height, width), dtype=torch.long)
    ff_gt = F.one_hot(cls_gt[:, :1, 0], num_classes=2)[..., 1:].permute(0, 1, 4, 2, 3).float()
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": "oracle_gt",
    }


class UNeXtDynaKeyTests(unittest.TestCase):
    def test_unext_backbone_shapes(self):
        model = UNeXtBackbone(in_channels=1, num_classes=2, base_dim=8, value_dim=16)
        out = model(torch.randn(2, 1, 32, 32))
        self.assertEqual(out["logits"].shape, (2, 2, 32, 32))
        self.assertEqual(out["low"].shape[-2:], (16, 16))
        self.assertEqual(out["mid"].shape[-2:], (8, 8))
        self.assertEqual(out["high"].shape[-2:], (4, 4))
        self.assertEqual(out["value"].shape, (2, 16, 8, 8))
        self.assertEqual(out["decoder_feature"].shape[-2:], (32, 32))

    def test_unext_dynakey_forward_with_temporal_refine(self):
        model = UNeXtDynaKeySegmenter(_cfg(temporal_refine=True).model)
        data = _batch()
        out = model(data)
        for t in range(3):
            self.assertEqual(out[f"logits_{t}"].shape, (2, 2, 32, 32))
            self.assertEqual(out[f"masks_{t}"].shape, (2, 1, 32, 32))
            self.assertIn("dynakey_aux", out[f"memory_aux_{t}"])
        self.assertEqual(out["num_objects"], [1, 1])

    def test_unext_dynakey_forward_without_temporal_refine(self):
        model = UNeXtDynaKeySegmenter(_cfg(temporal_refine=False).model)
        out = model(_batch(batch_size=1))
        self.assertEqual(out["logits_1"].shape, (1, 2, 32, 32))
        self.assertTrue(torch.isfinite(out["logits_1"]).all())

    def test_unext_dynakey_backward_has_expected_grads(self):
        cfg = _cfg(temporal_refine=True)
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        data["supervised_indices"] = torch.ones(2, 3, dtype=torch.bool)
        loss_cfg = cfg
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": True,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(loss_cfg, stage_cfg).compute(data, [1, 1])["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.temporal_refine_head[0].weight.grad)
        self.assertIsNotNone(model.temporal_delta_proj.weight.grad)

    def test_existing_dynakey_memory_core_still_runs(self):
        cfg = OmegaConf.create(
            {
                "type": "dynakey",
                "dynakey": {"BANK_SIZE": 2, "HIDDEN_DIM": 16},
            }
        )
        core = MemoryCore(
            value_dim=4,
            key_dim=2,
            temporal_memory_cfg=OmegaConf.create({"type": "dynakey", "dynakey": cfg.dynakey}),
            memory_core_cfg=cfg,
        )
        core.reset_state(1, 1, torch.device("cpu"))
        readout, aux = core(
            torch.randn(1, 1, 4, 2, 2),
            torch.randn(1, 2, 2, 2),
            torch.randn(1, 4, 2, 2),
            torch.ones(1, 1, 8, 8),
        )
        self.assertEqual(readout.shape, (1, 1, 4, 2, 2))
        self.assertEqual(aux["memory_type"], "dynakey")


if __name__ == "__main__":
    unittest.main()
