import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.losses import LossComputer
from model.spatial_dynakey import SpatialDynaKeyMemory, segmentation_gain_reward
from model.unext_dynakey import UNeXtDynaKeySegmenter


def _cfg(
    *,
    use_dynakey=True,
    memory_mode="spatial",
    refine=True,
    q_mode="off",
    dynamics="spatial",
    phase=True,
):
    return OmegaConf.create(
        {
            "model": {
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {
                    "type": "dynakey",
                    "dynakey": {
                        "BANK_SIZE": 3,
                        "HIDDEN_DIM": 16,
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
                    "use_dynakey": use_dynakey,
                    "dynakey_memory_mode": memory_mode,
                    "use_phase_retrieval": phase,
                    "readout_type": "spatial_gate" if memory_mode == "spatial" else "global_broadcast",
                    "dynamics_mode": dynamics,
                    "q_policy_mode": q_mode,
                    "enable_q_loss": q_mode == "training",
                    "lambda_q_ce": 1.0,
                    "use_temporal_refine": refine,
                    "use_mask_memory": memory_mode == "global" and use_dynakey,
                    "use_memory_readout": memory_mode == "global" and use_dynakey,
                    "temporal_residual_init_scale": 0.1,
                    "temporal_gate_bias": -2.0,
                    "spatial_memory_slots": 3,
                    "spatial_memory_size": 8,
                    "spatial_memory_confidence_threshold": 0.0,
                    "spatial_memory_fg_ratio_min": 0.0,
                    "spatial_memory_fg_ratio_max": 1.0,
                    "mask_memory_confidence_threshold": 0.0,
                    "mask_memory_fg_ratio_min": 0.0,
                    "mask_memory_fg_ratio_max": 1.0,
                },
            }
        }
    )


def _batch(batch_size=2, frames=3, height=32, width=40, init_mode="pred_or_zero"):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.zeros(batch_size, frames, 1, height, width, dtype=torch.long)
    cls_gt[..., height // 4 : height // 2, width // 4 : width // 2] = 1
    ff_gt = F.one_hot(cls_gt[:, :1, 0], num_classes=2)[..., 1:].permute(0, 1, 4, 2, 3).float()
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": init_mode,
        "current_iter": 2,
    }


class SpatialDynaKeyTests(unittest.TestCase):
    def test_spatial_memory_state_retrieval_and_update_shapes(self):
        mem = SpatialDynaKeyMemory(6, num_slots=2, spatial_size=5, confidence_threshold=0.0, fg_ratio_min=0.0, fg_ratio_max=1.0)
        value = torch.randn(2, 1, 6, 7, 9)
        mask = torch.rand(2, 1, 28, 36)
        read0 = mem.read(value, mask, frame_index=1, total_frames=4)
        self.assertEqual(read0.feature.shape, (2, 1, 6, 5, 5))
        self.assertEqual(read0.mask_prior.shape, (2, 1, 5, 5))
        aux = mem.update(value, mask, frame_index=1, total_frames=4)
        self.assertGreaterEqual(float(aux["spatial_memory_update_rate"]), 0.0)
        read1 = mem.read(value, mask, frame_index=2, total_frames=4)
        self.assertEqual(read1.weights.shape, (2, 1, 2))
        self.assertTrue(torch.isfinite(read1.feature).all())
        self.assertGreater(int(mem._valid.sum().item()), 0)

    def test_segmentation_gain_reward_sign(self):
        gt = torch.ones(2, 1, 8, 8)
        before = torch.full_like(gt, -2.0)
        better = torch.full_like(gt, 2.0)
        worse = torch.full_like(gt, -4.0)
        self.assertTrue((segmentation_gain_reward(before, better, gt) > 0).all())
        self.assertTrue((segmentation_gain_reward(before, worse, gt) < 0).all())
        self.assertIsNone(segmentation_gain_reward(before, better, None))

    def test_config_switches_forward_paths(self):
        cases = [
            ("off", _cfg(use_dynakey=False, memory_mode="global", refine=False)),
            ("global", _cfg(memory_mode="global", refine=True)),
            ("spatial", _cfg(memory_mode="spatial", refine=False)),
            ("spatial_refine", _cfg(memory_mode="spatial", refine=True)),
            ("q_diag", _cfg(memory_mode="spatial", refine=True, q_mode="diagnostic")),
            ("q_train", _cfg(memory_mode="spatial", refine=True, q_mode="training")),
        ]
        for name, cfg in cases:
            with self.subTest(name=name):
                model = UNeXtDynaKeySegmenter(cfg.model)
                out = model(_batch())
                self.assertEqual(out["logits_0"].shape, (2, 2, 32, 40))
                self.assertTrue(torch.isfinite(out["logits_2"]).all())
                aux = out["memory_aux_1"]
                if name == "off":
                    self.assertFalse(aux["dynakey_enabled"])
                if name.startswith("spatial") or name.startswith("q_"):
                    self.assertTrue(aux["spatial_memory_enabled"])
                    self.assertIn("spatial_memory_valid_slots", aux)
                if name == "q_diag":
                    self.assertIn("spatial_q_values", aux)
                    self.assertNotIn("spatial_q_target_action", aux)
                if name == "q_train":
                    self.assertIn("spatial_q_values", aux)
                    self.assertIn("spatial_q_target_action", aux)

    def test_spatial_q_training_backward(self):
        cfg = _cfg(memory_mode="spatial", refine=True, q_mode="training")
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
        out = model(data)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("spatial_q_total", losses)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertIsNotNone(model.q_policy_head[0].weight.grad)
        self.assertIsNotNone(model.spatial_memory_proj.weight.grad)

    def test_spatial_without_q_has_no_q_loss(self):
        cfg = _cfg(memory_mode="spatial", refine=True, q_mode="diagnostic")
        model = UNeXtDynaKeySegmenter(cfg.model)
        data = _batch()
        data.update(model(data))
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 32,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertNotIn("spatial_q_total", losses)


if __name__ == "__main__":
    unittest.main()
