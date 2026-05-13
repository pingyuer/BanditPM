import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.delay_ode import DelayODEKeyMapSegmenter
from model.losses import LossComputer


def _cfg(num_slots=8, use_low=True, use_mid=True, use_high=True):
    return OmegaConf.create(
        {
            "model": {
                "name": "delay_ode",
                "aux_loss": {
                    "sensory": {"weight": 0.0},
                    "query": {"weight": 0.0},
                },
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "delay_ode": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "delay_ode_num_slots": num_slots,
                    "delay_ode_key_dim": 12,
                    "delay_ode_value_dim": 16,
                    "delay_ode_state_dim": 20,
                    "delay_ode_temperature": 0.07,
                    "delay_ode_dt": 1.0,
                    "delay_ode_steps": 1,
                    "delay_ode_update_gate_max": 0.5,
                    "delay_ode_use_low": use_low,
                    "delay_ode_use_mid": use_mid,
                    "delay_ode_use_high": use_high,
                    "delay_ode_supervise_first_frame": False,
                    "delay_ode_lambda_selection_entropy": 0.001,
                    "delay_ode_lambda_gate_smooth": 0.01,
                    "delay_ode_lambda_latent_smooth": 0.01,
                    "delay_ode_lambda_state_smooth": 0.01,
                },
            }
        }
    )


def _batch(batch_size=2, frames=5, height=32, width=32, requires_grad=False):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    rgb.requires_grad_(requires_grad)
    cls_gt = torch.randint(0, 2, (batch_size, frames, 1, height, width), dtype=torch.long)
    ff_gt = torch.zeros(batch_size, 1, 1, height, width)
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "supervised_indices": torch.tensor([[False] + [True] * (frames - 1)] * batch_size),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": "pred_or_zero",
    }


class DelayODETests(unittest.TestCase):
    def test_forward_shapes(self):
        model = DelayODEKeyMapSegmenter(_cfg().model)
        out = model(_batch(batch_size=2, frames=5, height=112, width=112))
        self.assertEqual(out["logits_0"].shape, (2, 2, 112, 112))
        self.assertEqual(out["masks_0"].shape, (2, 1, 112, 112))
        self.assertEqual(out["logits_4"].shape, (2, 2, 112, 112))
        aux = out["memory_aux_4"]["delay_ode_aux"]
        for level in ("low", "mid", "high"):
            self.assertEqual(aux["keymap_weights"][level].shape, (2, 1, 4, 8))
            self.assertEqual(aux["update_gates"][level].shape[:3], (2, 1, 4))

    def test_current_feature_does_not_leak_to_current_logits(self):
        torch.manual_seed(3)
        model = DelayODEKeyMapSegmenter(_cfg().model).eval()
        data_a = _batch(batch_size=1, frames=4)
        data_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in data_a.items()}
        data_b["rgb"][:, 2] = torch.randn_like(data_b["rgb"][:, 2]) * 4.0
        out_a = model(data_a)
        out_b = model(data_b)
        self.assertTrue(torch.allclose(out_a["logits_2"], out_b["logits_2"], atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(out_a["logits_3"], out_b["logits_3"], atol=1e-5, rtol=1e-5))

    def test_future_loss_backprops_to_previous_frame_input(self):
        torch.manual_seed(4)
        model = DelayODEKeyMapSegmenter(_cfg().model)
        data = _batch(batch_size=1, frames=4, requires_grad=True)
        out = model(data)
        loss = out["logits_2"][:, 1].mean()
        loss.backward()
        self.assertGreater(data["rgb"].grad[:, 1].abs().sum().item(), 0.0)

    def test_current_loss_does_not_backprop_to_current_frame_input(self):
        torch.manual_seed(5)
        model = DelayODEKeyMapSegmenter(_cfg().model)
        data = _batch(batch_size=1, frames=4, requires_grad=True)
        out = model(data)
        loss = out["logits_2"][:, 1].mean()
        loss.backward()
        self.assertLess(data["rgb"].grad[:, 2].abs().max().item(), 1.0e-8)

    def test_warmup_frame_affects_first_predicted_frame(self):
        torch.manual_seed(6)
        model = DelayODEKeyMapSegmenter(_cfg().model).eval()
        data_a = _batch(batch_size=1, frames=3)
        data_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in data_a.items()}
        data_b["rgb"][:, 0] = data_b["rgb"][:, 0] + 2.0
        out_a = model(data_a)
        out_b = model(data_b)
        self.assertFalse(torch.allclose(out_a["logits_1"], out_b["logits_1"], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(out_a["logits_0"], torch.zeros_like(out_a["logits_0"])))

    def test_multi_scale_keymaps_participate(self):
        torch.manual_seed(7)
        data = _batch(batch_size=1, frames=4)
        full = DelayODEKeyMapSegmenter(_cfg().model).eval()
        no_mid = DelayODEKeyMapSegmenter(_cfg(use_mid=False).model).eval()
        out = full(data)
        aux = out["memory_aux_3"]["delay_ode_aux"]
        for level in ("low", "mid", "high"):
            self.assertIn(level, aux["keymap_weights"])
            self.assertIn(level, aux["update_gates"])
        out_no_mid = no_mid(data)
        self.assertFalse(torch.allclose(out["logits_3"], out_no_mid["logits_3"], atol=1e-5, rtol=1e-5))

    def test_slot_capacity_configs(self):
        for slots in (4, 6, 8, 12):
            with self.subTest(slots=slots):
                model = DelayODEKeyMapSegmenter(_cfg(num_slots=slots).model)
                out = model(_batch(batch_size=1, frames=3))
                aux = out["memory_aux_2"]["delay_ode_aux"]
                self.assertEqual(aux["keymap_weights"]["low"].shape[-1], slots)

    def test_train_smoke_with_loss_computer(self):
        cfg = _cfg()
        model = DelayODEKeyMapSegmenter(cfg.model)
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        data = _batch(batch_size=2, frames=4)
        out = model(data)
        data.update(out)
        stage_cfg = OmegaConf.create(
            {
                "point_supervision": False,
                "train_num_points": 64,
                "oversample_ratio": 1.0,
                "importance_sample_ratio": 0.5,
            }
        )
        loss = LossComputer(cfg, stage_cfg).compute(data, [1, 1])["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        opt.step()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
