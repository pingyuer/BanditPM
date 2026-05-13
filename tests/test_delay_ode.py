import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.delay_ode import DelayODEKeyMapSegmenter
from model.losses import LossComputer


def _cfg(
    num_slots=8,
    use_low=True,
    use_mid=True,
    use_high=True,
    *,
    gamma=0.1,
    steps=1,
    keymap_ema=0.85,
    first_frame_init="init_head",
    allow_current_feature=False,
):
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
                    "delay_ode_steps": steps,
                    "delay_ode_gamma": gamma,
                    "delay_ode_keymap_ema": keymap_ema,
                    "delay_ode_first_frame_init": first_frame_init,
                    "delay_ode_update_gate_max": 0.5,
                    "delay_ode_use_low": use_low,
                    "delay_ode_use_mid": use_mid,
                    "delay_ode_use_high": use_high,
                    "delay_ode_supervise_first_frame": False,
                    "delay_ode_allow_current_feature_for_current_mask": allow_current_feature,
                    "delay_ode_lambda_slot_balance": 0.001,
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
        with torch.no_grad():
            out_a = model(data_a)
            out_b = model(data_b)
        self.assertTrue(torch.allclose(out_a["logits_2"], out_b["logits_2"], atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(out_a["logits_3"], out_b["logits_3"], atol=1e-5, rtol=1e-5))

    def test_future_loss_backprops_to_previous_frame_input(self):
        torch.manual_seed(4)
        model = DelayODEKeyMapSegmenter(_cfg().model).train()
        data = _batch(batch_size=1, frames=4, requires_grad=True)
        out = model(data)
        loss = out["logits_2"][:, 1].mean()
        loss.backward()
        self.assertGreater(data["rgb"].grad[:, 1].abs().sum().item(), 0.0)

    def test_current_loss_does_not_backprop_to_current_frame_input(self):
        torch.manual_seed(5)
        model = DelayODEKeyMapSegmenter(_cfg().model).train()
        data = _batch(batch_size=1, frames=4, requires_grad=True)
        out = model(data)
        loss = out["logits_2"][:, 1].mean()
        loss.backward()
        self.assertLess(data["rgb"].grad[:, 2].abs().max().item(), 1.0e-8)

    def test_first_frame_init_mask_is_valid_and_affects_future(self):
        torch.manual_seed(6)
        model = DelayODEKeyMapSegmenter(_cfg().model).eval()
        data_a = _batch(batch_size=1, frames=3)
        data_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in data_a.items()}
        data_b["rgb"][:, 0] = data_b["rgb"][:, 0] + 2.0
        with torch.no_grad():
            out_a = model(data_a)
            out_b = model(data_b)
        self.assertFalse(torch.allclose(out_a["logits_1"], out_b["logits_1"], atol=1e-5, rtol=1e-5))
        self.assertGreater(out_a["logits_0"].abs().mean().item(), 0.0)
        self.assertGreater(out_a["masks_0"].mean().item(), 0.0)
        aux0 = out_a["memory_aux_0"]["delay_ode_aux"]
        self.assertTrue(torch.allclose(aux0["mask_stats"][:, :, 0, 0], out_a["masks_0"].mean(dim=(-2, -1)), atol=1.0e-5))

    def test_multi_scale_keymaps_participate(self):
        torch.manual_seed(7)
        data = _batch(batch_size=1, frames=4)
        full = DelayODEKeyMapSegmenter(_cfg().model).eval()
        with torch.no_grad():
            out = full(data)
        aux = out["memory_aux_3"]["delay_ode_aux"]
        for level in ("low", "mid", "high"):
            self.assertIn(level, aux["keymap_weights"])
            self.assertIn(level, aux["update_gates"])
        for level in ("low", "mid", "high"):
            ablated = DelayODEKeyMapSegmenter(_cfg().model).eval()
            ablated.load_state_dict(full.state_dict())
            ablated.level_enabled[level] = False
            with torch.no_grad():
                out_ablated = ablated(data)
            self.assertFalse(
                torch.allclose(out["logits_3"], out_ablated["logits_3"], atol=1e-5, rtol=1e-5),
                msg=f"{level} ablation did not affect logits",
            )

    def test_slot_capacity_configs(self):
        for slots in (4, 6, 8, 12):
            with self.subTest(slots=slots):
                model = DelayODEKeyMapSegmenter(_cfg(num_slots=slots).model)
                out = model(_batch(batch_size=1, frames=3))
                aux = out["memory_aux_2"]["delay_ode_aux"]
                self.assertEqual(aux["keymap_weights"]["low"].shape[-1], slots)

    def test_t_equals_one_outputs_empty_temporal_aux(self):
        model = DelayODEKeyMapSegmenter(_cfg().model)
        out = model(_batch(batch_size=1, frames=1))
        aux = out["memory_aux_0"]["delay_ode_aux"]
        self.assertEqual(aux["keymap_weights"]["low"].shape, (1, 1, 0, 8))
        self.assertEqual(aux["update_gates"]["low"].shape, (1, 1, 0, 1))
        self.assertTrue(torch.isfinite(out["logits_0"]).all())

    def test_gamma_controls_ode_update(self):
        torch.manual_seed(8)
        cfg_base = _cfg(gamma=0.1)
        model = DelayODEKeyMapSegmenter(cfg_base.model).eval()
        gamma0 = DelayODEKeyMapSegmenter(_cfg(gamma=0.0).model).eval()
        gamma0.load_state_dict(model.state_dict())
        gamma1 = DelayODEKeyMapSegmenter(_cfg(gamma=0.2).model).eval()
        gamma1.load_state_dict(model.state_dict())
        data = _batch(batch_size=1, frames=3)
        with torch.no_grad():
            out0 = gamma0(data)
            out1 = gamma1(data)
        self.assertFalse(torch.allclose(out0["logits_2"], out1["logits_2"], atol=1e-6, rtol=1e-6))

    def test_steps_use_sub_dt_not_longer_total_time(self):
        torch.manual_seed(9)
        one = DelayODEKeyMapSegmenter(_cfg(steps=1).model).eval()
        four = DelayODEKeyMapSegmenter(_cfg(steps=4).model).eval()
        four.load_state_dict(one.state_dict())
        data = _batch(batch_size=1, frames=3)
        with torch.no_grad():
            out1 = one(data)
            out4 = four(data)
        diff = (out1["logits_2"] - out4["logits_2"]).abs().mean()
        self.assertLess(diff.item(), 0.05)

    def test_keymap_ema_reduces_update_magnitude(self):
        torch.manual_seed(10)
        fast = DelayODEKeyMapSegmenter(_cfg(keymap_ema=0.0).model).eval()
        slow = DelayODEKeyMapSegmenter(_cfg(keymap_ema=0.95).model).eval()
        slow.load_state_dict(fast.state_dict())
        data = _batch(batch_size=1, frames=3)
        with torch.no_grad():
            fast_aux = fast(data)["memory_aux_2"]["delay_ode_aux"]
            slow_aux = slow(data)["memory_aux_2"]["delay_ode_aux"]
        self.assertGreater(
            fast_aux["effective_update_gates"]["mid"].mean().item(),
            slow_aux["effective_update_gates"]["mid"].mean().item(),
        )

    def test_slot_balance_loss_prefers_uniform_usage(self):
        model = DelayODEKeyMapSegmenter(_cfg(num_slots=4).model)
        uniform = [{"low": torch.full((1, 1, 4), 0.25), "mid": torch.full((1, 1, 4), 0.25), "high": torch.full((1, 1, 4), 0.25)}]
        collapsed = [{"low": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), "mid": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), "high": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])}]
        state = [torch.zeros(1, 1, model.state_dim)]
        latent = [{level: torch.zeros(1, 1, model.value_dim, 2, 2) for level in ("low", "mid", "high")}]
        gates = []
        self.assertGreater(
            model._regularizers(collapsed, gates, state, latent)["slot_balance"].item(),
            model._regularizers(uniform, gates, state, latent)["slot_balance"].item(),
        )

    def test_loss_dict_contains_delay_ode_aux_terms(self):
        cfg = _cfg()
        model = DelayODEKeyMapSegmenter(cfg.model)
        data = _batch(batch_size=1, frames=3)
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
        losses = LossComputer(cfg, stage_cfg).compute(data, [1])
        self.assertIn("aux_delay_ode_slot_balance", losses)
        self.assertIn("aux_delay_ode_total", losses)

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_smoke(self):
        cfg = _cfg()
        model = DelayODEKeyMapSegmenter(cfg.model).cuda()
        data = _batch(batch_size=1, frames=3)
        data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}
        data["info"] = {"num_objects": torch.ones(1, dtype=torch.long, device="cuda")}
        out = model(data)
        loss = out["logits_2"].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
