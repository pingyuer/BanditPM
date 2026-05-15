import unittest

import torch
from omegaconf import OmegaConf

from model.anchor_ode import UNeXtAnchorODEAffineSegmenter, UNeXtAnchorODESegmenter
from model.losses import LossComputer


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "anchor_ode",
                "aux_loss": {
                    "sensory": {"weight": 0.0},
                    "query": {"weight": 0.0},
                },
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
                    "ode_steps": 1,
                    "ode_gamma": 0.1,
                    "lambda_prior": 0.2,
                    "lambda_multiscale_prior": 0.02,
                    "lambda_geo": 0.05,
                    "lambda_temp_geo": 0.02,
                    "lambda_conf": 0.02,
                    "lambda_slot_balance": 0.001,
                },
            }
        }
    )


def _cfg_v2():
    cfg = _cfg()
    cfg.model.name = "anchor_ode_v2"
    cfg.model.anchor_ode.mode = "current_anchor_affine"
    cfg.model.anchor_ode.pop("anchor_size")
    cfg.model.anchor_ode.prior_residual_clip = 2.0
    cfg.model.anchor_ode.affine_max_translate = 0.12
    cfg.model.anchor_ode.affine_max_scale = 0.10
    cfg.model.anchor_ode.affine_max_rotate = 0.15
    cfg.model.anchor_ode.gate_warmup_iters = 0
    cfg.model.anchor_ode.lambda_base_seg = 0.5
    cfg.model.anchor_ode.lambda_warp_prior = 0.1
    cfg.model.anchor_ode.lambda_affine_reg = 0.02
    return cfg


def _batch(batch_size=2, frames=3, height=32, width=40):
    rgb = torch.randn(batch_size, frames, 1, height, width)
    cls_gt = torch.randint(0, 2, (batch_size, frames, 1, height, width), dtype=torch.long)
    ff_gt = torch.zeros(batch_size, 1, 1, height, width)
    return {
        "rgb": rgb,
        "ff_gt": ff_gt,
        "cls_gt": cls_gt,
        "label_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
        "init_mode": "pred_or_zero",
    }


class AnchorODETests(unittest.TestCase):
    def test_forward_shapes_and_skip_affines(self):
        torch.manual_seed(11)
        model = UNeXtAnchorODESegmenter(_cfg().model)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 40))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 40))
            aux = out[f"memory_aux_{ti}"]["anchor_ode_aux"]
            self.assertIn("slot_weights", aux)
            self.assertIn("prior_logits", aux)
            self.assertIn("confidence", aux)
            for name in ("affine_low", "affine_mid", "affine_high", "affine_dec"):
                self.assertEqual(aux[name].shape, (2, 1, 6))
                self.assertTrue(torch.isfinite(aux[name]).all())
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertEqual(aux["warped_priors"]["low"].shape[-2:], (16, 20))
            self.assertEqual(aux["warped_priors"]["mid"].shape[-2:], (8, 10))
            self.assertEqual(aux["warped_priors"]["high"].shape[-2:], (4, 5))
            self.assertEqual(aux["warped_priors"]["dec"].shape[-2:], (32, 40))

    def test_loss_backward_has_expected_grads(self):
        torch.manual_seed(12)
        cfg = _cfg()
        model = UNeXtAnchorODESegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
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
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("aux_anchor_ode_prior", losses)
        self.assertIn("aux_anchor_ode_geo", losses)
        loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.anchor_bank.anchors["dec"].grad)
        self.assertIsNotNone(model.affine_regressor.net[-1].weight.grad)
        self.assertIsNotNone(model.confidence.net[-1].weight.grad)


class AnchorODEV2Tests(unittest.TestCase):
    def test_current_anchor_affine_forward_contract(self):
        torch.manual_seed(21)
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 40))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 40))
            aux = out[f"memory_aux_{ti}"]["anchor_ode_aux"]
            self.assertEqual(aux["mode"], "current_anchor_affine")
            for key in (
                "base_object_logits",
                "guided_object_logits",
                "final_object_logits",
                "warped_anchor_low",
                "warped_anchor_mid",
                "warped_anchor_high",
                "warped_anchor_dec",
                "base_geometry",
                "final_geometry",
            ):
                self.assertIn(key, aux)
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertEqual(aux["warped_anchor_low"].shape[-2:], (16, 20))
            self.assertEqual(aux["warped_anchor_mid"].shape[-2:], (8, 10))
            self.assertEqual(aux["warped_anchor_high"].shape[-2:], (4, 5))
            self.assertEqual(aux["warped_anchor_dec"].shape[-2:], (32, 40))
            for name in ("affine_low", "affine_mid", "affine_high", "affine_dec"):
                affine = aux[name]
                self.assertTrue(torch.isfinite(affine).all())
                self.assertLessEqual(float(affine[..., 0:2].abs().max()), 0.1201)
                self.assertLessEqual(float((affine[..., 2:4] - 1.0).abs().max()), 0.1001)
                self.assertLessEqual(float(affine[..., 4:6].abs().max()), 0.1501)

    def test_identity_init_stays_close_to_base(self):
        torch.manual_seed(22)
        model = UNeXtAnchorODEAffineSegmenter(_cfg_v2().model)
        out = model(_batch(batch_size=1, frames=1))
        aux = out["memory_aux_0"]["anchor_ode_aux"]
        diff = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertLess(float(diff), 0.05)

    def test_v2_loss_backward_has_expected_grads(self):
        torch.manual_seed(23)
        cfg = _cfg_v2()
        model = UNeXtAnchorODEAffineSegmenter(cfg.model)
        data = _batch(batch_size=2, frames=3)
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
        losses = LossComputer(cfg, stage_cfg).compute(data, [1, 1])
        self.assertIn("aux_anchor_ode_base", losses)
        self.assertIn("aux_anchor_ode_prior", losses)
        self.assertIn("aux_anchor_ode_affine_reg", losses)
        loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.input_down[0].weight.grad)
        self.assertIsNotNone(model.ode_bank.affine_velocity.grad)
        self.assertIsNotNone(model.affine_regressor.net[-1].weight.grad)
        self.assertIsNotNone(model.gate_head[-1].weight.grad)
        self.assertIsNotNone(model.confidence.net[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
