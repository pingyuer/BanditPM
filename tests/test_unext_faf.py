import unittest

import torch
from omegaconf import OmegaConf

from model.unext_faf import UNeXtFAF
from models.registry import build_model
from training.parameter_groups import get_parameter_groups


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "unext_faf",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "unext_faf": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "num_anchors": 4,
                    "query_dim": 16,
                    "code_dim": 16,
                    "hidden_dim": 24,
                    "basis_dim": 4,
                    "anchor_size": 8,
                    "residual_clip": 0.25,
                    "trust_max": 0.8,
                    "retrieval_temperature": 0.25,
                    "memory_ema": 0.9,
                    "prediction_mode": "affine_mixture_safe",
                    "num_affine_slots": 4,
                    "identity_slot_index": 0,
                    "require_pretrained_unext": False,
                    "residual_scale": {"init": 0.0, "max": 0.05, "warmup_iters": 1500},
                    "selector": {"temperature_init": 1.0, "temperature_final": 0.35, "assignment_temperature": 0.15},
                    "confidence": {"enabled": True, "init": 0.10, "max": 0.35, "warmup_iters": 1500},
                    "residual": {"enabled": True, "init_scale": 0.0, "max_scale": 0.05, "clip": 0.15},
                    "temporal_update": {"enabled": True, "dt_init": 0.1, "dt_max": 0.6, "truncated_bptt_steps": 0},
                    "lambda_faf_affine": 0.001,
                    "lambda_faf_velocity": 0.001,
                    "lambda_faf_mixture": 0.3,
                    "lambda_faf_oracle": 0.5,
                    "lambda_faf_top1": 0.2,
                    "lambda_faf_base": 0.1,
                    "lambda_faf_coverage": 0.05,
                    "lambda_faf_sparse": 0.002,
                    "lambda_faf_diversity": 0.005,
                    "lambda_faf_temporal": 0.02,
                    "lambda_faf_write": 0.001,
                    "lambda_faf_residual_smallness": 0.05,
                    "feature_modulation": {"enabled": False},
                },
            }
        }
    )


def _ode_affine_cfg():
    cfg = _cfg()
    cfg.model.name = "unext_ode_affine"
    cfg.model.unext_faf.dense_momentum = {
        "enabled": True,
        "flow_size": 8,
        "hidden_dim": 16,
        "max_displacement": 0.08,
        "integration_steps": 2,
    }
    cfg.model.unext_faf.lambda_faf_dense_flow = 0.001
    cfg.model.unext_faf.lambda_faf_dense_smooth = 0.01
    return cfg


def _batch(batch_size=2, frames=3, height=32, width=32):
    return {
        "rgb": torch.rand(batch_size, frames, 1, height, width),
        "cls_gt": torch.randint(0, 2, (batch_size, frames, 1, height, width)),
        "supervised_indices": torch.ones(batch_size, frames, dtype=torch.bool),
        "info": {"num_objects": torch.ones(batch_size, dtype=torch.long)},
    }


class UNeXtFAFTests(unittest.TestCase):
    def test_registry_builds_and_forward_contract(self):
        torch.manual_seed(301)
        model = build_model(_cfg(), device="cpu")
        self.assertIsInstance(model, UNeXtFAF)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 32))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 32))
            aux = out[f"memory_aux_{ti}"]["faf_aux"]
            for key in (
                "base_logits",
                "final_logits",
                "mixture_logits",
                "anchor_logits",
                "warped_anchor_logits",
                "slot_weights",
                "slot_logits",
                "slot_confidence",
                "trust",
                "coverage_score",
                "effective_slot_number",
                "affine_delta",
                "affine_delta_norm",
                "selector_temperature",
                "residual_scale",
                "safety_residual_logits",
                "prediction_mode",
                "confidence_easy_mean",
                "confidence_hard_mean",
                "affine_identity_logits",
                "affine_top1_logits",
            ):
                self.assertIn(key, aux)
            self.assertEqual(aux["warped_anchor_logits"].shape, (2, 1, 4, 32, 32))
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertTrue(torch.isfinite(aux["effective_slot_number"]).all())
            self.assertTrue(torch.isfinite(aux["warped_anchor_logits"]).all())
            self.assertTrue(torch.allclose(aux["anchor_logits"], aux["base_object_logits"]))
            self.assertEqual(aux["prediction_mode"], "affine_mixture_safe")
            self.assertLessEqual(float(aux["affine_delta"][..., 0].abs().max()), 0.1501)
            self.assertLessEqual(float(aux["affine_delta"][..., 2].abs().max()), 0.1201)

    def test_affine_mixture_safe_initializes_close_to_base(self):
        torch.manual_seed(302)
        model = UNeXtFAF(_cfg().model)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        final_to_base = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        identity_to_base = (aux["affine_identity_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertLess(float(identity_to_base), 1.0e-5)
        self.assertTrue(torch.isfinite(final_to_base))
        self.assertLess(float(final_to_base), 0.05)
        self.assertGreater(float(aux["slot_weights"][..., 0].mean()), 0.5)
        self.assertLess(float(aux["confidence_mean"]), 0.12)

    def test_base_only_mode_outputs_base_logits(self):
        torch.manual_seed(302)
        cfg = _cfg()
        cfg.model.unext_faf.prediction_mode = "base_only"
        model = UNeXtFAF(cfg.model)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(aux["prediction_mode"], "base_only")
        self.assertTrue(torch.allclose(aux["final_object_logits"], aux["base_object_logits"], atol=1.0e-5))

    def test_ode_update_can_be_disabled(self):
        torch.manual_seed(303)
        cfg = _cfg()
        cfg.model.unext_faf.enable_memory_update = False
        model = UNeXtFAF(cfg.model)
        out = model(_batch(batch_size=1, frames=2))
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(float(aux["write_strength_mean"]), 0.0)
        self.assertEqual(float(aux["memory_update_norm"]), 0.0)

    def test_default_slot_count_initializes_identity_bias(self):
        torch.manual_seed(306)
        cfg = _cfg()
        cfg.model.unext_faf.num_anchors = 8
        cfg.model.unext_faf.num_affine_slots = 8
        model = UNeXtFAF(cfg.model)
        data = _batch(batch_size=1, frames=1)
        out = model(data)
        weights = out["memory_aux_0"]["faf_aux"]["slot_weights"]
        self.assertEqual(weights.shape[-1], 8)
        self.assertEqual(int(weights.argmax(dim=-1).item()), 0)

    def test_faf_residual_modules_use_residual_lr_group(self):
        model = UNeXtFAF(_cfg().model)
        stage_cfg = OmegaConf.create(
            {
                "learning_rate": 1.0e-4,
                "weight_decay": 1.0e-3,
                "embed_weight_decay": 0.0,
                "backbone_lr_ratio": 0.1,
                "unext_lr_ratio": 0.5,
                "functional_anchor_lr_ratio": 1.0,
                "residual_head_lr_mult": 5.0,
            }
        )
        groups = get_parameter_groups(model, stage_cfg)
        by_name = {group["name"]: group for group in groups}
        residual_ids = {id(param) for param in by_name["functional_anchor_residual_heads"]["params"]}
        method_ids = {id(param) for param in by_name["functional_anchor"]["params"]}
        self.assertAlmostEqual(by_name["functional_anchor_residual_heads"]["lr"], 5.0e-4)
        for param in model.faf.fusion.parameters():
            self.assertIn(id(param), residual_ids)
        for param in model.faf.affine_mixture.parameters():
            self.assertIn(id(param), method_ids)
        for param in model.faf.selector.parameters():
            self.assertIn(id(param), method_ids)

    def test_unext_ode_affine_dense_momentum_contract(self):
        torch.manual_seed(307)
        model = build_model(_ode_affine_cfg(), device="cpu")
        self.assertIsInstance(model, UNeXtFAF)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(aux["prediction_mode"], "affine_mixture_safe")
        self.assertEqual(float(aux["dense_momentum_enabled"]), 1.0)
        self.assertIn("affine_mixture_logits", aux)
        self.assertIn("dense_displacement", aux)
        self.assertIn("dense_flow_smoothness", aux)
        self.assertEqual(aux["dense_displacement"].shape, (1, 1, 2, 8, 8))
        self.assertTrue(torch.isfinite(aux["dense_displacement"]).all())
        self.assertLess(float(aux["dense_flow_abs_mean"]), 1.0e-6)
        self.assertTrue(torch.allclose(aux["mixture_logits"], aux["affine_mixture_logits"], atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
