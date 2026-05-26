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
                    "prediction_mode": "proposal_primary",
                    "residual_scale": {"init": 0.03, "max": 0.15, "warmup_iters": 1500},
                    "lambda_faf_affine": 0.001,
                    "lambda_faf_velocity": 0.001,
                    "lambda_faf_anchor": 0.3,
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
                "proposal_logits",
                "anchor_logits",
                "anchor_proposals",
                "active_weights",
                "trust",
                "coverage_score",
                "effective_anchor_number",
                "affine_delta",
                "affine_delta_norm",
                "retrieval_temperature",
                "residual_scale",
                "safety_residual_logits",
                "base_safety_logits",
                "proposal_corrected_logits",
                "prediction_mode",
                "trust_easy_mean",
                "trust_hard_mean",
                "anchor_area_separation",
                "canonical_sdf",
                "basis_weights",
                "function_codes",
            ):
                self.assertIn(key, aux)
            self.assertEqual(aux["anchor_proposals"].shape, (2, 1, 4, 32, 32))
            self.assertTrue(torch.allclose(aux["active_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertTrue(torch.isfinite(aux["effective_anchor_number"]).all())
            self.assertTrue(torch.isfinite(aux["anchor_proposals"]).all())
            self.assertTrue(torch.allclose(aux["anchor_logits"], aux["proposal_logits"]))
            self.assertEqual(aux["prediction_mode"], "proposal_primary")
            self.assertTrue(torch.allclose(aux["final_object_logits"], aux["proposal_corrected_logits"]))
            self.assertLessEqual(float(aux["affine_delta"][..., 0].abs().max()), 0.0801)
            self.assertLessEqual(float(aux["affine_delta"][..., 2].abs().max()), 0.0501)

    def test_proposal_primary_final_follows_proposal_not_base_safety(self):
        torch.manual_seed(302)
        model = UNeXtFAF(_cfg().model)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        final_to_proposal = (aux["final_object_logits"] - aux["proposal_corrected_logits"]).abs().mean()
        final_to_base = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        base_safety_to_base = (aux["base_safety_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertTrue(torch.isfinite(final_to_base))
        self.assertLess(float(final_to_proposal), 1.0e-6)
        self.assertGreater(float(final_to_base), float(base_safety_to_base) + 0.01)
        self.assertGreater(float(aux["trust"].mean()), 0.05)

    def test_base_safety_mode_preserves_previous_fusion_behavior(self):
        torch.manual_seed(302)
        cfg = _cfg()
        cfg.model.unext_faf.prediction_mode = "base_safety"
        model = UNeXtFAF(cfg.model)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(aux["prediction_mode"], "base_safety")
        self.assertTrue(torch.allclose(aux["final_object_logits"], aux["base_safety_logits"]))
        self.assertFalse(torch.allclose(aux["final_object_logits"], aux["proposal_corrected_logits"]))

    def test_ode_update_can_be_disabled(self):
        torch.manual_seed(303)
        cfg = _cfg()
        cfg.model.unext_faf.enable_memory_update = False
        model = UNeXtFAF(cfg.model)
        out = model(_batch(batch_size=1, frames=2))
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(float(aux["write_strength_mean"]), 0.0)
        self.assertEqual(float(aux["memory_update_norm"]), 0.0)

    def test_function_codes_are_diverse(self):
        torch.manual_seed(304)
        model = UNeXtFAF(_cfg().model)
        codes = model.faf.field_memory.anchor_function_codes
        self.assertEqual(codes.shape, (4, 16))
        # Function codes should be different (randomly initialized)
        pairwise_diff = (codes.unsqueeze(0) - codes.unsqueeze(1)).abs().mean(dim=-1)
        off_diag = pairwise_diff[~torch.eye(4, dtype=torch.bool)]
        self.assertGreater(float(off_diag.mean()), 0.001)

    def test_initial_anchor_fields_are_diverse(self):
        torch.manual_seed(305)
        model = UNeXtFAF(_cfg().model)
        canonical_sdf = model.faf.field_memory.decode_static_field()
        basis_weights = model.faf.field_memory.get_basis_weights()

        self.assertEqual(canonical_sdf.shape, (4, 1, 8, 8))
        self.assertEqual(basis_weights.shape, (4, 4))
        sdf_diff = (canonical_sdf.unsqueeze(0) - canonical_sdf.unsqueeze(1)).abs().mean(dim=(-1, -2, -3))
        weight_diff = (basis_weights.unsqueeze(0) - basis_weights.unsqueeze(1)).abs().mean(dim=-1)
        off_diag = ~torch.eye(4, dtype=torch.bool)
        self.assertGreater(float(sdf_diff[off_diag].mean()), 0.01)
        self.assertGreater(float(weight_diff[off_diag].mean()), 0.05)

    def test_default_anchor_count_does_not_duplicate_basis_weights(self):
        torch.manual_seed(306)
        cfg = _cfg()
        cfg.model.unext_faf.num_anchors = 8
        cfg.model.unext_faf.basis_dim = 6
        model = UNeXtFAF(cfg.model)
        basis_weights = model.faf.field_memory.get_basis_weights()
        weight_diff = (basis_weights.unsqueeze(0) - basis_weights.unsqueeze(1)).abs().mean(dim=-1)
        off_diag = ~torch.eye(8, dtype=torch.bool)
        self.assertGreater(float(weight_diff[off_diag].min()), 0.001)

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
        for param in model.faf.residual_refiner.parameters():
            self.assertIn(id(param), residual_ids)
        for param in model.faf.trust_gate_net.parameters():
            self.assertIn(id(param), residual_ids)
        for param in model.faf.proposal_generator.parameters():
            self.assertIn(id(param), method_ids)


if __name__ == "__main__":
    unittest.main()
