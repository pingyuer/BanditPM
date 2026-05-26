import unittest

import torch
from omegaconf import OmegaConf

from model.unext_faf import UNeXtFAF
from models.registry import build_model


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
                    "residual_clip": 0.5,
                    "trust_max": 0.6,
                    "retrieval_temperature": 0.4,
                    "memory_ema": 0.9,
                    "lambda_faf_affine": 0.001,
                    "lambda_faf_velocity": 0.001,
                    "lambda_faf_anchor": 0.3,
                    "lambda_faf_base": 0.2,
                    "lambda_faf_coverage": 0.05,
                    "lambda_faf_sparse": 0.002,
                    "lambda_faf_diversity": 0.001,
                    "lambda_faf_temporal": 0.02,
                    "lambda_faf_write": 0.001,
                    "lambda_faf_residual_smallness": 0.05,
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
                "anchor_proposals",
                "active_weights",
                "trust",
                "memory_stats",
                "coverage_score",
                "effective_anchor_number",
                "affine_delta",
                "affine_delta_norm",
                "affine_velocity_norm",
                "retrieval_temperature",
                "residual_scale",
                "decoder_object_logits",
                "safety_residual_logits",
                "feature_modulation",
                "feature_modulation_l1",
                "feature_modulation_l1_high",
                "trust_easy_mean",
                "trust_hard_mean",
                "anchor_area_separation",
            ):
                self.assertIn(key, aux)
            self.assertEqual(aux["anchor_proposals"].shape, (2, 1, 4, 32, 32))
            self.assertTrue(torch.allclose(aux["active_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))
            self.assertTrue(torch.isfinite(aux["effective_anchor_number"]).all())
            self.assertTrue(torch.isfinite(aux["anchor_proposals"]).all())
            self.assertTrue(torch.isfinite(aux["feature_modulation_l1"]))
            self.assertGreaterEqual(float(aux["feature_modulation_l1_high"]), 0.0)
            self.assertLessEqual(float(aux["affine_delta"][..., 0].abs().max()), 0.0801)
            self.assertLessEqual(float(aux["affine_delta"][..., 2].abs().max()), 0.0501)

    def test_feature_modulation_and_warmup_trust_are_active(self):
        torch.manual_seed(302)
        model = UNeXtFAF(_cfg().model)
        data = _batch(batch_size=1, frames=2)
        data["current_iter"] = 0
        out = model(data)
        aux = out["memory_aux_1"]["faf_aux"]
        diff = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertTrue(torch.isfinite(diff))
        self.assertGreater(float(aux["trust"].mean()), 0.09)
        self.assertGreater(float(aux["feature_modulation_l1"]), 0.0)

    def test_ode_update_can_be_disabled(self):
        torch.manual_seed(303)
        cfg = _cfg()
        cfg.model.unext_faf.enable_memory_update = False
        model = UNeXtFAF(cfg.model)
        out = model(_batch(batch_size=1, frames=2))
        aux = out["memory_aux_1"]["faf_aux"]
        self.assertEqual(float(aux["write_strength_mean"]), 0.0)
        self.assertEqual(float(aux["memory_update_norm"]), 0.0)


if __name__ == "__main__":
    unittest.main()
