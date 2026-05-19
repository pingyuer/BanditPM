import unittest

import torch
from omegaconf import OmegaConf

from model.functional_anchor import FunctionalAnchorSegmenter
from model.functional_anchor.confidence_fusion import ConfidenceFusion
from model.functional_anchor.residual_heads import ResidualHeads
from models.registry import build_model


def _cfg(prediction_mode="base_primary"):
    return OmegaConf.create(
        {
            "model": {
                "name": "functional_anchor",
                "aux_loss": {"sensory": {"weight": 0.0}, "query": {"weight": 0.0}},
                "memory_core": {"type": "none", "dynakey": {}},
                "temporal_memory": {"type": "none", "bpm": {}},
                "functional_anchor": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "base_dim": 8,
                    "value_dim": 16,
                    "num_slots": 5,
                    "state_dim": 24,
                    "phase_dim": 8,
                    "hidden_dim": 32,
                    "anchor_size": 8,
                    "residual_clip": 1.0,
                    "prediction_mode": prediction_mode,
                    "training_stage": "joint_residual",
                    "use_anchor_features_in_residual": True,
                    "temporal_state": {"detach_state": True, "detach_every": 3},
                    "lambda_anchor": 0.5,
                    "lambda_base_seg": 0.1,
                    "lambda_residual_smallness": 0.02,
                    "lambda_boundary_residual": 0.1,
                    "lambda_phase_consistency": 0.02,
                    "lambda_anchor_temporal": 0.02,
                    "lambda_slot_area_order": 0.01,
                    "lambda_phase_slot_correlation": 0.01,
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


class FunctionalAnchorForwardTests(unittest.TestCase):
    def test_registry_builds_and_forward_contract(self):
        torch.manual_seed(101)
        model = build_model(_cfg(), device="cpu")
        self.assertIsInstance(model, FunctionalAnchorSegmenter)
        out = model(_batch())
        for ti in range(3):
            self.assertEqual(out[f"logits_{ti}"].shape, (2, 2, 32, 32))
            self.assertEqual(out[f"masks_{ti}"].shape, (2, 1, 32, 32))
            aux = out[f"memory_aux_{ti}"]["functional_anchor_aux"]
            for key in (
                "base_object_logits",
                "anchor_logits",
                "shape_residual_logits",
                "boundary_residual_logits",
                "final_object_logits",
                "slot_weights",
                "phase_embed",
                "z_state",
                "confidence",
                "anchor_features",
                "trust_mean",
                "delta_abs_mean",
                "state_norm",
                "phase_source",
            ):
                self.assertIn(key, aux)
            self.assertEqual(aux["slot_weights"].shape, (2, 1, 5))
            self.assertTrue(torch.allclose(aux["slot_weights"].sum(dim=-1), torch.ones(2, 1), atol=1.0e-5))

    def test_default_is_base_primary_and_initial_output_stays_near_base(self):
        torch.manual_seed(1011)
        model = FunctionalAnchorSegmenter(_cfg().model)
        self.assertEqual(model.prediction_mode, "base_primary")
        out = model(_batch(batch_size=1, frames=2))
        aux = out["memory_aux_1"]["functional_anchor_aux"]
        diff = (aux["final_object_logits"] - aux["base_object_logits"]).abs().mean()
        self.assertLess(float(diff), 0.1)
        self.assertLess(float(aux["residual_abs_mean"]), 1.0e-6)

    def test_phase_changes_anchor_and_slot_selection(self):
        torch.manual_seed(102)
        model = FunctionalAnchorSegmenter(_cfg().model).eval()
        data = _batch(batch_size=1, frames=3)
        data_a = dict(data)
        data_b = dict(data)
        data_a["phase_override"] = torch.zeros(1, 3)
        data_b["phase_override"] = torch.full((1, 3), 0.5)
        with torch.no_grad():
            out_a = model(data_a)
            out_b = model(data_b)
        aux_a = out_a["memory_aux_1"]["functional_anchor_aux"]
        aux_b = out_b["memory_aux_1"]["functional_anchor_aux"]
        self.assertFalse(torch.allclose(aux_a["anchor_logits"], aux_b["anchor_logits"]))
        self.assertFalse(torch.allclose(aux_a["slot_weights"], aux_b["slot_weights"]))

    def test_anchor_generation_does_not_depend_on_base_logits_path(self):
        torch.manual_seed(103)
        model = FunctionalAnchorSegmenter(_cfg().model).eval()
        data = _batch(batch_size=1, frames=2)
        with torch.no_grad():
            out_normal = model(data)
            model.backbone.logit_head.weight.zero_()
            model.backbone.logit_head.bias.zero_()
            out_zero_base = model(data)
        anchor = out_zero_base["memory_aux_1"]["functional_anchor_aux"]["anchor_logits"]
        self.assertTrue(torch.isfinite(anchor).all())
        self.assertGreater(float(anchor.abs().mean()), 0.0)
        self.assertFalse(
            torch.allclose(
                out_normal["memory_aux_1"]["functional_anchor_aux"]["base_object_logits"],
                out_zero_base["memory_aux_1"]["functional_anchor_aux"]["base_object_logits"],
            )
        )

    def test_prediction_modes_have_distinct_paths(self):
        anchor = torch.ones(1, 1, 8, 8)
        base = torch.zeros(1, 1, 8, 8)
        shape = torch.full((1, 1, 8, 8), 0.4)
        boundary = torch.full((1, 1, 8, 8), 0.2)
        trust = torch.full((1, 1, 8, 8), 0.25)
        outputs = {}
        for mode in ("anchor_primary", "base_primary", "learned_blend", "residual_only"):
            outputs[mode], aux = ConfidenceFusion(mode, residual_clip=1.0)(
                anchor_logits=anchor,
                base_logits=base,
                shape_residual=shape,
                boundary_residual=boundary,
                anchor_trust=trust,
            )
            self.assertEqual(outputs[mode].shape, (1, 1, 8, 8))
            self.assertIn("residual_logits", aux)
        self.assertTrue(torch.allclose(outputs["anchor_primary"], torch.full((1, 1, 8, 8), 1.6)))
        self.assertTrue(torch.allclose(outputs["base_primary"], torch.full((1, 1, 8, 8), 0.4)))
        self.assertTrue(torch.allclose(outputs["learned_blend"], torch.full((1, 1, 8, 8), 0.4)))
        self.assertTrue(torch.allclose(outputs["residual_only"], torch.full((1, 1, 8, 8), 1.3)))

    def test_residual_head_anchor_feature_ablation_changes_inputs(self):
        torch.manual_seed(104)
        dims = {"low": 4, "mid": 8, "high": 16, "dec": 4}
        feats = {
            "low": torch.randn(1, 4, 16, 16),
            "mid": torch.randn(1, 8, 8, 8),
            "high": torch.randn(1, 16, 4, 4),
            "dec": torch.randn(1, 4, 32, 32),
        }
        anchor_features = {level: torch.randn(1, 1, dim, 8, 8) for level, dim in dims.items()}
        anchor = torch.randn(1, 1, 32, 32)
        base = torch.randn(1, 1, 32, 32)
        with_anchor = ResidualHeads(dims, 12, 1.0, use_anchor_features=True)
        without_anchor = ResidualHeads(dims, 12, 1.0, use_anchor_features=False)
        without_anchor.load_state_dict(with_anchor.state_dict())
        with torch.no_grad():
            with_anchor.shape_head[-1].weight.fill_(0.01)
            with_anchor.boundary_head[-1].weight.fill_(0.01)
            without_anchor.shape_head[-1].weight.fill_(0.01)
            without_anchor.boundary_head[-1].weight.fill_(0.01)
            out_a = with_anchor(feats, anchor, base, anchor_features)
            out_b = without_anchor(feats, anchor, base, anchor_features)
        self.assertFalse(torch.allclose(out_a["shape_residual_logits"], out_b["shape_residual_logits"]))
        self.assertFalse(torch.allclose(out_a["boundary_residual_logits"], out_b["boundary_residual_logits"]))

    def test_phase_source_priority_metadata_area_curve_then_normalized_time(self):
        torch.manual_seed(105)
        model = FunctionalAnchorSegmenter(_cfg().model).eval()
        metadata_batch = _batch(batch_size=1, frames=3)
        metadata_batch["ed_frame"] = torch.tensor([0])
        metadata_batch["es_frame"] = torch.tensor([2])
        with torch.no_grad():
            out_meta = model(metadata_batch)
            out_area = model(_batch(batch_size=1, frames=3))
            out_norm = model(_batch(batch_size=1, frames=1))
        self.assertEqual(float(out_meta["memory_aux_1"]["functional_anchor_aux"]["phase_source"].mean()), 0.0)
        self.assertEqual(float(out_area["memory_aux_1"]["functional_anchor_aux"]["phase_source"].mean()), 1.0)
        self.assertEqual(float(out_norm["memory_aux_0"]["functional_anchor_aux"]["phase_source"].mean()), 2.0)

    def test_detach_every_controls_state_gradient_history(self):
        cfg = _cfg()
        cfg.model.functional_anchor.temporal_state.detach_every = 0
        model_no_detach = FunctionalAnchorSegmenter(cfg.model)
        data = _batch(batch_size=1, frames=2)
        data["rgb"].requires_grad_(True)
        out_no_detach = model_no_detach(data)
        out_no_detach["logits_1"].mean().backward()
        grad_no_detach = data["rgb"].grad[:, 0].abs().sum()

        cfg.model.functional_anchor.temporal_state.detach_every = 1
        model_detach = FunctionalAnchorSegmenter(cfg.model)
        data = _batch(batch_size=1, frames=2)
        data["rgb"].requires_grad_(True)
        out_detach = model_detach(data)
        out_detach["logits_1"].mean().backward()
        grad_detach = data["rgb"].grad[:, 0].abs().sum()
        self.assertGreater(float(grad_no_detach), 0.0)
        self.assertEqual(float(grad_detach), 0.0)


if __name__ == "__main__":
    unittest.main()
