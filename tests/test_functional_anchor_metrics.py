import sys
import tempfile
import types
import unittest
from unittest import mock

from experiment import MLflowLogger


class FunctionalAnchorMetricsTests(unittest.TestCase):
    def test_functional_anchor_diagnostics_use_stable_metric_names(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_functional_anchor_diagnostics(
                {
                    "base_dice": 0.70,
                    "anchor_only_dice": 0.74,
                    "final_dice": 0.80,
                    "residual_l1": 0.03,
                    "residual_l2": 0.05,
                    "residual_abs_mean": 0.03,
                    "residual_abs_max": 0.09,
                    "delta_abs_mean": 0.12,
                    "residual_boundary_ratio": 0.62,
                    "area_curve_smoothness": 0.01,
                    "area_acceleration": 0.02,
                    "temporal_jitter": 0.03,
                    "anchor_temporal_consistency": 0.02,
                    "slot_entropy": 1.2,
                    "ed_slot_usage": 0.6,
                    "es_slot_usage": 0.5,
                    "slot_area_order_violation": 0.0,
                    "slot_order_loss": 0.0,
                    "slot_area_ed": 0.85,
                    "slot_area_early_systole": 0.62,
                    "slot_area_es": 0.30,
                    "slot_area_early_diastole": 0.58,
                    "slot_area_uncertain": 0.50,
                    "phase_source": 1.0,
                    "phase_reliability": 1.0,
                    "state_norm": 2.0,
                    "state_delta_norm": 0.2,
                    "ode_update_norm": 0.1,
                    "gate_mean_low": 0.4,
                    "gate_mean_mid": 0.5,
                    "gate_mean_high": 0.6,
                    "inject_gate_low": 0.4,
                    "inject_gate_mid": 0.5,
                    "inject_gate_high": 0.6,
                    "inject_gate_dec": 0.7,
                    "trust_mean": 0.7,
                    "trust_std": 0.1,
                    "anchor_trust_ratio": 0.7,
                },
                step=9,
            )
        self.assertIn(
            (
                "metrics",
                {
                    "functional_anchor/base_dice": 0.70,
                    "functional_anchor/anchor_only_dice": 0.74,
                    "functional_anchor/final_dice": 0.80,
                    "functional_anchor/final_minus_base": 0.10000000000000009,
                    "functional_anchor/final_minus_anchor": 0.06000000000000005,
                    "functional_anchor/residual_l1": 0.03,
                    "functional_anchor/residual_l2": 0.05,
                    "functional_anchor/residual_abs_mean": 0.03,
                    "functional_anchor/residual_abs_max": 0.09,
                    "functional_anchor/delta_abs_mean": 0.12,
                    "functional_anchor/residual_boundary_ratio": 0.62,
                    "functional_anchor/area_curve_smoothness": 0.01,
                    "functional_anchor/area_acceleration": 0.02,
                    "functional_anchor/temporal_jitter": 0.03,
                    "functional_anchor/anchor_temporal_consistency": 0.02,
                    "functional_anchor/slot_entropy": 1.2,
                    "functional_anchor/ED_slot_usage": 0.6,
                    "functional_anchor/ES_slot_usage": 0.5,
                    "functional_anchor/slot_area_order_violation": 0.0,
                    "functional_anchor/slot_order_loss": 0.0,
                    "functional_anchor/slot_area_ed": 0.85,
                    "functional_anchor/slot_area_early_systole": 0.62,
                    "functional_anchor/slot_area_es": 0.30,
                    "functional_anchor/slot_area_early_diastole": 0.58,
                    "functional_anchor/slot_area_uncertain": 0.50,
                    "functional_anchor/phase_source": 1.0,
                    "functional_anchor/phase_reliability": 1.0,
                    "functional_anchor/state_norm": 2.0,
                    "functional_anchor/state_delta_norm": 0.2,
                    "functional_anchor/ode_update_norm": 0.1,
                    "functional_anchor/gate_mean_low": 0.4,
                    "functional_anchor/gate_mean_mid": 0.5,
                    "functional_anchor/gate_mean_high": 0.6,
                    "functional_anchor/inject_gate_low": 0.4,
                    "functional_anchor/inject_gate_mid": 0.5,
                    "functional_anchor/inject_gate_high": 0.6,
                    "functional_anchor/inject_gate_dec": 0.7,
                    "functional_anchor/trust_mean": 0.7,
                    "functional_anchor/trust_std": 0.1,
                    "functional_anchor/anchor_trust_ratio": 0.7,
                },
                9,
            ),
            calls,
        )

    def test_train_step_maps_functional_anchor_losses_under_train_namespace(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_train_step(
                {
                    "total_loss": 1.0,
                    "aux_functional_anchor_anchor": 0.2,
                    "aux_functional_anchor_residual_l1": 0.01,
                },
                step=3,
            )
        self.assertIn(
            (
                "metrics",
                {
                    "train/loss/total": 1.0,
                    "train/loss/weighted/functional_anchor/anchor": 0.2,
                    "train/loss/weighted/functional_anchor/residual_l1": 0.01,
                },
                3,
            ),
            calls,
        )

    def test_eval_summary_uses_split_functional_namespace(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_eval_summary(
                {
                    "dice_frame_mean": 0.8,
                    "iou_frame_mean": 0.7,
                    "functional_anchor/base_dice": 0.72,
                    "functional_anchor/anchor_only_dice": 0.61,
                    "functional_anchor/proposal_dice": 0.77,
                    "functional_anchor/final_dice": 0.8,
                    "functional_anchor/trust_mean": 0.5,
                },
                mode="val",
                step=4,
            )
        merged = {}
        for _, metrics, _ in calls:
            merged.update(metrics)
        self.assertEqual(merged["val/functional_anchor/base_dice"], 0.72)
        self.assertEqual(merged["val/functional_anchor/anchor_only_dice"], 0.61)
        self.assertEqual(merged["val/functional_anchor/proposal_dice"], 0.77)
        self.assertEqual(merged["val/functional_anchor/final_dice"], 0.8)
        self.assertNotIn("functional_anchor/final_dice", merged)
        self.assertFalse(any(key.startswith("val/anchor_ode/") for key in merged))

    def test_faf_diagnostics_use_faf_namespace(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_faf_diagnostics(
                {
                    "affine_oracle_dice": 0.81,
                    "effective_slot_number": 2.4,
                    "coverage_score": 0.7,
                    "base_dice": 0.72,
                    "final_dice": 0.8,
                    "confidence_hard_mean": 0.3,
                    "feature_modulation_l1": 0.02,
                    "hard_frame_final_minus_base": 0.04,
                },
                step=6,
            )
        self.assertIn(
            (
                "metrics",
                {
                    "faf/base_dice": 0.72,
                    "faf/affine_oracle_dice": 0.81,
                    "faf/final_dice": 0.8,
                    "faf/final_minus_base_dice": 0.08000000000000007,
                    "faf/effective_slot_number": 2.4,
                    "faf/coverage_score": 0.7,
                    "faf/confidence_hard_mean": 0.3,
                    "faf/feature_modulation_l1": 0.02,
                    "faf/hard_frame_final_minus_base": 0.04,
                },
                6,
            ),
            calls,
        )

    def test_eval_summary_uses_split_faf_namespace(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_eval_summary(
                {
                    "dice_frame_mean": 0.8,
                    "faf/base_dice": 0.72,
                    "faf/affine_oracle_dice": 0.77,
                    "faf/final_minus_base_dice": 0.08,
                },
                mode="val",
                step=4,
            )
        merged = {}
        for _, metrics, _ in calls:
            merged.update(metrics)
        self.assertEqual(merged["val/faf/base_dice"], 0.72)
        self.assertEqual(merged["val/faf/affine_oracle_dice"], 0.77)
        self.assertEqual(merged["val/faf/final_minus_base_dice"], 0.08)
        self.assertNotIn("faf/affine_oracle_dice", merged)

    def test_plain_eval_summary_does_not_emit_method_diagnostics(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_eval_summary({"dice_frame_mean": 0.8, "base_only_dice_frame_mean": 0.0}, mode="test", step=5)
        merged = {}
        for _, metrics, _ in calls:
            merged.update(metrics)
        self.assertEqual(merged["test/dice"], 0.8)
        self.assertFalse(any("functional_anchor" in key for key in merged))
        self.assertFalse(any("anchor_ode" in key for key in merged))


if __name__ == "__main__":
    unittest.main()
