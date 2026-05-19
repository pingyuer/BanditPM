import sys
import tempfile
import types
import unittest
from unittest import mock

from utils.mlflow_logger import MLflowLogger


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
                    "residual_boundary_ratio": 0.62,
                    "area_curve_smoothness": 0.01,
                    "anchor_temporal_consistency": 0.02,
                    "slot_entropy": 1.2,
                    "ed_slot_usage": 0.6,
                    "es_slot_usage": 0.5,
                    "slot_area_order_violation": 0.0,
                    "gate_mean_low": 0.4,
                    "gate_mean_mid": 0.5,
                    "gate_mean_high": 0.6,
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
                    "functional_anchor/residual_boundary_ratio": 0.62,
                    "functional_anchor/area_curve_smoothness": 0.01,
                    "functional_anchor/anchor_temporal_consistency": 0.02,
                    "functional_anchor/slot_entropy": 1.2,
                    "functional_anchor/ED_slot_usage": 0.6,
                    "functional_anchor/ES_slot_usage": 0.5,
                    "functional_anchor/slot_area_order_violation": 0.0,
                    "functional_anchor/gate_mean_low": 0.4,
                    "functional_anchor/gate_mean_mid": 0.5,
                    "functional_anchor/gate_mean_high": 0.6,
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
                    "train/functional_anchor/anchor": 0.2,
                    "train/functional_anchor/residual_l1": 0.01,
                },
                3,
            ),
            calls,
        )


if __name__ == "__main__":
    unittest.main()
