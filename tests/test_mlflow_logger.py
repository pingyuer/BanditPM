import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from experiment import MLflowLogger
from evaluation import EvaluationResult


class _FakeRun:
    info = types.SimpleNamespace(run_id="run-123")


class MLflowLoggerTests(unittest.TestCase):
    def test_start_resume_and_log_calls(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
            set_experiment=lambda name: calls.append(("experiment", name)),
            start_run=lambda **kwargs: calls.append(("start_run", kwargs)) or _FakeRun(),
            end_run=lambda status="FINISHED": calls.append(("end_run", status)),
            log_params=lambda params: calls.append(("params", params)),
            log_param=lambda key, value: calls.append(("param", key, value)),
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
            log_artifacts=lambda path, artifact_path=None: calls.append(("artifacts", Path(path).name, artifact_path)),
            set_tags=lambda tags: calls.append(("tags", tags)),
            set_tag=lambda key, value: calls.append(("tag", key, value)),
        )
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "tracking_uri": "http://test-mlflow-server:5000",
                "experiment_name": "anchor_ode",
                "run_name": None,
                "resume_run_id": "abc",
                "required": True,
                "artifacts_required": True,
            }
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger(cfg, run_dir=tmp, enabled=True, main_process=True)
            logger.start_run()
            logger.log_config(OmegaConf.create({"model": {"name": "debug"}, "seed": 7}))
            logger.log_metrics({"dice": 0.9, "bad": float("nan"), "proposal/top5_cover_rate@0.85": 1.0}, step=3, prefix="val")
            logger.mark_failed()

        self.assertIn(("tracking_uri", "http://test-mlflow-server:5000"), calls)
        self.assertIn(("experiment", "anchor_ode"), calls)
        self.assertIn(("start_run", {"run_id": "abc"}), calls)
        self.assertIn(("metrics", {"val/dice": 0.9, "val/proposal/top5_cover_rate_0.85": 1.0}, 3), calls)
        self.assertIn(("end_run", "FAILED"), calls)
        self.assertTrue(any(call == ("artifact", "config.yaml", "configs") for call in calls))
        self.assertTrue(any(call == ("artifact", "config_resolved.yaml", "configs") for call in calls))
        self.assertTrue(any(call == ("artifact", "overrides.txt", "configs") for call in calls))

    def test_log_config_uses_param_whitelist(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_params=lambda params: calls.append(("params", params)),
            log_param=lambda key, value: calls.append(("param", key, value)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
        )
        cfg = OmegaConf.create(
            {
                "model": {"name": "debug", "huge_blob": {"nested": "kept out"}},
                "main_training": {"learning_rate": 1.0e-4, "lr_schedule_steps": [100, 200]},
                "hydra": {"runtime": {"output_dir": "/tmp/should_not_be_param"}},
                "callbacks": {"private": "nope"},
                "seed": 7,
            }
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_config(cfg)
        merged_params = {}
        for call in calls:
            if call[0] == "params":
                merged_params.update(call[1])
        self.assertEqual(merged_params, {})
        self.assertNotIn("hydra.runtime.output_dir", merged_params)
        self.assertNotIn("callbacks.private", merged_params)
        self.assertTrue(any(call == ("artifact", "config_resolved.yaml", "configs") for call in calls))

    def test_log_run_metadata_writes_tags_and_core_params(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            set_tags=lambda tags: calls.append(("tags", tags)),
            log_params=lambda params: calls.append(("params", params)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_run_metadata(
                tags={"run_type": "train", "method": "gdkvm"},
                params={"train.lr": 1.0e-4},
            )
        self.assertIn(("tags", {"run_type": "train", "method": "gdkvm"}), calls)
        self.assertIn(("params", {"train.lr": "0.0001"}), calls)

    def test_structured_metric_helpers(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_train_step({"total_loss": 1.0, "dice_loss": 0.2, "lr": 1.0e-4}, step=5)
            logger.log_eval_summary({"dice_frame_mean": 0.8, "iou_frame_mean": 0.7}, mode="val", step=6)
            logger.log_best({"dice_frame_mean": 0.8, "iou_frame_mean": 0.7, "hd95": 2.0}, epoch=1, iteration=6)
            logger.log_cardia_diagnostics(
                {"stage2_head_usage_entropy": 0.5, "stage2_flow_smooth": 0.01, "boundary_edge_gate_mean": 0.3},
                step=6,
            )
        self.assertIn(("metrics", {"train/loss/total": 1.0, "train/loss/dice": 0.2, "train/lr": 1.0e-4}, 5), calls)
        self.assertTrue(
            any(
                call[0] == "metrics"
                and call[1].get("val/dice") == 0.8
                and call[1].get("val/iou") == 0.7
                and call[1].get("val/overall/Dice") == 0.8
                for call in calls
            )
        )
        self.assertIn(("metrics", {"best/val_dice": 0.8, "best/val_iou": 0.7, "best/val_hd95": 2.0, "best/epoch": 1.0, "best/iter": 6.0}, 6), calls)
        self.assertTrue(
            any(call[0] == "metrics" and call[1].get("cardia/stage2/head_usage_entropy") == 0.5 for call in calls)
        )

    def test_eval_summary_logs_phase_and_overall_metrics(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_eval_summary(
                {
                    "ed_dice": 0.91,
                    "es_dice": 0.82,
                    "ed_hd95": 3.0,
                    "es_hd95": 5.0,
                    "overall_dice": 0.87,
                    "overall_hd95": 4.0,
                },
                mode="test",
                step=12,
            )
        merged = {}
        for _, metrics, _ in calls:
            merged.update(metrics)
        self.assertEqual(merged["test/phase/ED_Dice"], 0.91)
        self.assertEqual(merged["test/phase/ES_Dice"], 0.82)
        self.assertEqual(merged["test/phase/ED_HD95"], 3.0)
        self.assertEqual(merged["test/phase/ES_HD95"], 5.0)
        self.assertEqual(merged["test/overall/Dice"], 0.87)
        self.assertEqual(merged["test/overall/HD95"], 4.0)

    def test_evaluation_result_artifacts_are_opt_in(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
        )
        result = EvaluationResult(
            mode="val",
            iteration=10,
            epoch=1,
            summary_metrics={"dice": 0.8},
            threshold_sweep={"0.50": 0.8},
            per_video_metrics=[{"video": "a", "dice": 0.8}],
            per_frame_metrics=[{"video": "a", "frame": 0, "dice": 0.8}],
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger({"required": True}, run_dir=tmp, enabled=True, main_process=True)
            logger.log_evaluation_result(result, step=10, log_artifacts=False)
            logger.log_evaluation_result(result, step=11, log_artifacts=True)
        self.assertTrue(any(call[0] == "metrics" and call[1].get("val/dice") == 0.8 for call in calls if call[2] == 10))
        self.assertTrue(any(call[0] == "metrics" and call[1].get("val/dice") == 0.8 for call in calls if call[2] == 11))
        artifact_calls = [call for call in calls if call[0] == "artifact"]
        self.assertTrue(any(call == ("artifact", "summary.json", "eval") for call in artifact_calls))
        self.assertEqual(
            [call for call in calls if call[0] == "artifact" and call[1] == "summary.json"],
            [("artifact", "summary.json", "eval")],
        )

    def test_geomaskformer_evaluation_result_suppresses_standard_aliases(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_metrics=lambda metrics, step=None: calls.append(("metrics", metrics, step)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
        )
        result = EvaluationResult(
            mode="val",
            iteration=10,
            epoch=1,
            summary_metrics={
                "dice": 0.8,
                "iou": 0.7,
                "ed_dice": 0.9,
                "es_dice": 0.75,
                "area_smoothness": 0.01,
                "temporal_drift": 0.2,
            },
        )
        cfg = OmegaConf.create({"required": True, "model": {"name": "geomaskformer"}})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            logger = MLflowLogger(cfg, run_dir=tmp, enabled=True, main_process=True)
            logger.log_evaluation_result(result, step=10, log_artifacts=False)
        logged = {}
        for _, metrics, _ in calls:
            logged.update(metrics)
        assert "val/dice" not in logged
        assert "val/overall/Dice" not in logged
        assert "val/phase/ED_Dice" not in logged
        assert "val/area_smoothness" not in logged
        assert "val/temporal_drift" not in logged

    def test_run_logs_uploads_train_and_command_logs(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            tmp_path = Path(tmp)
            train_log = tmp_path / "train.log"
            command_log = tmp_path / "command.log"
            train_log.write_text("train\n", encoding="utf-8")
            command_log.write_text("command\n", encoding="utf-8")
            logger = MLflowLogger(
                {"required": True, "command_log_path": str(command_log)},
                run_dir=tmp,
                enabled=True,
                main_process=True,
            )
            logger.log_run_logs()
        self.assertIn(("artifact", "train.log", "logs"), calls)
        self.assertIn(("artifact", "command.log", "logs"), calls)

    def test_artifact_paths_and_eval_tags_are_flat(self):
        calls = []
        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
            set_experiment=lambda name: calls.append(("experiment", name)),
            start_run=lambda **kwargs: calls.append(("start_run", kwargs)) or _FakeRun(),
            set_tags=lambda tags: calls.append(("tags", tags)),
            log_artifact=lambda path, artifact_path=None: calls.append(("artifact", Path(path).name, artifact_path)),
        )
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "tracking_uri": "http://test-mlflow-server:5000",
                "experiment_name": "evals",
                "run_name": "eval-debug",
                "resume_run_id": None,
                "required": True,
                "artifacts_required": True,
            }
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            tmp_path = Path(tmp)
            ckpt = tmp_path / "best_raw.pth"
            ckpt.write_text("checkpoint", encoding="utf-8")
            logger = MLflowLogger(cfg, run_dir=tmp, enabled=True, main_process=True)
            logger.start_eval_run(
                source_run_id="train-1",
                source_checkpoint="best_raw.pth",
                eval_mode="tta",
                dataset="echonet",
                protocol="ed2es",
            )
            logger.log_checkpoint(ckpt, artifact_name="best_raw.pth")
            logger.log_env_info()
            logger.log_git_info()

        self.assertIn(("artifact", "best_raw.pth", "checkpoints"), calls)
        self.assertTrue(any(call == ("artifact", "runtime.json", "env") for call in calls))
        self.assertTrue(any(call == ("artifact", "git.json", "source") for call in calls))
        self.assertTrue(
            any(
                call[0] == "tags"
                and call[1]["run_type"] == "eval"
                and call[1]["source_run_id"] == "train-1"
                and call[1]["source_checkpoint"] == "best_raw.pth"
                for call in calls
            )
        )

    def test_preflight_success_and_failure(self):
        calls = []

        class FakeMLflow:
            def set_tracking_uri(self, uri):
                calls.append(("tracking_uri", uri))

            def set_experiment(self, name):
                calls.append(("experiment", name))

            def start_run(self, **kwargs):
                calls.append(("start_run", kwargs))
                return self

            def __enter__(self):
                return _FakeRun()

            def __exit__(self, exc_type, exc, tb):
                calls.append(("end_context", exc_type))

            def set_tag(self, key, value):
                calls.append(("tag", key, value))

            def log_metric(self, key, value, step=None):
                calls.append(("metric", key, value, step))

            def log_artifact(self, path, artifact_path=None):
                calls.append(("artifact", Path(path).name, artifact_path))

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": FakeMLflow()}):
            logger = MLflowLogger({"tracking_uri": "uri", "experiment_name": "exp", "required": True}, run_dir=tmp)
            logger.preflight()
        self.assertTrue(any(call == ("metric", "preflight/alive", 1.0, 0) for call in calls))

        failing = types.SimpleNamespace(set_tracking_uri=lambda uri: None, set_experiment=lambda name: None, start_run=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"mlflow": failing}):
            logger = MLflowLogger({"required": True}, run_dir=tmp)
            with self.assertRaises(RuntimeError):
                logger.preflight()

    def test_non_main_process_is_noop(self):
        cfg = OmegaConf.create({"enabled": True})
        logger = MLflowLogger(cfg, run_dir=".", enabled=True, main_process=False)
        logger.start_run()
        logger.log_metrics({"loss": 1.0})
        self.assertIsNone(logger.run_id)


if __name__ == "__main__":
    unittest.main()
