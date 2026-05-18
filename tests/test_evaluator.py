import unittest

from evaluation import EvaluationResult, Evaluator


class _TrainerStub:
    def _run_evaluation_impl(self, data_loader, mode, epoch, run_path, it):
        return EvaluationResult(
            mode=mode,
            epoch=epoch,
            iteration=it,
            summary_metrics={"dice": 0.8},
            per_video_metrics=[{"video": "a", "dice": 0.8}],
            per_frame_metrics=[{"video": "a", "frame": 0, "dice": 0.8}],
            threshold_sweep={"0.50": 0.8},
            postprocess={"enabled": True},
        )


class EvaluatorTests(unittest.TestCase):
    def test_returns_structured_result(self):
        result = Evaluator(_TrainerStub()).evaluate([], "val", 2, "/tmp/run", 10)
        self.assertEqual(result.mode, "val")
        self.assertEqual(result.summary_metrics["dice"], 0.8)
        self.assertEqual(result.per_video_metrics[0]["video"], "a")
        self.assertEqual(result.per_frame_metrics[0]["frame"], 0)
        self.assertEqual(result.threshold_sweep["0.50"], 0.8)
        self.assertTrue(result.postprocess["enabled"])


if __name__ == "__main__":
    unittest.main()
