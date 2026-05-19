import re
import unittest

from omegaconf import OmegaConf

from experiment.metadata import resolve_mlflow_experiment_name, resolve_mlflow_run_name


def _cfg(model_name, *, exp_id="exp", memory_type="none", use_dynakey=False):
    return OmegaConf.create(
        {
            "model_name": model_name,
            "exp_id": exp_id,
            "dataset_name": "echonet",
            "seed": 7,
            "data": {"protocol_name": "ed2es"},
            "mlflow": {"experiment_name": None, "run_name": None},
            "model": {
                "name": model_name,
                "memory_core": {"type": memory_type},
                "unext_dynakey": {"use_dynakey": use_dynakey},
            },
        }
    )


class MLflowNamingTests(unittest.TestCase):
    def test_experiment_name_method_families(self):
        cases = [
            (_cfg("anchor_ode_v2"), "anchor_ode"),
            (_cfg("functional_anchor"), "functional_anchor"),
            (_cfg("gdkvm"), "gdkvm"),
            (_cfg("BanditPM"), "gdkvm"),
            (_cfg("kpff"), "kpff"),
            (_cfg("unext_fusion"), "unext_fusion"),
            (_cfg("delay_ode"), "delay_ode"),
            (_cfg("unext_fusion", memory_type="dynakey"), "dynakey"),
            (_cfg("gdkvm", memory_type="dynakey"), "dynakey"),
            (_cfg("unext_only"), "unext_baseline"),
        ]
        for cfg, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(resolve_mlflow_experiment_name(cfg), expected)

    def test_explicit_experiment_name_wins(self):
        cfg = _cfg("gdkvm")
        cfg.mlflow.experiment_name = "manual"
        self.assertEqual(resolve_mlflow_experiment_name(cfg), "manual")

    def test_run_name_includes_protocol_timestamp_and_git(self):
        cfg = _cfg("gdkvm", exp_id="gdkvm_echo")
        run_name = resolve_mlflow_run_name(cfg, timestamp="0519-1032", git_hash="abc1234")
        self.assertEqual(
            run_name,
            "gdkvm_echonet_ed2es_train_s7_0519-1032_abc1234",
        )
        self.assertRegex(run_name, re.compile(r".*_s7_\d{4}-\d{4}_[0-9a-z]+$"))


if __name__ == "__main__":
    unittest.main()
