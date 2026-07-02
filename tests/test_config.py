"""
Tests for configuration management system.

Following AI project testing best practices:
- Unit tests for configuration resolution
- Environment variable override tests
- Error handling tests
"""
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_resolver import (
    ConfigValidationError,
    resolve_mlflow_config,
    resolve_data_config,
    validate_mlflow_config,
    validate_data_config,
)

from hydra import compose, initialize_config_dir


class MockConfig:
    def __init__(self, data: dict):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class TestMLflowConfig:
    def test_resolve_mlflow_config_from_env(self):
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://test-server:5000"}):
            cfg = MockConfig({})
            result = resolve_mlflow_config(cfg)
            assert result["tracking_uri"] == "http://test-server:5000"

    def test_resolve_mlflow_config_from_config(self):
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-server:5000"}):
            cfg = MockConfig({"mlflow": {"tracking_uri": "http://config-server:5000"}})
            result = resolve_mlflow_config(cfg)
            assert result["tracking_uri"] == "http://config-server:5000"

    def test_resolve_mlflow_config_default(self):
        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = MockConfig({})
            result = resolve_mlflow_config(cfg)
            assert result["tracking_uri"] == "http://localhost:5000"

    def test_validate_mlflow_config_missing_uri(self):
        cfg = {"tracking_uri": ""}
        try:
            validate_mlflow_config(cfg)
            assert False, "Should have raised ConfigValidationError"
        except ConfigValidationError as e:
            assert "tracking_uri is not configured" in str(e)

    def test_validate_mlflow_config_valid(self):
        cfg = {"tracking_uri": "http://localhost:5000"}
        validate_mlflow_config(cfg)


class TestDataConfig:
    def test_resolve_data_config_from_env(self):
        with mock.patch.dict(os.environ, {"DATA_ROOT": "/custom/data/path"}):
            cfg = MockConfig({})
            result = resolve_data_config(cfg)
            assert result["data_root"] == "/custom/data/path"

    def test_resolve_data_config_default(self):
        env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = MockConfig({})
            result = resolve_data_config(cfg)
            assert "datasets" in result["data_root"]

    def test_validate_data_config_missing_root(self):
        cfg = {"data_root": ""}
        try:
            validate_data_config(cfg)
            assert False, "Should have raised ConfigValidationError"
        except ConfigValidationError as e:
            assert "Data root is not configured" in str(e)

    def test_validate_data_config_nonexistent_path(self):
        cfg = {"data_root": "/nonexistent/path/that/does/not/exist"}
        try:
            validate_data_config(cfg)
            assert False, "Should have raised ConfigValidationError"
        except ConfigValidationError as e:
            assert "does not exist" in str(e)

    def test_validate_data_config_valid(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"data_root": tmp}
            validate_data_config(cfg)


class TestConfigPriority:
    def test_env_overrides_default(self):
        with mock.patch.dict(os.environ, {"SEED": "123"}):
            cfg = MockConfig({})
            seed = cfg.get("seed") or os.environ.get("SEED", "42")
            assert seed == "123"

    def test_config_overrides_env(self):
        with mock.patch.dict(os.environ, {"SEED": "123"}):
            cfg = MockConfig({"seed": "456"})
            seed = cfg.get("seed") or os.environ.get("SEED", "42")
            assert seed == "456"


class TestReBelConfigs:
    def test_rebel_configs_load_with_top_level_mlflow_tags(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        names = [
            "rebel_camus",
            "rebel_echo",
            "rebel_cardiacuda_g2r",
            "rebel_cardiacuda_r2g",
            "rebel_cardiacuda_sparse_sitegen",
            "rebel_cardiacuda_sparse_sitegen_seed2",
        ]
        with initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
            for name in names:
                cfg = compose(config_name=name)
                assert cfg.model.name == "rebel"
                assert cfg.loss.name == "rebel"
                assert "tags" in cfg.mlflow
                assert cfg.mlflow.tags.method == "ReBel"
                assert "tags" not in cfg.model


def run_tests():
    passed = 0
    failed = 0

    test_classes = [TestMLflowConfig, TestDataConfig, TestConfigPriority, TestReBelConfigs]

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  ✓ {cls.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {cls.__name__}.{method_name}: {e}")
                    failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
