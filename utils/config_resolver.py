"""
Configuration resolver utilities for AI project setup.

This module provides tools for validating and resolving configuration,
following AI project best practices:
- Configuration injection via environment variables
- No hardcoded values for sensitive information
- Clear error messages for missing configurations
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def resolve_mlflow_config(cfg: Any) -> dict[str, Any]:
    """
    Resolve MLflow configuration from various sources.

    Priority order:
    1. YAML config (via Hydra)
    2. Environment variables
    3. Sensible defaults (localhost only)

    Args:
        cfg: Hydra DictConfig or similar config object

    Returns:
        Dictionary with resolved MLflow configuration

    Raises:
        ConfigValidationError: If required configuration is missing
    """
    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}

    tracking_uri = (
        mlflow_cfg.get("tracking_uri")
        or os.environ.get("MLFLOW_TRACKING_URI")
        or "http://localhost:5000"
    )

    experiment_name = (
        mlflow_cfg.get("experiment_name")
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    )

    run_name = (
        mlflow_cfg.get("run_name")
        or os.environ.get("MLFLOW_RUN_NAME")
    )

    return {
        "enabled": mlflow_cfg.get("enabled", True),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "stage": mlflow_cfg.get("stage", "full"),
        "required": mlflow_cfg.get("required", True),
    }


def resolve_data_config(cfg: Any) -> dict[str, Any]:
    """
    Resolve data configuration from various sources.

    Args:
        cfg: Hydra DictConfig or similar config object

    Returns:
        Dictionary with resolved data configuration
    """
    data_root = (
        cfg.get("data_root")
        or os.environ.get("DATA_ROOT")
        or str(Path.home() / "datasets")
    )

    processed_root = cfg.get("processed_root") or f"{data_root}/processed"

    return {
        "data_root": data_root,
        "processed_root": processed_root,
    }


def validate_mlflow_config(cfg: Any) -> None:
    """
    Validate MLflow configuration.

    Args:
        cfg: Resolved MLflow configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    tracking_uri = cfg.get("tracking_uri")
    if not tracking_uri:
        raise ConfigValidationError(
            "MLflow tracking_uri is not configured. "
            "Please set MLFLOW_TRACKING_URI environment variable "
            "or mlflow.tracking_uri in your YAML config."
        )


def validate_data_config(cfg: Any) -> None:
    """
    Validate data configuration.

    Args:
        cfg: Resolved data configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    data_root = cfg.get("data_root")
    if not data_root:
        raise ConfigValidationError(
            "Data root is not configured. "
            "Please set DATA_ROOT environment variable "
            "or data_root in your YAML config."
        )

    data_path = Path(data_root)
    if not data_path.exists():
        raise ConfigValidationError(
            f"Data root does not exist: {data_root}. "
            "Please ensure the path is correct or set DATA_ROOT "
            "environment variable."
        )
