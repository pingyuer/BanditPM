from __future__ import annotations


SUPPORTED_MODELS = {"gdkvm", "banditpm", "dpfr", "dual_prompt_flow_refinement"}


def validate_project_config(cfg) -> None:
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    model_name = str(model_cfg.get("name", "")).lower() if hasattr(model_cfg, "get") else ""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. This project exposes only gdkvm and dpfr."
        )
