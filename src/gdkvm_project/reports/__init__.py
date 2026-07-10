"""Report export extension points for experiment comparisons."""

from importlib import import_module


def __getattr__(name: str):
    if name in {"compare_runs", "format_markdown", "load_run_artifacts"}:
        compare_module = import_module("gdkvm_project.reports.compare_runs")
        return getattr(compare_module, name)
    raise AttributeError(name)

__all__ = ["compare_runs", "format_markdown", "load_run_artifacts"]
