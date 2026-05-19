from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    mode: str
    iteration: int
    epoch: int
    summary_metrics: dict[str, float] = field(default_factory=dict)
    per_video_metrics: list[dict[str, Any]] = field(default_factory=list)
    per_frame_metrics: list[dict[str, Any]] = field(default_factory=list)
    threshold_sweep: dict[str, float] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)
    visual_artifacts: list[Path] = field(default_factory=list)


class Evaluator:
    """Evaluation boundary that returns structured results for experiment logging."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def evaluate(
        self,
        data_loader,
        mode: str,
        epoch: int,
        run_path: str | Path,
        it: int,
        *,
        full_eval: bool = False,
    ) -> EvaluationResult:
        return self.trainer._run_evaluation_impl(
            data_loader,
            mode,
            epoch,
            run_path,
            it,
            full_eval=full_eval,
        )
