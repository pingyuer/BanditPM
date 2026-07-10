import json

import torch

from gdkvm_project.evaluation import align_logits_to_target, binary_dice_iou
from gdkvm_project.reports.compare_runs import compare_runs
from gdkvm_project.tracking import RunRecorder
from gdkvm_project.visualization import render_dpfr_diagnostic_panel, render_sequence_panel


def test_run_recorder_writes_required_artifacts(tmp_path):
    recorder = RunRecorder(tmp_path)
    recorder.log_config({"model": {"name": "dpfr"}})
    recorder.log_runtime()
    recorder.log_git()
    recorder.log_metrics({"dice": 0.5}, step=1, split="val")
    recorder.log_summary({"status": "finished"})
    recorder.log_data_flow_summary({"dataset": "synthetic"})
    recorder.ensure_required_files()
    for name in recorder.REQUIRED_FILES:
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "finished"


def test_render_sequence_panel_smoke(tmp_path):
    batch = {
        "rgb": torch.rand(1, 2, 1, 16, 16),
        "cls_gt": torch.randint(0, 2, (1, 2, 1, 16, 16)),
    }
    output = {"logits": torch.rand(1, 2, 2, 16, 16)}
    path = tmp_path / "panel.png"
    fig = render_sequence_panel(batch, output, save_path=path)
    assert path.exists()
    fig.clf()


def test_render_dpfr_diagnostic_panel_smoke(tmp_path):
    batch = {
        "rgb": torch.rand(1, 2, 1, 16, 16),
        "cls_gt": torch.randint(0, 2, (1, 2, 1, 16, 16)),
    }
    output = {
        "logits": torch.rand(1, 2, 2, 16, 16),
        "final_logits": torch.rand(1, 2, 2, 16, 16),
        "anchor_logits": torch.rand(1, 2, 2, 16, 16),
        "prompt_logits": torch.rand(1, 2, 2, 16, 16),
        "flow_logits": torch.rand(1, 2, 2, 16, 16),
        "flow_grid": torch.rand(1, 2, 2, 16, 16) * 0.1,
    }
    path = tmp_path / "dpfr_panel.png"
    fig = render_dpfr_diagnostic_panel(batch, output, save_path=path)
    assert path.exists()
    fig.clf()


def test_compare_runs_reports_differences(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    RunRecorder(left).log_config({"exp_id": "left", "model": {"name": "dpfr"}})
    RunRecorder(right).log_config({"exp_id": "right", "model": {"name": "dpfr"}})
    (left / "summary.json").write_text(json.dumps({"dice_frame_mean": 0.9}), encoding="utf-8")
    (right / "summary.json").write_text(json.dumps({"dice_frame_mean": 0.8}), encoding="utf-8")
    report = compare_runs(left, right)
    assert any(row["field"] == "config.exp_id" for row in report["differences"])
    assert any(row["field"] == "summary.dice_frame_mean" for row in report["differences"])


def test_eval_metric_helpers_align_to_target_size():
    logits = torch.randn(2, 3, 2, 8, 8)
    target = torch.randint(0, 2, (2, 3, 16, 16))
    aligned = align_logits_to_target(logits, target)
    assert aligned.shape == (2, 3, 2, 16, 16)
    dice, iou = binary_dice_iou(torch.ones(4, 4), torch.ones(4, 4))
    assert dice == 1.0
    assert iou == 1.0
