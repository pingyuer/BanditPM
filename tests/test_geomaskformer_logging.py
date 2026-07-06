from __future__ import annotations

import torch
from omegaconf import OmegaConf

from geomaskformer import GeoMaskFormer
from training.trainer import Trainer


def _trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = OmegaConf.create(
        {
            "model": {
                "name": "geomaskformer",
                "geomaskformer": {"loss": {"topk": 4}, "depth": 1, "heads": 4, "dim": 32, "num_queries": 4},
            },
            "evaluation": {"metric_space": "original"},
        }
    )
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    return trainer


def test_geomaskformer_eval_logging_has_no_old_aliases():
    trainer = _trainer()
    logged = trainer._geomaskformer_eval_log_metrics(
        {
            "dice": 0.8,
            "iou": 0.7,
            "hd95": 3.0,
            "assd": 1.2,
            "boundary_dice": 0.4,
            "area_smoothness": 0.01,
            "temporal_drift": 0.2,
            "centroid_jitter": 0.03,
            "geomaskformer/proposal_oracle_top4_mean_dice": 0.9,
            "geomaskformer/proposal_oracle_top5_best_dice": 0.91,
            "geomaskformer/proposal_oracle_top5_mean_dice": 0.88,
            "geomaskformer/proposal_top5_cover_rate_0p85": 1.0,
            "geomaskformer/proposal_top5_cover_rate_0p90": 0.5,
        }
    )
    assert "temporal/area_second_difference_abs" in logged
    assert "temporal/metrics_on_dice_ge_threshold" not in logged
    assert logged["temporal/dice_ge_threshold_ratio"] == 0.0
    assert "proposal/oracle_top4_mean_dice" in logged
    assert logged["proposal/oracle_top5_best_dice"] == 0.91
    assert logged["proposal/oracle_top5_mean_dice"] == 0.88
    assert logged["proposal/top5_cover_rate_0p85"] == 1.0
    assert logged["proposal/top5_cover_rate_0p90"] == 0.5
    assert not any("@" in key for key in logged)
    forbidden = {"area_acceleration", "temporal_jitter", "best_query_mean", "overall/Dice", "val/dice"}
    assert not any(any(alias in key for alias in forbidden) for key in logged)


def test_geomaskformer_parameter_accounting_closes():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "geomaskformer",
                "geomaskformer": {
                    "in_channels": 1,
                    "num_classes": 2,
                    "dim": 32,
                    "base_channels": 16,
                    "depth": 1,
                    "heads": 4,
                    "num_queries": 4,
                    "decoder_layers": 1,
                    "max_frames": 8,
                },
            }
        }
    )
    trainer = _trainer()
    trainer.model = GeoMaskFormer(cfg.model)
    metrics = trainer._geomaskformer_capacity_metrics()
    parts = (
        metrics["model/params_m_image_tokenizer"]
        + metrics["model/params_m_mask_tokenizer"]
        + metrics["model/params_m_prompt_query_adapter"]
        + metrics["model/params_m_dual_stream_transformer"]
        + metrics["model/params_m_pixel_decoder"]
        + metrics["model/params_m_proposal_decoder"]
        + metrics["model/params_m_quality_head"]
        + metrics["model/params_m_unclassified"]
    )
    assert abs(metrics["model/params_m_total"] - parts) < 1.0e-6
    assert not any("parameters_m_faf" in key or "parameters_m_memory" in key for key in metrics)


def test_geomaskformer_protocol_stats_use_condition_visibility():
    trainer = _trainer()
    totals = trainer._metric_totals_template()
    eval_indices = torch.tensor([[False, True, True], [True, True, False]])
    condition_visibility = torch.tensor([[1, 0, 0], [0, 1, 0]])
    trainer._accumulate_geomaskformer_protocol_metrics(totals, {}, eval_indices, condition_visibility)
    metrics = trainer._reduce_metric_totals(totals)
    assert metrics["geomaskformer/visible_frame_count"] == 1.0
    assert metrics["geomaskformer/samples_with_visible_condition_ratio"] == 1.0
    assert metrics["geomaskformer/visible_and_supervised_ratio"] == 0.25
    assert metrics["geomaskformer/masked_and_supervised_ratio"] == 0.75


def test_geomaskformer_proposal_selection_gap_is_oracle_minus_selected():
    trainer = _trainer()
    trainer.cfg.evaluation.geomaskformer_temporal_dice_threshold = 0.75
    totals = trainer._metric_totals_template()
    gt = torch.zeros(1, 1, 8, 8)
    gt[:, :, 2:6, 2:6] = 1.0
    proposals = torch.full((1, 1, 2, 8, 8), -6.0)
    proposals[:, :, 0, 0:4, 0:4] = 6.0
    proposals[:, :, 1, 2:6, 2:6] = 6.0
    out = {"proposal_logits": proposals, "quality_scores": torch.tensor([[[6.0, -6.0]]])}
    trainer._accumulate_geomaskformer_frame_metrics(totals, out, 0, 0, gt, 0.0, 0.5, "all_mask_invisible")
    metrics = trainer._reduce_metric_totals(totals)
    assert metrics["geomaskformer/proposal_oracle_best_dice"] > metrics["geomaskformer/proposal_selected_dice"]
    assert metrics["geomaskformer/proposal_selection_gap"] > 0.0
    assert metrics["geomaskformer/temporal_dice_threshold_frame_count"] == 1.0
    assert metrics["geomaskformer/temporal_dice_ge_threshold_count"] == 0.0
    assert metrics["geomaskformer/temporal_dice_ge_threshold_ratio"] == 0.0
