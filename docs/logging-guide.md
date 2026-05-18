# 日志与指标

## 输出位置

Hydra run dir 由 config 决定，默认在：

```text
outputs/BanditPM/<exp_id>/<date>/<time>
```

每个 run 可包含：

- MLflow run。
- `summary.csv`。
- checkpoint / weights。
- configs / eval / visuals / env / source artifacts。

历史 run 可汇总到：

```text
outputs/EXPERIMENT_SUMMARY.csv
```

命令：

```bash
python scripts/summarize_and_clean_outputs.py --clean
```

## MLflow 配置

入口在 `train.py`：

- `mlflow.enabled`: 是否启用 MLflow。
- `mlflow.tracking_uri`: tracking server。
- `mlflow.experiment_name`: 实验名。
- `mlflow.run_name`: run 名，留空时自动生成。
- `mlflow.resume_run_id`: 恢复已有 run。

默认 tracking server：

```bash
http://172.16.240.77:5000
```

## Trainer 日志流

主要位置：

- `Trainer._log_train_metrics()`: loss dict。
- `Trainer._log_dynakey_stats()`: DynaKey / UNeXt-DynaKey aux 指标。
- `Trainer._log_final_metrics()`: val/test metrics。
- `Trainer._write_summary_row()`: `summary.csv`。

新增指标推荐流程：

1. 模块 forward 返回 tensor aux，尽量 `.detach()`。
2. 主模型把 aux 放到 `aux_t` 或 `memory_aux_t`。
3. `Trainer` 聚合 batch/time 维度，写 MLflow。
4. 如果是最终评估指标，再写入 `summary.csv`。

## 命名约定

推荐 prefix：

```text
loss/...              主 loss 或辅助 loss
metrics/...           val/test 指标
dynakey/...           legacy/global DynaKey dictionary
unext_dynakey/...     UNeXt-DynaKey refine/fusion/memory
q_policy/...          Q policy 专用诊断，后续可拆出
protocol/...          no-leak、init、label_valid 等协议信息
```

已有常见字段：

```text
dynakey/occupancy_ratio
dynakey/active_key_count
dynakey/retrieval_entropy
dynakey/prediction_error
dynakey/action_count_*
unext_dynakey/gate_mean
unext_dynakey/residual_abs_mean
unext_dynakey/memory_update_rate
unext_dynakey/mid_memory_gate_mean
unext_dynakey/spatial_memory_entropy
```

## summary.csv 扩展原则

`summary.csv` 用于实验表格，不要记录过细的 step-level debug。

适合写入 summary：

- final/best Dice、IoU、HD95、ASD。
- best threshold。
- `exclude_init_frame`、`init_mode`、`oracle_gt_used`。
- temporal consistency 类最终指标。

不适合写入 summary：

- 每 step 的 gate mean。
- 每 frame 的 slot id。
- 大 tensor 或 histogram。

这类信息应写 MLflow 或 debug script 输出。
