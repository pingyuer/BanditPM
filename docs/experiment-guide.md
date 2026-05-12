# 实验指南

## 推荐矩阵

当前推荐的 3 x 3 canonical matrix：

```text
gdkvm_echo          gdkvm_camus          gdkvm_domain
kpff_echo           kpff_camus           kpff_domain
unext_fusion_echo   unext_fusion_camus   unext_fusion_domain
```

其中：

- `echo`: EchoNet endpoint clips。
- `camus`: CAMUS short dense clips。
- `domain`: CardiacUDA A4C LV dense domain data。

## 单实验

```bash
DATASETS_ROOT=$HOME/datasets \
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run python train.py \
  --config-name unext_fusion_echo
```

## 矩阵脚本

统一入口：

```bash
bash scripts/run_canonical_matrix.sh
```

子矩阵：

```bash
METHOD=gdkvm DATASET=all bash scripts/run_canonical_matrix.sh
METHOD=unext_fusion DATASET=echo bash scripts/run_canonical_matrix.sh
METHOD=all DATASET=domain bash scripts/run_canonical_matrix.sh
```

兼容入口：

```bash
bash scripts/run_gdkvm_matrix.sh
bash scripts/run_kpff_matrix.sh
bash scripts/run_unext_fusion_matrix.sh
```

脚本支持环境变量：

```bash
PROJECT_DIR=/home/tahara/GDKVM
UV_BIN=/home/tahara/miniconda3/bin/uv
DATASETS_ROOT=$HOME/datasets
CUDA_VISIBLE_DEVICES=0
WANDB_MODE=online
LOG_DIR=outputs/BanditPM/tmux_logs
```

## smoke run

修改代码或配置后，先跑 1-iter smoke：

```bash
PYTHONPATH=. HYDRA_FULL_ERROR=1 WANDB_MODE=disabled \
/home/tahara/miniconda3/bin/uv run python train.py \
  --config-name unext_fusion_echo \
  wandb_mode=disabled \
  main_training.num_iterations=1 \
  main_training.batch_size=1 \
  main_training.num_workers=0 \
  main_training.amp=false \
  eval_stage.eval_interval=999 \
  hydra.run.dir=/tmp/unext_fusion_echo_smoke
```

smoke 通过只能说明入口链路可跑，不代表指标有效。

## full run

正式实验建议：

- `wandb_mode=online`。
- 保持 `evaluation.exclude_init_frame=true`。
- 保持 `evaluation.init_mode=pred_or_zero`。
- 保持 `model.allow_oracle_init_when_requested=false`，除非是 oracle upper-bound。
- 记录 git 状态和 config diff。

## tmux 建议

双卡时可手动分配：

```bash
tmux new -s unext_echo 'CUDA_VISIBLE_DEVICES=0 METHOD=unext_fusion DATASET=echo bash scripts/run_canonical_matrix.sh'
tmux new -s gdkvm_domain 'CUDA_VISIBLE_DEVICES=1 METHOD=gdkvm DATASET=domain bash scripts/run_canonical_matrix.sh'
```

查看日志：

```bash
tail -n 50 outputs/BanditPM/tmux_logs/unext_fusion_echo.log
```

## 结果整理

汇总历史 outputs：

```bash
python scripts/summarize_and_clean_outputs.py
```

清理旧 run 但保留总表：

```bash
python scripts/summarize_and_clean_outputs.py --clean
```

总表路径：

```text
outputs/EXPERIMENT_SUMMARY.csv
```
