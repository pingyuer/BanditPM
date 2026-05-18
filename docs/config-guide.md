# 配置指南

## 推荐配置结构

新实验优先使用 canonical config，而不是直接修改 legacy config。

基础层：

```text
config/_base_/models/*.yaml
config/_base_/datasets/*.yaml
config/_base_/runtime/default_runtime.yaml
config/_base_/schedules/default_3k.yaml
```

实验层：

```text
gdkvm_echo.yaml
kpff_camus.yaml
unext_fusion_domain.yaml
```

每个 base 文件顶部应使用：

```yaml
# @package _global_
```

这样 Hydra 会把字段合并到顶层，而不是放到 `_base_` 子节点里。

## canonical config 示例

```yaml
defaults:
  - config_unext_dynakey_spatial_memory_primary
  - _base_/datasets/echo
  - _base_/models/unext_fusion
  - _base_/runtime/default_runtime
  - _base_/schedules/default_3k
  - _self_

exp_id: "unext_fusion_echo"

MLflow:
  group: "canonical_unext_fusion"
  tags: ["canonical", "unext_fusion", "echo", "no_leak"]
```

`_self_` 放在最后，表示本文件 override base。

## 常用 override

命令行覆盖训练步数和 batch：

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run python train.py \
  --config-name unext_fusion_echo \
  main_training.num_iterations=10 \
  main_training.batch_size=2
```

切换 MLflow：

```bash
mlflow.enabled=true
mlflow.enabled=false
mlflow.enabled=false
```

指定输出目录：

```bash
hydra.run.dir=/tmp/unext_fusion_smoke
```

## no-leak 关键字段

主结果默认应使用：

```yaml
phase_init:
  train: "pred_or_zero"
  val: "pred_or_zero"
  test: "pred_or_zero"

evaluation:
  init_mode: "pred_or_zero"
  exclude_init_frame: true
  protocol_version: "v3_canonical_no_leak"

model:
  allow_oracle_init_when_requested: false
  use_first_frame_gt_init: false
```

oracle GT 初始化只能作为 upper bound / sanity check，配置名和 MLflow tag 必须显式标记 `oracle`。

## legacy config

这些配置保留用于复现实验和消融：

```text
config_unext_*
config_dynakey_*
config_gdkvm_*
*_oracle.yaml
```

新实验不要在 legacy config 上继续堆长期分支。若某个 legacy 设置值得保留，应抽成 `_base_` 或新的 canonical config。
