# 架构与数据流

## 顶层入口

训练入口是 `train.py`。它只负责：

- 读取 Hydra config。
- 解析 dataset class。
- 构造 dataloader。
- 初始化 `Trainer`。
- 初始化 wandb/TensorBoard 输出。

训练循环、eval、checkpoint、summary 写入集中在 `model/trainer.py`。

## Registry 边界

本项目保留 Hydra/YAML，但用轻量 registry 替代硬编码 if/else。

- `utils/registry.py`: 最小 `Registry` 实现。
- `model/registry.py`: `MODEL_REGISTRY`，注册 `gdkvm`、`kpff`、`unext_fusion` 和 legacy aliases。
- `dataset/registry.py`: `DATASET_REGISTRY`，注册 `echo/echonet`、`camus`、`domain/cardiacuda`。

兼容 facade 仍然存在：

- `model.trainer.build_model_from_cfg(cfg, device)`
- `train.resolve_dataset_class(cfg)`

二次开发时应扩展 registry，而不是继续扩展这些 facade。

## Config 分层

canonical config 采用 Hydra `_base_` 风格：

```text
config/_base_/models/*.yaml
config/_base_/datasets/*.yaml
config/_base_/runtime/default_runtime.yaml
config/_base_/schedules/default_3k.yaml
```

例如 `unext_fusion_echo.yaml` 表示：

```text
UNeXt fusion model base
+ Echo dataset base
+ default runtime
+ default schedule
+ experiment-specific overrides
```

## 训练数据流

标准 batch contract：

```text
rgb: [B, T, C, H, W]
cls_gt: [B, T, 1, H, W]
ff_gt: [B, 1, N, H, W]
label_valid: [B, T]
supervised_indices: optional [B, T]
eval_valid: optional [B, T]
info: metadata dict
```

模型 forward 输出按时间步写入：

```text
logits_0, logits_1, ...
aux_0, aux_1, ...
memory_aux_0, memory_aux_1, ...
```

`LossComputer` 读取 logits、GT 和 supervision mask。`Trainer` 读取 loss dict、model aux、memory aux 并写日志。

## 方法线

- `gdkvm`: `model/gdkvm01.py`，保留原 GDKVM/KPFF/GDR 路径。
- `kpff`: 仍由 `GDKVM` 构建，但 `memory_core` 与 `temporal_memory` 设为 `none`。
- `unext_fusion`: `model/unext_dynakey.py`，使用 UNeXt、DynaKey/spatial memory、mid-level fusion、可选 Q policy。

legacy/global DynaKey 和 spatial-phase DynaKey 都应通过配置切换，不应破坏旧路径。
