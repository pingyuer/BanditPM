# 架构与数据流

## 顶层入口

训练入口是 `train.py`。它只负责：

- 读取 Hydra config。
- 解析 dataset class。
- 构造 dataloader。
- 初始化 `training.Trainer`。
- 初始化 `experiment.MLflowLogger` 并管理 run 生命周期。

训练循环位于 `training/`，评估入口位于 `evaluation/`，loss 聚合位于 `losses/`，MLflow 实验档案位于 `experiment/`。旧迁移 shim 已移除，新代码只使用这些公共包边界。

## 模块边界

- `training/`: `Trainer`、EMA、参数组、训练期本地日志和 checkpoint 触发。
- `evaluation/`: validation/test 入口、`EvaluationResult`、指标聚合和后续 sweep/postprocess 承载点。
- `losses/`: 公共 segmentation loss 与方法族 loss facade，`LossComputer` 仍保持统一调用面。
- `experiment/`: MLflow API 封装、experiment/run 命名、metadata tags/params、env/source/config artifact。
- `models/`: 模型 registry 公共入口；现有实现文件仍在 `model/` 下，逐步迁移。
- `visualization/`: sequence/sample 可视化与方法族诊断面板。

## Registry 边界

本项目保留 Hydra/YAML，但用轻量 registry 替代硬编码 if/else。

- `utils/registry.py`: 最小 `Registry` 实现。
- `models/registry.py`: `MODEL_REGISTRY`，注册 `gdkvm`、`kpff`、`unext_fusion`、`functional_anchor` 和 legacy aliases。
- `dataset/registry.py`: `DATASET_REGISTRY`，注册 `echo/echonet`、`camus`、`domain/cardiacuda`。

公共 facade：

- `training.build_model_from_cfg(cfg, device)`
- `models.registry.build_model(cfg, device=device)`
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

`losses.LossComputer` 读取 logits、GT 和 supervision mask。`training.Trainer` 读取 loss dict、model aux、memory aux，交给训练期 logger / evaluator / MLflow facade 记录。

## 方法线

- `gdkvm`: `model/gdkvm01.py`，保留原 GDKVM/KPFF/GDR 路径。
- `kpff`: 仍由 `GDKVM` 构建，但 `memory_core` 与 `temporal_memory` 设为 `none`。
- `unext_fusion`: `model/unext_dynakey.py`，使用 UNeXt、DynaKey/spatial memory、mid-level fusion、可选 Q policy。
- `anchor_ode`: `model/anchor_ode.py`，作为 Anchor-ODE 对照方法族保留。
- `functional_anchor`: `model/functional_anchor/`，以模块化方式实现 phase/state/anchor/residual/fusion。

legacy/global DynaKey 和 spatial-phase DynaKey 都应通过配置切换，不应破坏旧路径。
