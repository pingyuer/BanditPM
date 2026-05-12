# BanditPM / GDKVM Research Workspace

这是一个面向超声心动图视频分割的研究代码库。当前主线已经整理为三条方法线和三类数据集，推荐通过 canonical config 与本地 registry 扩展，而不是继续在旧脚本里堆分支。

## 方法线

- `gdkvm`: 原 GDKVM 风格路径，包含 KPFF 与 GDR memory。
- `kpff`: 只保留 KPFF 空间融合，关闭 temporal/memory，用作轻量 baseline。
- `unext_fusion`: UNeXt 单帧分割主干 + DynaKey/spatial memory/mid-level fusion。

## 数据集

- `echo`: EchoNet 风格 endpoint clips。
- `camus`: CAMUS dense/short clips。
- `domain`: CardiacUDA A4C LV domain 数据，在代码中映射到现有 `cardiacuda` loader 路径。

数据根目录通过环境变量控制：

```bash
export DATASETS_ROOT="${DATASETS_ROOT:-$HOME/datasets}"
```

## 快速运行

单个 canonical 实验：

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run python train.py --config-name unext_fusion_echo
```

运行一个子矩阵：

```bash
METHOD=unext_fusion DATASET=echo bash scripts/run_canonical_matrix.sh
METHOD=all DATASET=domain bash scripts/run_canonical_matrix.sh
```

兼容的分方法入口仍保留：

```bash
bash scripts/run_gdkvm_matrix.sh
bash scripts/run_kpff_matrix.sh
bash scripts/run_unext_fusion_matrix.sh
```

常用测试：

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run pytest -q
```

数据协议检查：

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run python scripts/check_dataset_protocol.py \
  --dataset echonet \
  --data_path "$DATASETS_ROOT/processed/echonet_png128_10f"
```

## 配置入口

推荐新实验使用 canonical config：

```text
gdkvm_echo          gdkvm_camus          gdkvm_domain
kpff_echo           kpff_camus           kpff_domain
unext_fusion_echo   unext_fusion_camus   unext_fusion_domain
```

这些配置按 Hydra `_base_` 风格组合：

- `config/_base_/models/{gdkvm,kpff,unext_fusion}.yaml`
- `config/_base_/datasets/{echo,camus,domain}.yaml`
- `config/_base_/runtime/default_runtime.yaml`
- `config/_base_/schedules/default_3k.yaml`

旧的 `config_unext_*`、`config_dynakey_*`、`config_gdkvm_*` 文件保留为 legacy/ablation，不作为新实验首选入口。

## 二次开发

详细开发说明在 `docs/`：

- [文档总览](docs/README.md)
- [架构与数据流](docs/architecture.md)
- [添加模型/模块](docs/develop-modules.md)
- [添加数据集](docs/develop-datasets.md)
- [配置指南](docs/config-guide.md)
- [日志与指标](docs/logging-guide.md)
- [实验指南](docs/experiment-guide.md)
- [工程规范](docs/project-guidelines.md)

核心扩展原则：

- 新模型通过 `MODEL_REGISTRY` 注册。
- 新数据集通过 `DATASET_REGISTRY` 注册。
- 新实验优先新增 `_base_` 与 canonical config。
- 新日志指标优先从 model aux 返回，再由 `Trainer` 聚合到 TensorBoard / wandb / `summary.csv`。

## 输出与记录

历史实验输出已整理为：

```text
outputs/EXPERIMENT_SUMMARY.csv
```

如需重新汇总并清理旧 run：

```bash
python scripts/summarize_and_clean_outputs.py --clean
```
