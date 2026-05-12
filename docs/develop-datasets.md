# 添加数据集

## Dataset contract

新数据集应返回和现有训练链路兼容的字段：

```text
rgb: [T, C, H, W]
cls_gt: [T, 1, H, W]
ff_gt: [1, N, H, W]
label_valid: [T]
info: dict
```

可选字段：

```text
eval_valid: [T]
supervised_indices: [T]
init_mode: str
```

关键语义：

- `label_valid[t] = true` 表示第 `t` 帧有可监督 mask。
- sparse ED/ES 数据不要给无标签中间帧算 supervised loss。
- no-leak 主协议下，val/test 不应使用首帧 GT 初始化，除非 explicit oracle upper-bound config。

## 添加 Dataset class

推荐流程：

1. 在 `dataset/` 下实现 `torch.utils.data.Dataset`。
2. 确认 `mode=train|val|test`、`seq_length`、`max_num_obj`、`size` 参数兼容 `train.py`。
3. 在 `dataset/registry.py` 注册别名。
4. 新增 `config/_base_/datasets/<name>.yaml`。
5. 新增一个 canonical config，例如 `unext_fusion_<name>.yaml`。
6. 写 dataset 解析测试和 dataloader smoke。

注册示例：

```python
DATASET_REGISTRY.register("my_dataset", module=MyDataset)
```

配置示例：

```yaml
# @package _global_
dataset_name: my_dataset
data_path: "${processed_root}/my_dataset_processed"

data:
  protocol_name: "my_protocol"

evaluation:
  init_mode: "pred_or_zero"
  exclude_init_frame: true
  protocol_version: "v3_canonical_no_leak"
```

## 标签和帧索引

EchoNet/CardiacUDA 风格数据通常有 sparse labels。优先复用：

- `dataset/frame_index.py`
- `scripts/check_dataset_protocol.py`

标签名应支持常见形式：

```text
000.png
000001_mask.png
frame_000.png
frame001.png
ED.png
ES.png
```

如果原始帧号和 clip 内局部帧号不同，metadata 中应提供 `source_frames`，并在解析时映射到 local index。

## 协议检查

新增数据集后先跑：

```bash
PYTHONPATH=. /home/tahara/miniconda3/bin/uv run python scripts/check_dataset_protocol.py \
  --dataset echonet \
  --data_path "$DATASETS_ROOT/processed/your_dataset"
```

检查重点：

- split 样本数。
- 每个样本帧数。
- `label_valid` 分布。
- 是否存在首帧 GT。
- 空 mask / 尺寸不匹配。
- ED/ES 或 sparse label 是否正确映射。

如果出现 `label_valid all zero`，应修数据解析，不要在 loss 里绕过。
