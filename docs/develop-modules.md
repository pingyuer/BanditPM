# 添加模型或模块

## 添加一个新模型变体

推荐流程：

1. 在 `model/` 或 `model/modules/` 下实现模型类或 builder 所需模块。
2. 在 `model/registry.py` 中注册一个清晰的 `model.name`。
3. 新增 `config/_base_/models/<name>.yaml`。
4. 新增一个 canonical config，例如 `<name>_echo.yaml`。
5. 新增 builder/forward smoke test。

示例注册形式：

```python
@MODEL_REGISTRY.register("my_model")
def build_my_model(cfg, *, device):
    model_cfg = cfg.get("model", cfg)
    return MyModel(model_cfg).to(device)
```

配置中必须能通过下面字段选择：

```yaml
model:
  name: my_model
```

## 添加子模块

子模块优先保持局部边界清晰：

- backbone / encoder 放在 `model/modules/` 或已有模型文件附近。
- memory/readout/policy 放在 `model/modules/` 或专题子目录。
- loss 汇总放在 `model/losses.py`，不要让模型直接修改总 loss。
- logging 只返回 aux，不直接在模块里调用 wandb。

模块 forward 推荐返回：

```python
output, aux = module(...)
```

或在主模型输出中写：

```python
out[f"aux_{t}"] = {...}
out[f"memory_aux_{t}"] = {...}
```

## 接入 loss

主分割 loss 仍由 `LossComputer.compute()` 聚合。新增 loss 时：

1. 模型 forward 返回必要 aux，例如 `memory_only_logits`、`spatial_q_target_action`。
2. `LossComputer` 从 `data` 中读取 aux。
3. loss dict 使用可读字段名，例如 `aux_memory_only_ce`、`spatial_q_total`。
4. 只有配置明确启用时，辅助 loss 才加入 `total_loss`。

不要把 diagnostic reward 和 train loss 混在同一个字段里。

## 接入日志

模型或模块不直接写 wandb。新增指标先放入 aux：

```python
memory_aux = {
    "gate_mean": gate.mean().detach(),
    "retrieval_entropy": entropy.detach(),
}
```

再在 `Trainer._log_dynakey_stats()` 或合适的 trainer logging helper 中统一聚合。

命名建议：

- `dynakey/...`: legacy/global DynaKey dictionary 指标。
- `unext_dynakey/...`: UNeXt-DynaKey refine/fusion 指标。
- `q_policy/...`: 若后续拆分 Q policy 专用日志。

## 测试模板

至少补一个 synthetic test：

- 构造小输入，例如 `B=2, T=3, H=32, W=40`。
- 初始化模型。
- 跑 forward。
- 检查 logits shape、aux 是否存在、无 NaN/Inf。
- 如果引入 trainable loss，跑 `loss.backward()` 并检查关键参数有梯度。

推荐放在：

- `tests/test_registry_builders.py`: registry/builder 能否构建。
- `tests/test_spatial_dynakey.py`: UNeXt-DynaKey/spatial memory 行为。
- 新模块复杂时新增 `tests/test_<module>.py`。
