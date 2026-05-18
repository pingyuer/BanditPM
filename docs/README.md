# Docs 总览

这里是 BanditPM/GDKVM 工作区的二次开发文档。README 只保留快速入口，具体开发流程放在本目录。

## 推荐阅读顺序

1. [architecture.md](architecture.md): 先理解训练入口、registry、config 与数据流。
2. [config-guide.md](config-guide.md): 再理解如何组合 canonical config。
3. [develop-modules.md](develop-modules.md): 添加模型、memory、fusion、loss 或 policy。
4. [develop-datasets.md](develop-datasets.md): 添加新数据集或新协议。
5. [logging-guide.md](logging-guide.md): 添加 MLflow/summary 指标。
6. [experiment-guide.md](experiment-guide.md): 运行三方法 x 三数据集实验矩阵。
7. [project-guidelines.md](project-guidelines.md): 工程约定和测试策略。

## 专题文档

DynaKey 相关设计和审阅资料已经归档到：

- [dynakey/design.md](dynakey/design.md)
- [dynakey/code-review-guide.md](dynakey/code-review-guide.md)

## 新功能最小闭环

添加任何新能力时，按这个顺序做：

1. 明确它属于 model、dataset、loss、logging、script 还是 config。
2. 在对应 registry 或入口处接入，避免在 `train.py` 里继续写硬编码分支。
3. 新增 `_base_` 配置或 canonical config，旧 config 只做兼容。
4. 增加一个 synthetic smoke test，确认 forward/backward 或 dataloader 可跑。
5. 跑 `pytest -q`，再启动真实数据 smoke。
