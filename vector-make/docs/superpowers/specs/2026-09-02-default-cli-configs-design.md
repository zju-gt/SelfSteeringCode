# 默认 CLI 配置设计

## 背景

当前八个阶段脚本都要求重复传入以下参数：

```bash
--config configs/model.yaml \
--config configs/data.yaml \
--config configs/experiment.yaml
```

这三份文件是项目正式实验的固定基础配置。强制逐次显式传入会使命令冗长，也容易在某个阶段遗漏配置。

## 目标

- 未传入 `--config` 时，自动加载项目内的 `model.yaml`、`data.yaml` 和 `experiment.yaml`。
- 显式传入一个或多个 `--config` 时，完全使用显式配置列表，不混入默认配置。
- 保留现有的左到右 YAML 合并、`--override` 和 `--limit` 行为。
- 默认配置路径不依赖调用脚本时的当前工作目录。

## CLI 行为

日常运行无需配置参数：

```bash
python scripts/00_prepare_data.py
python scripts/06_run_steering.py --override experiment.target_layer=18
```

三个默认文件按以下顺序合并：

1. `configs/model.yaml`
2. `configs/data.yaml`
3. `configs/experiment.yaml`

显式配置完全替换默认列表：

```bash
python scripts/00_prepare_data.py --config custom.yaml
```

上例只加载 `custom.yaml`。多个显式配置仍按命令行顺序从左到右合并。

## 实现边界

在 `scripts/_common.py` 中根据该文件位置计算项目根目录，并定义默认配置路径。`--config` 不再使用 `required=True`，其解析默认值保持为 `None`。`resolved_config()` 根据是否提供显式配置，在显式列表和默认列表之间二选一，再调用现有的 `load_config()`。

默认路径使用绝对路径，确保从项目目录之外调用编号脚本时仍能找到配置。配置文件缺失、内容非法或合并后配置不完整时，继续复用现有 `ConfigError` 行为。

本次不增加新的入口脚本，不改变 `load_config()` 的公共接口，也不修改历史设计与计划文档。

## 测试与文档

CLI 测试需要覆盖：

- 不传 `--config` 时可以加载三份默认配置并运行轻量本地数据流程。
- 显式传入配置时，解析结果不包含默认配置中的独有字段，以证明默认列表被完全替换。
- `--help` 继续展示 `--config`，并说明其缺省行为和可重复使用方式。
- 多个显式配置、`--override` 和 `--limit` 的既有行为保持通过。

README 和中文实验指南将改用无 `BASE_CONFIG` 的简化命令，同时保留显式自定义配置示例。
