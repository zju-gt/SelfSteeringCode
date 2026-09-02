# Self-Steering 实验运行指南设计

## 目标

新增一份面向服务器实验执行的中文运行指南，帮助使用者理解 `00–07` 脚本的职责、源码入口、输入输出和需要调整的超参数，并能从小规模 smoke test 平滑切换到正式实验。

目标读者已经了解本项目的研究目的，但不应被要求先阅读全部源码。文档保持简明，不做逐行代码解释，也不重复设计规格中的理论背景。

## 文档位置

最终文档保存为：

`docs/self_steering_experiment_guide.md`

## 组织结构

文档采用流水线顺序组织：

1. 运行前准备：环境、工作目录、API key、安装与基本检查。
2. 配置总览：解释三份 YAML 的合并关系和 `--override`、`--limit`。
3. 流程总览：展示 `00_prepare_data` 到 `07_score_generations` 的依赖顺序。
4. 分阶段说明：每个脚本包含用途、核心源码入口、主要输入、主要输出、关键超参数、资源需求和重跑注意事项。
5. 超参数速查：按模型、数据、标注、向量提取和 steering 评测分组。
6. 两套命令：小规模 smoke test 和正式实验。
7. 结果检查与常见问题：产物位置、失败记录、resume、配置变更后的重跑起点。

## 内容边界

文档会覆盖：

- 当前代码真实支持的 MMLU、MATH500、AIME 2024/2025/2026、ARC-C 和 OBQA；
- 默认使用 MMLU 提取特征、MATH500 做 steering 评测；
- Qwen2.5-7B-Instruct、固定 revision、默认第 19 层、四项 capability、阈值和 alpha；
- DeLeAn API 并发、重试和 rubric 文件；
- activation capture、向量聚合、相似度、generation 和 specificity 结果；
- 哪些步骤需要 OpenAI API、GPU 或只需要 CPU；
- 可复制命令和安全的 smoke-test 参数。

文档不会覆盖：

- 新功能或新配置项；
- 集群调度器、容器化、分布式推理或自动 OOM 调参；
- 逐函数 API 文档；
- 尚未由当前代码实现的实验变体。

## 准确性要求

所有脚本名称、函数名称、配置键和输出路径以当前 `main` 代码为准。命令从 `vector-make` 目录执行。文档会明确说明 `--limit` 的生效阶段、增量恢复行为、模型 revision 必须固定，以及本地离线测试不能替代真实 7B GPU 与在线 API 验证。
