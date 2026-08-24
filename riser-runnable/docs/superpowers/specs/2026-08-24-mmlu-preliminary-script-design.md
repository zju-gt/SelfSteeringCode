# MMLU Steering 预实验脚本设计

## 目标

提供一个面向 Linux 服务器的最小预实验入口，完整执行：

```text
MMLU 数学题采样
  → positive/negative prompt 激活提取
  → activation difference 聚类
  → primitive library 保存
  → 固定 primitive steering
  → baseline/steered MMLU 对比
```

该实验用于验证 activation vector 是否能对模型行为产生初步因果影响，不包含 Router 训练，也不声称已经完成模型自主 feature controller 验证。

## 文件设计

### `scripts/run_mmlu_preliminary.sh`

Linux Bash 主入口，使用 `set -euo pipefail`。脚本顶部集中放置用户需要修改的配置：

- `MODEL_PATH`：本地 Qwen 模型目录或 Hugging Face 模型名；
- `DEVICE`、`DTYPE`：运行设备和模型精度；
- `LAYERS`：提取和注入的 Transformer 层；
- `NUM_SAMPLES`、`SUBJECTS`、`SPLIT`：MMLU 数据配置；
- `CLUSTERS`：primitive 数量，默认 6；
- `FIXED_PRIMITIVES`、`FIXED_STRENGTHS`：固定 steering route；
- `MAX_NEW_TOKENS`、`OUTPUT_DIR`：生成长度和结果目录。

脚本依次调用现有的 `examples/collect_mmlu_math_vectors.py`、辅助转换脚本和评测 CLI，并把中间产物及最终 JSONL 结果保存到同一实验目录。脚本不修改源代码，不训练 Router。

### `scripts/prepare_mmlu_eval.py`

将向量采集阶段的 JSONL 转换为评测输入格式：

- 使用 positive prompt 作为评测 Prompt；
- 将 MMLU 的整数答案转换为 `A/B/C/D`；
- 保留 subject、question、choices 等 metadata；
- 提供基于最后一个有效选项字母的 MMLU accuracy 解析，避免对单字母直接做 substring match 造成误判。

该脚本既可被 Bash 主入口调用，也可以单独运行检查数据转换结果。

### `docs/mmlu_preliminary_experiment_zh.md`

中文操作文档，包含环境安装、配置项、完整执行命令、输出文件结构、strength/layer/primitive 修改方式和常见错误说明。

## 数据流和输出

实验目录默认包含：

```text
prompt_pairs.jsonl       # MMLU 正负 Prompt 和答案
vectors.pt               # positive/negative/difference activations
primitives.pt            # 聚类后的 primitive library
primitives.json          # primitive metadata
evaluation.jsonl         # baseline/steered 评测输入
results.jsonl            # 评测输出、token usage、latency、routing info
```

评测默认使用固定 route，因此 `results.jsonl` 中的 steering 配置是可复现的。需要动态 Router 时，后续可以在同一入口增加 `ROUTER_CHECKPOINT` 配置，不改变数据转换格式。

## 错误处理和兼容性

- Bash 在任一子命令失败时立即退出；
- 检查 Python、模型路径/模型名、CUDA 请求和固定 primitive/strength 长度；
- 默认使用单层 `last` aggregation，避免 `concat` 向量无法注入单层 hidden state 的维度问题；
- CPU 默认使用 `float32`，GPU 可显式设置 `float16` 或 `bfloat16`；
- 所有路径支持通过脚本顶部变量修改，不依赖当前工作目录之外的隐式路径。

## 验证方式

实现前先为辅助转换脚本添加离线单元测试，覆盖：

1. MMLU 整数答案到选项字母的转换；
2. 从生成文本中提取最后一个有效选项字母；
3. JSONL metadata 保留；
4. 空行和非法答案的错误提示。

随后运行项目离线测试和辅助脚本测试；不在测试阶段下载模型或数据集。
