# MMLU Steering 预实验说明

这个预实验验证最小闭环：

```text
MMLU 数学题
  → positive/negative Prompt
  → activation difference
  → 6 个 primitive vectors
  → 固定 vector steering
  → baseline / steered MMLU accuracy
```

当前脚本不训练 Router。默认使用固定 primitive 和固定 strength，适合先验证向量注入是否会因果改变模型行为。

## 1. 环境准备

在 Linux 服务器进入本目录：

```bash
cd /path/to/SelfSteeringCode/riser-runnable
python -m pip install -e .
python -m pip install datasets
```

建议 Python 3.10+，并使用项目要求的 `numpy<2`。如果使用 GPU，请确认 CUDA 版 PyTorch 已安装：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

## 2. 设置实验信息

打开：

```text
scripts/run_mmlu_preliminary.sh
```

重点修改脚本顶部的配置：

```bash
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"
DEVICE="cuda"
DTYPE="float16"
LAYERS_RAW="20"
INJECT_LAYER="20"
NUM_SAMPLES="500"
CLUSTERS="6"
FIXED_PRIMITIVES_RAW="0"
FIXED_STRENGTHS_RAW="1.0"
```

### 必须确认的配置

- `MODEL_PATH`：Qwen 模型的本地目录，或者 Hugging Face 模型名，例如 `Qwen/Qwen2.5-7B-Instruct`。
- `DEVICE`：GPU 使用 `cuda`，CPU 使用 `cpu`。
- `DTYPE`：GPU 可以用 `float16` 或 `bfloat16`；CPU 建议 `float32`。
- `LAYERS_RAW`：提取 activation 的层，可写多个层，例如 `19 20`。
- `INJECT_LAYER`：实际注入层。当前预实验建议只使用一个层，并与 `LAYERS_RAW` 中的层一致。
- `NUM_SAMPLES`：用于提取 vector 的 MMLU 样本数。第一次建议使用 `50` 或 `100` 做 smoke test，确认无误后再改成 `500`。
- `CLUSTERS`：primitive 数量，默认 `6`。
- `FIXED_PRIMITIVES_RAW`：要注入的 primitive 行号，例如 `0 2`。
- `FIXED_STRENGTHS_RAW`：对应的 strength，例如 `0.5 1.0`。两个列表长度必须相同，最大 strength 默认不超过 `2.0`。

默认数据集配置是：

```bash
DATASET_NAME="cais/mmlu"
SUBJECTS_RAW="abstract_algebra college_mathematics elementary_mathematics high_school_mathematics"
SPLIT="test"
```

也可以只选一个学科进行快速测试，例如：

```bash
SUBJECTS_RAW="elementary_mathematics"
NUM_SAMPLES="20"
```

## 3. 执行

脚本会自动执行三步：

1. 采样 MMLU、构造正负 Prompt、提取 activation、聚类 primitive；
2. 将 MMLU 答案转换为评测 JSONL；
3. 使用固定 primitive 运行 baseline/steered 对比。

运行：

```bash
bash scripts/run_mmlu_preliminary.sh
```

也可以不修改脚本，临时覆盖配置：

```bash
MODEL_PATH=/data/models/Qwen2.5-7B-Instruct \
NUM_SAMPLES=50 \
FIXED_PRIMITIVES_RAW="0" \
FIXED_STRENGTHS_RAW="1.0" \
bash scripts/run_mmlu_preliminary.sh
```

## 4. 输出文件

默认输出到：

```text
artifacts/mmlu_preliminary/
```

主要文件：

```text
prompt_pairs.jsonl  # 每个 MMLU 题目的正负 Prompt、答案和 metadata
vectors.pt          # positive/negative/difference activation tensors
primitives.pt       # 聚类后的 primitive library
primitives.json     # primitive 数量、维度、来源等 metadata
evaluation.jsonl    # 实际用于 baseline/steered 生成的 Prompt 和答案
results.jsonl       # 两种生成结果、MMLU accuracy、token 数、latency、routing info
```

`results.jsonl` 中重点查看：

```json
{
  "metrics": {
    "mmlu_accuracy": {
      "baseline": 0.0,
      "steered": 1.0
    }
  },
  "routing": {
    "selected_primitives": [[0]],
    "selected_strengths": [[1.0]]
  }
}
```

MMLU accuracy 使用生成文本中最后一个独立的 `A/B/C/D` 选项作为模型答案，而不是对单字母做 substring match。

评测现在采用即时写入：每完成一个样本的 baseline 和 steered 两次生成，就立即追加一行并 flush 到 `results.jsonl`。如果进程中途被终止，已经完成的样本仍会保留；重新运行同一命令时，结果文件会从头覆盖，不会自动续跑。

## 5. strength 和 primitive 对比

固定 vector 模式下，可以重复运行不同配置：

```bash
FIXED_PRIMITIVES_RAW="0" FIXED_STRENGTHS_RAW="0.5" \
  OUTPUT_DIR=artifacts/mmlu_p0_s05 bash scripts/run_mmlu_preliminary.sh

FIXED_PRIMITIVES_RAW="0" FIXED_STRENGTHS_RAW="1.0" \
  OUTPUT_DIR=artifacts/mmlu_p0_s10 bash scripts/run_mmlu_preliminary.sh

FIXED_PRIMITIVES_RAW="0" FIXED_STRENGTHS_RAW="2.0" \
  OUTPUT_DIR=artifacts/mmlu_p0_s20 bash scripts/run_mmlu_preliminary.sh
```

多个 primitive 可以组合：

```bash
FIXED_PRIMITIVES_RAW="0 2" \
FIXED_STRENGTHS_RAW="1.0 0.5" \
bash scripts/run_mmlu_preliminary.sh
```

建议的初步对照包括：

```text
baseline
primitive 0, strength 0.5 / 1.0 / 2.0
primitive 1, strength 0.5 / 1.0 / 2.0
...
random 或打乱后的 vector（后续可补充）
```

## 6. 当前实验边界

- 当前使用 `FixedRouter`，primitive 的选择与输入 hidden state 无关；因此只能验证 vector 的因果 steering 效果，不能证明模型已经学会自主选择 feature。
- 当前默认只对一个 layer 注入；`aggregation=concat` 会产生多层拼接向量，不适合直接注入单层 hidden state，建议使用 `last`。
- 第一次运行建议使用 `NUM_SAMPLES=20` 或 `50` 检查模型路径、层号和显存，再扩大到 500。
- 如果需要真正的 Router controller 实验，下一步需要收集 Prompt hidden state、cluster/primitive 标签，并训练 Router checkpoint。

## 7. 单独调用辅助脚本

如果已经有采集结果，也可以单独转换评测输入：

```bash
python scripts/prepare_mmlu_eval.py \
  --input artifacts/mmlu_preliminary/prompt_pairs.jsonl \
  --output artifacts/mmlu_preliminary/evaluation.jsonl
```

单独查看评测入口参数：

```bash
python scripts/evaluate_mmlu.py --help
```
