# Self-Steering MVP 实验运行指南

本文面向准备在服务器上运行实验的使用者，重点说明 `scripts/00` 到 `scripts/07` 分别做什么、对应哪些核心代码、需要设置哪些超参数，以及如何运行真实数据正式实验。

命令默认从 `vector-make` 目录执行。

## 1. 流程总览

```text
00 准备数据
  ↓
01 DeLeAn 四维需求标注
  ↓
02 构造 MMLU 提取集和外部评测集
  ↓
03 捕获 generic/capability prompt 的激活差
  ↓
04 聚合 capability vectors
  ↓
05 分析向量相似度和内部一致性
  ↓
06 在生成过程中注入向量
  ↓
07 统计 accuracy、demand slice 和 specificity
```

| 脚本 | 核心函数 | 作用 | 主要输入 | 主要输出 | 资源 |
|---|---|---|---|---|---|
| `00_prepare_data.py` | `pipeline.prepare_data` | 下载或读取数据并统一格式 | `configs/data.yaml` | `data/processed/*.jsonl` | CPU、网络 |
| `01_score_demands.py` | `pipeline.score_demands` | 对每题标注四维 DeLeAn demand | processed JSONL、rubrics | `data/scored/*_delean_long.jsonl` 和 `*_with_4d_demands.jsonl` | OpenAI API、CPU |
| `02_prepare_items.py` | `pipeline.prepare_items` | 筛选向量提取题和 steering 评测题 | scored JSONL | `processed/extraction/`、`processed/evaluation/` | CPU |
| `03_capture_contrasts.py` | `pipeline.capture_contrasts` | 捕获指定层的 prompt 激活差 | MMLU extraction、模型 | `outputs/activations/<id>/` | GPU |
| `04_extract_vectors.py` | `pipeline.extract_vectors` | 聚合四个 capability vector | activation shards | `outputs/vectors/<id>/` | CPU、内存 |
| `05_analyze_similarity.py` | `pipeline.analyze_similarity` | 计算 coherence 和 cosine similarity | vectors、activation shards | `outputs/metrics/<id>_vector_*.json` | CPU |
| `06_run_steering.py` | `pipeline.run_steering` | 生成并持续注入 steering vector | evaluation JSONL、vectors、模型 | `outputs/generations/<id>.jsonl` | GPU |
| `07_score_generations.py` | `pipeline.score_generations` | 计算最终评测指标 | generations JSONL | `outputs/metrics/<id>_steering_metrics.*` | CPU |

## 2. 运行前准备

建议在服务器创建干净的 Python 3.10+ 环境，并先安装与服务器 CUDA 匹配的 PyTorch：

```bash
cd /path/to/vector-make
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

stage 01 需要 OpenAI API：

```bash
export OPENAI_API_KEY="your-key"
```

Hugging Face 公共数据集和模型通常不要求登录；服务器或镜像环境需要时可设置：

```bash
export HF_TOKEN="your-token"
```

运行前建议先检查：

```bash
python -m pytest -q
python scripts/00_prepare_data.py --help
```

离线测试通过只表示代码逻辑可运行，不等于已经完成真实 7B GPU 或在线 API 验证。

## 3. 配置方式

每个脚本都按从左到右的顺序合并三份配置：

```bash
BASE_CONFIG=(
  --config configs/model.yaml
  --config configs/data.yaml
  --config configs/experiment.yaml
)
```

调用示例：

```bash
python scripts/00_prepare_data.py "${BASE_CONFIG[@]}"
```

临时修改参数可使用重复的 `--override key=value`，不需要编辑 YAML：

```bash
python scripts/06_run_steering.py "${BASE_CONFIG[@]}" \
  --override experiment.target_layer=18 \
  --override 'experiment.alphas=[0.0,0.5,1.0]'
```

注意：

- 同一轮实验各阶段应使用一致的模型、数据、layer、threshold 和路径配置。
- stage 06 使用了 `alphas` 或 `max_new_tokens` override 时，stage 07 必须重复相同 override，否则会寻找另一个 generation run ID。
- 所有脚本都接受 `--limit`，但当前只有 stage 00、01、03、06 实际读取它。
- stage 00/01 的 limit 是“每个数据集最多 N 题”；stage 03 是“每个 capability 最多 N 题”；stage 06 是“每个评测数据集最多 N 题”。

## 4. 各阶段说明

### 4.1 Stage 00：准备 canonical 数据

```bash
python scripts/00_prepare_data.py "${BASE_CONFIG[@]}"
```

核心代码：

- 入口：`src/self_steering/pipeline.py::prepare_data`
- 数据注册：`src/self_steering/datasets/registry.py`
- schema 转换：`src/self_steering/datasets/adapters.py`
- 统一数据类型：`src/self_steering/datasets/types.py::CanonicalItem`

该阶段始终准备 MMLU，并准备 `data.enabled_steering_datasets` 中启用的评测集。默认只启用 MATH500；AIME 2024/2025/2026、ARC-C 和 OBQA 可以通过配置扩展。

每个样本被转换为统一字段，例如 `item_id`、`prompt`、`gold_answer`、`answer_type`、`choices` 和 `metadata`。如果配置了 `local_path`，本地 JSONL 优先于 Hugging Face 下载。

主要产物：

```text
data/processed/mmlu.jsonl
data/processed/math500.jsonl
data/processed/aime2024.jsonl       # 启用时
...
```

重点参数：`enabled_steering_datasets`、各数据源的 `path/name/split/local_path`、`data.cache_dir`、`experiment.paths.data_dir`。

### 4.2 Stage 01：DeLeAn demand 标注

```bash
python scripts/01_score_demands.py "${BASE_CONFIG[@]}"
```

核心代码：

- 流水线：`pipeline.score_demands`
- prompt、schema、API 调用：`datasets/delean_labeler.py`
- 并发、resume 和 long-to-wide：`datasets/scoring.py`
- 四份 rubric：`rubrics/QLl.txt`、`QLq.txt`、`CL.txt`、`MCr.txt`

每道题会分别标注四个 capability，分数范围为 0–5：

- `QLl`：Logical Reasoning
- `QLq`：Quantitative Reasoning
- `CL`：Conceptualisation, Learning and Abstraction
- `MCr`：Identifying Relevant Information

主要产物：

```text
data/scored/<dataset>_delean_long.jsonl
data/scored/<dataset>_with_4d_demands.jsonl
```

long 文件每行是一题的一个维度标注；wide 文件把四维分数合并到题目的 `demand_scores`。

该阶段支持并发和增量恢复。成功记录的缓存身份包含题目、rubric、标注模型以及完整 annotation prompt/schema；修改这些内容后不会错误复用旧标注。失败调用写为 `status: error`，重跑时会重新尝试未成功项。

重点参数：

- `experiment.annotation.model`
- `max_workers`：API 并发数，服务器或账号限流时应降低。
- `max_attempts`：超时、限流和部分服务端错误的最大尝试次数。
- `initial_backoff_seconds`：指数退避的初始等待时间。
- `experiment.paths.rubrics_dir`

API 调用规模约为：

```text
(MMLU 题数 + 所有启用评测集题数) × 4
```

### 4.3 Stage 02：构造提取集和评测集

```bash
python scripts/02_prepare_items.py "${BASE_CONFIG[@]}"
```

核心代码：`pipeline.prepare_items` 和 `datasets/filtering.py`。

MMLU 用于向量提取。对 capability `k`，默认选取：

```text
d_k >= high_demand_threshold = 4
```

外部评测集会记录每道题属于哪些 high/low demand slice：

```text
high: d_k >= 4
low:  d_k <= 1
```

处于中间分数的维度不会产生 membership；如果一道题在所有维度都既非 high 也非 low，该题不会进入 evaluation 文件。不同 capability 之间允许重叠，不做 domain 平衡。

主要产物：

```text
data/processed/extraction/QLl.jsonl
data/processed/extraction/QLq.jsonl
data/processed/extraction/CL.jsonl
data/processed/extraction/MCr.jsonl
data/processed/evaluation/<dataset>.jsonl
```

运行后建议检查数量：

```bash
wc -l data/processed/extraction/*.jsonl
wc -l data/processed/evaluation/*.jsonl
```

stage 05 要求每个 capability 至少有两个成功 capture，因此小规模运行时四个 extraction 文件都应至少有 2 行。

### 4.4 Stage 03：捕获 prompt contrast

```bash
python scripts/03_capture_contrasts.py "${BASE_CONFIG[@]}"
```

核心代码：

- `pipeline.capture_contrasts`
- `prompts/serialization.py` 和 `prompts/templates.py`
- `vectors/capture.py`
- `hooks/capture.py`

对每个 capability 的每道 MMLU 高需求题，代码构造两份相同题目、不同 reasoning instruction 的 prompt：

```text
generic prompt
capability-specific prompt
```

两者都以 assistant 消息 `Reasoning:` 作为预填充起点。代码读取目标 decoder layer 最后一个 token 位置的 residual output，并保存：

```text
delta = capability_activation - generic_activation
```

`target_layer: 19` 是零基索引，即模型的第 20 个 decoder block。每题保存一个 safetensors shard，并通过 `index.json` 建立索引：

```text
outputs/activations/<capture_id>/<capability>/*.safetensors
outputs/activations/<capture_id>/index.json
outputs/activations/<capture_id>/errors.jsonl       # 有失败时
```

已有成功 shard 会被复用。CUDA OOM 等异常按题记录，不会把失败 shard 写入 index。

重点参数：模型、revision、dtype、device map、attention implementation、`target_layer`、capability prompts，以及 `--limit`。

### 4.5 Stage 04：聚合 capability vectors

```bash
python scripts/04_extract_vectors.py "${BASE_CONFIG[@]}"
```

核心代码：`pipeline.extract_vectors`、`vectors/extract.py` 和 `vectors/storage.py`。

对 capability `k`，先计算所有 per-item delta 的均值：

```text
raw_k = mean(delta_i)
```

随后保存三种形式：

- `raw`：原始均值向量。
- `unit`：单位长度向量。
- `steering`：单位向量乘以四个 raw vector 的平均范数，即默认 `mean_norm` 形式。

主要产物：

```text
outputs/vectors/<capture_id>/capability_vectors.safetensors
outputs/vectors/<capture_id>/capability_vectors.json
```

如果某个 capability 没有成功 shard，阶段会直接报错，不会生成虚假向量。

### 4.6 Stage 05：分析向量质量

```bash
python scripts/05_analyze_similarity.py "${BASE_CONFIG[@]}"
```

核心代码：`pipeline.analyze_similarity` 和 `vectors/similarity.py`。

该阶段输出：

- 四个 unit vector 的 cosine similarity matrix；
- 每个 capability 内部所有 per-item delta 两两 cosine similarity 的平均值，即 coherence。

```text
outputs/metrics/<capture_id>_vector_similarity.json
outputs/metrics/<capture_id>_vector_coherence.json
```

coherence 没有固定“通过阈值”，主要用于比较 capability 内部稳定性；每个 capability 至少需要两个非零、有限的 delta。

### 4.7 Stage 06：运行 steering generation

```bash
python scripts/06_run_steering.py "${BASE_CONFIG[@]}"
```

核心代码：

- `pipeline.run_steering`
- `models/generation.py`
- `hooks/intervention.py`
- `evaluation/answers.py`

模型先接收 generic reasoning prompt，并从 `Reasoning:` 开始生成。在 prefill 和后续 cached decoding 的每次目标层调用中，代码都对当前序列最后一个位置加入：

```text
hidden[:, -1, :] += alpha × vector
```

默认对四个 capability 和五个 alpha 分别生成：

```text
[-1.0, -0.5, 0.0, 0.5, 1.0]
```

`alpha = 0` 是无 steering baseline。同一道题的 baseline 文本会在 capability 之间复用，但仍会为每个 capability 保存一行，以便后续配对。

当前 generation 固定为 greedy decoding，并在代码中使用：

```text
do_sample = False
use_cache = True
```

`max_new_tokens` 控制最大生成长度。允许输出 CoT，但最终答案应遵循 prompt 中的 `Final Answer:` 格式。选择题提取字母；MATH500 使用规范化字符串 exact match；AIME 转换为整数比较。

主要产物：

```text
outputs/generations/<steering_run_id>.jsonl
```

成功行包含 run/vector/model/generation 参数和正确性；失败行包含完整 item/capability/alpha 身份。CUDA OOM 会标记为 `error_type: cuda_oom`。重跑时只跳过相同身份的最新成功行，失败项会再次执行。

重点参数：`vector_scaling`、`alphas`、`max_new_tokens`、`target_layer`、enabled datasets 和 `--limit`。

### 4.8 Stage 07：计算评测指标

```bash
python scripts/07_score_generations.py "${BASE_CONFIG[@]}"
```

核心代码：`pipeline.score_generations` 和 `evaluation/metrics.py`。

该阶段只使用最新的成功 generation，并要求同一题拥有配置中全部 alpha 的完整配对记录。主要统计：

- 每个 steering capability、每个 alpha 的 accuracy 和相对 baseline delta；
- high/low demand slice 的 accuracy；
- complete items 和因 alpha 不完整被排除的 items；
- steering capability × demand capability 的 specificity matrix；
- 每个 specificity cell 的样本数和缺失 cell；
- 只有完整且所有 cell 非空时才计算 diagonal dominance，否则写为 `null`。

主要产物：

```text
outputs/metrics/<steering_run_id>_steering_metrics.json
outputs/metrics/<steering_run_id>_steering_metrics.csv
```

## 5. 超参数速查

### 5.1 模型参数：`configs/model.yaml`

| 参数 | 默认值 | 作用和建议 |
|---|---|---|
| `name` | `Qwen/Qwen2.5-7B-Instruct` | 模型和 tokenizer 名称；改变后从 stage 03 重跑。 |
| `revision` | 固定 commit SHA | 必须使用 commit 或稳定 release tag，不能使用 `main/master`。 |
| `cache_dir` | `null` | 模型缓存目录；服务器可改为高速共享盘。 |
| `num_hidden_layers` | `28` | 用于配置校验；应与模型实际层数一致。 |
| `dtype` | `bfloat16` | 支持 `bfloat16/float16/float32`；按 GPU 支持情况选择。 |
| `device_map` | `auto` | Transformers 设备分配；单卡显存不足时可自动切分。 |
| `attention_implementation` | `sdpa` | 注意力实现；服务器依赖不支持时可修改或移除。 |
| `trust_remote_code` | `false` | Qwen2.5 当前无需开启。 |
| `max_new_tokens` | `2048` | stage 06 最大生成长度；可按任务所需的推理和答案长度调整。 |
| `do_sample` | `false` | 当前 generation 代码固定为 greedy；修改此 YAML 不会开启采样。 |
| `use_cache` | `true` | 当前 generation 代码固定使用 cache；修改此 YAML 不会关闭 cache。 |

### 5.2 数据参数：`configs/data.yaml`

| 参数 | 默认值 | 作用和建议 |
|---|---|---|
| `enabled_steering_datasets` | `[math500]` | 需要标注和 steering 的数据集；可加入 AIME、ARC-C、OBQA。 |
| `cache_dir` | `.cache/huggingface` | Hugging Face dataset cache。 |
| `sources.<name>.path` | 各 HF dataset ID | 远程数据源。 |
| `sources.<name>.name` | 数据集配置名或无 | 例如 ARC-Challenge、OBQA main。 |
| `sources.<name>.split` | 当前数据集默认 split | 更换 split 后从 stage 00 重跑。 |
| `sources.<name>.local_path` | `null` | 设置后优先读取本地 canonical/raw JSONL。 |
| `prepared_dir` | `data/processed` | 当前 pipeline 未读取该键；实际路径由 `experiment.paths.data_dir` 决定。 |

支持的数据集名称为：

```text
math500, aime2024, aime2025, aime2026, arc_c, obqa
```

MMLU 始终作为 extraction 数据集，不写入 `enabled_steering_datasets`。

### 5.3 实验参数：`configs/experiment.yaml`

| 参数 | 默认值 | 作用和建议 |
|---|---|---|
| `capabilities` | `QLl, QLq, CL, MCr` | MVP 固定要求四项齐全且不重复，不支持只选子集。 |
| `target_layer` | `19` | 零基 decoder layer；必须小于 `num_hidden_layers`。 |
| `high_demand_threshold` | `4` | MMLU extraction 和 high evaluation slice 的下限。 |
| `low_demand_threshold` | `1` | low evaluation slice 的上限；必须小于 high threshold。 |
| `vector_scaling` | `mean_norm` | `raw/unit/mean_norm`；只改变注入形式时可从 stage 06 重跑。 |
| `alphas` | `[-1,-0.5,0,0.5,1]` | 必须有限、唯一并包含 0；范围可在 pilot 后调整。 |
| `seed` | `42` | 模型阶段统一设置 Python/PyTorch seed；当前 greedy 主流程基本确定。 |
| `annotation.model` | `gpt-5.6-terra` | DeLeAn 标注模型；改变后 stage 01 会生成新的标注身份。 |
| `annotation.max_workers` | `8` | API 并发；遇到 429 或连接压力时降低。 |
| `annotation.max_attempts` | `5` | transient error 最大尝试次数。 |
| `annotation.initial_backoff_seconds` | `1.0` | 指数退避初始秒数。 |
| `paths.rubrics_dir` | `rubrics` | 四份 capability rubric 所在目录。 |
| `paths.data_dir` | `data` | processed/scored 数据根目录。 |
| `paths.outputs_dir` | `outputs` | activations/vectors/generations/metrics/manifests 根目录。 |

## 6. 正式实验命令

### 6.1 默认：只评测 MATH500

```bash
python scripts/00_prepare_data.py "${BASE_CONFIG[@]}"
python scripts/01_score_demands.py "${BASE_CONFIG[@]}"
python scripts/02_prepare_items.py "${BASE_CONFIG[@]}"
python scripts/03_capture_contrasts.py "${BASE_CONFIG[@]}"
python scripts/04_extract_vectors.py "${BASE_CONFIG[@]}"
python scripts/05_analyze_similarity.py "${BASE_CONFIG[@]}"
python scripts/06_run_steering.py "${BASE_CONFIG[@]}"
python scripts/07_score_generations.py "${BASE_CONFIG[@]}"
```

### 6.2 扩展到 MATH500 和 AIME 2024/2025/2026

```bash
EVALS=(
  --override 'data.enabled_steering_datasets=[math500,aime2024,aime2025,aime2026]'
)

python scripts/00_prepare_data.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/01_score_demands.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/02_prepare_items.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/03_capture_contrasts.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/04_extract_vectors.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/05_analyze_similarity.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/06_run_steering.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
python scripts/07_score_generations.py "${BASE_CONFIG[@]}" "${EVALS[@]}"
```

如需 ARC-C 和 OBQA，可继续追加 `arc_c,obqa`。启用更多数据集意味着 stage 01 会为这些数据集的每道题增加四次 API 标注，并显著增加 stage 06 的生成量。

## 7. 如何检查运行结果

### 数据和标注

```bash
wc -l data/processed/*.jsonl
wc -l data/scored/*_delean_long.jsonl
wc -l data/processed/extraction/*.jsonl
wc -l data/processed/evaluation/*.jsonl
```

long annotation 文件中可检查失败项：

```bash
rg '"status": "error"' data/scored/*_delean_long.jsonl
```

### Capture 和 generation 错误

```bash
find outputs/activations -name errors.jsonl -print
rg '"status": "error"' outputs/generations/*.jsonl
```

### Manifest

每个阶段会写：

```text
outputs/manifests/<stage>.json
outputs/manifests/<stage>/<run_id>.json
```

其中包含 resolved config、文件哈希、seed、prompt/rubric hash、包版本和可获得的模型/tokenizer commit，用于核对实验身份。

## 8. 参数变化后从哪里重跑

| 变化 | 建议起始阶段 | 原因 |
|---|---:|---|
| 数据源、split、local path、启用数据集 | 00 | canonical 数据和后续标注都可能改变。 |
| rubric、annotation model 或 annotation prompt/schema | 01 | 标注 cache identity 会变化。 |
| high/low threshold | 02 | scored 数据不变，只需重新筛选。 |
| 模型名称/revision/dtype、target layer、generic/capability prompt | 03 | activation 和 vector identity 改变。 |
| 向量聚合算法代码 | 04 | capture shard 可以复用。 |
| 仅修改 `vector_scaling` | 06 | stage 04 已同时保存 raw/unit/steering。 |
| alphas、max_new_tokens | 06，然后用相同配置运行 07 | steering run ID 和完整 alpha population 改变。 |
| 只修改统计代码 | 07 | generation 可以复用。 |
| max_workers、retry、backoff | 01 | 已成功标注仍会恢复，只影响未完成调用。 |
| data/output path | 对新目录从最早所需阶段开始 | 新目录不会自动引用旧目录产物。 |

## 9. 常见问题

### Stage 04 报某个 capability 没有 activation shards

先检查 stage 02 的 extraction 文件是否非空，再检查 stage 03 的 `errors.jsonl`。提高 MMLU 数量或处理 capture 错误后重跑 stage 03。

### Stage 05 报 coherence 至少需要两个 delta

该 capability 成功 capture 少于 2 个。提高 stage 03 的 limit 或补足成功 shard。

### Stage 06 CUDA OOM

generation OOM 时可先降低 `model.max_new_tokens`；同时检查 GPU 是否被其他任务占用。stage 03 的单题 capture OOM 通常需要更多显存、调整 device map/dtype，或减少单题 prompt 长度；仅降低题目数量不会降低单题峰值显存。

### Stage 07 找不到 generation 文件

通常是 stage 06 和 07 的 `alphas`、`max_new_tokens`、enabled datasets、vector scaling 或路径 override 不一致。使用完全相同的 resolved config 重跑 stage 07。

### 修改配置后为什么生成了新的 `<id>` 目录

activation、vector 和 generation 路径使用内容身份。影响实验语义的配置或输入文件改变后创建新 ID 是预期行为，可避免把旧产物混入新实验。

### 能否只运行部分 capability

当前 MVP 配置校验要求 `QLl`、`QLq`、`CL`、`MCr` 四项完整且唯一。若只想运行子集，需要先修改代码和指标假设，不能只改 YAML。
