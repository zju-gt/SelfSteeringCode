# Self-Steering 实验运行指南实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写一份简明、可直接用于服务器实验的中文 `00–07` 流水线运行指南。

**Architecture:** 只新增一份独立指南，不修改实验代码和配置。内容以当前 `main` 中的脚本、pipeline 函数和 YAML 配置为唯一事实来源，按运行顺序串联输入、输出、超参数和重跑边界。

**Tech Stack:** Markdown、Python CLI、YAML、pytest。

---

### Task 1: 建立脚本与源码映射

**Files:**
- Read: `scripts/00_prepare_data.py` through `scripts/07_score_generations.py`
- Read: `src/self_steering/pipeline.py`
- Read: `configs/model.yaml`
- Read: `configs/data.yaml`
- Read: `configs/experiment.yaml`
- Create: `docs/self_steering_experiment_guide.md`

- [x] **Step 1: 核对八个脚本的 pipeline 入口**

运行：

```bash
rg -n "from self_steering.pipeline import|print_paths" scripts/[0-7][0-7]_*.py
```

预期：每个编号脚本只调用一个同名职责的 pipeline 入口。

- [x] **Step 2: 写入指南标题、适用范围、运行前准备和流程总览**

正文必须说明命令从 `vector-make` 执行，阶段顺序为：

```text
00 prepare data → 01 score demands → 02 prepare items
→ 03 capture contrasts → 04 extract vectors
→ 05 analyze similarity → 06 run steering → 07 score generations
```

- [x] **Step 3: 写入脚本映射表**

表格列固定为：脚本、核心函数、作用、主要输入、主要输出、资源需求。

### Task 2: 编写分阶段运行说明

**Files:**
- Modify: `docs/self_steering_experiment_guide.md`
- Reference: `src/self_steering/pipeline.py`
- Reference: `src/self_steering/datasets/delean_labeler.py`
- Reference: `src/self_steering/hooks/intervention.py`

- [x] **Step 1: 编写 00–02 数据阶段**

必须覆盖 canonical JSONL、DeLeAn 四维标注、rubric、API 并发/重试、`d_k >= 4` 提取集和 high/low evaluation memberships。

- [x] **Step 2: 编写 03–05 向量阶段**

必须覆盖第 19 层的零基索引、generic/capability prompt 对比、per-item safetensors、raw/unit/mean-norm 三种向量以及 coherence/cosine similarity。

- [x] **Step 3: 编写 06–07 steering 与指标阶段**

必须覆盖从 `Reasoning:` 开始持续注入、alpha-zero baseline、答案解析、增量恢复、错误记录、paired population、specificity coverage 和 diagonal dominance 完整性要求。

### Task 3: 编写超参数与可复制命令

**Files:**
- Modify: `docs/self_steering_experiment_guide.md`
- Reference: `configs/model.yaml`
- Reference: `configs/data.yaml`
- Reference: `configs/experiment.yaml`
- Reference: `scripts/_common.py`

- [x] **Step 1: 写超参数速查表**

按模型、数据、能力标注、提取/注入和路径五组列出当前配置键、默认值、作用、常见调整及受影响的起始阶段。

- [x] **Step 2: 写 smoke-test 命令**

使用同一组基础配置，并在 00、01 阶段从 `--limit 20` 起步；stage 02 后用 `wc -l data/processed/extraction/*.jsonl` 确认每个 capability 至少有 2 个样本，否则提高 limit 后增量重跑。stage 03 和 06 可用 `--limit 2` 控制实际 GPU 工作量。明确 smoke test 仍会在 stage 01 调用 OpenAI API、在 stage 03/06 加载 7B 模型。

- [x] **Step 3: 写正式实验命令**

提供默认 MATH500 流程，以及通过：

```bash
--override data.enabled_steering_datasets=[math500,aime2024,aime2025,aime2026]
```

扩展评测数据集的示例。

- [x] **Step 4: 写配置变更后的重跑表**

至少覆盖数据源、rubric/annotation model、threshold、target layer/prompt、vector scaling、alpha/max tokens 六类变更。

### Task 4: 一致性验证与提交

**Files:**
- Verify: `docs/self_steering_experiment_guide.md`

- [x] **Step 1: 检查占位符和脚本名称**

运行：

```bash
rg -n "TB[D]|TO[D]O|待补充|00_prepare_data|07_score_generations" docs/self_steering_experiment_guide.md
```

预期：无占位符，并能找到起止脚本。

- [x] **Step 2: 验证 CLI 示例所用的公共参数**

运行：

```bash
python scripts/00_prepare_data.py --help
python scripts/06_run_steering.py --help
```

预期：两者均显示 `--config`、`--override` 和 `--limit`。

- [x] **Step 3: 验证文档未引入代码回归**

运行：

```bash
pytest -q
git diff --check
```

预期：离线测试无失败，diff 无 whitespace error；真实 Qwen 测试可因当前工作站依赖不兼容明确跳过。

- [x] **Step 4: 提交指南**

```bash
git add docs/self_steering_experiment_guide.md
git commit -m "docs: add self-steering experiment guide"
```
