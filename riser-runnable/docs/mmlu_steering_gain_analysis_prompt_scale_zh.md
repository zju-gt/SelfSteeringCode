# Steering 提升不明显的两个原因

## 1. 评测 baseline 不应使用向量提取的 positive prompt

当前实现中，`scripts/prepare_mmlu_eval.py` 会把向量提取阶段的 `positive_prompt` 直接作为评测输入，因此实际比较的是：

```text
positive prompt
positive prompt + steering vector
```

这会让 baseline 预先获得“仔细推理、检查计算、解释过程”等显式 reasoning 指令，削弱 activation steering 的边际收益。需要区分两个阶段：

- 向量提取：使用 RISER 的 positive/negative prompt，构造对比激活差异；
- 下游评测：不使用上述 reasoning 指令，采用 neutral zero-shot prompt，让 baseline 和 steered 模型接收完全相同的原始题目。

RISER 论文将 Base Model 定义为 zero-shot/standard inference，并将 CoT 作为单独的 baseline；因此评测应改为：

```text
neutral zero-shot prompt
neutral zero-shot prompt + steering vector
```

建议的 MMLU neutral prompt 只保留题目、选项和答案位置，不加入“仔细推理”“展示过程”等指令：

```text
Question:
{{QUESTION}}

Choices:
A. {{CHOICE_A}}
B. {{CHOICE_B}}
C. {{CHOICE_C}}
D. {{CHOICE_D}}

Answer:
```

如果需要衡量显式 prompting 的影响，应另行增加 `CoT` 与 `CoT + steering` 两组，不能把 positive prompt 当作 RISER 风格的 baseline。

## 2. 当前注入强度相对 hidden state 偏小

`riser/primitives/clustering.py` 会把每个 primitive 归一化为单位向量；随后 `FixedRouter` 在 `strength=1.0` 时只执行：

```python
injected_state = hidden_state + primitive
```

当前 primitive 范数约为 `1`，而模型 hidden state 范数约为 `100` 左右，因此实际扰动只有约 `1%`。这种小幅变化可能改变部分生成 token，但不一定跨过最终答案的决策边界，导致 steering 文本变化与 MMLU accuracy 提升不一致。

## 3. RISER 中使用的 positive / negative prompt

RISER 论文 Appendix G.1 使用“Reasoning Fidelity Contrast”设计：positive 强调可验证、证明级的推导；negative 要求模型基于表面关联生成听起来合理但未经验证的回答。以下按 RISER 原文列出。

来源： [RISER 论文 Appendix G.1](https://aclanthology.org/2026.findings-acl.226.pdf)。

### Positive prompt

```text
Role: You are a meticulous logician focused on absolute precision.
Task: Derive the answer to the following question using proof-level rigor.
Instructions:
• Derive: Do not just state facts; deduce them from axioms or given data.
• Verify: Check each intermediate calculation or logic jump for errors.
• Precision: Prioritize correctness over fluency. Reject any heuristic shortcuts.
• Output: Provide a sound explanation.
Question: {{QUESTION}}
Rigorous Derivation:
```

### Negative prompt

```text
Role: You are a fluent conversationalist acting on "autopilot".
Task: Provide a plausibly sounding answer based on surface-level associations.
Instructions:
• Flow: Write whatever comes to mind first based on language patterns.
• Approximate: Do not perform actual calculations or verification. Use "ballpark" figures.
• Plausibility: The answer should sound correct to a layperson, even if the logic is flawed.
• Output: Generate a coherent but unverified response (simulate a hallucination if necessary).
Question: {{QUESTION}}
Plausible Response:
```

与当前实现相比，主要变化不是简单地把 positive 改成“多写 reasoning”，而是让两种状态在“可验证推导”与“表面合理但未验证”之间形成更明确的质量差异。后续仍应配合 RISER 的 LLM-Judge 过滤，只保留 positive 确实严谨、negative 确实缺乏有效推理的样本对。

## 4. 代码更新需求：批量并行推理与时间戳结果文件

以下是后续代码更新要求，本节暂时只记录需求，不在本次修改中实现。

### 4.1 使用 Transformers batch inference

当前 `EvaluationRunner._generate()` 每次只接收一个 prompt，`run()` 逐条调用 baseline 和 steered 生成。需要新增项目级 `run_batch()`，使用 Hugging Face Transformers 的批量接口，而不是启动多个 Python 进程：

```python
encoded = tokenizer(
    prompts,
    padding=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**encoded, **generation_kwargs)
```

具体要求：

- shell 和 Python CLI 都支持 `BATCH_SIZE` / `--batch-size`，由用户指定每次并行推理的样本数；
- 对 decoder-only 模型使用左 padding，正确处理不同 prompt 长度；
- baseline 和 steered 分别以 batch 运行，不能在同一模型 forward 中混用 hook；
- 保持输入顺序，逐样本解码并记录 input/output/total tokens、latency、metrics 和 routing；
- `ActivationInjectionHook`、`RouterInference` 和 `SteeredModel` 必须正确处理 `[batch, hidden]`；动态 Router 要保留每个样本各自的路由信息；
- 每完成一个 batch 的 baseline/steered 对，就逐条写入 JSONL 并 `flush()`，保留现有的中断保护能力；
- `BATCH_SIZE=1` 的输出应与当前逐条实现一致，作为回归测试。

这里的“并行”指 Transformers 在一个 batch 内并行执行 GPU 推理，不等同于多进程或多 GPU 数据并行。

### 4.2 结果文件命名

如果用户没有显式指定输出文件，默认使用本地时间生成：

```python
datetime.now().strftime("results_%y%m%d_%H%M.jsonl")
```

例如：

```text
results_250825_0815.jsonl
```

要求：

- 默认输出目录仍为 `artifacts/mmlu_preliminary/`；
- `--output` 或 `RESULTS_OUTPUT` 显式指定时优先使用用户路径；
- 提供可复现实验的时间戳/文件名覆盖选项，避免同一分钟重复运行产生歧义；
- 结果记录中同步保存 `batch_size`、`max_new_tokens`、layer、primitive、strength 和生成时间戳，不能只依赖文件名推断配置。
