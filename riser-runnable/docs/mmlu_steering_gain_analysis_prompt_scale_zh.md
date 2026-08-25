# MMLU Steering 提升不明显的两个主要原因

本文只总结当前已经确认的两点原因：

1. 评测 baseline 已经使用了 reasoning-oriented positive prompt，导致 steering 的边际空间较小；
2. primitive 经过单位归一化后，当前 `strength=1.0` 带来的 hidden-state 扰动相对较小。

本文基于当前 `riser-runnable` 代码和已经生成的 `results.jsonl`，不涉及 Router 训练、LLM-Judge 或其他尚未验证的因素。

## 1. Positive prompt 已经提前激活 reasoning 行为

### 1.1 向量提取使用了正负 instruction 对比

在 `examples/collect_mmlu_math_vectors.py` 中，positive prompt 明确要求模型：

```text
Work through the problem carefully, check the calculation,
explain the reasoning, and then give the final answer as one choice letter.
```

negative prompt 则要求：

```text
Return only the final choice letter and do not show any reasoning.
```

代码位置：

- `examples/collect_mmlu_math_vectors.py` 的 `make_prompt_pair()`；
- `riser/primitives/extractor.py` 中的 `ActivationPair.difference`，即 `positive_activation - negative_activation`。

因此，提取到的向量意图上是“从只回答案转向完整 reasoning”的方向。

### 1.2 但评测只使用 positive prompt

`scripts/prepare_mmlu_eval.py` 在转换评测数据时，只把 `positive_prompt` 写入最终的 `evaluation.jsonl`：

```python
"prompt": positive_prompt
```

所以当前实验实际比较的是：

```text
Baseline: positive prompt
Steered:  positive prompt + steering vector
```

而不是更能体现因果作用的：

```text
Baseline: negative/neutral prompt
Steered:  negative/neutral prompt + reasoning vector
```

这会导致 baseline 已经被语言指令强烈要求进行 reasoning。此时 vector 再次激活 reasoning feature，属于在已经激活的行为上继续叠加，边际收益自然可能很小。

### 1.3 当前结果与该解释一致

`results.jsonl` 中的 306 条已完成样本显示：

- Baseline accuracy：`80.07%`；
- Steered accuracy：`80.72%`；
- 净提升：只有 `2` 个正确样本，即 `+0.65` 个百分点；
- 但有 `134/306` 个样本的生成文本发生了变化。

这说明 steering 确实改变了部分生成过程，但由于 baseline 已经处于 reasoning prompt 条件下，文本变化没有稳定转化为最终答案变化。

### 1.4 更合适的验证矩阵

下一步应该至少加入以下四组：

| Prompt 条件 | 无 steering | 有 reasoning vector |
|---|---:|---:|
| negative prompt | baseline-negative | negative + vector |
| positive prompt | baseline-positive | positive + vector |

其中最关键的是 `negative prompt` 行。如果 `negative + vector` 明显接近 `positive`，才能更直接证明 vector 本身在激活 reasoning feature，而不是 positive instruction 在发挥主要作用。

更严格的版本可以再加入不包含 reasoning/answer-only 指令的 task-only prompt，并保持题目内容、选项和生成参数完全一致。

## 2. 当前注入幅度相对 hidden state 偏小

### 2.1 primitive 在聚类后被强制归一化

`riser/primitives/clustering.py` 的代表向量在保存前执行：

```python
norm = representative.norm()
if norm > 0:
    representative = representative / norm
```

因此每个 primitive 的范数约等于 `1`。

### 2.2 strength=1.0 只加入一个单位向量

当前脚本使用：

```bash
FIXED_PRIMITIVES_RAW="0"
FIXED_STRENGTHS_RAW="1.0"
```

`FixedRouter` 最终通过 `RouterInference.inject_activation()` 执行：

```python
injected_state = routing_hidden + v_inject
```

当前 `strength=1.0` 时，`v_inject` 基本就是一个范数为 1 的向量；代码没有根据当前 hidden-state 范数对注入向量进行额外放大或归一化。

### 2.3 当前 artifact 的数量级证据

从 `artifacts/mmlu_preliminary/` 中保存的 activation 可以看到：

- positive activation 平均范数约为 `103.6`；
- negative activation 平均范数约为 `113.6`；
- primitive 范数约为 `1.0`。

所以当前 strength=1 的注入量大约只有 hidden-state 范数的 `1%` 左右；即使使用允许的最大 strength=2，整体扰动也大约只有 `2%` 量级。

这并不意味着注入完全无效，但它更可能只改变部分 token 的 logit 排序，而不会稳定地改变最终选项。

### 2.4 当前结果也符合“小扰动”特征

在 306 条结果中：

- `172/306` 个样本的 baseline/steered 文本完全相同；
- `134/306` 个样本文本发生变化；
- 最终答案只翻转了 12 次，其中提升 7 次、下降 5 次。

这说明注入已经进入模型计算路径，但当前幅度不足以产生一致的答案级别变化。

### 2.5 建议进行 strength sweep

在不改变模型、题目、层和 prompt 的情况下，运行：

```text
strength = 0.0
strength = 0.5
strength = 1.0
strength = 1.5
strength = 2.0
```

每个配置都记录：

- baseline/steered MMLU accuracy；
- 输出文本变化比例；
- 最终答案翻转比例；
- 注入向量范数与目标 hidden-state 范数的比值；
- A/B/C/D 最终答案 logit 或概率（如果实现 logit-level evaluation）。

如果 accuracy 或 answer-logit margin 随 strength 呈现单调变化，才能说明当前问题主要是干预幅度不足；如果 strength=2 仍没有趋势，则需要回到向量语义和 prompt 对照设计。

## 综合判断

当前 steering 提升不明显，最合理的解释是：

```text
positive prompt 已经激活 reasoning
        +
strength=1 的单位向量只造成约 1% hidden-state 扰动
        ↓
文本有部分变化，但最终答案提升不稳定
```

因此下一步优先级应是：

1. 加入 negative/neutral prompt 对照，验证 vector 是否能够独立激活 reasoning；
2. 在 strength=0–2 范围内做系统 sweep；
3. 记录注入范数比例和答案 logit margin，而不只看最终 MMLU accuracy。

当前结论不应表述为“steering 没有效果”，更准确的表述是：

> 在已经使用 reasoning-oriented positive prompt 且注入强度较小的条件下，固定 steering vector 能改变部分生成文本，但尚未产生稳定的答案级收益。
