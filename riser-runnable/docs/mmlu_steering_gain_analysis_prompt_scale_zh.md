# Steering 提升不明显的两个原因

## 1. Baseline 已经被 positive prompt 激活了 reasoning

向量提取时，positive prompt 要求模型“仔细推理并解释过程”，negative prompt 才要求“只回答案”。但评测阶段在 `scripts/prepare_mmlu_eval.py` 中只使用 `positive_prompt`，因此实际比较的是：

```text
positive prompt
positive prompt + steering vector
```

Baseline 本身已经处于 reasoning 状态，steering vector 的边际作用很小，所以文本可能发生变化，但最终答案准确率提升不明显。

## 2. 当前注入强度相对 hidden state 偏小

`riser/primitives/clustering.py` 会把每个 primitive 归一化为单位向量；随后 `FixedRouter` 在 `strength=1.0` 时只执行：

```python
injected_state = hidden_state + primitive
```

当前 primitive 范数约为 `1`，而模型 hidden state 范数约为 `100` 左右，因此实际扰动只有约 `1%`。这种小幅变化可能改变部分生成 token，但不一定跨过最终答案的决策边界，导致 steering 文本变化与 MMLU accuracy 提升不一致。
