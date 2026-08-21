# RISER Runnable Foundation

This directory is an isolated runnable copy of the official RISER code. It
contains the primitive extraction and clustering modules from the clone plus a
minimal Router, generation-time steering wrapper, and baseline-versus-steered
evaluation runner.

## Scope

The foundation intentionally does not download datasets, call the LLM-Judge,
or train the Router. Those components can be added later using the stable APIs
implemented here. The optional Judge dependency is loaded only when
`LLMJudgeFilter` is instantiated.

## Installation

```text
python -m pip install -e .
python -m pip install -e ".[dev]"
```

For Judge support:

```text
python -m pip install -e ".[judge]"
```

The NumPy upper bound is intentional: the Transformers version used by this
project requires NumPy below 2.0.

## Current pipeline

```text
prompt pairs
  -> ActivationExtractor
  -> positive_activation - negative_activation
  -> PrimitiveClustering / PrimitiveLibrary
  -> Router
  -> ActivationInjectionHook
  -> SteeredModel.generate
  -> EvaluationRunner
```

The Router follows RISER's two-head MLP design. It selects primitive vectors
with a default probability threshold of `0.7` and bounds each strength in
`[0, 2]`. `SteeredModel` caches the composed vector for one generation by
default, matching the paper's prefill-and-reuse behavior.

## Verification

The offline test suite does not download a checkpoint:

```text
python -m unittest discover -s tests -v
```

After installing the development extra, the same tests can also be run with:

```text
python -m pytest -q
```

## Input formats

Primitive extraction expects one JSON object per line:

```json
{"id": "q1", "positive_prompt": "...", "negative_prompt": "...", "task": "..."}
```

Evaluation expects:

```json
{"id": "q1", "prompt": "...", "reference": "..."}
```

## Runnable examples

Build a library from prompt pairs (the model must already be available to
Transformers):

```text
python examples/build_primitives.py \
  --input prompt_pairs.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layers 20 \
  --output primitives.pt
```

Generate one steered answer using an existing Router checkpoint:

```text
python examples/run_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --router router.pt \
  --library primitives.pt \
  --layer 20 \
  --prompt "Solve this problem carefully: ..."
```

Run the initial baseline-versus-steered evaluator:

```text
python -m riser.evaluation.cli \
  --model Qwen/Qwen2.5-7B-Instruct \
  --router router.pt \
  --library primitives.pt \
  --layer 20 \
  --input evaluation.jsonl \
  --output results.jsonl
```

The evaluator records both generations, input/output/total token counts,
latency, exact/substring reference metrics, and the Router's selected
primitive IDs and strengths. It does not train a Router or call an external
LLM-Judge; a Router checkpoint must therefore be supplied by the caller.
