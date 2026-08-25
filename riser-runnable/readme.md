# RISER Runnable Foundation

This directory is an isolated runnable copy of the official RISER code. It
contains the primitive extraction and clustering modules from the clone plus a
minimal Router, generation-time steering wrapper, and baseline-versus-steered
evaluation runner.

## Scope

The main steering/evaluation path does not download datasets, call the
LLM-Judge, or train the Router. The opt-in MMLU collection example downloads
the selected mathematics subjects and builds prompt pairs and vectors. The
optional Judge dependency is loaded only when `LLMJudgeFilter` is instantiated.

## Installation

```text
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[data]"
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

All runnable model-generation paths use the tokenizer's Hugging Face
`chat_template`: the readable prompt is sent as one `user` message and
`add_generation_prompt=True` is used for the assistant turn. This applies to
vector collection, MMLU evaluation, the steering example, and the optional
Judge generation path. The JSONL files keep the readable user content, while
the runtime tokenizer call applies the template.

The MMLU evaluation prompt asks the model to write concise reasoning first
and finish with an explicit `Final answer: A/B/C/D` line. The evaluator parses
that explicit final-answer marker before using its legacy standalone-choice
fallback.

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

Collect 500 contrastive activation vectors from MMLU mathematics subjects
and build a primitive library in one step.  This requires the `data` extra:

```text
python examples/collect_mmlu_math_vectors.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --layers 20 \
  --device cuda \
  --prompt-pairs-output artifacts/mmlu_math_500_prompt_pairs.jsonl \
  --vectors-output artifacts/mmlu_math_500_vectors.pt \
  --library-output artifacts/mmlu_math_500_primitives.pt
```

The positive prompts request careful reasoning and the negative prompts ask
for only the answer letter.  The script does not use the optional LLM Judge.

Generate one steered answer using an existing Router checkpoint:

```text
python examples/run_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --router router.pt \
  --library primitives.pt \
  --layer 20 \
  --prompt "Solve this problem carefully: ..."
```

For a preliminary experiment, a Router checkpoint is optional.  Omit
`--router` and set primitive IDs and strengths manually:

```text
python examples/run_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --library primitives.pt \
  --layer 20 \
  --fixed-primitives 0 2 \
  --fixed-strengths 1.0 0.5 \
  --prompt "Solve this problem carefully: ..."
```

When `--router` is omitted, the route is deterministic and independent of
the hidden state.  If no fixed primitives are supplied, the model runs with
an empty route, which is useful as a no-injection control.  Fixed primitive
IDs refer to the row positions in the saved library and fixed strengths are
bounded by `--fixed-max-strength` (default `2.0`).

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

The evaluator accepts the same fixed-route options and can therefore run
without `--router`:

```text
python -m riser.evaluation.cli \
  --model Qwen/Qwen2.5-7B-Instruct \
  --library primitives.pt \
  --layer 20 \
  --fixed-primitives 0 \
  --fixed-strengths 1.0 \
  --input evaluation.jsonl \
  --output results.jsonl
```

The evaluator records both generations, input/output/total token counts,
latency, exact/substring reference metrics, and the Router's selected
primitive IDs and strengths. It does not train a Router or call an external
LLM-Judge. The current runnable foundation intentionally uses an external
MLP Router when a checkpoint is supplied; self-generated steering
configurations are reserved for a later experiment.
