# RISER Runnable Foundation Design

## Goal

Create an isolated, runnable foundation for the cloned RISER repository that supports primitive-vector preparation, Router-based activation steering, and baseline-versus-steered evaluation without requiring dataset loaders, LLM-Judge calls, or Router training.

## Architecture

The implementation preserves the existing primitive extraction, clustering, library, and hook APIs. It fills the missing Router and steered-model modules, adds a small evaluation package, makes the optional LLM-Judge dependency non-blocking, and documents a reproducible smoke-test path.

The runtime path is:

```text
prompt pairs
  -> ActivationExtractor
  -> ActivationPair.difference
  -> PrimitiveClustering / PrimitiveLibrary
  -> Router
  -> ActivationInjectionHook
  -> SteeredModel.generate
  -> EvaluationRunner
```

The Router uses the RISER paper architecture: a shared `d -> 1024 -> 1024` SiLU MLP, a sigmoid selection head with a configurable hard threshold (default `0.7`), and a bounded strength head with maximum strength `10.0` by default. It supports both `[d]` and `[batch, d]` hidden-state inputs and composes vectors as `sum(mask_i * strength_i * primitive_i)`.

The steered model registers the existing target-layer hook for generation, computes and caches the route during the first applicable forward pass by default, reuses the same injection during decoding, and always removes the hook in a `finally` path. A refresh option remains available for experiments that need per-forward routing.

## Components and responsibilities

### Router implementation

Create `riser/router/model.py` with:

- `RouterConfig` dataclass containing hidden size, primitive count, bottleneck size, selection threshold, maximum strength, and optional temperature settings.
- `Router(nn.Module)` exposing `forward(hidden_state, hard=False)` with the five-value return shape expected by `RouterInference`.
- `Router.route(hidden_state, primitive_library, hard=False)` returning an injection tensor and structured routing information.
- deterministic hard selection for inference and a straight-through-compatible path for future training, without implementing a training loop in this phase.
- checkpoint-compatible configuration serialization through a plain dictionary.

### Steered-model wrapper

Create `riser/inference/steered_model.py` with:

- a wrapper around any Hugging Face causal language model;
- `generate` and `forward` delegation;
- hook lifecycle management;
- route-cache reset and routing-information accessors;
- validation that the primitive-library hidden dimension matches the model/router dimension;
- device and dtype handling that does not mutate the caller's model unexpectedly.

### Evaluation package

Create `riser/evaluation/` with:

- `records.py`: dataclasses for examples and generation results;
- `metrics.py`: exact-match, substring-match, output-length, and token-count metrics;
- `runner.py`: paired baseline/steered generation with optional user-supplied metric callables;
- a small CLI or module entry point accepting JSON/JSONL examples and writing JSONL results.

The evaluator will not call an external Judge. It will record input tokens, generated tokens, total tokens, latency, selected primitive IDs, and strengths so later ablations can compare steering against baseline compute.

### Optional dependencies and documentation

- Make `anthropic` optional at import time; using `LLMJudgeFilter` without the package should raise a focused installation error.
- Add a minimal `requirements.txt` for the runnable path and a `pyproject.toml` for editable installation.
- Update `readme.md` with installation, primitive-library, steering, and evaluation examples.
- Add a small `examples/` smoke-test path that can run with a tiny test model and does not download a large checkpoint automatically.

## Testing strategy

Tests will be written before implementation for:

1. Router output shapes, hard-threshold behavior, strength bounds, composition, and checkpoint round-trip.
2. Hook injection of only the final sequence position and guaranteed hook removal.
3. Steered-model generation lifecycle and route-cache behavior using a tiny local causal-model double.
4. Evaluation records, token accounting, metric functions, and JSONL serialization.
5. Top-level imports with and without the optional `anthropic` package.

An optional integration smoke test will be documented for a real Hugging Face causal model; the default test suite will remain offline and deterministic.

## Scope exclusions

This phase deliberately excludes automatic dataset loading, LLM-Judge execution, supervised Router warm-up, GRPO, distributed training, and full paper benchmark reproduction. Those can consume the stable interfaces created here in a later phase.

## Acceptance criteria

- The original cloned directory is unchanged.
- The `-runnable` copy imports successfully without `anthropic` installed.
- A tiny local model can execute baseline and steered generation through the same public API.
- A primitive library can be saved, loaded, and consumed by RouterInference.
- The evaluation command produces paired JSONL results with token-use and routing metadata.
- Offline tests pass and README commands match the implemented interfaces.
