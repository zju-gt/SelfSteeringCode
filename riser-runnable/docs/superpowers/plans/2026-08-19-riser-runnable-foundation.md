# RISER Runnable Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete an isolated runnable RISER foundation with an API-compatible Router, generation-time activation steering, and a dataset-agnostic baseline-versus-steered evaluator.

**Architecture:** Preserve the existing extractor, clustering, library, and hook interfaces. Add a two-head MLP Router and a `SteeredModel` wrapper that caches one composed injection vector per generation by default. Add a small evaluation package and CLI that records outputs, token counts, routing decisions, and simple reference-based metrics; keep datasets, LLM-Judge calls, and Router training out of scope.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, scikit-learn (existing clustering), pytest for offline tests, JSON/JSONL for evaluation data.

---

### Task 1: Establish package metadata and optional Judge dependency

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Modify: `riser/primitives/filtering.py:1-20`
- Modify: `readme.md`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write import and dependency tests**

Create a test that imports `riser`, `riser.router`, `riser.inference`, and `riser.evaluation`. Assert that importing the package does not require an Anthropic client. If `anthropic` is unavailable, instantiate `LLMJudgeFilter` and assert it raises a focused `ImportError`/`RuntimeError` mentioning `anthropic` rather than failing during module import.

- [ ] **Step 2: Run the import tests and confirm the current failure**

Run:

```text
python -m pytest tests/test_imports.py -q
```

Expected before implementation: failure because `riser.router.model`, `riser.inference.steered_model`, and/or the top-level `anthropic` import is unavailable.

- [ ] **Step 3: Add package metadata**

Add a `pyproject.toml` using setuptools with package discovery for `riser*` and dependencies `torch`, `transformers`, `scikit-learn`, and `matplotlib`. Keep `anthropic` out of the base dependencies and expose it as an optional `judge` extra. Add `pytest` as a `dev` extra.

Add `requirements.txt` containing the base runtime dependencies and a comment that `anthropic` is only needed for the optional Judge.

- [ ] **Step 4: Make `anthropic` lazy**

Remove the module-level hard dependency from `riser/primitives/filtering.py`. Import `anthropic` inside `LLMJudgeFilter.__init__`; if unavailable, raise:

```python
raise ImportError(
    "LLMJudgeFilter requires the optional 'anthropic' package. "
    "Install it with: pip install '.[judge]'"
)
```

Store the imported module/client on the instance without changing the existing public methods.

- [ ] **Step 5: Update README installation and scope**

Document:

```text
pip install -e .
pip install -e ".[dev]"
```

State explicitly that dataset loading, LLM-Judge evaluation, and Router training are not included in the runnable foundation.

- [ ] **Step 6: Run the import tests**

Run `python -m pytest tests/test_imports.py -q`; expected: PASS when base dependencies are installed, with the Judge test skipped or passing according to whether `anthropic` exists.

- [ ] **Step 7: Commit the packaging change**

```text
git add pyproject.toml requirements.txt riser/primitives/filtering.py readme.md tests/test_imports.py
git commit -m "build: make RISER foundation importable without Judge extras"
```

### Task 2: Implement the API-compatible Router

**Files:**
- Create: `riser/router/model.py`
- Modify: `riser/router/__init__.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write Router behavior tests**

Add tests using `hidden_size=4`, `num_primitives=3`, and `bottleneck_dim=8` that verify:

```python
config = RouterConfig(hidden_size=4, num_primitives=3, bottleneck_dim=8)
router = Router(config)
hidden = torch.zeros(2, 4)
mask, strength, probs, logits, features = router(hidden, hard=True)
assert mask.shape == (2, 3)
assert strength.shape == (2, 3)
assert probs.shape == (2, 3)
assert logits.shape == (2, 3)
assert features.shape == (2, 8)
assert torch.all((strength >= 0) & (strength <= 2))
```

Patch the selection head bias to produce probabilities below and above `0.7`, then assert the hard mask follows the threshold. Test that a one-dimensional `[4]` input is accepted and returns a batch dimension of one.

Add a composition test with an explicit `[3, 4]` primitive matrix and assert `route()` equals the weighted matrix product `((mask * strength) @ primitives)`.

Add a `RouterConfig.to_dict()`/`from_dict()` round-trip and a `torch.save`/`load_state_dict` round-trip.

- [ ] **Step 2: Run Router tests and confirm the missing-module failure**

Run `python -m pytest tests/test_router.py -q`; expected before implementation: import failure for `riser.router.model`.

- [ ] **Step 3: Implement `RouterConfig`**

Define a dataclass with:

```python
hidden_size: int
num_primitives: int
bottleneck_dim: int = 1024
selection_threshold: float = 0.7
max_strength: float = 2.0
strength_temperature: float = 1.0
```

Validate positive dimensions, threshold in `[0, 1]`, and positive maximum strength/temperature. Implement `to_dict()` and `from_dict()` returning ordinary serializable values.

- [ ] **Step 4: Implement the shared MLP and heads**

Implement:

```python
self.feature_extractor = nn.Sequential(
    nn.Linear(hidden_size, bottleneck_dim),
    nn.SiLU(),
    nn.Linear(bottleneck_dim, bottleneck_dim),
    nn.SiLU(),
)
self.selection_head = nn.Linear(bottleneck_dim, num_primitives)
self.strength_head = nn.Linear(bottleneck_dim, num_primitives)
```

In `forward`, normalize a `[d]` input to `[1, d]`, reject other ranks or wrong hidden dimensions, compute sigmoid selection probabilities, and compute strengths as `sigmoid(strength_logits / temperature) * max_strength`. With `hard=True`, use `(probs >= threshold).to(probs.dtype)` for the mask; otherwise use probabilities as soft weights. Return `(mask, strength, probs, logits, features)`.

- [ ] **Step 5: Implement vector composition**

Implement `route(hidden_state, primitive_library, hard=False)` with primitive shape `[K, d]`. Accept a tensor or a dictionary whose keys are primitive IDs; sort dictionary keys before stacking. Validate `K` and `d`, move/cast the library to the hidden-state device/dtype, call `forward`, and compute:

```python
injection = (selection_mask * strength) @ primitive_library
```

Return the injection and a dictionary containing masks, strengths, probabilities, selected IDs, and the unmodified feature tensor. Remove the temporary batch dimension only when the caller supplied a one-dimensional hidden state.

- [ ] **Step 6: Export the Router**

Update `riser/router/__init__.py` to export `Router`, `RouterConfig`, and `RouterInference` without introducing circular imports.

- [ ] **Step 7: Run Router tests**

Run `python -m pytest tests/test_router.py -q`; expected: all Router tests pass.

- [ ] **Step 8: Commit the Router**

```text
git add riser/router/model.py riser/router/__init__.py tests/test_router.py
git commit -m "feat: implement RISER routing model"
```

### Task 3: Complete generation-time steering and hook lifecycle

**Files:**
- Create: `riser/inference/steered_model.py`
- Modify: `riser/inference/hooks.py`
- Modify: `riser/inference/__init__.py`
- Test: `tests/test_hooks.py`
- Test: `tests/test_steered_model.py`

- [ ] **Step 1: Write hook tests**

Create a tiny module with a `model.layers` `ModuleList` whose layer returns a `[batch, sequence, hidden]` tensor. Register `ActivationInjectionHook` with a deterministic injection function and assert only `hidden[:, -1, :]` changes while earlier positions are bitwise equal. Assert `remove()` makes subsequent calls unchanged.

- [ ] **Step 2: Write wrapper lifecycle tests**

Create a tiny local causal-model double with a `model.layers` list, a tokenizer double, and a `generate` method that executes the target layer. Assert `SteeredModel.generate()` registers a hook during generation, removes it afterward, exposes routing info, and removes the hook even if the fake model raises. Assert cache reuse calls the router once per generation and `clear_route_cache()` permits a new route.

- [ ] **Step 3: Run the steering tests and confirm the missing-wrapper failure**

Run `python -m pytest tests/test_hooks.py tests/test_steered_model.py -q`; expected before implementation: import failure for `riser.inference.steered_model` and lifecycle tests failing.

- [ ] **Step 4: Implement `SteeredModel`**

The wrapper will hold `base_model`, `router_inference`, `target_layer`, `cache_routing=True`, and private cached vector/info fields. Its injection closure will call `RouterInference.inject_activation()` on the first hidden state, cache `injected - hidden` and routing info, then return `hidden + cached_vector` on later forwards. `generate()` will clear the cache, register `ActivationInjectionHook`, delegate all keyword arguments to the base model, and remove the hook in `finally`. `forward()` will use the same lifecycle for one forward call. Add `get_last_routing_info()` and `clear_route_cache()`.

- [ ] **Step 5: Harden hook output handling**

Preserve tuple outputs and clone the hidden-state tensor before changing the final sequence position. Validate that the injection function returns a tensor with the same `[batch, hidden]` shape. Keep existing model architecture detection for `model.layers` and `transformer.h`.

- [ ] **Step 6: Export and test**

Update `riser/inference/__init__.py`, run both test files, and verify the complete import path `from riser.inference import SteeredModel`.

- [ ] **Step 7: Commit steering support**

```text
git add riser/inference/steered_model.py riser/inference/hooks.py riser/inference/__init__.py tests/test_hooks.py tests/test_steered_model.py
git commit -m "feat: add cached generation-time activation steering"
```

### Task 4: Add evaluation records, metrics, and runner

**Files:**
- Create: `riser/evaluation/records.py`
- Create: `riser/evaluation/metrics.py`
- Create: `riser/evaluation/runner.py`
- Create: `riser/evaluation/__init__.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write metric and serialization tests**

Test `exact_match(" The Answer ", "the answer")`, `substring_match`, output-token counting from generated IDs, and JSON serialization of an evaluation result. Assert missing references produce `None` metrics rather than an exception.

- [ ] **Step 2: Write paired-runner tests**

Use fake baseline and steered models plus a tokenizer double. Run two examples and assert the result contains both outputs, input/output/total token counts, latency fields, metric values, and the last routing information from the steered model. Assert JSONL writing creates one JSON object per line.

- [ ] **Step 3: Implement evaluation data classes**

Define `EvaluationExample(example_id, prompt, reference=None, metadata=None)` and `EvaluationResult` containing prompt, both outputs, token counts, elapsed times, metrics, routing metadata, and metadata. Add `to_dict()` using JSON-safe primitive values.

- [ ] **Step 4: Implement metrics**

Implement normalized case-insensitive exact match, normalized substring match, and a `callable` metric adapter. Keep metrics deterministic and independent of any external Judge.

- [ ] **Step 5: Implement paired generation**

Implement `EvaluationRunner(baseline_model, steered_model, tokenizer)` with `run(examples, generation_kwargs=None, metrics=None)` and `write_jsonl(results, path)`. Tokenize each prompt once for input-token accounting, call each model's `generate`, decode only newly generated tokens when possible, count generated tokens from the returned sequence, and attach routing info from the steered wrapper.

- [ ] **Step 6: Export and run tests**

Export the records, metrics, and runner from `riser/evaluation/__init__.py`. Run `python -m pytest tests/test_evaluation.py -q`; expected: all tests pass.

- [ ] **Step 7: Commit evaluation support**

```text
git add riser/evaluation tests/test_evaluation.py
git commit -m "feat: add baseline and steered evaluation runner"
```

### Task 5: Add examples and command-line evaluation

**Files:**
- Create: `examples/build_primitives.py`
- Create: `examples/run_steering.py`
- Create: `examples/evaluate_steering.py`
- Modify: `readme.md`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Write CLI parsing tests**

Test that the evaluation CLI accepts `--model`, `--router`, `--library`, `--layer`, `--input`, `--output`, `--max-new-tokens`, and `--device`, and that a missing input file produces a clear error.

- [ ] **Step 2: Implement primitive-building example**

Accept a JSONL file containing `positive_prompt`, `negative_prompt`, `task`, and `task_id`; call the existing extractor, clustering, and library APIs; save the `.pt` library and optional metadata. Do not download a dataset automatically.

- [ ] **Step 3: Implement steering example**

Load a Hugging Face causal model, a saved Router checkpoint, and a primitive library; construct `RouterInference` and `SteeredModel`; generate one prompt and print routing information plus output.

- [ ] **Step 4: Implement evaluation CLI**

Load JSONL examples, construct baseline and steered model instances, run `EvaluationRunner`, and write paired JSONL results. Include `--no-cache-routing` for the per-forward experimental mode.

- [ ] **Step 5: Update README with exact commands**

Document the JSONL schemas, a fake-model offline test command, primitive-building command, steering command, and evaluation command. State that a Router checkpoint must already exist because training is intentionally out of scope.

- [ ] **Step 6: Run CLI tests and commit examples**

Run `python -m pytest tests/test_examples.py -q`, then:

```text
git add examples readme.md tests/test_examples.py
git commit -m "docs: add runnable RISER examples and evaluation CLI"
```

### Task 6: Full verification and handoff

**Files:**
- Modify: `readme.md` only if verification finds command mismatches

- [ ] **Step 1: Run the complete offline suite**

Run:

```text
python -m pytest -q
```

Expected: all offline tests pass without downloading a model or calling an external API.

- [ ] **Step 2: Run syntax and import checks**

Run:

```text
python -m compileall -q riser examples
python -c "import riser; from riser.router import Router, RouterConfig; from riser.inference import SteeredModel; from riser.evaluation import EvaluationRunner; print('imports ok')"
```

Expected: both commands exit successfully.

- [ ] **Step 3: Verify the source copy is unchanged**

Run `git -C ../RISER-Orchestrating-Latent-Reasoning-Skills-for-Adaptive-Activation-Steering status --short`; expected: no output.

- [ ] **Step 4: Review the runnable copy status and summarize limitations**

Report the test command and result, the new public APIs, and explicitly note that data loading, Judge calls, and Router training remain deferred.

- [ ] **Step 5: Commit any final verification-only documentation fix**

Only if README commands required correction, commit with:

```text
git add readme.md
git commit -m "docs: correct runnable verification instructions"
```
