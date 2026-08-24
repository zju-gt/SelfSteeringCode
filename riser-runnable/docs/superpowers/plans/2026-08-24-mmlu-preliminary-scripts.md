# MMLU Steering Preliminary Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Add a Linux Bash entry point and small Python helpers that run the MMLU vector-extraction → fixed steering → accuracy evaluation pilot with clearly documented configuration.

**Architecture:** Reuse the existing MMLU collector and activation-steering classes. A conversion helper turns collector JSONL into evaluation JSONL and exposes a robust last-choice parser; a dedicated MMLU evaluator uses `EvaluationRunner` with that parser; a Bash script orchestrates both stages and stores artifacts in one directory. Router training remains out of scope.

**Tech Stack:** Bash (`set -euo pipefail`), Python 3.10+, standard-library `json/re`, PyTorch, Transformers, existing `riser` package, unittest.

---

### Task 1: Add failing tests for MMLU conversion and answer parsing

**Files:**
- Create: `tests/test_mmlu_scripts.py`
- Test: `scripts/prepare_mmlu_eval.py` (to be created in Task 2)

- [ ] **Step 1: Write the failing tests**

Add tests for integer-to-letter conversion, last-choice extraction, JSONL conversion, metadata preservation, and invalid answers. The tests import `answer_to_letter`, `extract_last_choice`, `convert_rows`, and `mmlu_choice_match` from `scripts.prepare_mmlu_eval`.

- [ ] **Step 2: Run the focused tests to verify the expected failure**

Run:

```bash
python -m unittest tests.test_mmlu_scripts -v
```

Expected: import failure because `scripts/prepare_mmlu_eval.py` does not yet exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add -- tests/test_mmlu_scripts.py
git commit -m "test: specify MMLU evaluation conversion behavior"
```

### Task 2: Implement the MMLU conversion and metric helper

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/prepare_mmlu_eval.py`

- [ ] **Step 1: Implement `answer_to_letter`**

Accept integer `0..3` or letter strings `A..D`, return uppercase letters, and raise `ValueError` for all other values.

- [ ] **Step 2: Implement `extract_last_choice`**

Use a case-insensitive regular expression to collect standalone `A/B/C/D` tokens, including forms such as `B`, `B.`, and `(B)`, and return the last match or `None` when no valid choice occurs.

- [ ] **Step 3: Implement `mmlu_choice_match` and `convert_rows`**

`mmlu_choice_match(prediction, reference)` returns `1.0` only when the last extracted choice equals the normalized reference letter, otherwise `0.0` (or `None` when the reference is missing). `convert_rows` reads collector JSONL, uses `positive_prompt` and converted `answer`, and writes one evaluation object per line while retaining subject/question/choices/task IDs in metadata.

- [ ] **Step 4: Add the CLI**

Support:

```text
python scripts/prepare_mmlu_eval.py --input prompt_pairs.jsonl --output evaluation.jsonl
```

Report the number of converted examples and fail with a line-numbered error for malformed JSON or missing required fields.

- [ ] **Step 5: Run the focused tests to verify they pass**

Run:

```bash
python -m unittest tests.test_mmlu_scripts -v
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit the helper**

```bash
git add -- scripts/__init__.py scripts/prepare_mmlu_eval.py tests/test_mmlu_scripts.py
git commit -m "feat: add MMLU evaluation conversion helper"
```

### Task 3: Make Router inference dtype-safe for half/bfloat16 pilots

**Files:**
- Modify: `riser/router/inference.py:35-72`
- Modify: `tests/test_router.py`

- [ ] **Step 1: Write a failing dtype-compatibility test**

Add a test that creates a float32 Router and passes a float16 hidden state to `RouterInference.inject_activation`, asserting that the call succeeds and the returned tensor preserves the input dtype.

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
python -m unittest tests.test_router.RouterTests.test_inference_preserves_hidden_dtype -v
```

Expected: a dtype mismatch error from the Router linear layer.

- [ ] **Step 3: Implement minimal dtype alignment**

Before calling `router.route` or `router.forward`, cast the routing input to the Router parameter/buffer dtype; after composing the injection, cast the returned hidden state back to the original device and dtype. Keep routing metadata on CPU as before.

- [ ] **Step 4: Run focused and existing Router tests**

Run:

```bash
python -m unittest tests.test_router -v
```

Expected: all Router tests pass.

- [ ] **Step 5: Commit the dtype fix**

```bash
git add -- riser/router/inference.py tests/test_router.py
git commit -m "fix: align Router inference with model dtype"
```

### Task 4: Add a dedicated MMLU evaluator entry point

**Files:**
- Create: `scripts/evaluate_mmlu.py`
- Modify: `tests/test_mmlu_scripts.py`

- [ ] **Step 1: Write a parser smoke test**

Test that the evaluator parser accepts model, library, layer, input, output, fixed primitive IDs, strengths, device, dtype, and max-new-tokens without loading a checkpoint.

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
python -m unittest tests.test_mmlu_scripts.MMLUEvaluatorTests -v
```

Expected: import failure because `scripts/evaluate_mmlu.py` does not yet exist.

- [ ] **Step 3: Implement the evaluator**

Load the tokenizer/model, construct `RouterInference.from_pretrained` and `SteeredModel`, load evaluation examples, run baseline and steered generations through `EvaluationRunner`, and register `mmlu_choice_match` as the metric. Support fixed routing when `--router` is omitted and write JSONL results.

- [ ] **Step 4: Run all offline tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass, with only the existing optional Judge skip.

- [ ] **Step 5: Commit the evaluator**

```bash
git add -- scripts/evaluate_mmlu.py tests/test_mmlu_scripts.py
git commit -m "feat: add MMLU steering evaluator"
```

### Task 5: Add the Linux orchestration script

**Files:**
- Create: `scripts/run_mmlu_preliminary.sh`

- [ ] **Step 1: Implement robust path and configuration handling**

Use `SCRIPT_DIR`/`ROOT_DIR`, `set -euo pipefail`, configurable `PYTHON_BIN`, `MODEL_PATH`, `DEVICE`, `DTYPE`, `LAYERS`, `INJECT_LAYER`, `SUBJECTS`, `SPLIT`, `NUM_SAMPLES`, `CLUSTERS`, `FIXED_PRIMITIVES`, `FIXED_STRENGTHS`, `MAX_NEW_TOKENS`, and `OUTPUT_DIR`.

- [ ] **Step 2: Invoke the three pipeline stages**

Run the collector, call `prepare_mmlu_eval.py`, and call `evaluate_mmlu.py` with fixed routing. Create the output directory before the first write and print each output path at the end.

- [ ] **Step 3: Add shell validation and syntax verification**

Reject an empty model path, mismatched primitive/strength counts, and a CUDA request when `torch.cuda.is_available()` is false. Verify syntax with:

```bash
bash -n scripts/run_mmlu_preliminary.sh
```

- [ ] **Step 4: Commit the orchestration script**

```bash
git add -- scripts/run_mmlu_preliminary.sh
git commit -m "feat: add Linux MMLU preliminary experiment runner"
```

### Task 6: Write the concise Chinese usage document

**Files:**
- Create: `docs/mmlu_preliminary_experiment_zh.md`

- [ ] **Step 1: Document configuration locations**

Explain exactly where to set the Qwen model path, device, dtype, layer, MMLU subjects/split, sample count, primitive IDs, strengths, and output directory.

- [ ] **Step 2: Document execution and outputs**

Include installation commands, `bash scripts/run_mmlu_preliminary.sh`, artifact descriptions, strength/layer changes, and fixed-router limitations.

- [ ] **Step 3: Commit the documentation**

```bash
git add -- docs/mmlu_preliminary_experiment_zh.md
git commit -m "docs: add Chinese MMLU pilot instructions"
```

### Task 7: Final verification

**Files:**
- Verify: `scripts/run_mmlu_preliminary.sh`, `scripts/prepare_mmlu_eval.py`, `scripts/evaluate_mmlu.py`, `tests/`, `docs/mmlu_preliminary_experiment_zh.md`

- [ ] **Step 1: Run all offline Python tests**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 2: Run compile and shell syntax checks**

```bash
python -m compileall -q riser scripts tests
bash -n scripts/run_mmlu_preliminary.sh
```

- [ ] **Step 3: Run CLI help smoke checks without downloading a model**

```bash
python scripts/prepare_mmlu_eval.py --help
python scripts/evaluate_mmlu.py --help
```

- [ ] **Step 4: Inspect the final diff and status**

```bash
git diff HEAD~6..HEAD --stat
git status --short --branch
```

Report that the offline checks pass. Do not claim a real MMLU run until the user supplies a model and executes the GPU/data-dependent command.
