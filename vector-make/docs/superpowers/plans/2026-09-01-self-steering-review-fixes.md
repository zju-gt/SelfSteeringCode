# Self-Steering MVP Review Fixes Implementation Plan

> Implement each task with a failing focused test before the smallest production change.

**Goal:** Repair the confirmed correctness and reproducibility defects without restructuring the existing Self-Steering MVP pipeline.

**Architecture:** Keep the current modules and numbered stages. Add small identity/provenance helpers, normalize data at adapter boundaries, make metric completeness explicit, and catch failures only at independent item boundaries.

**Tech Stack:** Python 3.10+, PyTorch, Transformers 4.x, datasets, safetensors, pytest.

---

### Task 1: Stable generation identity and canonical rows

**Files:**
- Modify: `src/self_steering/pipeline.py`
- Modify: `src/self_steering/evaluation/metrics.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_metrics.py`

1. Add failing tests proving pending and stored rows have identical keys and duplicate successful keys are scored once.
2. Add one item-identity helper and one generation-record constructor.
3. Canonicalize successful generation rows by key, keeping the last success.
4. Update resume, baseline cache, and metric pairing to use the canonical identity.
5. Run the focused tests.

### Task 2: Official OBQA and ARC schemas

**Files:**
- Modify: `src/self_steering/datasets/adapters.py`
- Test: `tests/test_datasets.py`

1. Add official-shape OBQA and numeric-label ARC regression tests.
2. Accept `question_stem` and normalize labels positionally.
3. Map the source gold label to its canonical letter.
4. Run dataset and type tests.

### Task 3: Complete specificity reporting

**Files:**
- Modify: `src/self_steering/evaluation/metrics.py`
- Modify: `src/self_steering/pipeline.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_pipeline.py`

1. Add failing tests for sparse and complete configured matrices.
2. Return values, counts, and missing cells for all configured capabilities.
3. Compute diagonal dominance only for a complete non-empty matrix.
4. Update report serialization and run focused tests.

### Task 4: Minimal provenance and strict config validation

**Files:**
- Modify: `src/self_steering/config.py`
- Modify: `src/self_steering/datasets/delean_labeler.py`
- Modify: `src/self_steering/pipeline.py`
- Modify: `src/self_steering/utils/manifest.py`
- Test: `tests/test_config.py`
- Test: `tests/test_delean.py`
- Test: `tests/test_pipeline.py`

1. Add failing validation, annotation-version, and manifest provenance tests.
2. Enforce the existing MVP config constraints.
3. Include prompt/schema hashes in annotation identity.
4. Hash stage inputs and record effective seed/revisions in manifests and run identities.
5. Invoke deterministic seeding before model-dependent work.
6. Run focused tests.

### Task 5: Structured item failures and hook conversion cache

**Files:**
- Modify: `src/self_steering/hooks/intervention.py`
- Modify: `src/self_steering/pipeline.py`
- Test: `tests/test_hooks.py`
- Test: `tests/test_capture_pipeline.py`
- Test: `tests/test_pipeline.py`

1. Add failing tests for one conversion per device/dtype and per-item capture/generation errors.
2. Cache converted/scaled vectors inside the hook context.
3. Append structured error records with CUDA OOM classification and continue independent work.
4. Exclude failed capture shards from the index.
5. Run focused tests.

### Task 6: Real cached-decode integration and hermetic CLI tests

**Files:**
- Modify: `tests/test_cli.py`
- Add: `tests/test_qwen_integration.py`

1. Make CLI subprocesses set `PYTHONPATH` to this checkout's `src`.
2. Add a no-download tiny Qwen2 cached-generation test using the real Transformers model.
3. Skip with an explicit reason only when Qwen2 cannot be imported/constructed in the environment.
4. Run focused tests.

### Task 7: Full verification

1. Run `pytest -q` with `PYTHONPATH=src`.
2. Run package compilation.
3. Inspect the diff for scope creep and unrelated files.
4. Commit only the review-fix files on the isolated branch.
