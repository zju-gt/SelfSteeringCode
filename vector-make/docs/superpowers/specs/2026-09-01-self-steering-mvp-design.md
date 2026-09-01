# Self-Steering MVP Design

## 1. Objective

Build a reproducible, configuration-driven experiment pipeline that tests whether four externally defined cognitive capabilities correspond to stable and causally steerable activation directions in `Qwen/Qwen2.5-7B-Instruct`.

The four capabilities are:

- `QLl`: Logical Reasoning
- `QLq`: Quantitative Reasoning
- `CL`: Conceptualisation, Learning and Abstraction
- `MCr`: Identifying Relevant Information

The MVP uses MMLU items for vector extraction. Both MMLU and every enabled steering dataset receive DeLeAn demand annotations: MMLU scores select extraction items, while evaluation-dataset scores define high-demand and low-demand steering slices. Steering evaluation is extensible through dataset adapters, with MATH500 enabled by default and AIME 2024, AIME 2025, AIME 2026, ARC-C, and OpenBookQA available as optional evaluation datasets.

## 2. Scope

### Included

- Canonical dataset representation and an extensible dataset registry.
- Hugging Face dataset adapters for MMLU, MATH500, AIME 2024, AIME 2025, AIME 2026, ARC-C, and OpenBookQA.
- Local JSONL overrides for every dataset.
- Concurrent, resumable DeLeAn annotation for `QLl`, `QLq`, `CL`, and `MCr`.
- Filtering MMLU independently for each capability with `d_k >= 4`.
- Filtering enabled steering datasets independently into high-demand (`d_k >= 4`) and low-demand (`d_k <= 1`) evaluation slices.
- Same-question capability-prompt contrast extraction.
- Activation capture at a configurable Qwen decoder layer, defaulting to zero-based layer index 19.
- Raw, unit, and mean-norm-calibrated steering vectors.
- Continuous generation-time intervention from the `Reasoning:` assistant prefill onward.
- Greedy CoT generation, answer extraction, exact-match evaluation, incremental outputs, and run manifests.
- Unit tests and offline smoke tests that do not require the 7B checkpoint, network access, or an OpenAI key.

### Excluded

- Repeat annotation and manual annotation audit.
- Cross-domain vector extraction or validation.
- Multiple extraction templates and held-out prompt robustness.
- Demand-overlap removal or purity filtering.
- Statistical significance thresholds or automatic hyperparameter tuning.
- Full 18-dimensional extraction, trajectory analysis, vector composition, routing, and the Self-Steering controller.

## 3. Chosen Architecture

Use native Hugging Face Transformers and PyTorch hooks. Do not depend on TransformerLens or NNSight.

The code is divided into focused modules with stable interfaces:

```text
configs and CLI scripts
        |
        v
DatasetRegistry -> CanonicalItem JSONL
        |
        +-> DeLeAn annotation -> long and wide demand JSONL
        |
        +-> high-demand filtering -> extraction sets
        |
        +-> activation capture -> per-item contrast shards
        |
        +-> vector aggregation -> raw/unit/mean-norm safetensors
        |
        +-> steering generation -> generation JSONL
        |
        `-> answer grading -> metrics JSON
```

Native hooks are preferred because they make the exact residual-stream location and token positions explicit and keep Qwen2.5 compatibility under project control.

## 4. Configuration

Three YAML files provide defaults:

- `configs/model.yaml`: checkpoint, revision, dtype, device map, attention implementation, and generation settings.
- `configs/data.yaml`: dataset sources, splits, local overrides, cache directory, and enabled evaluation datasets.
- `configs/experiment.yaml`: target capabilities, target layer, demand threshold, prompt definitions, alpha sweep, seeds, concurrency, retries, and output paths.

Important defaults are:

```yaml
model_name: Qwen/Qwen2.5-7B-Instruct
target_layer: 19
high_demand_threshold: 4
low_demand_threshold: 1
enabled_steering_datasets:
  - math500
alphas:
  - -1.0
  - -0.5
  - 0.0
  - 0.5
  - 1.0
vector_scaling: mean_norm
do_sample: false
```

`target_layer` is a zero-based decoder-block index and resolves by default to `model.model.layers[19]`. It remains configurable.

## 5. Dataset Layer

### 5.1 Canonical representation

Every adapter produces a `CanonicalItem` with:

```text
item_id: globally stable string
dataset: registry name
split: source split
prompt: complete task instance without the gold answer
gold_answer: normalized reference answer
answer_type: choice | math
choices: optional ordered mapping of answer letters to text
metadata: source-specific JSON-compatible fields
```

Dataset adapters validate required fields, reject duplicate item IDs, and normalize answer types before writing canonical JSONL.

### 5.2 Registry

The registry maps stable names to adapters:

```text
mmlu
math500
aime2024
aime2025
aime2026
arc_c
obqa
```

Every adapter accepts either a configured Hugging Face dataset identifier or a local JSONL override. Source-specific schema differences remain inside the adapter.

MMLU is used for annotation and extraction, but never for steering evaluation. Enabled steering datasets are annotated only to select their evaluation slices; their activations never contribute to the vectors. The default steering dataset list contains only MATH500. AIME 2024, AIME 2025, AIME 2026, ARC-C, and OpenBookQA can be enabled by editing configuration or passing CLI overrides.

### 5.3 Answer rules

- ARC-C, OpenBookQA, and MMLU use an uppercase option letter.
- MATH500 uses a final mathematical answer extracted from CoT.
- AIME datasets use an integer in `[0, 999]`; leading zeros are ignored for correctness.
- Math generation requests a final `\boxed{...}` answer. The extractor also recognizes a `Final Answer:` fallback.
- Evaluation is deterministic exact match after dataset-specific normalization. Symbolic-equivalence grading is outside this MVP.

## 6. Prompt Serialization

There are exactly five reasoning conditions: one generic instruction and one instruction for each of the four capabilities. Each capability vector uses a single same-question contrast against the generic condition.

For Qwen2.5-Instruct, prompts use the official chat template:

```text
user:
[reasoning instruction]

Question:
[canonical task prompt]

[dataset-specific final-answer format]

assistant:
Reasoning:
```

The assistant message is a prefill and is serialized with `continue_final_message=True`. No additional generation prompt is appended. Consequently, the final input position is the final token of `Reasoning:`.

The generic and capability-specific conditions differ only in the reasoning instruction. Question text and answer-format instructions remain identical within a contrast pair.

## 7. DeLeAn Annotation

The four official rubric files are sourced from the cloned `ADeLe-AI-Evaluation/ADeLe-AIEvaluation` repository at commit `b896a55d916f1701cbb5e211a20267cd640b6479`.

Annotation runs over MMLU and all enabled steering datasets. It uses one request for one item and one dimension. The request includes only the task instruction, question, choices, and solving context. It never includes the gold answer, target-model response, correctness, or confidence.

The structured response is:

```json
{
  "score": 4,
  "brief_justification": "Requires several interacting deductive steps."
}
```

Requests run through a bounded thread pool. Retriable failures use exponential backoff up to a configured attempt count. Worker threads return records to the main thread, which is the only JSONL writer.

The durable long-form key includes:

```text
item_id
dimension
task_sha256
rubric_sha256
annotator_model_requested
```

An old row is reusable only when the full key matches and `status == "ok"`. Errors are written with their status and can be retried. A completed long-form file is joined with the canonical source to produce wide-form items with four demand scores.

Repeat annotation and manual audit are not implemented.

## 8. Activation Capture and Vector Extraction

For capability `k` and MMLU item `i`, capture the target block's output residual at the final prefill token:

```text
delta_i,k = h_19(q_i, p_k) - h_19(q_i, p_generic)
```

MMLU extraction items are selected independently per capability using `d_i,k >= 4`. No domain grouping or cross-capability purity constraint is applied.

The block hook accepts the tuple-style output used by Qwen decoder layers, reads its hidden-state tensor, and leaves the forward pass unchanged during capture. Hook registration and removal are controlled by a context manager.

Per-item deltas are written as safetensors shards with JSONL metadata. Sharding bounds memory use and permits extraction resume without repeating complete samples.

For each capability:

```text
raw_k  = mean_i(delta_i,k)
unit_k = raw_k / ||raw_k||
scale  = mean_j(||raw_j||), for the four capabilities at layer 19
steer_k = unit_k * scale
```

`unit_k` is used for cosine analysis. `steer_k` is the default intervention vector. Zero-norm vectors are rejected with an explicit error. Raw, unit, and mean-norm vectors are saved together with their norms and source metadata.

## 9. Steering

Steering evaluation always uses the generic reasoning prompt. Prompt wording is held constant while capability, alpha, and dataset vary.

At the target layer:

```text
h' = h + alpha * steer_k
```

During the prefill pass, the hook changes only the final `Reasoning:` token. This changes the logits for the first generated token while leaving earlier prompt-token activations untouched. With KV caching enabled, each later decoder call contains the current generated position, and the hook adds the same vector to that position. Intervention therefore persists through every generated token.

The default alpha sweep is `[-1, -0.5, 0, 0.5, 1]`. Alpha zero is generated once per item and can be reused as the common baseline across capability rows.

For each enabled steering dataset and demand capability `j`, evaluation items are selected directly with `d_j >= 4` for the high-demand slice and `d_j <= 1` for the low-demand control. Dimension overlap is allowed and no purity filter is applied. Every steering vector `i` is evaluated on every high-demand slice `j`; the matching low-demand slice is evaluated as the selectivity control.

Generation uses greedy decoding with a configurable maximum number of new tokens. Each `item x steering capability x alpha` result is appended to JSONL and contains its demand-slice memberships, raw output, extracted answer, correctness, generation settings, vector identity, and run identity.

## 10. Metrics

For every enabled dataset and capability, report:

- Item count and baseline accuracy.
- Accuracy at every alpha.
- Accuracy change relative to alpha zero.
- Positive and negative steering results on every high-demand capability slice.
- Low-demand control effects for each capability.
- A causal specificity matrix whose cell `(i, j)` is the performance change from steering vector `i` on the `d_j >= 4` evaluation slice.

The MVP reports descriptive values without imposing numerical pass/fail thresholds or significance tests.

## 11. Persistence and Reproducibility

Primary formats are:

- JSONL for canonical items, annotations, filtered item metadata, and generations.
- Safetensors for activation differences and vectors.
- JSON for manifests and aggregate metrics.
- CSV generated from metrics for convenient plotting and inspection.

Every run manifest records:

- Resolved configuration.
- Model name and requested revision.
- Dataset source identifiers, splits, and available fingerprints.
- Rubric hashes.
- Prompt hashes.
- Random seed.
- Software versions.
- Start time and run identifier.

Output keys include the configuration-dependent identities needed to avoid reusing stale artifacts.

## 12. Failure Handling

- Invalid configurations, unknown datasets, out-of-range layers, duplicate IDs, and malformed gold answers fail during validation.
- Dataset download and schema failures identify the dataset and source field that failed.
- Annotation transport and rate-limit failures are retried; permanent schema failures are recorded without retry loops.
- Hooks are always removed in `finally` paths through context managers.
- CUDA OOM errors identify the failed item and recommend explicit configuration changes; the software never silently changes batch size, dtype, or token limits.
- Interrupted JSONL stages resume from valid completed keys.
- Safetensors shards are written to temporary files and renamed only after a complete write.

## 13. Testing Strategy

Development follows test-driven development: each production behavior is preceded by a focused failing test.

The default test suite is offline and covers:

- Registry lookup, canonicalization, source-schema normalization, duplicate detection, and local JSONL overrides.
- Qwen chat serialization and the `Reasoning:` assistant-prefill boundary.
- Choice, boxed-answer, `Final Answer:`, AIME integer, and leading-zero extraction.
- Annotation keying, rubric/task hashes, retry behavior, resume behavior, and single-writer output.
- High-demand filtering at exactly the threshold boundary.
- Capture and intervention behavior using small Torch modules or a tiny locally instantiated Qwen configuration.
- Prefill-last-token-only intervention and one-position cached decoding intervention.
- Raw, unit, and mean-norm vector calculations and zero-norm errors.
- Incremental generation keys and accuracy aggregation.
- CLI smoke flows using local fixtures and fake injectable clients/models.

Real OpenAI requests, full Hugging Face downloads, and Qwen2.5-7B GPU runs are explicit integration commands and are not part of the default unit test suite.

## 14. Command-Line Workflow

The planned workflow is:

```text
scripts/00_prepare_data.py
scripts/01_score_demands.py
scripts/02_prepare_items.py
scripts/03_capture_contrasts.py
scripts/04_extract_vectors.py
scripts/05_analyze_similarity.py
scripts/06_run_steering.py
scripts/07_score_generations.py
```

Every script loads the same resolved YAML configuration, supports sample limits for smoke runs, writes a manifest, and performs one pipeline stage. The Python modules remain usable independently of the scripts.

## 15. Completion Criteria

Implementation is complete when:

- The offline test suite passes.
- All CLI entry points provide help without importing API-only or GPU-only dependencies prematurely.
- A fixture-based smoke pipeline prepares data, filters demands, aggregates vectors, and scores generations.
- MMLU extraction and MATH500 steering configurations are present as runnable defaults.
- AIME 2024, AIME 2025, AIME 2026, ARC-C, and OpenBookQA can be enabled through configuration without editing steering code.
- The real-run commands and required environment variables are documented.
