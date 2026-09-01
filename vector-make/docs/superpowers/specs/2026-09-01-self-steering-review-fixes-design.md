# Self-Steering MVP Review Fixes Design

## Scope

This change fixes only the defects confirmed in the post-merge code review. It keeps the numbered pipeline, configuration shape, output layout, and native Transformers implementation intact.

In scope:

- make generation resume, deduplication, and metric pairing use one stable item identity;
- normalize official OBQA and ARC schemas;
- report specificity coverage and suppress diagonal dominance for incomplete matrices;
- record the minimum provenance needed to distinguish reusable artifacts;
- validate the existing MVP configuration contract;
- emit structured capture/generation failures;
- avoid repeated steering-vector device/dtype conversion;
- add focused regression and tiny-Qwen integration tests.

Out of scope:

- a new run database or orchestration framework;
- changing prompt content, capability definitions, vector mathematics, or benchmark scoring;
- automatic OOM recovery, batch-size tuning, distributed execution, or interactive auditing;
- redesigning all artifact directories around immutable run snapshots.

## Identity and resume behavior

Generation rows will carry an `item_identity` derived from dataset, item id, prompt, gold answer, answer type, choices, and demand memberships. A generation key is `(run_id, item_identity, steering_capability, alpha)`. Pending rows and persisted rows are created through the same helper so the identity cannot diverge.

When an output file already contains duplicate keys, the latest successful row is canonical. Resume skips that key, and scoring consumes only canonical successful rows. This repairs existing polluted files without rewriting or deleting them.

The steering `run_id` remains a short content hash, but its identity includes the vector file hash, effective generation parameters, enabled datasets, capabilities, prompt/instruction hashes, and the resolved model/tokenizer revisions when available.

## Dataset normalization

The multiple-choice adapter accepts `question_stem` in addition to the existing question fields. For both ARC and OBQA, source labels are mapped positionally to `A`, `B`, and so on. The gold label is resolved against the source labels before being converted to its canonical letter. This supports ARC rows whose labels are `1` through `4` while preserving letter-only prompts and answers downstream.

## Specificity metrics

Specificity reporting receives the configured capability list. It returns:

- a full capability-by-capability matrix whose missing cells are `null`;
- a parallel count matrix;
- a list of missing cells;
- diagonal dominance only when every configured cell has paired baseline and steered observations.

Existing accuracy and demand-slice calculations remain unchanged.

## Provenance and validation

Stage manifests keep their current filenames but gain a deterministic artifact/run id, seed, hashes for existing stage inputs, prompt hashes, rubric hashes where applicable, and resolved model/tokenizer revisions when the loaded objects expose them. The pipeline invokes the existing seed helper at model-dependent stage entry points.

The annotation cache key gains a hash of the rendered annotation prompt and JSON schema. Existing annotation rows without that field are treated as stale and are regenerated.

Configuration validation enforces the current MVP contract: capabilities are exactly `QLl`, `QLq`, `CL`, and `MCr` with no duplicates; alphas are finite and unique and include zero; target layer is non-negative; generation length, annotation worker count, retry attempts, and retry backoff are positive.

## Failure handling

Capture and generation wrap each item operation. On failure they append a structured error record containing the stage identity and item/capability/alpha context. CUDA OOM errors receive an explicit `error_type` and concise mitigation hint. Processing continues to the next independent item. Vector extraction still fails clearly when a capability has no successful shards, rather than fabricating a vector.

## Testing

Focused tests first reproduce every confirmed defect. A tiny randomly initialized `Qwen2ForCausalLM` test exercises cached decoding and the continuous layer hook without downloading weights; it skips only when the installed Transformers/Torch combination cannot provide Qwen2. CLI subprocess tests explicitly expose the repository `src` directory, eliminating dependence on an old editable installation.
