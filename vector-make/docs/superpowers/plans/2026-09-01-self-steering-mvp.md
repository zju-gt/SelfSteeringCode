# Self-Steering MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, configuration-driven pipeline that labels cognitive demand, extracts four layer-19 capability vectors from MMLU on Qwen2.5-7B-Instruct, and evaluates continuous generation-time steering on MATH500 plus optional AIME/ARC-C/OpenBookQA datasets.

**Architecture:** A Python package under `src/self_steering` exposes focused configuration, dataset, annotation, prompting, hook, vector, generation, and evaluation APIs. Thin numbered scripts compose those APIs into resumable JSONL/safetensors stages. Native Transformers hooks capture and edit the Qwen decoder-block residual stream without third-party interpretability frameworks.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers/Datasets, OpenAI Responses API, PyYAML, safetensors, pytest, standard-library JSONL/concurrency utilities.

---

## File Map

```text
vector-make/
├── pyproject.toml
├── README.md
├── configs/{model,data,experiment}.yaml
├── rubrics/{QLl,QLq,CL,MCr}.txt
├── src/self_steering/
│   ├── config.py
│   ├── datasets/{types,registry,adapters,delean_labeler,scoring,filtering}.py
│   ├── prompts/{templates,serialization}.py
│   ├── models/{loader,generation}.py
│   ├── hooks/{capture,intervention}.py
│   ├── vectors/{extract,similarity,storage}.py
│   ├── evaluation/{answers,metrics}.py
│   └── utils/{io,manifest,seed}.py
├── scripts/00_prepare_data.py
├── scripts/01_score_demands.py
├── scripts/02_prepare_items.py
├── scripts/03_capture_contrasts.py
├── scripts/04_extract_vectors.py
├── scripts/05_analyze_similarity.py
├── scripts/06_run_steering.py
├── scripts/07_score_generations.py
└── tests/
    ├── fixtures/*.jsonl
    ├── test_config.py
    ├── test_datasets.py
    ├── test_prompts.py
    ├── test_answers.py
    ├── test_delean.py
    ├── test_filtering.py
    ├── test_hooks.py
    ├── test_vectors.py
    ├── test_metrics.py
    └── test_smoke_pipeline.py
```

### Task 1: Package scaffold and resolved configuration

**Files:**
- Create: `vector-make/pyproject.toml`
- Create: `vector-make/src/self_steering/__init__.py`
- Create: `vector-make/src/self_steering/config.py`
- Create: `vector-make/configs/model.yaml`
- Create: `vector-make/configs/data.yaml`
- Create: `vector-make/configs/experiment.yaml`
- Test: `vector-make/tests/test_config.py`

- [ ] **Step 1: Add package metadata and YAML defaults, then install development dependencies**

`pyproject.toml` must define the `src` package layout, runtime dependencies, a `dev` extra containing pytest, and pytest `pythonpath = ["src"]`. This is build configuration rather than production behavior.

Run: `cd vector-make; python -m pip install -e ".[dev]"`

Expected: installation succeeds and `python -m pytest --version` exits with code 0.

- [ ] **Step 2: Write failing configuration tests**

```python
from pathlib import Path

import pytest

from self_steering.config import ConfigError, load_config


def test_load_config_deep_merges_three_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "model.yaml").write_text("model:\n  name: qwen\n  dtype: bfloat16\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("data:\n  enabled_steering_datasets: [math500]\n", encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text("experiment:\n  target_layer: 19\n", encoding="utf-8")
    config = load_config([tmp_path / "model.yaml", tmp_path / "data.yaml", tmp_path / "experiment.yaml"])
    assert config["model"]["name"] == "qwen"
    assert config["data"]["enabled_steering_datasets"] == ["math500"]
    assert config["experiment"]["target_layer"] == 19


def test_load_config_rejects_out_of_range_layer(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  num_hidden_layers: 28\nexperiment:\n  target_layer: 28\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="target_layer"):
        load_config([path])
```

- [ ] **Step 3: Run tests and verify the import/behavior fails**

Run: `cd vector-make; python -m pytest tests/test_config.py -q`

Expected: FAIL because `self_steering.config` does not exist.

- [ ] **Step 4: Add deep merge, dot-path overrides, and validation**

The public configuration API must be:

```python
class ConfigError(ValueError):
    pass


def load_config(paths: list[Path], overrides: list[str] | None = None) -> dict:
    """Merge YAML files left-to-right, apply key=value dot-path overrides, and validate."""


def validate_config(config: dict) -> None:
    """Validate dataset names, layer bounds, thresholds, alphas, and vector scaling."""
```

- [ ] **Step 5: Run the focused tests**

Run: `cd vector-make; python -m pytest tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -- vector-make/pyproject.toml vector-make/configs vector-make/src/self_steering/__init__.py vector-make/src/self_steering/config.py vector-make/tests/test_config.py
git commit -m "feat: scaffold self-steering configuration"
```

### Task 2: Durable IO, manifests, and canonical dataset types

**Files:**
- Create: `vector-make/src/self_steering/utils/__init__.py`
- Create: `vector-make/src/self_steering/utils/io.py`
- Create: `vector-make/src/self_steering/utils/manifest.py`
- Create: `vector-make/src/self_steering/utils/seed.py`
- Create: `vector-make/src/self_steering/datasets/__init__.py`
- Create: `vector-make/src/self_steering/datasets/types.py`
- Test: `vector-make/tests/test_io.py`
- Test: `vector-make/tests/test_types.py`

- [ ] **Step 1: Write failing JSONL and canonical-item tests**

```python
from pathlib import Path

import pytest

from self_steering.datasets.types import CanonicalItem
from self_steering.utils.io import append_jsonl, read_jsonl


def test_jsonl_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"text": "数量推理"})
    assert list(read_jsonl(path)) == [{"text": "数量推理"}]


def test_canonical_item_rejects_invalid_aime_answer() -> None:
    with pytest.raises(ValueError, match="AIME"):
        CanonicalItem(
            item_id="aime2026_1",
            dataset="aime2026",
            split="test",
            prompt="problem",
            gold_answer="1000",
            answer_type="math",
            metadata={"competition": "AIME"},
        ).validate()
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_io.py tests/test_types.py -q`

Expected: FAIL because IO and canonical types do not exist.

- [ ] **Step 3: Implement atomic JSON/safetensors helpers, hashing, seeding, manifests, and `CanonicalItem`**

Required type contract:

```python
@dataclass(frozen=True)
class CanonicalItem:
    item_id: str
    dataset: str
    split: str
    prompt: str
    gold_answer: str
    answer_type: Literal["choice", "math"]
    choices: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject empty IDs/prompts, malformed choices, and invalid AIME answers."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
```

Atomic tensor writes must create a sibling temporary file and replace the destination only after `safetensors.torch.save_file` succeeds.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_io.py tests/test_types.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/utils vector-make/src/self_steering/datasets/__init__.py vector-make/src/self_steering/datasets/types.py vector-make/tests/test_io.py vector-make/tests/test_types.py
git commit -m "feat: add durable artifacts and canonical items"
```

### Task 3: Extensible dataset registry and adapters

**Files:**
- Create: `vector-make/src/self_steering/datasets/registry.py`
- Create: `vector-make/src/self_steering/datasets/adapters.py`
- Create: `vector-make/tests/fixtures/math_items.jsonl`
- Test: `vector-make/tests/test_datasets.py`

- [ ] **Step 1: Write failing registry and schema-normalization tests**

```python
import pytest

from self_steering.datasets.adapters import adapt_aime, adapt_math500, adapt_multiple_choice
from self_steering.datasets.registry import DatasetRegistry


def test_registry_contains_all_supported_datasets() -> None:
    registry = DatasetRegistry.default()
    assert set(registry.names()) == {"mmlu", "math500", "aime2024", "aime2025", "aime2026", "arc_c", "obqa"}


def test_aime2026_schema_is_canonicalized() -> None:
    item = adapt_aime(
        {"source_problem_id": "aime_2026__000", "problem": "Compute.", "ground_truth": 7},
        dataset="aime2026",
        split="test",
        index=0,
    )
    assert item.item_id == "aime_2026__000"
    assert item.gold_answer == "7"


def test_multiple_choice_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="choices"):
        adapt_multiple_choice(
            {"question": "Q", "choices": {"label": ["A"], "text": ["x", "y"]}, "answerKey": "A"},
            dataset="obqa",
            split="test",
            index=0,
        )
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_datasets.py -q`

Expected: FAIL because the registry and adapters do not exist.

- [ ] **Step 3: Implement lazy Hugging Face loading, local JSONL overrides, adapters, duplicate checks, and prepared JSONL output**

Required registry API:

```python
Loader = Callable[[dict[str, Any]], Iterable[CanonicalItem]]


class DatasetRegistry:
    def register(self, name: str, loader: Loader) -> None:
        """Register one stable dataset name and reject duplicates."""

    def load(self, name: str, config: dict[str, Any]) -> list[CanonicalItem]:
        """Load, validate, and reject duplicate item IDs."""

    def names(self) -> list[str]:
        """Return sorted registry names."""

    @classmethod
    def default(cls) -> "DatasetRegistry":
        """Build the seven-adapter default registry."""
```

Hugging Face defaults are configured rather than hard-coded in steering logic: `cais/mmlu`, `HuggingFaceH4/MATH-500`, `OpenRLHF/aime-2024`, `test-time-compute/aime_2025`, `96kevinli29/aime2026-en`, `allenai/ai2_arc`, and `allenai/openbookqa`.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_datasets.py -q`

Expected: PASS without importing `datasets` until a Hugging Face source is actually loaded.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/datasets/registry.py vector-make/src/self_steering/datasets/adapters.py vector-make/tests/fixtures vector-make/tests/test_datasets.py vector-make/configs/data.yaml
git commit -m "feat: add extensible benchmark adapters"
```

### Task 4: Prompt definitions, Qwen serialization, and answer extraction

**Files:**
- Create: `vector-make/src/self_steering/prompts/__init__.py`
- Create: `vector-make/src/self_steering/prompts/templates.py`
- Create: `vector-make/src/self_steering/prompts/serialization.py`
- Create: `vector-make/src/self_steering/evaluation/__init__.py`
- Create: `vector-make/src/self_steering/evaluation/answers.py`
- Test: `vector-make/tests/test_prompts.py`
- Test: `vector-make/tests/test_answers.py`

- [ ] **Step 1: Write failing prompt and grading tests**

```python
from self_steering.evaluation.answers import extract_answer, is_correct
from self_steering.prompts.serialization import build_chat_messages, serialize_reasoning_prefill


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [10, 20, 30]


def test_reasoning_prefill_uses_continue_final_message() -> None:
    tokenizer = RecordingTokenizer()
    ids = serialize_reasoning_prefill(tokenizer, "instruction", "question", "Return boxed math.")
    assert ids == [10, 20, 30]
    assert tokenizer.messages[-1] == {"role": "assistant", "content": "Reasoning:"}
    assert tokenizer.kwargs["continue_final_message"] is True
    assert tokenizer.kwargs["add_generation_prompt"] is False


def test_extracts_last_boxed_answer() -> None:
    assert extract_answer("work \\boxed{12} more \\boxed{34}", "math") == "34"


def test_aime_leading_zeros_compare_as_integers() -> None:
    assert is_correct("007", "7", dataset="aime2026", answer_type="math")
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_prompts.py tests/test_answers.py -q`

Expected: FAIL because prompting and answer extraction do not exist.

- [ ] **Step 3: Implement exactly five prompt conditions, assistant-prefill serialization, dataset answer instructions, robust final-answer parsing, and exact-match normalization**

Required prompt API:

```python
CAPABILITY_PROMPTS: dict[str, str]
GENERIC_PROMPT: str


def build_chat_messages(reasoning_instruction: str, question: str, answer_instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"{reasoning_instruction}\n\nQuestion:\n{question}\n\n{answer_instruction}"},
        {"role": "assistant", "content": "Reasoning:"},
    ]
```

The math parser checks the last boxed expression first, then the last `Final Answer:` occurrence, then the final standalone numeric token. Choice parsing accepts only `A` through `Z` and normalizes case.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_prompts.py tests/test_answers.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/prompts vector-make/src/self_steering/evaluation vector-make/tests/test_prompts.py vector-make/tests/test_answers.py
git commit -m "feat: add capability prompts and answer parsing"
```

### Task 5: Concurrent resumable DeLeAn annotation

**Files:**
- Copy: `vector-make/external/ADeLe-AIEvaluation/rubrics/{QLl,QLq,CL,MCr}.txt`
- Create: `vector-make/src/self_steering/datasets/delean_labeler.py`
- Create: `vector-make/src/self_steering/datasets/scoring.py`
- Test: `vector-make/tests/test_delean.py`

- [ ] **Step 1: Write failing annotation, resume-key, and retry tests with an injected fake client**

```python
from types import SimpleNamespace

from self_steering.datasets.delean_labeler import AnnotationRequest, label_one_dimension
from self_steering.datasets.scoring import completed_annotation_keys


class FakeResponses:
    def create(self, **kwargs):
        return SimpleNamespace(output_text='{"score":4,"brief_justification":"multi-step"}', id="resp_1", model="judge")


def test_label_one_dimension_returns_hashed_audit_record(tmp_path) -> None:
    rubric = tmp_path / "QLl.txt"
    rubric.write_text("rubric", encoding="utf-8")
    client = SimpleNamespace(responses=FakeResponses())
    row = label_one_dimension(
        client,
        AnnotationRequest(item_id="i1", dataset="mmlu", split="test", prompt="Q", dimension="QLl"),
        rubric,
        model="judge",
    )
    assert row["level"] == 4
    assert row["status"] == "ok"
    assert len(row["task_sha256"]) == 64
    assert len(row["rubric_sha256"]) == 64


def test_completed_keys_change_when_task_hash_changes() -> None:
    rows = [{"item_id": "i1", "demand": "QLl", "task_sha256": "old", "rubric_sha256": "r", "annotator_model_requested": "m", "status": "ok"}]
    assert ("i1", "QLl", "new", "r", "m") not in completed_annotation_keys(rows)
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_delean.py -q`

Expected: FAIL because annotation modules do not exist.

- [ ] **Step 3: Copy the four official rubric files and implement structured Responses API requests, full resume keys, bounded concurrency, exponential retry, one-writer JSONL, and long-to-wide joining**

Required orchestration API:

```python
def score_items(
    items: Iterable[CanonicalItem],
    dimensions: Sequence[str],
    output_path: Path,
    label_fn: Callable[[CanonicalItem, str], dict[str, Any]],
    max_workers: int,
) -> None:
    """Submit missing keys concurrently and append completed/error rows on the caller thread."""


def annotations_to_wide(items: Iterable[CanonicalItem], rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require one successful row per item and target dimension, then attach demand_scores."""
```

- [ ] **Step 4: Verify GREEN and rubric byte identity**

Run: `cd vector-make; python -m pytest tests/test_delean.py -q`

Run: `cd vector-make; python -c "from pathlib import Path; import hashlib; pairs=[('QLl','cf39f364760795dcdce1d6d21c7a58ddfb4ca54313d0ce4783497f29a7932b9b'),('QLq','e10bdaebb65109c4862743bfb25a7c10532859932af36a0f2266cf64f95bd9a1'),('CL','67576c056cd035c8f3f143e27cb3a6c1ca25559bd20c8c1c487079114ab9888d'),('MCr','f7964b27a61ed3def15eb71ed5f529313479adb6188e2849ccfe3e3ff127669a')]; assert all(hashlib.sha256(Path(f'rubrics/{n}.txt').read_bytes()).hexdigest()==h for n,h in pairs)"`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/rubrics vector-make/src/self_steering/datasets/delean_labeler.py vector-make/src/self_steering/datasets/scoring.py vector-make/tests/test_delean.py
git commit -m "feat: add concurrent DeLeAn scoring"
```

### Task 6: Demand slicing and vector mathematics

**Files:**
- Create: `vector-make/src/self_steering/datasets/filtering.py`
- Create: `vector-make/src/self_steering/vectors/__init__.py`
- Create: `vector-make/src/self_steering/vectors/extract.py`
- Create: `vector-make/src/self_steering/vectors/similarity.py`
- Create: `vector-make/src/self_steering/vectors/storage.py`
- Test: `vector-make/tests/test_filtering.py`
- Test: `vector-make/tests/test_vectors.py`

- [ ] **Step 1: Write failing boundary and vector-scaling tests**

```python
import pytest
import torch

from self_steering.datasets.filtering import demand_slice
from self_steering.vectors.extract import aggregate_capability_vectors


def test_demand_slice_includes_threshold_boundaries() -> None:
    rows = [
        {"item_id": "low", "demand_scores": {"QLl": 1}},
        {"item_id": "middle", "demand_scores": {"QLl": 2}},
        {"item_id": "high", "demand_scores": {"QLl": 4}},
    ]
    assert [x["item_id"] for x in demand_slice(rows, "QLl", "high", 4, 1)] == ["high"]
    assert [x["item_id"] for x in demand_slice(rows, "QLl", "low", 4, 1)] == ["low"]


def test_mean_norm_vectors_share_the_mean_raw_norm() -> None:
    deltas = {"QLl": torch.tensor([[3.0, 0.0]]), "QLq": torch.tensor([[0.0, 1.0]])}
    vectors = aggregate_capability_vectors(deltas)
    assert torch.allclose(vectors["QLl"]["steering"].norm(), torch.tensor(2.0))
    assert torch.allclose(vectors["QLq"]["steering"].norm(), torch.tensor(2.0))


def test_zero_norm_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        aggregate_capability_vectors({"QLl": torch.zeros(2, 3)})
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_filtering.py tests/test_vectors.py -q`

Expected: FAIL because filtering and vector modules do not exist.

- [ ] **Step 3: Implement high/low slicing, mean delta aggregation, raw/unit/mean-norm outputs, cosine matrices, and safetensors metadata storage**

Required aggregation contract:

```python
def aggregate_capability_vectors(
    deltas: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Return raw, unit, and steering tensors for every capability."""
```

The function first computes all raw means and validates their norms, then computes one mean raw norm across the available capabilities, and finally scales each unit vector to that shared norm.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_filtering.py tests/test_vectors.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/datasets/filtering.py vector-make/src/self_steering/vectors vector-make/tests/test_filtering.py vector-make/tests/test_vectors.py
git commit -m "feat: add demand slicing and vector scaling"
```

### Task 7: Native residual capture and continuous intervention hooks

**Files:**
- Create: `vector-make/src/self_steering/hooks/__init__.py`
- Create: `vector-make/src/self_steering/hooks/capture.py`
- Create: `vector-make/src/self_steering/hooks/intervention.py`
- Test: `vector-make/tests/test_hooks.py`

- [ ] **Step 1: Write failing hook tests on a tiny tuple-returning Torch block**

```python
import torch
from torch import nn

from self_steering.hooks.capture import capture_last_token
from self_steering.hooks.intervention import add_steering_vector


class TupleBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1, "cache")


def test_capture_reads_last_token_without_changing_output() -> None:
    block = TupleBlock()
    hidden = torch.zeros(1, 3, 2)
    with capture_last_token(block) as captured:
        output = block(hidden)
    assert torch.equal(captured.value, torch.ones(2))
    assert torch.equal(output[0], torch.ones_like(hidden))


def test_prefill_intervention_changes_only_last_position() -> None:
    block = TupleBlock()
    hidden = torch.zeros(1, 3, 2)
    with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=1.0):
        output = block(hidden)[0]
    assert torch.equal(output[0, 0], torch.ones(2))
    assert torch.equal(output[0, -1], torch.tensor([3.0, 4.0]))


def test_cached_decode_intervention_changes_single_current_position() -> None:
    block = TupleBlock()
    with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=-1.0):
        output = block(torch.zeros(1, 1, 2))[0]
    assert torch.equal(output[0, 0], torch.tensor([-1.0, -2.0]))
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_hooks.py -q`

Expected: FAIL because hook modules do not exist.

- [ ] **Step 3: Implement exception-safe context managers that preserve tensor/tuple output types, capture detached CPU vectors, and modify only `hidden[:, -1, :]`**

Required APIs:

```python
@contextmanager
def capture_last_token(layer: nn.Module) -> Iterator[CaptureBuffer]:
    """Capture the layer-output residual at the final sequence position."""


@contextmanager
def add_steering_vector(layer: nn.Module, vector: torch.Tensor, alpha: float) -> Iterator[None]:
    """Add alpha*vector to the final position on prefill and cached decode calls."""
```

Both context managers must remove their hook in `finally`, including when the wrapped model raises.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_hooks.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/hooks vector-make/tests/test_hooks.py
git commit -m "feat: add residual capture and steering hooks"
```

### Task 8: Model loading, contrast capture, and steered generation

**Files:**
- Create: `vector-make/src/self_steering/models/__init__.py`
- Create: `vector-make/src/self_steering/models/loader.py`
- Create: `vector-make/src/self_steering/models/generation.py`
- Create: `vector-make/src/self_steering/vectors/capture.py`
- Test: `vector-make/tests/test_generation.py`
- Test: `vector-make/tests/test_capture_pipeline.py`

- [ ] **Step 1: Write failing tests with injected fake tokenizer/model/layer dependencies**

```python
import torch

from self_steering.models.loader import resolve_decoder_layer
from self_steering.vectors.capture import contrast_delta


class Nested:
    pass


def test_resolve_decoder_layer_uses_zero_based_index() -> None:
    model = Nested()
    model.model = Nested()
    model.model.layers = [object(), object(), object()]
    assert resolve_decoder_layer(model, 1) is model.model.layers[1]


def test_contrast_delta_subtracts_generic_from_capability() -> None:
    generic = torch.tensor([1.0, 2.0])
    capability = torch.tensor([4.0, 8.0])
    assert torch.equal(contrast_delta(capability, generic), torch.tensor([3.0, 6.0]))
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_generation.py tests/test_capture_pipeline.py -q`

Expected: FAIL because model and capture pipeline modules do not exist.

- [ ] **Step 3: Implement lazy Qwen loading, tokenizer prefill encoding, layer resolution, no-grad capture passes, greedy generation, alpha-zero baseline reuse, and incremental generation keys**

Required model API:

```python
def load_model_and_tokenizer(config: dict[str, Any]):
    """Load AutoTokenizer and AutoModelForCausalLM using configured revision/dtype/device map."""


def resolve_decoder_layer(model: Any, index: int) -> nn.Module:
    """Resolve model.model.layers[index] and validate bounds."""


def generate_with_optional_steering(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    layer: nn.Module,
    vector: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
) -> str:
    """Run deterministic generation with use_cache=True and decode only new tokens."""
```

Capture runs generic and capability prompts independently, moves the two captured states to CPU float32, and subtracts generic from capability.

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_generation.py tests/test_capture_pipeline.py -q`

Expected: PASS without downloading a model.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/models vector-make/src/self_steering/vectors/capture.py vector-make/tests/test_generation.py vector-make/tests/test_capture_pipeline.py
git commit -m "feat: add Qwen capture and steered generation"
```

### Task 9: Metrics, specificity matrix, and resumable stage services

**Files:**
- Create: `vector-make/src/self_steering/evaluation/metrics.py`
- Create: `vector-make/src/self_steering/pipeline.py`
- Test: `vector-make/tests/test_metrics.py`
- Test: `vector-make/tests/test_pipeline.py`

- [ ] **Step 1: Write failing paired-effect and specificity tests**

```python
from self_steering.evaluation.metrics import accuracy_by_alpha, specificity_matrix


def test_accuracy_by_alpha_reports_change_from_zero() -> None:
    rows = [
        {"alpha": 0.0, "correct": False},
        {"alpha": 1.0, "correct": True},
    ]
    result = accuracy_by_alpha(rows)
    assert result[0.0] == {"count": 1, "accuracy": 0.0, "delta": 0.0}
    assert result[1.0] == {"count": 1, "accuracy": 1.0, "delta": 1.0}


def test_specificity_matrix_indexes_steering_and_demand_capabilities() -> None:
    rows = [
        {"steering_capability": "QLl", "demand_capability": "QLq", "alpha": 0.0, "correct": False},
        {"steering_capability": "QLl", "demand_capability": "QLq", "alpha": 1.0, "correct": True},
    ]
    assert specificity_matrix(rows, alpha=1.0)["QLl"]["QLq"] == 1.0
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_metrics.py tests/test_pipeline.py -q`

Expected: FAIL because metrics and stage services do not exist.

- [ ] **Step 3: Implement accuracy aggregation, high/low effects, specificity matrices, completed generation keys, and one service function per numbered stage**

Stage services must accept resolved config plus injectable registry/client/model dependencies, making them unit-testable without network or GPU:

```python
def prepare_data(config: dict, registry: DatasetRegistry) -> dict[str, Path]:
    """Write canonical JSONL for MMLU and enabled steering datasets."""


def score_demands(config: dict, label_fn: Callable) -> dict[str, Path]:
    """Annotate all prepared datasets and write long/wide outputs."""


def prepare_items(config: dict) -> dict[str, Path]:
    """Write MMLU extraction sets and external high/low evaluation memberships."""


def capture_contrasts(config: dict, model: Any, tokenizer: Any) -> list[Path]:
    """Capture resumable MMLU contrast shards."""


def extract_vectors(config: dict) -> Path:
    """Aggregate contrast shards and persist all vector forms."""


def analyze_similarity(config: dict) -> dict[str, Path]:
    """Write capability cosine matrices and coherence summaries."""


def run_steering(config: dict, model: Any, tokenizer: Any) -> Path:
    """Generate resumable external-dataset interventions."""


def score_generations(config: dict) -> dict[str, Path]:
    """Write JSON/CSV descriptive metrics and specificity matrices."""
```

- [ ] **Step 4: Verify GREEN**

Run: `cd vector-make; python -m pytest tests/test_metrics.py tests/test_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- vector-make/src/self_steering/evaluation/metrics.py vector-make/src/self_steering/pipeline.py vector-make/tests/test_metrics.py vector-make/tests/test_pipeline.py
git commit -m "feat: add experiment stages and causal metrics"
```

### Task 10: CLI scripts, smoke pipeline, and operator documentation

**Files:**
- Create: `vector-make/scripts/_common.py`
- Create: `vector-make/scripts/00_prepare_data.py`
- Create: `vector-make/scripts/01_score_demands.py`
- Create: `vector-make/scripts/02_prepare_items.py`
- Create: `vector-make/scripts/03_capture_contrasts.py`
- Create: `vector-make/scripts/04_extract_vectors.py`
- Create: `vector-make/scripts/05_analyze_similarity.py`
- Create: `vector-make/scripts/06_run_steering.py`
- Create: `vector-make/scripts/07_score_generations.py`
- Create: `vector-make/tests/test_cli.py`
- Create: `vector-make/tests/test_smoke_pipeline.py`
- Create: `vector-make/README.md`

- [ ] **Step 1: Write failing CLI-help and fixture smoke tests**

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_cli_provides_help_without_loading_model() -> None:
    for name in [
        "00_prepare_data.py", "01_score_demands.py", "02_prepare_items.py",
        "03_capture_contrasts.py", "04_extract_vectors.py", "05_analyze_similarity.py",
        "06_run_steering.py", "07_score_generations.py",
    ]:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--config" in result.stdout
```

- [ ] **Step 2: Verify RED**

Run: `cd vector-make; python -m pytest tests/test_cli.py tests/test_smoke_pipeline.py -q`

Expected: FAIL because CLI scripts do not exist.

- [ ] **Step 3: Implement thin CLIs with shared `--config`, repeatable `--override`, and `--limit`; add a local-fixture smoke test and document installation, environment variables, stage commands, output layout, AIME enablement, and real Qwen run requirements**

Every CLI must defer API/model imports until after argument parsing so `--help` works in a minimal environment. The documented default workflow is:

```powershell
python scripts/00_prepare_data.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/01_score_demands.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/02_prepare_items.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/03_capture_contrasts.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/04_extract_vectors.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/05_analyze_similarity.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/06_run_steering.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
python scripts/07_score_generations.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml
```

- [ ] **Step 4: Run the complete offline suite and smoke workflow**

Run: `cd vector-make; python -m pytest -q`

Expected: all tests PASS with no model download or API call.

Run: `cd vector-make; python scripts/00_prepare_data.py --config configs/model.yaml --config configs/data.yaml --config configs/experiment.yaml --override data.sources.math500.local_path=tests/fixtures/math_items.jsonl --limit 2`

Expected: exit code 0 and a two-row canonical MATH500 JSONL plus manifest.

- [ ] **Step 5: Run static sanity checks**

Run: `cd vector-make; python -m compileall -q src scripts tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 6: Commit**

```powershell
git add -- vector-make/scripts vector-make/tests/test_cli.py vector-make/tests/test_smoke_pipeline.py vector-make/README.md
git commit -m "feat: complete self-steering MVP workflow"
```

### Task 11: Final integration verification

**Files:**
- Modify only files implicated by verification failures.

- [ ] **Step 1: Install the project development environment**

Run: `cd vector-make; python -m pip install -e ".[dev]"`

Expected: installation succeeds and imports resolve from `src/self_steering`.

- [ ] **Step 2: Run all offline tests**

Run: `cd vector-make; python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 3: Run compilation and CLI help checks**

Run: `cd vector-make; python -m compileall -q src scripts tests`

Expected: exit code 0.

Run: `cd vector-make; python scripts/06_run_steering.py --help`

Expected: exit code 0 and no model load.

- [ ] **Step 4: Verify repository hygiene and intended changes**

Run: `git status --short`

Expected: only the previously existing unrelated files plus intentional Self-Steering project artifacts are present.

Run: `git diff --check HEAD~1..HEAD`

Expected: no output.

- [ ] **Step 5: Record the verification evidence in the handoff**

Report the exact test count, commands, commit IDs, unexecuted GPU/API integration steps, and any data-source caveats. Do not claim the Qwen/OpenAI end-to-end run passed unless it was actually executed with the required credentials and hardware.
