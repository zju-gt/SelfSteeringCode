import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.types import CanonicalItem
from self_steering.pipeline import (
    canonical_generation_rows,
    capture_contrasts,
    capture_artifact_id,
    generation_identity_record,
    generation_key,
    prepare_data,
    prepare_items,
    run_steering,
    score_demands,
    steering_artifact_id,
)
from self_steering.utils.io import read_jsonl, sha256_file, write_jsonl
from self_steering.vectors.storage import save_vector_library


def base_config(tmp_path: Path) -> dict:
    return {
        "model": {"name": "fake", "num_hidden_layers": 28},
        "data": {
            "enabled_steering_datasets": ["math500"],
            "sources": {"mmlu": {}, "math500": {}},
        },
        "experiment": {
            "capabilities": ["QLl", "QLq"],
            "target_layer": 19,
            "high_demand_threshold": 4,
            "low_demand_threshold": 1,
            "vector_scaling": "mean_norm",
            "alphas": [0, 1],
            "seed": 42,
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "outputs_dir": str(tmp_path / "outputs"),
                "rubrics_dir": str(tmp_path / "rubrics"),
            },
        },
    }


def test_prepare_data_writes_mmlu_and_enabled_steering_dataset(tmp_path: Path) -> None:
    registry = DatasetRegistry()
    registry.register(
        "mmlu",
        lambda config: [
            CanonicalItem("m1", "mmlu", "test", "q", "A", "choice", {"A": "x"})
        ],
    )
    registry.register(
        "math500",
        lambda config: [CanonicalItem("x1", "math500", "test", "q", "1", "math")],
    )
    paths = prepare_data(base_config(tmp_path), registry)
    assert set(paths) == {"mmlu", "math500"}
    assert list(read_jsonl(paths["mmlu"]))[0]["item_id"] == "m1"
    manifest = json.loads(
        (tmp_path / "outputs" / "manifests" / "00_prepare_data.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset_fingerprints"]["mmlu"] == sha256_file(paths["mmlu"])
    assert manifest["seed"] == 42
    assert set(manifest["prompt_hashes"]) == {"capability_prompts", "generic_prompt"}
    assert (tmp_path / "outputs" / "manifests" / "00_prepare_data").is_dir()


def test_prepare_items_writes_extraction_and_external_memberships(
    tmp_path: Path,
) -> None:
    config = base_config(tmp_path)
    scored = tmp_path / "data" / "scored"
    scored.mkdir(parents=True)
    write_jsonl(
        scored / "mmlu_with_4d_demands.jsonl",
        [
            {
                "item_id": "m1",
                "dataset": "mmlu",
                "split": "test",
                "prompt": "q",
                "gold_answer": "A",
                "answer_type": "choice",
                "choices": {"A": "x"},
                "metadata": {},
                "demand_scores": {"QLl": 4, "QLq": 2},
            }
        ],
    )
    write_jsonl(
        scored / "math500_with_4d_demands.jsonl",
        [
            {
                "item_id": "x1",
                "dataset": "math500",
                "split": "test",
                "prompt": "q",
                "gold_answer": "1",
                "answer_type": "math",
                "choices": None,
                "metadata": {},
                "demand_scores": {"QLl": 1, "QLq": 4},
            }
        ],
    )
    paths = prepare_items(config)
    assert [row["item_id"] for row in read_jsonl(paths["extraction_QLl"])] == ["m1"]
    evaluation = list(read_jsonl(paths["evaluation_math500"]))
    assert evaluation[0]["demand_memberships"] == {"QLl": "low", "QLq": "high"}


def test_capture_artifact_id_changes_with_layer(tmp_path: Path) -> None:
    first = base_config(tmp_path)
    second = base_config(tmp_path)
    second["experiment"]["target_layer"] = 18
    assert capture_artifact_id(first) != capture_artifact_id(second)


def test_generation_key_changes_with_task_content() -> None:
    base = {
        "dataset": "math500",
        "item_id": "x",
        "prompt": "old",
        "gold_answer": "1",
        "steering_capability": "QLl",
        "alpha": 1.0,
    }
    assert generation_key(base) != generation_key(dict(base, prompt="new"))
    assert generation_key(base) != generation_key(
        dict(base, demand_memberships={"QLl": "high"})
    )


def test_generation_identity_record_matches_persisted_row_key() -> None:
    item = CanonicalItem("x", "math500", "test", "q", "1", "math")
    source = {"demand_memberships": {"QLl": "high"}}
    pending = generation_identity_record(
        run_id="run-1",
        dataset="math500",
        item=item,
        source_row=source,
        steering_capability="QLl",
        alpha=1.0,
    )
    persisted = dict(pending, raw_output="1", status="ok")
    assert pending["demand_memberships"] == {"QLl": "high"}
    assert generation_key(pending) == generation_key(persisted)


def test_canonical_generation_rows_keep_latest_success_per_key() -> None:
    identity = {
        "run_id": "run-1",
        "dataset": "math500",
        "item_id": "x",
        "item_identity": "identity-1",
        "steering_capability": "QLl",
        "alpha": 1.0,
    }
    rows = [
        dict(identity, raw_output="old", status="ok"),
        dict(identity, raw_output="failed", status="error"),
        dict(identity, raw_output="new", status="ok"),
    ]
    assert canonical_generation_rows(rows) == [rows[-1]]


def test_canonical_generation_rows_ignores_other_run_ids() -> None:
    current = {
        "run_id": "current",
        "dataset": "math500",
        "item_id": "x",
        "prompt": "q",
        "gold_answer": "1",
        "answer_type": "math",
        "steering_capability": "QLl",
        "alpha": 0.0,
        "status": "ok",
    }
    stale = dict(current, run_id="stale")
    assert canonical_generation_rows([stale, current], run_id="current") == [
        dict(current, item_identity=generation_key(current)[1])
    ]


def test_steering_artifact_id_changes_with_vector_content(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    vector_path = (
        tmp_path
        / "outputs"
        / "vectors"
        / capture_artifact_id(config)
        / "capability_vectors.safetensors"
    )
    vector_path.parent.mkdir(parents=True)
    vector_path.write_bytes(b"first")
    first = steering_artifact_id(config)
    vector_path.write_bytes(b"second")
    assert steering_artifact_id(config) != first


def test_score_demands_honors_runtime_limit(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    config["_runtime"] = {"limit": 1}
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for dataset in ("mmlu", "math500"):
        write_jsonl(
            processed / f"{dataset}.jsonl",
            [
                CanonicalItem(
                    f"{dataset}-{index}", dataset, "test", "q", "1", "math"
                ).to_dict()
                for index in range(2)
            ],
        )

    calls = []

    def label(item: CanonicalItem, dimension: str) -> dict:
        calls.append((item.item_id, dimension))
        return {
            "item_id": item.item_id,
            "demand": dimension,
            "level": 4,
            "status": "ok",
        }

    score_demands(config, label)
    assert len(calls) == 4


def test_score_demands_preserves_errors_and_explains_how_to_resume(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for dataset in ("mmlu", "math500"):
        write_jsonl(
            processed / f"{dataset}.jsonl",
            [CanonicalItem(f"{dataset}-1", dataset, "test", "q", "1", "math").to_dict()],
        )

    def fail(item: CanonicalItem, dimension: str) -> dict:
        raise RuntimeError("invalid API key")

    with pytest.raises(RuntimeError, match="annotations are incomplete.*rerun"):
        score_demands(config, fail)

    failed_rows = list(read_jsonl(tmp_path / "data" / "scored" / "mmlu_delean_long.jsonl"))
    assert len(failed_rows) == 2
    assert {row["status"] for row in failed_rows} == {"error"}


class MinimalTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [1]


class FailingGenerationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.model = SimpleNamespace(layers=[nn.Identity() for _ in range(20)])

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def test_capture_contrasts_records_cuda_oom_per_item(tmp_path: Path, monkeypatch) -> None:
    config = base_config(tmp_path)
    extraction = tmp_path / "data" / "processed" / "extraction"
    extraction.mkdir(parents=True)
    for capability in config["experiment"]["capabilities"]:
        write_jsonl(
            extraction / f"{capability}.jsonl",
            [CanonicalItem("x", "mmlu", "test", "q", "A", "choice", {"A": "x"}).to_dict()],
        )
    monkeypatch.setattr(
        "self_steering.pipeline.capture_prompt_contrast",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("CUDA out of memory")
        ),
    )
    model = FailingGenerationModel()
    assert capture_contrasts(config, model, MinimalTokenizer()) == []
    error_path = (
        tmp_path
        / "outputs"
        / "activations"
        / capture_artifact_id(config)
        / "errors.jsonl"
    )
    errors = list(read_jsonl(error_path))
    assert len(errors) == 2
    assert {row["error_type"] for row in errors} == {"cuda_oom"}
    assert all("reduce" in row["hint"].lower() for row in errors)


def test_run_steering_records_cuda_oom_for_each_generation(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    evaluation = tmp_path / "data" / "processed" / "evaluation"
    evaluation.mkdir(parents=True)
    write_jsonl(
        evaluation / "math500.jsonl",
        [
            dict(
                CanonicalItem("x", "math500", "test", "q", "1", "math").to_dict(),
                demand_memberships={"QLl": "high"},
            )
        ],
    )
    vector_root = (
        tmp_path / "outputs" / "vectors" / capture_artifact_id(config)
    )
    vectors = {
        capability: {
            "raw": torch.ones(2),
            "unit": torch.ones(2),
            "steering": torch.ones(2),
        }
        for capability in config["experiment"]["capabilities"]
    }
    save_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
        vectors,
        metadata={},
    )
    output = run_steering(config, FailingGenerationModel(), MinimalTokenizer())
    rows = list(read_jsonl(output))
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"error"}
    assert {row["error_type"] for row in rows} == {"cuda_oom"}
    assert all(row["item_identity"] for row in rows)


def test_run_steering_reports_resumed_generation_progress(
    tmp_path: Path, monkeypatch
) -> None:
    config = base_config(tmp_path)
    evaluation = tmp_path / "data" / "processed" / "evaluation"
    evaluation.mkdir(parents=True)
    write_jsonl(
        evaluation / "math500.jsonl",
        [
            dict(
                CanonicalItem("x", "math500", "test", "q", "1", "math").to_dict(),
                demand_memberships={"QLl": "high"},
            )
        ],
    )
    vector_root = tmp_path / "outputs" / "vectors" / capture_artifact_id(config)
    vectors = {
        capability: {
            "raw": torch.ones(2),
            "unit": torch.ones(2),
            "steering": torch.ones(2),
        }
        for capability in config["experiment"]["capabilities"]
    }
    save_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
        vectors,
        metadata={},
    )
    progress_instances = []

    class FakeProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.closed = False
            progress_instances.append(self)

        def update(self, amount):
            self.updates.append(amount)

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr("self_steering.pipeline.tqdm", FakeProgress, raising=False)
    monkeypatch.setattr(
        "self_steering.pipeline.generate_batch_with_optional_steering",
        lambda *args, **kwargs: [r"\boxed{1}" for _ in args[2]],
    )

    run_steering(config, FailingGenerationModel(), MinimalTokenizer())
    run_steering(config, FailingGenerationModel(), MinimalTokenizer())

    assert progress_instances[0].kwargs["total"] == 4
    assert progress_instances[0].kwargs["initial"] == 0
    assert progress_instances[0].updates == [1, 1, 1, 1]
    assert progress_instances[0].closed
    assert progress_instances[1].kwargs["initial"] == 4
    assert progress_instances[1].updates == []
    assert progress_instances[1].closed


def test_run_steering_batches_matching_capability_alpha_work(
    tmp_path: Path, monkeypatch
) -> None:
    config = base_config(tmp_path)
    config["experiment"]["generation"] = {"batch_size": 2}
    evaluation = tmp_path / "data" / "processed" / "evaluation"
    evaluation.mkdir(parents=True)
    write_jsonl(
        evaluation / "math500.jsonl",
        [
            dict(
                CanonicalItem(item_id, "math500", "test", "q", "1", "math").to_dict(),
                demand_memberships={"QLl": "high"},
            )
            for item_id in ("x1", "x2")
        ],
    )
    vector_root = tmp_path / "outputs" / "vectors" / capture_artifact_id(config)
    vectors = {
        capability: {
            "raw": torch.ones(2),
            "unit": torch.ones(2),
            "steering": torch.ones(2),
        }
        for capability in config["experiment"]["capabilities"]
    }
    save_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
        vectors,
        metadata={},
    )
    batch_sizes = []

    def generate_batch(*args, **kwargs):
        inputs = args[2]
        batch_sizes.append(len(inputs))
        return [r"\boxed{1}" for _ in inputs]

    monkeypatch.setattr(
        "self_steering.pipeline.generate_batch_with_optional_steering",
        generate_batch,
        raising=False,
    )

    output = run_steering(config, FailingGenerationModel(), MinimalTokenizer())

    assert batch_sizes == [2, 2, 2]
    assert len(list(read_jsonl(output))) == 8
