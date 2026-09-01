from pathlib import Path

from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.types import CanonicalItem
from self_steering.pipeline import (
    capture_artifact_id,
    generation_key,
    prepare_data,
    prepare_items,
    score_demands,
    steering_artifact_id,
)
from self_steering.utils.io import read_jsonl, write_jsonl


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
