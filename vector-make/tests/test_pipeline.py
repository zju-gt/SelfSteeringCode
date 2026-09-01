import json
from pathlib import Path

from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.types import CanonicalItem
from self_steering.pipeline import prepare_data, prepare_items
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
    registry.register("mmlu", lambda config: [
        CanonicalItem("m1", "mmlu", "test", "q", "A", "choice", {"A": "x"})
    ])
    registry.register("math500", lambda config: [
        CanonicalItem("x1", "math500", "test", "q", "1", "math")
    ])
    paths = prepare_data(base_config(tmp_path), registry)
    assert set(paths) == {"mmlu", "math500"}
    assert list(read_jsonl(paths["mmlu"]))[0]["item_id"] == "m1"


def test_prepare_items_writes_extraction_and_external_memberships(tmp_path: Path) -> None:
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

