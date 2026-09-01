import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_steering.datasets.delean_labeler import (
    AnnotationRequest,
    annotation_key,
    label_one_dimension,
)
from self_steering.datasets.scoring import (
    annotations_to_wide,
    completed_annotation_keys,
    score_items,
)
from self_steering.datasets.types import CanonicalItem
from self_steering.utils.io import read_jsonl


class FakeResponses:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text='{"score":4,"brief_justification":"multi-step"}',
            id="resp_1",
            model="judge-returned",
        )


def test_label_one_dimension_returns_hashed_audit_record(tmp_path: Path) -> None:
    rubric = tmp_path / "QLl.txt"
    rubric.write_text("rubric", encoding="utf-8")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    row = label_one_dimension(
        client,
        AnnotationRequest(
            item_id="i1",
            dataset="mmlu",
            split="test",
            prompt="Q",
            dimension="QLl",
        ),
        rubric,
        model="judge",
    )
    assert row["level"] == 4
    assert row["status"] == "ok"
    assert row["annotator_model_returned"] == "judge-returned"
    assert len(row["task_sha256"]) == 64
    assert len(row["rubric_sha256"]) == 64
    assert responses.kwargs["text"]["format"]["strict"] is True


def test_annotation_key_changes_when_task_hash_changes() -> None:
    old = {
        "item_id": "i1",
        "demand": "QLl",
        "task_sha256": "old",
        "rubric_sha256": "r",
        "annotator_model_requested": "m",
        "status": "ok",
    }
    new = dict(old, task_sha256="new")
    assert annotation_key(old) != annotation_key(new)
    assert annotation_key(new) not in completed_annotation_keys([old])


def test_score_items_does_not_reuse_stale_hash(tmp_path: Path) -> None:
    output = tmp_path / "annotations.jsonl"
    old = {
        "item_id": "i1",
        "demand": "QLl",
        "task_sha256": "old",
        "rubric_sha256": "r",
        "annotator_model_requested": "m",
        "status": "ok",
    }
    output.write_text(json.dumps(old) + "\n", encoding="utf-8")
    item = CanonicalItem("i1", "mmlu", "test", "new task", "A", "choice", {"A": "x"})
    expected = ("i1", "QLl", "new", "r", "m")
    calls = []

    def label_fn(current, dimension):
        calls.append((current.item_id, dimension))
        return dict(
            old,
            task_sha256="new",
            level=4,
            brief_justification="ok",
        )

    score_items(
        [item],
        ["QLl"],
        output,
        label_fn,
        max_workers=2,
        expected_key_fn=lambda current, dimension: expected,
    )
    assert calls == [("i1", "QLl")]
    assert len(list(read_jsonl(output))) == 2


def test_score_items_records_worker_errors(tmp_path: Path) -> None:
    output = tmp_path / "annotations.jsonl"
    item = CanonicalItem("i1", "mmlu", "test", "task", "A", "choice", {"A": "x"})

    def fail(current, dimension):
        raise RuntimeError("rate limited")

    score_items([item], ["QLl"], output, fail, max_workers=1)
    row = list(read_jsonl(output))[0]
    assert row["status"] == "error"
    assert "rate limited" in row["error"]


def test_annotations_to_wide_requires_all_dimensions() -> None:
    item = CanonicalItem("i1", "mmlu", "test", "task", "A", "choice", {"A": "x"})
    rows = [
        {"item_id": "i1", "demand": "QLl", "level": 4, "status": "ok"},
        {"item_id": "i1", "demand": "QLq", "level": 3, "status": "ok"},
    ]
    with pytest.raises(ValueError, match="missing successful annotations"):
        annotations_to_wide([item], rows, dimensions=["QLl", "QLq", "CL", "MCr"])


def test_annotations_to_wide_attaches_scores() -> None:
    item = CanonicalItem("i1", "mmlu", "test", "task", "A", "choice", {"A": "x"})
    rows = [
        {"item_id": "i1", "demand": dim, "level": level, "status": "ok"}
        for dim, level in zip(["QLl", "QLq", "CL", "MCr"], [4, 3, 2, 1])
    ]
    wide = annotations_to_wide([item], rows, dimensions=["QLl", "QLq", "CL", "MCr"])
    assert wide[0]["demand_scores"] == {"QLl": 4, "QLq": 3, "CL": 2, "MCr": 1}

