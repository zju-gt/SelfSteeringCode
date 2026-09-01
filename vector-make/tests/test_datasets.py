from pathlib import Path

import pytest

from self_steering.datasets.adapters import (
    adapt_aime,
    adapt_math500,
    adapt_mmlu,
    adapt_multiple_choice,
)
from self_steering.datasets.registry import DatasetRegistry


def test_registry_contains_all_supported_datasets() -> None:
    registry = DatasetRegistry.default()
    assert set(registry.names()) == {
        "mmlu",
        "math500",
        "aime2024",
        "aime2025",
        "aime2026",
        "arc_c",
        "obqa",
    }


def test_aime2026_schema_is_canonicalized() -> None:
    item = adapt_aime(
        {
            "source_problem_id": "aime_2026__000",
            "problem": "Compute.",
            "ground_truth": 7,
            "topic": "algebra",
        },
        dataset="aime2026",
        split="test",
        index=0,
    )
    assert item.item_id == "aime_2026__000"
    assert item.gold_answer == "7"
    assert item.metadata["competition"] == "AIME"


def test_aime2024_chat_prompt_and_label_are_canonicalized() -> None:
    item = adapt_aime(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem step by step. Please put your "
                        "final answer within \\boxed{}.\n\nCompute 1+1.\n\nRemember to "
                        "put your final answer within \\boxed{}."
                    ),
                }
            ],
            "label": "002",
        },
        dataset="aime2024",
        split="train",
        index=0,
    )
    assert item.prompt == "Compute 1+1."
    assert item.gold_answer == "002"


def test_math500_schema_prefers_answer_field() -> None:
    item = adapt_math500(
        {"problem": "Compute.", "answer": "42", "solution": "work", "unique_id": "x"},
        split="test",
        index=0,
    )
    assert item.item_id == "math500_x"
    assert item.prompt == "Compute."
    assert item.gold_answer == "42"


def test_mmlu_renders_lettered_choices() -> None:
    item = adapt_mmlu(
        {
            "question": "Pick.",
            "choices": ["one", "two"],
            "answer": 1,
            "subject": "logic",
        },
        split="test",
        index=3,
    )
    assert item.gold_answer == "B"
    assert "A. one" in item.prompt
    assert "B. two" in item.prompt


def test_multiple_choice_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="choices"):
        adapt_multiple_choice(
            {
                "question": "Q",
                "choices": {"label": ["A"], "text": ["x", "y"]},
                "answerKey": "A",
            },
            dataset="obqa",
            split="test",
            index=0,
        )


def test_obqa_official_schema_uses_question_stem() -> None:
    item = adapt_multiple_choice(
        {
            "id": "obqa-1",
            "question_stem": "What conducts electricity?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["rubber", "glass", "copper", "wood"],
            },
            "answerKey": "C",
        },
        dataset="obqa",
        split="test",
        index=0,
    )
    assert item.item_id == "obqa-1"
    assert item.gold_answer == "C"
    assert "C. copper" in item.prompt


def test_arc_numeric_source_labels_are_mapped_to_letters() -> None:
    item = adapt_multiple_choice(
        {
            "id": "arc-1",
            "question": "Pick the second option.",
            "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]},
            "answerKey": "2",
        },
        dataset="arc_c",
        split="test",
        index=0,
    )
    assert item.choices == {"A": "a", "B": "b", "C": "c", "D": "d"}
    assert item.gold_answer == "B"
    assert "1." not in item.prompt


def test_local_override_reads_canonical_jsonl() -> None:
    fixture = Path(__file__).parent / "fixtures" / "math_items.jsonl"
    registry = DatasetRegistry.default()
    items = registry.load("math500", {"local_path": str(fixture), "split": "test"})
    assert [item.item_id for item in items] == ["math500_test_0", "math500_test_1"]


def test_registry_rejects_duplicate_item_ids() -> None:
    registry = DatasetRegistry()
    registry.register(
        "duplicate",
        lambda config: [
            adapt_math500(
                {"problem": "a", "answer": "1", "unique_id": "same"},
                split="test",
                index=0,
            ),
            adapt_math500(
                {"problem": "b", "answer": "2", "unique_id": "same"},
                split="test",
                index=1,
            ),
        ],
    )
    with pytest.raises(ValueError, match="duplicate item_id"):
        registry.load("duplicate", {})
