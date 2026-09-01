import pytest

from self_steering.datasets.types import CanonicalItem


def test_canonical_item_round_trip() -> None:
    item = CanonicalItem(
        item_id="arc_c_test_1",
        dataset="arc_c",
        split="test",
        prompt="Which answer?",
        gold_answer="B",
        answer_type="choice",
        choices={"A": "first", "B": "second"},
        metadata={"subject": "science"},
    )
    item.validate()
    assert CanonicalItem.from_dict(item.to_dict()) == item


def test_choice_item_requires_gold_letter_in_choices() -> None:
    item = CanonicalItem(
        item_id="bad",
        dataset="obqa",
        split="test",
        prompt="Q",
        gold_answer="C",
        answer_type="choice",
        choices={"A": "first", "B": "second"},
    )
    with pytest.raises(ValueError, match="gold answer"):
        item.validate()


def test_canonical_item_rejects_invalid_aime_answer() -> None:
    item = CanonicalItem(
        item_id="aime2026_1",
        dataset="aime2026",
        split="test",
        prompt="problem",
        gold_answer="1000",
        answer_type="math",
        metadata={"competition": "AIME"},
    )
    with pytest.raises(ValueError, match="AIME"):
        item.validate()
