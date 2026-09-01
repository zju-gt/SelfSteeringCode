import pytest

from self_steering.datasets.filtering import demand_memberships, demand_slice


ROWS = [
    {"item_id": "low", "demand_scores": {"QLl": 1, "QLq": 4}},
    {"item_id": "middle", "demand_scores": {"QLl": 2, "QLq": 2}},
    {"item_id": "high", "demand_scores": {"QLl": 4, "QLq": 4}},
]


def test_demand_slice_includes_threshold_boundaries() -> None:
    assert [x["item_id"] for x in demand_slice(ROWS, "QLl", "high", 4, 1)] == ["high"]
    assert [x["item_id"] for x in demand_slice(ROWS, "QLl", "low", 4, 1)] == ["low"]


def test_memberships_allow_dimension_overlap() -> None:
    memberships = demand_memberships(ROWS[2], ["QLl", "QLq"], 4, 1)
    assert memberships == {"QLl": "high", "QLq": "high"}


def test_demand_slice_rejects_missing_dimension() -> None:
    with pytest.raises(ValueError, match="missing demand score"):
        demand_slice([{"item_id": "x", "demand_scores": {}}], "QLl", "high", 4, 1)

