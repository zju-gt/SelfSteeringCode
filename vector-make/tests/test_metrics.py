import pytest

from self_steering.evaluation.metrics import (
    accuracy_by_alpha,
    diagonal_dominance,
    specificity_matrix,
)


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
        {
            "steering_capability": "QLl",
            "demand_capability": "QLq",
            "alpha": 0.0,
            "correct": False,
        },
        {
            "steering_capability": "QLl",
            "demand_capability": "QLq",
            "alpha": 1.0,
            "correct": True,
        },
    ]
    assert specificity_matrix(rows, alpha=1.0)["QLl"]["QLq"] == 1.0


def test_diagonal_dominance_subtracts_off_diagonal_mean() -> None:
    matrix = {"a": {"a": 1.0, "b": 0.0}, "b": {"a": 0.0, "b": 0.5}}
    assert diagonal_dominance(matrix) == pytest.approx(0.75)

