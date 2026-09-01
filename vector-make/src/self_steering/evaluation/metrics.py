"""Descriptive steering effects and causal specificity matrices."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def accuracy_by_alpha(rows: Iterable[Mapping[str, Any]]) -> dict[float, dict[str, float | int]]:
    grouped: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[float(row["alpha"])].append(bool(row["correct"]))
    if 0.0 not in grouped:
        raise ValueError("alpha-zero baseline is required")
    accuracies = {
        alpha: sum(values) / len(values) for alpha, values in grouped.items() if values
    }
    baseline = accuracies[0.0]
    return {
        alpha: {
            "count": len(grouped[alpha]),
            "accuracy": accuracy,
            "delta": accuracy - baseline,
        }
        for alpha, accuracy in sorted(accuracies.items())
    }


def _high_demand_capabilities(row: Mapping[str, Any]) -> list[str]:
    explicit = row.get("demand_capability")
    if explicit:
        return [str(explicit)]
    memberships = row.get("demand_memberships", {})
    if not isinstance(memberships, Mapping):
        return []
    return [str(name) for name, level in memberships.items() if level == "high"]


def specificity_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    cells: dict[tuple[str, str, float], list[bool]] = defaultdict(list)
    for row in materialized:
        steering = str(row["steering_capability"])
        row_alpha = float(row["alpha"])
        for demand in _high_demand_capabilities(row):
            cells[(steering, demand, row_alpha)].append(bool(row["correct"]))

    steering_names = sorted({key[0] for key in cells})
    demand_names = sorted({key[1] for key in cells})
    matrix: dict[str, dict[str, float]] = {}
    for steering in steering_names:
        matrix[steering] = {}
        for demand in demand_names:
            baseline_values = cells.get((steering, demand, 0.0), [])
            steered_values = cells.get((steering, demand, float(alpha)), [])
            if not baseline_values or not steered_values:
                continue
            matrix[steering][demand] = (
                sum(steered_values) / len(steered_values)
                - sum(baseline_values) / len(baseline_values)
            )
    return matrix


def diagonal_dominance(matrix: Mapping[str, Mapping[str, float]]) -> float:
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    for row_name, row in matrix.items():
        for column_name, value in row.items():
            if row_name == column_name:
                diagonal.append(float(value))
            else:
                off_diagonal.append(float(value))
    if not diagonal:
        raise ValueError("matrix has no diagonal values")
    return sum(diagonal) / len(diagonal) - (
        sum(off_diagonal) / len(off_diagonal) if off_diagonal else 0.0
    )

