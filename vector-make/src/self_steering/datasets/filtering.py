"""Demand-based extraction and evaluation slices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _score(row: Mapping[str, Any], capability: str) -> int:
    scores = row.get("demand_scores")
    if not isinstance(scores, Mapping) or capability not in scores:
        raise ValueError(
            f"missing demand score {capability} for item {row.get('item_id', '<unknown>')}"
        )
    score = scores[capability]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
        raise ValueError(f"invalid demand score {capability}={score!r}")
    return score


def demand_slice(
    rows: Iterable[dict[str, Any]],
    capability: str,
    slice_name: str,
    high_threshold: int,
    low_threshold: int,
) -> list[dict[str, Any]]:
    if slice_name not in {"high", "low"}:
        raise ValueError("slice_name must be 'high' or 'low'")
    selected: list[dict[str, Any]] = []
    for row in rows:
        score = _score(row, capability)
        if slice_name == "high" and score >= high_threshold:
            selected.append(row)
        elif slice_name == "low" and score <= low_threshold:
            selected.append(row)
    return selected


def demand_memberships(
    row: Mapping[str, Any],
    capabilities: Sequence[str],
    high_threshold: int,
    low_threshold: int,
) -> dict[str, str]:
    memberships: dict[str, str] = {}
    for capability in capabilities:
        score = _score(row, capability)
        if score >= high_threshold:
            memberships[capability] = "high"
        elif score <= low_threshold:
            memberships[capability] = "low"
    return memberships
