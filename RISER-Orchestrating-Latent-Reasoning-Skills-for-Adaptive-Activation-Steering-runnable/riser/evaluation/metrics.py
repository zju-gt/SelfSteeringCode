"""Deterministic reference-based metrics for initial steering experiments."""

from __future__ import annotations

from typing import Optional


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(prediction: str, reference: Optional[str]) -> Optional[float]:
    if reference is None:
        return None
    return float(_normalize(prediction) == _normalize(reference))


def substring_match(prediction: str, reference: Optional[str]) -> Optional[float]:
    if reference is None:
        return None
    return float(_normalize(reference) in _normalize(prediction))
