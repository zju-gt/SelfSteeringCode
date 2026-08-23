"""Initial dataset-agnostic evaluation API."""

from .metrics import exact_match, substring_match
from .records import EvaluationExample, EvaluationResult
from .runner import EvaluationRunner

__all__ = [
    "EvaluationExample",
    "EvaluationResult",
    "EvaluationRunner",
    "exact_match",
    "substring_match",
]
