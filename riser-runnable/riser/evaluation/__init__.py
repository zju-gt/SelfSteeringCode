"""Initial dataset-agnostic evaluation API."""

from .metrics import exact_match, substring_match
from .paths import DEFAULT_RESULTS_DIR, resolve_results_path
from .records import EvaluationExample, EvaluationResult
from .runner import EvaluationRunner

__all__ = [
    "EvaluationExample",
    "EvaluationResult",
    "DEFAULT_RESULTS_DIR",
    "EvaluationRunner",
    "exact_match",
    "resolve_results_path",
    "substring_match",
]
