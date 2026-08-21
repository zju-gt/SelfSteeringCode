"""Serializable records used by the lightweight evaluation runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationExample:
    example_id: str
    prompt: str
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "EvaluationExample":
        values = dict(values)
        return cls(
            example_id=str(values.pop("id", values.pop("example_id", ""))),
            prompt=str(values.pop("prompt")),
            reference=values.pop("reference", None),
            metadata=values.pop("metadata", values),
        )


@dataclass
class EvaluationResult:
    example_id: str
    prompt: str
    reference: Optional[str]
    baseline_output: str
    steered_output: str
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_total_tokens: int
    steered_input_tokens: int
    steered_output_tokens: int
    steered_total_tokens: int
    baseline_latency_seconds: float
    steered_latency_seconds: float
    metrics: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    routing: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
