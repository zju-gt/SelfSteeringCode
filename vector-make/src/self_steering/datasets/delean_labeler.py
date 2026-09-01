"""One-item, one-dimension DeLeAn annotation through Responses API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from self_steering.utils.io import sha256_text


TARGET_DIMS = {
    "QLl": "Logical Reasoning",
    "QLq": "Quantitative Reasoning",
    "CL": "Conceptualisation, Learning and Abstraction",
    "MCr": "Identifying Relevant Information",
}

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "brief_justification": {"type": "string"},
    },
    "required": ["score", "brief_justification"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AnnotationRequest:
    item_id: str
    dataset: str
    split: str
    prompt: str
    dimension: str
    domain: str | None = None


def build_annotation_prompt(request: AnnotationRequest, rubric: str) -> str:
    if request.dimension not in TARGET_DIMS:
        raise ValueError(f"unknown DeLeAn dimension: {request.dimension}")
    dimension_name = TARGET_DIMS[request.dimension]
    return f"""You are annotating the cognitive demand of a task.

DIMENSION:
{dimension_name} ({request.dimension})

OFFICIAL RUBRIC:
{rubric}

TASK INSTANCE:
{request.prompt}

Assess only how much {dimension_name} is required to solve the task correctly.

Rules:
- Judge task demand, not model difficulty.
- Do not use gold answers or model responses.
- Follow the rubric literally.
- Map the highest 5+ category to score 5.
- Return an integer score from 0 to 5.
- Give a short justification."""


def _annotation_contract_hash(prompt: str) -> str:
    serialized = json.dumps(
        {"prompt": prompt, "schema": ANNOTATION_SCHEMA},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(serialized)


def annotation_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["item_id"]),
        str(row["demand"]),
        str(row["task_sha256"]),
        str(row["rubric_sha256"]),
        str(row["annotator_model_requested"]),
        str(row["annotation_contract_sha256"]),
    )


def expected_annotation_key(
    request: AnnotationRequest,
    rubric: str,
    model: str,
) -> tuple[str, str, str, str, str, str]:
    prompt = build_annotation_prompt(request, rubric)
    return (
        request.item_id,
        request.dimension,
        sha256_text(request.prompt),
        sha256_text(rubric),
        model,
        _annotation_contract_hash(prompt),
    )


def label_one_dimension(
    client: Any,
    request: AnnotationRequest,
    rubric_path: Path | str,
    *,
    model: str,
) -> dict[str, Any]:
    rubric = Path(rubric_path).read_text(encoding="utf-8")
    prompt = build_annotation_prompt(request, rubric)
    schema_serialized = json.dumps(
        ANNOTATION_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "delean_annotation",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            }
        },
    )
    parsed = json.loads(response.output_text)
    score = parsed.get("score")
    justification = parsed.get("brief_justification")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
        raise ValueError(f"annotator returned invalid score: {score!r}")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("annotator returned an empty brief_justification")
    return {
        "item_id": request.item_id,
        "dataset": request.dataset,
        "split": request.split,
        "domain": request.domain,
        "demand": request.dimension,
        "demand_name": TARGET_DIMS[request.dimension],
        "level": score,
        "brief_justification": justification.strip(),
        "annotator_provider": "openai",
        "annotator_model_requested": model,
        "annotator_model_returned": getattr(response, "model", model),
        "response_id": getattr(response, "id", None),
        "rubric_version": "DeLeAn-v1.0",
        "rubric_sha256": sha256_text(rubric),
        "task_sha256": sha256_text(request.prompt),
        "annotation_prompt_sha256": sha256_text(prompt),
        "annotation_schema_sha256": sha256_text(schema_serialized),
        "annotation_contract_sha256": _annotation_contract_hash(prompt),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
    }


T = TypeVar("T")


def _is_transient_exception(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in {408, 409, 429} or status_code >= 500
    ):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def retry_call(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    initial_backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if not _is_transient_exception(error) or attempt == max_attempts:
                raise
            sleep(initial_backoff_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")
