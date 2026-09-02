"""Concurrent, resumable demand scoring and long-to-wide conversion."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm.auto import tqdm

from self_steering.datasets.delean_labeler import annotation_key
from self_steering.datasets.types import CanonicalItem
from self_steering.utils.io import append_jsonl, read_jsonl


AnnotationKey = tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class ScoreSummary:
    total: int
    resumed: int
    attempted: int
    errors: int


def completed_annotation_keys(rows: Iterable[dict[str, Any]]) -> set[AnnotationKey]:
    completed: set[AnnotationKey] = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            completed.add(annotation_key(row))
        except KeyError:
            continue
    return completed


def current_annotation_rows(
    rows: Iterable[dict[str, Any]],
    expected_keys: set[AnnotationKey],
) -> list[dict[str, Any]]:
    """Return only the latest successful row for each current full hash key."""

    current: dict[AnnotationKey, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            key = annotation_key(row)
        except KeyError:
            continue
        if key in expected_keys:
            current[key] = row
    return list(current.values())


def score_items(
    items: Iterable[CanonicalItem],
    dimensions: Sequence[str],
    output_path: Path | str,
    label_fn: Callable[[CanonicalItem, str], dict[str, Any]],
    *,
    max_workers: int,
    expected_key_fn: Callable[[CanonicalItem, str], AnnotationKey] | None = None,
) -> ScoreSummary:
    items = list(items)
    destination = Path(output_path)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed_full = completed_annotation_keys(existing)
    completed_pairs = {
        (str(row.get("item_id")), str(row.get("demand")))
        for row in existing
        if row.get("status") == "ok"
    }
    jobs: list[tuple[CanonicalItem, str]] = []
    for item in items:
        for dimension in dimensions:
            if expected_key_fn is not None:
                if expected_key_fn(item, dimension) in completed_full:
                    continue
            elif (item.item_id, dimension) in completed_pairs:
                continue
            jobs.append((item, dimension))

    total = len(items) * len(dimensions)
    resumed = total - len(jobs)
    errors = 0
    progress = tqdm(total=total, initial=resumed, desc=destination.stem)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[dict[str, Any]], tuple[CanonicalItem, str]] = {
            executor.submit(label_fn, item, dimension): (item, dimension)
            for item, dimension in jobs
        }
        for future in as_completed(futures):
            item, dimension = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "item_id": item.item_id,
                    "dataset": item.dataset,
                    "split": item.split,
                    "demand": dimension,
                    "status": "error",
                    "error": repr(exc),
                }
            if row.get("status") != "ok":
                errors += 1
            append_jsonl(destination, row)
            progress.update(1)
            progress.set_postfix(resumed=resumed, failed=errors)
    progress.close()
    return ScoreSummary(
        total=total,
        resumed=resumed,
        attempted=len(jobs),
        errors=errors,
    )


def annotations_to_wide(
    items: Iterable[CanonicalItem],
    rows: Iterable[dict[str, Any]],
    *,
    dimensions: Sequence[str],
) -> list[dict[str, Any]]:
    scores: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("demand") not in dimensions:
            continue
        scores.setdefault(str(row["item_id"]), {})[str(row["demand"])] = int(
            row["level"]
        )

    wide: list[dict[str, Any]] = []
    required = set(dimensions)
    for item in items:
        item_scores = scores.get(item.item_id, {})
        missing = required - set(item_scores)
        if missing:
            raise ValueError(
                f"missing successful annotations for {item.item_id}: {', '.join(sorted(missing))}"
            )
        record = item.to_dict()
        record["demand_scores"] = {
            dimension: item_scores[dimension] for dimension in dimensions
        }
        wide.append(record)
    return wide
