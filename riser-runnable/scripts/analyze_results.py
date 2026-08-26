"""Summarize baseline and steered metrics from an evaluation JSONL file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load result objects from JSONL, reporting the source line on errors."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results JSONL not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )
            records.append(value)
    return records


def _numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summarize_metric(
    records: Iterable[Mapping[str, Any]], metric_name: str
) -> dict[str, Any]:
    baseline_values: list[float] = []
    steered_values: list[float] = []
    paired_values: list[tuple[float, float]] = []

    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        pair = metrics.get(metric_name)
        if not isinstance(pair, Mapping):
            continue
        baseline = _numeric_score(pair.get("baseline"))
        steered = _numeric_score(pair.get("steered"))
        if baseline is not None:
            baseline_values.append(baseline)
        if steered is not None:
            steered_values.append(steered)
        if baseline is not None and steered is not None:
            paired_values.append((baseline, steered))

    baseline_mean = (
        sum(baseline_values) / len(baseline_values)
        if baseline_values
        else None
    )
    steered_mean = (
        sum(steered_values) / len(steered_values)
        if steered_values
        else None
    )
    delta = (
        steered_mean - baseline_mean
        if baseline_mean is not None and steered_mean is not None
        else None
    )

    return {
        "baseline_count": len(baseline_values),
        "steered_count": len(steered_values),
        "paired_count": len(paired_values),
        "baseline_mean": baseline_mean,
        "steered_mean": steered_mean,
        "delta": delta,
        "both_correct": sum(
            baseline == 1.0 and steered == 1.0
            for baseline, steered in paired_values
        ),
        "both_wrong": sum(
            baseline == 0.0 and steered == 0.0
            for baseline, steered in paired_values
        ),
        "steered_gain": sum(
            baseline == 0.0 and steered == 1.0
            for baseline, steered in paired_values
        ),
        "steered_loss": sum(
            baseline == 1.0 and steered == 0.0
            for baseline, steered in paired_values
        ),
    }


def summarize_results(
    records: Sequence[Mapping[str, Any]],
    metric_names: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Return overall and per-subject summaries for all requested metrics."""

    discovered = sorted(
        {
            str(metric_name)
            for record in records
            for metric_name in (
                record.get("metrics", {}).keys()
                if isinstance(record.get("metrics"), Mapping)
                else ()
            )
        }
    )
    selected = discovered if metric_names is None else list(metric_names)

    metrics = {
        name: _summarize_metric(records, name)
        for name in selected
    }

    subject_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        subject = metadata.get("subject")
        if subject is None:
            continue
        subject_records.setdefault(str(subject), []).append(record)

    subjects = {
        subject: {
            name: _summarize_metric(subject_rows, name)
            for name in selected
        }
        for subject, subject_rows in sorted(subject_records.items())
    }
    return {"count": len(records), "metrics": metrics, "subjects": subjects}


def _percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _delta_percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{100.0 * value:+.2f}pp"


def format_summary(summary: Mapping[str, Any], source: str | Path | None = None) -> str:
    """Format a summary as concise, human-readable Chinese console output."""

    lines = []
    if source is not None:
        lines.append(f"结果文件: {source}")
    lines.append(f"样本数: {summary['count']}")
    lines.append("")
    lines.append("整体指标:")
    for name, stats in summary["metrics"].items():
        lines.append(
            f"  {name}: baseline={_percent(stats['baseline_mean'])}, "
            f"steered={_percent(stats['steered_mean'])}, "
            f"变化={_delta_percent(stats['delta'])} "
            f"(n={stats['paired_count']})"
        )

    primary_name = (
        "mmlu_accuracy"
        if "mmlu_accuracy" in summary["metrics"]
        else next(iter(summary["metrics"]), None)
    )
    if primary_name is not None:
        primary = summary["metrics"][primary_name]
        lines.extend(
            [
                "",
                f"配对结果 ({primary_name}):",
                f"  两者都正确: {primary['both_correct']}",
                f"  两者都错误: {primary['both_wrong']}",
                f"  steered 修正错误: {primary['steered_gain']}",
                f"  steered 导致退化: {primary['steered_loss']}",
            ]
        )

    if summary["subjects"] and primary_name is not None:
        lines.extend(["", f"分科指标 ({primary_name}):"])
        for subject, subject_metrics in summary["subjects"].items():
            stats = subject_metrics[primary_name]
            lines.append(
                f"  {subject}: baseline={_percent(stats['baseline_mean'])}, "
                f"steered={_percent(stats['steered_mean'])}, "
                f"变化={_delta_percent(stats['delta'])} "
                f"(n={stats['paired_count']})"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print baseline and steered metrics from an evaluation JSONL."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Results JSONL path (positional form)",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        help="Results JSONL path (option form)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metric_names",
        help="Only print this metric; repeat for multiple metrics",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_path and args.input_option:
        parser.error("provide the input path either positionally or with --input")
    input_path = args.input_option or args.input_path
    if not input_path:
        parser.error("an input JSONL path is required")

    records = load_results(input_path)
    summary = summarize_results(records, metric_names=args.metric_names)
    if not summary["metrics"]:
        raise ValueError("No numeric baseline/steered metrics were found in the JSONL")
    print(format_summary(summary, source=input_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "format_summary",
    "load_results",
    "main",
    "summarize_results",
]
