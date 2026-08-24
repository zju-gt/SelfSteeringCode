"""Convert collected MMLU prompt pairs into evaluation JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


_CHOICE_PATTERN = re.compile(
    r"(?<![A-Z])([ABCD])(?=(?:\s|[.\)\]}:;,!?]|$))",
    re.IGNORECASE,
)


def answer_to_letter(answer: Any) -> str:
    """Normalize an MMLU answer index or choice letter to ``A``-``D``."""

    if isinstance(answer, bool):
        raise ValueError(f"Invalid MMLU answer: {answer!r}")
    if isinstance(answer, int):
        if 0 <= answer <= 3:
            return chr(ord("A") + answer)
        raise ValueError(f"MMLU answer index must be in [0, 3], got {answer!r}")
    if isinstance(answer, str):
        normalized = answer.strip().upper()
        if normalized in {"A", "B", "C", "D"}:
            return normalized
    raise ValueError(f"Invalid MMLU answer: {answer!r}")


def extract_last_choice(text: str) -> Optional[str]:
    """Extract the final standalone A/B/C/D token from generated text."""

    matches = _CHOICE_PATTERN.findall(str(text).upper())
    return matches[-1].upper() if matches else None


def mmlu_choice_match(prediction: str, reference: Optional[str]) -> Optional[float]:
    """Score a generation by comparing its final choice with the reference."""

    if reference is None:
        return None
    expected = answer_to_letter(reference)
    predicted = extract_last_choice(prediction)
    return float(predicted == expected)


def _converted_record(values: dict[str, Any], line_number: int) -> dict[str, Any]:
    positive_prompt = values.get("positive_prompt")
    if not isinstance(positive_prompt, str) or not positive_prompt.strip():
        raise ValueError("missing non-empty 'positive_prompt'")
    if "answer" not in values:
        raise ValueError("missing 'answer'")

    example_id = str(values.get("id", values.get("task_id", line_number)))
    metadata = {
        key: value
        for key, value in values.items()
        if key not in {"positive_prompt", "negative_prompt"}
    }
    return {
        "id": example_id,
        "prompt": positive_prompt,
        "reference": answer_to_letter(values["answer"]),
        "metadata": metadata,
    }


def convert_rows(input_path: str | Path, output_path: str | Path) -> int:
    """Convert collector JSONL rows and return the number of written rows."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"MMLU prompt-pair input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}"
                ) from exc
            if not isinstance(values, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {input_path}"
                )
            try:
                record = _converted_record(values, line_number)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MMLU record on line {line_number} of {input_path}: {exc}"
                ) from exc
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Collector prompt-pair JSONL")
    parser.add_argument("--output", required=True, help="Evaluation JSONL output")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    count = convert_rows(args.input, args.output)
    print(f"Wrote {count} MMLU evaluation examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "answer_to_letter",
    "convert_rows",
    "extract_last_choice",
    "mmlu_choice_match",
]
