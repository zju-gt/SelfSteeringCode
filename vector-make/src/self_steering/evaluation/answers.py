"""Dataset-aware final-answer extraction and exact-match grading."""

from __future__ import annotations

import re


def _boxed_contents(text: str) -> list[str]:
    marker = r"\boxed{"
    results: list[str] = []
    cursor = 0
    while True:
        marker_index = text.find(marker, cursor)
        if marker_index < 0:
            return results
        start = marker_index + len(marker)
        depth = 1
        position = start
        while position < len(text) and depth:
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
            position += 1
        if depth == 0:
            results.append(text[start : position - 1].strip())
            cursor = position
        else:
            return results


def extract_answer(text: str, answer_type: str) -> str | None:
    if not text or not text.strip():
        return None
    boxes = _boxed_contents(text)
    if answer_type == "choice":
        if boxes and re.fullmatch(r"[A-Za-z]", boxes[-1].strip()):
            return boxes[-1].strip().upper()
        matches = re.findall(
            r"(?im)final\s+answer\s*:\s*(?:option\s*)?\(?([A-Za-z])\)?\s*[.)]?\s*$",
            text,
        )
        if matches:
            return matches[-1].upper()
        terminal = re.search(r"(?i)(?:^|\n)\s*\(?([A-Za-z])\)?\s*[.)]?\s*$", text)
        return terminal.group(1).upper() if terminal else None
    if answer_type != "math":
        raise ValueError(f"unsupported answer type: {answer_type!r}")
    if boxes:
        return boxes[-1]
    final_matches = re.findall(r"(?im)final\s+answer\s*:\s*(.+?)\s*$", text)
    if final_matches:
        return final_matches[-1].strip().rstrip(".")
    numeric = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*[.]?\s*$", text)
    return numeric.group(1) if numeric else None


def normalize_math_answer(answer: str) -> str:
    normalized = answer.strip().replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(",", "").replace(" ", "")
    return normalized.rstrip(".")


def is_correct(
    predicted: str | None,
    gold: str,
    *,
    dataset: str,
    answer_type: str,
) -> bool:
    if predicted is None:
        return False
    if answer_type == "choice":
        return predicted.strip().upper() == gold.strip().upper()
    predicted_normalized = normalize_math_answer(predicted)
    gold_normalized = normalize_math_answer(gold)
    if dataset.lower().startswith("aime"):
        try:
            return int(predicted_normalized) == int(gold_normalized)
        except ValueError:
            return False
    return predicted_normalized == gold_normalized

