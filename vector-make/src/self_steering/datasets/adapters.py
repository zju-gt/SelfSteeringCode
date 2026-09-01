"""Source-specific normalization into :class:`CanonicalItem`."""

from __future__ import annotations

import re
from string import ascii_uppercase
from typing import Any, Mapping, Sequence

from self_steering.datasets.types import CanonicalItem


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    raise ValueError(f"none of the required fields are present: {', '.join(names)}")


def _last_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def _prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        messages = [message for message in value if isinstance(message, Mapping)]
        user_messages = [
            message for message in messages if message.get("role") == "user"
        ]
        candidates = user_messages or messages
        if candidates and isinstance(candidates[-1].get("content"), str):
            return candidates[-1]["content"].strip()
    raise ValueError("prompt must be a string or a chat message list")


def _strip_aime_wrapper(text: str) -> str:
    prefix = (
        "Solve the following math problem step by step. Please put your final answer "
        "within \\boxed{}.\n\n"
    )
    suffix = "\n\nRemember to put your final answer within \\boxed{}."
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.strip()


def _render_choice_prompt(question: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {text}" for label, text in choices.items())
    return f"{question.strip()}\n\nChoices:\n{rendered}"


def adapt_mmlu(row: Mapping[str, Any], *, split: str, index: int) -> CanonicalItem:
    question = str(_first_present(row, ["question", "prompt"]))
    raw_choices = _first_present(row, ["choices", "options"])
    if not isinstance(raw_choices, (list, tuple)) or not raw_choices:
        raise ValueError("MMLU choices must be a non-empty list")
    if len(raw_choices) > len(ascii_uppercase):
        raise ValueError("MMLU has too many choices")
    choices = {ascii_uppercase[i]: str(text) for i, text in enumerate(raw_choices)}
    raw_answer = _first_present(row, ["answer", "answerKey"])
    if isinstance(raw_answer, int):
        if not 0 <= raw_answer < len(choices):
            raise ValueError("MMLU answer index is out of range")
        answer = ascii_uppercase[raw_answer]
    else:
        answer = str(raw_answer).strip().upper()
    subject = str(row.get("subject", "unknown"))
    item = CanonicalItem(
        item_id=str(row.get("item_id") or f"mmlu_{subject}_{split}_{index:05d}"),
        dataset="mmlu",
        split=split,
        prompt=_render_choice_prompt(question, choices),
        gold_answer=answer,
        answer_type="choice",
        choices=choices,
        metadata={"subject": subject},
    )
    item.validate()
    return item


def adapt_math500(row: Mapping[str, Any], *, split: str, index: int) -> CanonicalItem:
    problem = str(_first_present(row, ["problem", "question", "prompt"]))
    try:
        raw_answer = _first_present(row, ["answer", "gold_answer"])
        answer = str(raw_answer).strip()
    except ValueError:
        solution = str(_first_present(row, ["solution"]))
        answer = _last_boxed(solution) or ""
    unique_id = row.get("unique_id", row.get("id", f"{split}_{index:05d}"))
    metadata = {
        key: row[key]
        for key in ("subject", "level")
        if key in row and row[key] is not None
    }
    item = CanonicalItem(
        item_id=str(row.get("item_id") or f"math500_{unique_id}"),
        dataset="math500",
        split=split,
        prompt=problem,
        gold_answer=answer,
        answer_type="math",
        metadata=metadata,
    )
    item.validate()
    return item


def adapt_aime(
    row: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    index: int,
) -> CanonicalItem:
    raw_problem = _first_present(row, ["problem", "question", "prompt"])
    problem = _strip_aime_wrapper(_prompt_text(raw_problem))
    try:
        raw_answer = _first_present(
            row, ["ground_truth", "answer", "gold_answer", "label"]
        )
        answer = str(raw_answer).strip()
    except ValueError:
        solution = str(_first_present(row, ["solution", "response"]))
        answer = _last_boxed(solution) or ""
    source_id = row.get(
        "source_problem_id",
        row.get("problem_idx", row.get("id", f"{split}_{index:03d}")),
    )
    item_id = str(row.get("item_id") or source_id)
    if not item_id.lower().startswith("aime"):
        item_id = f"{dataset}_{item_id}"
    metadata = {"competition": "AIME"}
    for key in ("topic", "problem_type", "difficulty", "language"):
        if key in row and row[key] is not None:
            metadata[key] = row[key]
    item = CanonicalItem(
        item_id=item_id,
        dataset=dataset,
        split=split,
        prompt=problem,
        gold_answer=answer,
        answer_type="math",
        metadata=metadata,
    )
    item.validate()
    return item


def adapt_multiple_choice(
    row: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    index: int,
) -> CanonicalItem:
    question = str(_first_present(row, ["question", "prompt"]))
    raw_choices = _first_present(row, ["choices", "options"])
    if isinstance(raw_choices, Mapping):
        labels = raw_choices.get("label")
        texts = raw_choices.get("text")
        if not isinstance(labels, (list, tuple)) or not isinstance(
            texts, (list, tuple)
        ):
            raise ValueError("choices must contain label and text lists")
        if len(labels) != len(texts) or not labels:
            raise ValueError(
                "choices label and text lists must have equal non-zero length"
            )
        choices = {str(label).upper(): str(text) for label, text in zip(labels, texts)}
    elif isinstance(raw_choices, (list, tuple)):
        choices = {ascii_uppercase[i]: str(text) for i, text in enumerate(raw_choices)}
    else:
        raise ValueError("choices must be a mapping or list")
    answer = (
        str(_first_present(row, ["answerKey", "answer", "gold_answer"])).strip().upper()
    )
    item = CanonicalItem(
        item_id=str(
            row.get("item_id") or row.get("id") or f"{dataset}_{split}_{index:05d}"
        ),
        dataset=dataset,
        split=split,
        prompt=_render_choice_prompt(question, choices),
        gold_answer=answer,
        answer_type="choice",
        choices=choices,
        metadata={key: row[key] for key in ("fact1",) if key in row},
    )
    item.validate()
    return item
