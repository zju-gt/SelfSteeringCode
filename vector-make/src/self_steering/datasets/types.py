"""Canonical benchmark item shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CanonicalItem:
    item_id: str
    dataset: str
    split: str
    prompt: str
    gold_answer: str
    answer_type: Literal["choice", "math"]
    choices: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must be non-empty")
        if not self.dataset.strip():
            raise ValueError("dataset must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"prompt must be non-empty for {self.item_id}")
        if self.answer_type not in {"choice", "math"}:
            raise ValueError(f"unsupported answer_type: {self.answer_type!r}")
        if not self.gold_answer.strip():
            raise ValueError(f"gold_answer must be non-empty for {self.item_id}")

        if self.answer_type == "choice":
            if not self.choices:
                raise ValueError(f"choice item requires choices: {self.item_id}")
            if any(not key.isalpha() or key != key.upper() for key in self.choices):
                raise ValueError(
                    f"choice labels must be uppercase letters: {self.item_id}"
                )
            if self.gold_answer.upper() not in self.choices:
                raise ValueError(
                    f"gold answer is not present in choices: {self.item_id}"
                )

        if (
            self.dataset.lower().startswith("aime")
            or self.metadata.get("competition") == "AIME"
        ):
            try:
                answer = int(self.gold_answer)
            except ValueError as exc:
                raise ValueError(
                    f"AIME answer must be an integer: {self.item_id}"
                ) from exc
            if not 0 <= answer <= 999:
                raise ValueError(f"AIME answer must be in [0, 999]: {self.item_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CanonicalItem":
        item = cls(
            item_id=str(row["item_id"]),
            dataset=str(row["dataset"]),
            split=str(row.get("split", "unknown")),
            prompt=str(row["prompt"]),
            gold_answer=str(row["gold_answer"]),
            answer_type=row["answer_type"],
            choices=(
                {str(key): str(value) for key, value in row["choices"].items()}
                if row.get("choices") is not None
                else None
            ),
            metadata=dict(row.get("metadata", {})),
        )
        item.validate()
        return item
