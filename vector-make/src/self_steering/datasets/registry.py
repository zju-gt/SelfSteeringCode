"""Extensible dataset registry with lazy remote loading."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

from self_steering.datasets.adapters import (
    adapt_aime,
    adapt_math500,
    adapt_mmlu,
    adapt_multiple_choice,
)
from self_steering.datasets.types import CanonicalItem
from self_steering.utils.io import read_jsonl


Loader = Callable[[dict[str, Any]], Iterable[CanonicalItem]]
Adapter = Callable[..., CanonicalItem]


class DatasetRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, Loader] = {}

    def register(self, name: str, loader: Loader) -> None:
        if name in self._loaders:
            raise ValueError(f"dataset is already registered: {name}")
        self._loaders[name] = loader

    def load(self, name: str, config: dict[str, Any]) -> list[CanonicalItem]:
        if name not in self._loaders:
            raise KeyError(f"unknown dataset: {name}")
        items = list(self._loaders[name](config))
        seen: set[str] = set()
        for item in items:
            item.validate()
            if item.item_id in seen:
                raise ValueError(f"duplicate item_id in {name}: {item.item_id}")
            seen.add(item.item_id)
        return items

    def names(self) -> list[str]:
        return sorted(self._loaders)

    @classmethod
    def default(cls) -> "DatasetRegistry":
        registry = cls()
        registry.register(
            "mmlu", partial(_load_source, dataset="mmlu", adapter=adapt_mmlu)
        )
        registry.register(
            "math500", partial(_load_source, dataset="math500", adapter=adapt_math500)
        )
        for name in ("aime2024", "aime2025", "aime2026"):
            registry.register(
                name,
                partial(
                    _load_source,
                    dataset=name,
                    adapter=partial(adapt_aime, dataset=name),
                ),
            )
        registry.register(
            "arc_c",
            partial(
                _load_source,
                dataset="arc_c",
                adapter=partial(adapt_multiple_choice, dataset="arc_c"),
            ),
        )
        registry.register(
            "obqa",
            partial(
                _load_source,
                dataset="obqa",
                adapter=partial(adapt_multiple_choice, dataset="obqa"),
            ),
        )
        return registry


def _load_source(
    config: dict[str, Any],
    *,
    dataset: str,
    adapter: Adapter,
) -> Iterable[CanonicalItem]:
    split = str(config.get("split", "test"))
    local_path = config.get("local_path")
    if local_path:
        rows = list(read_jsonl(Path(local_path)))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face datasets is required for remote dataset sources"
            ) from exc
        kwargs: dict[str, Any] = {
            "path": config["path"],
            "split": split,
        }
        if config.get("name"):
            kwargs["name"] = config["name"]
        if config.get("cache_dir"):
            kwargs["cache_dir"] = config["cache_dir"]
        rows = load_dataset(**kwargs)

    for index, row in enumerate(rows):
        record = dict(row)
        if {
            "item_id",
            "dataset",
            "prompt",
            "gold_answer",
            "answer_type",
        }.issubset(record):
            yield CanonicalItem.from_dict(record)
        else:
            yield adapter(record, split=split, index=index)
