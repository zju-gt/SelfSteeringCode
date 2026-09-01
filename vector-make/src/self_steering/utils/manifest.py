"""Run-manifest construction."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable


def build_manifest(
    config: dict[str, Any],
    *,
    run_id: str,
    dataset_fingerprints: dict[str, str | None] | None = None,
    rubric_hashes: dict[str, str] | None = None,
    prompt_hashes: dict[str, str] | None = None,
    packages: Iterable[str] = ("torch", "transformers", "datasets", "openai"),
) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": versions,
        "config": config,
        "dataset_fingerprints": dataset_fingerprints or {},
        "rubric_hashes": rubric_hashes or {},
        "prompt_hashes": prompt_hashes or {},
    }
