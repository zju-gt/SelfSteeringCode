"""Output-path helpers for reproducible evaluation runs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
DEFAULT_RESULTS_DIR = Path("artifacts/mmlu_preliminary")


def resolve_results_path(
    output: Optional[PathLike] = None,
    output_dir: PathLike = DEFAULT_RESULTS_DIR,
    now: Optional[datetime] = None,
) -> Path:
    """Resolve an explicit or timestamped JSONL output path.

    Explicit ``output`` takes precedence over ``RESULTS_OUTPUT``.  When neither
    is provided, the filename is generated as ``results_%y%m%d_%H%M.jsonl`` in
    ``output_dir``.
    """

    if output is not None and str(output).strip():
        return Path(output)
    environment_output = os.environ.get("RESULTS_OUTPUT", "").strip()
    if environment_output:
        return Path(environment_output)

    timestamp = now or datetime.now()
    filename = timestamp.strftime("results_%y%m%d_%H%M.jsonl")
    return Path(output_dir) / filename


__all__ = ["DEFAULT_RESULTS_DIR", "resolve_results_path"]
