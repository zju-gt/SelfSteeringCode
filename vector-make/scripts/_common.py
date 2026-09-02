"""Shared argument parsing for numbered experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_steering.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "model.yaml",
    PROJECT_ROOT / "configs" / "data.yaml",
    PROJECT_ROOT / "configs" / "experiment.yaml",
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help=(
            "YAML file; repeat to merge files left-to-right. Supplying any "
            "--config replaces the default model/data/experiment configs"
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Resolved dot.path=value override; repeat as needed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit items per dataset for a smoke run",
    )
    return parser


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    config_paths = args.config if args.config is not None else DEFAULT_CONFIG_PATHS
    config = load_config(config_paths, args.override)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least one")
        config["_runtime"] = {"limit": args.limit}
    return config


def print_paths(paths: Any) -> None:
    if isinstance(paths, dict):
        payload = {key: str(value) for key, value in paths.items()}
    elif isinstance(paths, list):
        payload = [str(value) for value in paths]
    else:
        payload = str(paths)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
