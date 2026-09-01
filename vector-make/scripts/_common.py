"""Shared argument parsing for numbered experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_steering.config import load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        required=True,
        help="YAML file; repeat to merge files left-to-right",
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
    config = load_config(args.config, args.override)
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
