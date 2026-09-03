"""Loading and validation for experiment YAML configuration."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


SUPPORTED_DATASETS = {
    "mmlu",
    "math500",
    "aime2024",
    "aime2025",
    "aime2026",
    "arc_c",
    "obqa",
}
SUPPORTED_VECTOR_SCALINGS = {"raw", "unit", "mean_norm"}
MVP_CAPABILITIES = {"QLl", "QLq", "CL", "MCr"}


class ConfigError(ValueError):
    """Raised when a resolved experiment configuration is invalid."""


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must use key=value syntax: {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ConfigError(f"override has an empty key: {override!r}")
    value = yaml.safe_load(raw_value)
    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set nested key below non-mapping {part!r}")
        current = child
    current[parts[-1]] = value


def load_config(
    paths: Iterable[Path | str],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Merge YAML files left-to-right, apply overrides, and validate."""

    resolved: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"configuration file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            raise ConfigError(f"configuration root must be a mapping: {path}")
        resolved = _deep_merge(resolved, loaded)

    for override in overrides or []:
        _apply_override(resolved, override)

    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate stable cross-stage configuration invariants."""

    model = config.get("model")
    data = config.get("data")
    experiment = config.get("experiment")
    if not isinstance(model, dict):
        raise ConfigError("missing model configuration")
    if not isinstance(data, dict):
        raise ConfigError("missing data configuration")
    if not isinstance(experiment, dict):
        raise ConfigError("missing experiment configuration")

    num_layers = model.get("num_hidden_layers")
    target_layer = experiment.get("target_layer")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ConfigError("model.num_hidden_layers must be a positive integer")
    revision = model.get("revision")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().lower() in {"main", "master"}
    ):
        raise ConfigError(
            "model.revision must be pinned to an immutable commit or release tag"
        )
    if not isinstance(target_layer, int) or not 0 <= target_layer < num_layers:
        raise ConfigError(
            f"experiment.target_layer must be in [0, {num_layers - 1}], got {target_layer!r}"
        )

    capabilities = experiment.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or len(capabilities) != len(MVP_CAPABILITIES)
        or not all(isinstance(name, str) for name in capabilities)
        or set(capabilities) != MVP_CAPABILITIES
    ):
        raise ConfigError(
            "experiment.capabilities must contain each of QLl, QLq, CL, and MCr exactly once"
        )

    enabled = data.get("enabled_steering_datasets", [])
    if not isinstance(enabled, list) or not all(
        isinstance(name, str) for name in enabled
    ):
        raise ConfigError("data.enabled_steering_datasets must be a list of names")
    unknown = sorted(set(enabled) - (SUPPORTED_DATASETS - {"mmlu"}))
    if unknown:
        raise ConfigError(f"unknown dataset(s): {', '.join(unknown)}")

    high = experiment.get("high_demand_threshold")
    low = experiment.get("low_demand_threshold")
    if not isinstance(high, int) or not 0 <= high <= 5:
        raise ConfigError(
            "experiment.high_demand_threshold must be an integer from 0 to 5"
        )
    if not isinstance(low, int) or not 0 <= low <= 5:
        raise ConfigError(
            "experiment.low_demand_threshold must be an integer from 0 to 5"
        )
    if low >= high:
        raise ConfigError(
            "low_demand_threshold must be smaller than high_demand_threshold"
        )

    scaling = experiment.get("vector_scaling")
    if scaling not in SUPPORTED_VECTOR_SCALINGS:
        raise ConfigError(
            f"experiment.vector_scaling must be one of {sorted(SUPPORTED_VECTOR_SCALINGS)}"
        )

    alphas = experiment.get("alphas")
    if (
        not isinstance(alphas, list)
        or not alphas
        or not all(
            isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            for alpha in alphas
        )
    ):
        raise ConfigError("experiment.alphas must be a non-empty list of numbers")
    if not any(float(alpha) == 0.0 for alpha in alphas):
        raise ConfigError("experiment.alphas must include zero for the baseline")
    normalized_alphas = [float(alpha) for alpha in alphas]
    if not all(math.isfinite(alpha) for alpha in normalized_alphas):
        raise ConfigError("experiment.alphas must contain only finite numbers")
    if len(normalized_alphas) != len(set(normalized_alphas)):
        raise ConfigError("experiment.alphas must contain unique values")

    max_new_tokens = model.get("max_new_tokens", 2048)
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ConfigError("model.max_new_tokens must be a positive integer")

    generation = experiment.get("generation", {})
    if not isinstance(generation, dict):
        raise ConfigError("experiment.generation must be a mapping")
    batch_size = generation.get("batch_size", 1)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ConfigError("experiment.generation.batch_size must be a positive integer")

    annotation = experiment.get("annotation", {})
    if not isinstance(annotation, dict):
        raise ConfigError("experiment.annotation must be a mapping")
    for name, default in (("max_workers", 1), ("max_attempts", 1)):
        value = annotation.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"experiment.annotation.{name} must be a positive integer")
    backoff = annotation.get("initial_backoff_seconds", 1.0)
    if (
        not isinstance(backoff, (int, float))
        or isinstance(backoff, bool)
        or not math.isfinite(float(backoff))
        or float(backoff) <= 0
    ):
        raise ConfigError(
            "experiment.annotation.initial_backoff_seconds must be a positive finite number"
        )
