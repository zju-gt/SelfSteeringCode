"""Composable services for every numbered Self-Steering MVP stage."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file

from self_steering.datasets.filtering import demand_memberships, demand_slice
from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.scoring import annotations_to_wide, score_items
from self_steering.datasets.types import CanonicalItem
from self_steering.evaluation.answers import extract_answer, is_correct
from self_steering.evaluation.metrics import accuracy_by_alpha, specificity_matrix
from self_steering.models.generation import generate_with_optional_steering
from self_steering.models.loader import resolve_decoder_layer
from self_steering.prompts.serialization import answer_instruction, serialize_reasoning_prefill
from self_steering.prompts.templates import CAPABILITY_PROMPTS, GENERIC_PROMPT
from self_steering.utils.io import (
    append_jsonl,
    atomic_save_json,
    atomic_save_tensors,
    read_jsonl,
    write_jsonl,
)
from self_steering.utils.manifest import build_manifest
from self_steering.vectors.capture import capture_prompt_contrast
from self_steering.vectors.extract import aggregate_capability_vectors
from self_steering.vectors.similarity import cosine_similarity_matrix, vector_coherence
from self_steering.vectors.storage import load_vector_library, save_vector_library


def _directories(config: dict[str, Any]) -> tuple[Path, Path]:
    paths = config["experiment"].get("paths", {})
    return Path(paths.get("data_dir", "data")), Path(paths.get("outputs_dir", "outputs"))


def _limit(config: dict[str, Any], rows: list[Any]) -> list[Any]:
    limit = config.get("_runtime", {}).get("limit")
    return rows[: int(limit)] if limit is not None else rows


def _write_manifest(config: dict[str, Any], outputs_dir: Path, stage: str) -> None:
    path = outputs_dir / "manifests" / f"{stage}.json"
    atomic_save_json(path, build_manifest(config, run_id=stage))


def prepare_data(config: dict, registry: DatasetRegistry) -> dict[str, Path]:
    data_dir, outputs_dir = _directories(config)
    names = ["mmlu", *config["data"].get("enabled_steering_datasets", [])]
    paths: dict[str, Path] = {}
    for name in dict.fromkeys(names):
        source = dict(config["data"].get("sources", {}).get(name, {}))
        source.setdefault("cache_dir", config["data"].get("cache_dir"))
        items = _limit(config, registry.load(name, source))
        destination = data_dir / "processed" / f"{name}.jsonl"
        write_jsonl(destination, (item.to_dict() for item in items))
        paths[name] = destination
    _write_manifest(config, outputs_dir, "00_prepare_data")
    return paths


def score_demands(
    config: dict,
    label_fn: Callable[[CanonicalItem, str], dict[str, Any]],
    *,
    expected_key_fn: Callable[[CanonicalItem, str], tuple[str, str, str, str, str]] | None = None,
) -> dict[str, Path]:
    data_dir, outputs_dir = _directories(config)
    dimensions = list(config["experiment"]["capabilities"])
    names = ["mmlu", *config["data"].get("enabled_steering_datasets", [])]
    result: dict[str, Path] = {}
    max_workers = int(config["experiment"].get("annotation", {}).get("max_workers", 1))
    for name in dict.fromkeys(names):
        items = [CanonicalItem.from_dict(row) for row in read_jsonl(data_dir / "processed" / f"{name}.jsonl")]
        long_path = data_dir / "scored" / f"{name}_delean_long.jsonl"
        score_items(
            items,
            dimensions,
            long_path,
            label_fn,
            max_workers=max_workers,
            expected_key_fn=expected_key_fn,
        )
        wide = annotations_to_wide(items, read_jsonl(long_path), dimensions=dimensions)
        wide_path = data_dir / "scored" / f"{name}_with_4d_demands.jsonl"
        write_jsonl(wide_path, wide)
        result[f"{name}_long"] = long_path
        result[f"{name}_wide"] = wide_path
    _write_manifest(config, outputs_dir, "01_score_demands")
    return result


def prepare_items(config: dict) -> dict[str, Path]:
    data_dir, outputs_dir = _directories(config)
    experiment = config["experiment"]
    capabilities = list(experiment["capabilities"])
    high = int(experiment["high_demand_threshold"])
    low = int(experiment["low_demand_threshold"])
    result: dict[str, Path] = {}
    mmlu = list(read_jsonl(data_dir / "scored" / "mmlu_with_4d_demands.jsonl"))
    for capability in capabilities:
        selected = demand_slice(mmlu, capability, "high", high, low)
        path = data_dir / "processed" / "extraction" / f"{capability}.jsonl"
        write_jsonl(path, selected)
        result[f"extraction_{capability}"] = path

    for dataset in config["data"].get("enabled_steering_datasets", []):
        rows = list(read_jsonl(data_dir / "scored" / f"{dataset}_with_4d_demands.jsonl"))
        selected: list[dict[str, Any]] = []
        for row in rows:
            memberships = demand_memberships(row, capabilities, high, low)
            if memberships:
                enriched = dict(row)
                enriched["demand_memberships"] = memberships
                selected.append(enriched)
        path = data_dir / "processed" / "evaluation" / f"{dataset}.jsonl"
        write_jsonl(path, selected)
        result[f"evaluation_{dataset}"] = path
    _write_manifest(config, outputs_dir, "02_prepare_items")
    return result


def _tensor_input(token_ids: Any) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        return token_ids.unsqueeze(0) if token_ids.ndim == 1 else token_ids
    return torch.tensor([token_ids], dtype=torch.long)


def _safe_id(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)


def capture_contrasts(config: dict, model: Any, tokenizer: Any) -> list[Path]:
    data_dir, outputs_dir = _directories(config)
    layer = resolve_decoder_layer(model, int(config["experiment"]["target_layer"]))
    written: list[Path] = []
    for capability in config["experiment"]["capabilities"]:
        for row in read_jsonl(data_dir / "processed" / "extraction" / f"{capability}.jsonl"):
            item = CanonicalItem.from_dict(row)
            path = outputs_dir / "activations" / capability / f"{_safe_id(item.item_id)}.safetensors"
            if path.exists():
                written.append(path)
                continue
            instruction = answer_instruction(item.answer_type, item.dataset)
            generic_ids = _tensor_input(
                serialize_reasoning_prefill(tokenizer, GENERIC_PROMPT, item.prompt, instruction)
            )
            capability_ids = _tensor_input(
                serialize_reasoning_prefill(
                    tokenizer,
                    CAPABILITY_PROMPTS[capability],
                    item.prompt,
                    instruction,
                )
            )
            delta = capture_prompt_contrast(model, layer, generic_ids, capability_ids)
            atomic_save_tensors(
                path,
                {"delta": delta},
                metadata={"item_id": item.item_id, "capability": capability},
            )
            written.append(path)
    _write_manifest(config, outputs_dir, "03_capture_contrasts")
    return written


def extract_vectors(config: dict) -> Path:
    _, outputs_dir = _directories(config)
    deltas: dict[str, torch.Tensor] = {}
    for capability in config["experiment"]["capabilities"]:
        paths = sorted((outputs_dir / "activations" / capability).glob("*.safetensors"))
        if not paths:
            raise ValueError(f"no activation shards found for {capability}")
        deltas[capability] = torch.stack([load_file(str(path))["delta"] for path in paths])
    vectors = aggregate_capability_vectors(deltas)
    tensor_path = outputs_dir / "vectors" / "capability_vectors.safetensors"
    metadata_path = outputs_dir / "vectors" / "capability_vectors.json"
    save_vector_library(
        tensor_path,
        metadata_path,
        vectors,
        metadata={
            "layer": int(config["experiment"]["target_layer"]),
            "scaling": "mean_norm",
            "counts": {name: int(values.shape[0]) for name, values in deltas.items()},
        },
    )
    _write_manifest(config, outputs_dir, "04_extract_vectors")
    return tensor_path


def analyze_similarity(config: dict) -> dict[str, Path]:
    _, outputs_dir = _directories(config)
    vectors, _ = load_vector_library(
        outputs_dir / "vectors" / "capability_vectors.safetensors",
        outputs_dir / "vectors" / "capability_vectors.json",
    )
    units = {name: forms["unit"] for name, forms in vectors.items()}
    coherence: dict[str, float] = {}
    for capability in units:
        shards = sorted((outputs_dir / "activations" / capability).glob("*.safetensors"))
        values = torch.stack([load_file(str(path))["delta"] for path in shards])
        coherence[capability] = vector_coherence(values)
    similarity_path = outputs_dir / "metrics" / "vector_similarity.json"
    coherence_path = outputs_dir / "metrics" / "vector_coherence.json"
    atomic_save_json(similarity_path, cosine_similarity_matrix(units))
    atomic_save_json(coherence_path, coherence)
    _write_manifest(config, outputs_dir, "05_analyze_similarity")
    return {"similarity": similarity_path, "coherence": coherence_path}


def run_steering(config: dict, model: Any, tokenizer: Any) -> Path:
    data_dir, outputs_dir = _directories(config)
    vectors, _ = load_vector_library(
        outputs_dir / "vectors" / "capability_vectors.safetensors",
        outputs_dir / "vectors" / "capability_vectors.json",
    )
    scaling = str(config["experiment"].get("vector_scaling", "mean_norm"))
    form = {"mean_norm": "steering", "unit": "unit", "raw": "raw"}[scaling]
    layer = resolve_decoder_layer(model, int(config["experiment"]["target_layer"]))
    output_path = outputs_dir / "generations" / "steering.jsonl"
    existing = list(read_jsonl(output_path)) if output_path.exists() else []
    completed = {
        (row["dataset"], row["item_id"], row["steering_capability"], float(row["alpha"]))
        for row in existing
        if row.get("status") == "ok"
    }
    baseline_cache = {
        (row["dataset"], row["item_id"]): row["raw_output"]
        for row in existing
        if row.get("status") == "ok" and float(row["alpha"]) == 0.0
    }
    for dataset in config["data"].get("enabled_steering_datasets", []):
        for row in read_jsonl(data_dir / "processed" / "evaluation" / f"{dataset}.jsonl"):
            item = CanonicalItem.from_dict(row)
            final_instruction = answer_instruction(item.answer_type, item.dataset)
            input_ids = _tensor_input(
                serialize_reasoning_prefill(
                    tokenizer,
                    GENERIC_PROMPT,
                    item.prompt,
                    final_instruction,
                )
            )
            for capability in config["experiment"]["capabilities"]:
                for raw_alpha in config["experiment"]["alphas"]:
                    alpha = float(raw_alpha)
                    key = (dataset, item.item_id, capability, alpha)
                    if key in completed:
                        continue
                    cache_key = (dataset, item.item_id)
                    if alpha == 0.0 and cache_key in baseline_cache:
                        output = baseline_cache[cache_key]
                    else:
                        output = generate_with_optional_steering(
                            model,
                            tokenizer,
                            input_ids,
                            layer,
                            vector=None if alpha == 0.0 else vectors[capability][form],
                            alpha=alpha,
                            max_new_tokens=int(config["model"].get("max_new_tokens", 2048)),
                        )
                        if alpha == 0.0:
                            baseline_cache[cache_key] = output
                    predicted = extract_answer(output, item.answer_type)
                    append_jsonl(
                        output_path,
                        {
                            "dataset": dataset,
                            "item_id": item.item_id,
                            "steering_capability": capability,
                            "alpha": alpha,
                            "demand_memberships": row.get("demand_memberships", {}),
                            "raw_output": output,
                            "predicted_answer": predicted,
                            "gold_answer": item.gold_answer,
                            "answer_type": item.answer_type,
                            "correct": is_correct(
                                predicted,
                                item.gold_answer,
                                dataset=dataset,
                                answer_type=item.answer_type,
                            ),
                            "status": "ok",
                        },
                    )
    _write_manifest(config, outputs_dir, "06_run_steering")
    return output_path


def score_generations(config: dict) -> dict[str, Path]:
    _, outputs_dir = _directories(config)
    generation_path = outputs_dir / "generations" / "steering.jsonl"
    rows = [row for row in read_jsonl(generation_path) if row.get("status") == "ok"]
    report: dict[str, Any] = {"datasets": {}}
    csv_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_report: dict[str, Any] = {"by_capability": {}, "specificity": {}}
        for capability in sorted({row["steering_capability"] for row in dataset_rows}):
            capability_rows = [
                row for row in dataset_rows if row["steering_capability"] == capability
            ]
            effects = accuracy_by_alpha(capability_rows)
            dataset_report["by_capability"][capability] = effects
            for alpha, values in effects.items():
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "steering_capability": capability,
                        "alpha": alpha,
                        **values,
                    }
                )
        for raw_alpha in config["experiment"]["alphas"]:
            alpha = float(raw_alpha)
            if alpha == 0.0:
                continue
            dataset_report["specificity"][str(alpha)] = specificity_matrix(
                dataset_rows, alpha=alpha
            )
        report["datasets"][dataset] = dataset_report
    json_path = outputs_dir / "metrics" / "steering_metrics.json"
    csv_path = outputs_dir / "metrics" / "steering_metrics.csv"
    atomic_save_json(json_path, report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "steering_capability",
            "alpha",
            "count",
            "accuracy",
            "delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_manifest(config, outputs_dir, "07_score_generations")
    return {"json": json_path, "csv": csv_path}

