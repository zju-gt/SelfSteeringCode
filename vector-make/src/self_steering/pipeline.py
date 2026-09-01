"""Composable services for every numbered Self-Steering MVP stage."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors.torch import load_file

from self_steering.datasets.filtering import demand_memberships, demand_slice
from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.scoring import (
    annotations_to_wide,
    current_annotation_rows,
    score_items,
)
from self_steering.datasets.types import CanonicalItem
from self_steering.evaluation.answers import extract_answer, is_correct
from self_steering.evaluation.metrics import (
    accuracy_by_alpha,
    accuracy_by_demand_slice,
    diagonal_dominance,
    paired_alpha_rows,
    specificity_matrix,
)
from self_steering.models.generation import generate_with_optional_steering
from self_steering.models.loader import resolve_decoder_layer
from self_steering.prompts.serialization import (
    answer_instruction,
    serialize_reasoning_prefill,
)
from self_steering.prompts.templates import CAPABILITY_PROMPTS, GENERIC_PROMPT
from self_steering.utils.io import (
    append_jsonl,
    atomic_save_json,
    atomic_save_tensors,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_jsonl,
)
from self_steering.utils.manifest import build_manifest
from self_steering.vectors.capture import capture_prompt_contrast
from self_steering.vectors.extract import aggregate_capability_vectors
from self_steering.vectors.similarity import cosine_similarity_matrix, vector_coherence
from self_steering.vectors.storage import load_vector_library, save_vector_library


def _directories(config: dict[str, Any]) -> tuple[Path, Path]:
    paths = config["experiment"].get("paths", {})
    return Path(paths.get("data_dir", "data")), Path(
        paths.get("outputs_dir", "outputs")
    )


def _limit(config: dict[str, Any], rows: list[Any]) -> list[Any]:
    limit = config.get("_runtime", {}).get("limit")
    return rows[: int(limit)] if limit is not None else rows


def _write_manifest(config: dict[str, Any], outputs_dir: Path, stage: str) -> None:
    path = outputs_dir / "manifests" / f"{stage}.json"
    atomic_save_json(path, build_manifest(config, run_id=stage))


def capture_artifact_id(config: dict[str, Any]) -> str:
    identity = {
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "dtype": config["model"].get("dtype"),
        "target_layer": config["experiment"]["target_layer"],
        "generic_prompt": GENERIC_PROMPT,
        "capability_prompts": CAPABILITY_PROMPTS,
    }
    return sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:16]


def generation_key(row: dict[str, Any]) -> tuple[str, str, str, str, float]:
    memberships = json.dumps(
        row.get("demand_memberships", {}), sort_keys=True, ensure_ascii=False
    )
    item_identity = sha256_text(
        f"{row.get('prompt', '')}\0{row.get('gold_answer', '')}\0{memberships}"
    )
    return (
        str(row["dataset"]),
        str(row["item_id"]),
        item_identity,
        str(row["steering_capability"]),
        float(row["alpha"]),
    )


def steering_artifact_id(config: dict[str, Any]) -> str:
    _, outputs_dir = _directories(config)
    vector_path = (
        outputs_dir
        / "vectors"
        / capture_artifact_id(config)
        / "capability_vectors.safetensors"
    )
    identity = {
        "capture_artifact_id": capture_artifact_id(config),
        "vector_sha256": sha256_file(vector_path),
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "max_new_tokens": config["model"].get("max_new_tokens"),
        "vector_scaling": config["experiment"].get("vector_scaling"),
        "alphas": config["experiment"].get("alphas"),
        "enabled_steering_datasets": config["data"].get(
            "enabled_steering_datasets", []
        ),
        "generic_prompt": GENERIC_PROMPT,
        "answer_instructions": {
            "choice": answer_instruction("choice", "mmlu"),
            "math": answer_instruction("math", "math500"),
            "aime": answer_instruction("math", "aime2024"),
        },
    }
    return sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:16]


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
    expected_key_fn: Callable[[CanonicalItem, str], tuple[str, str, str, str, str]]
    | None = None,
) -> dict[str, Path]:
    data_dir, outputs_dir = _directories(config)
    dimensions = list(config["experiment"]["capabilities"])
    names = ["mmlu", *config["data"].get("enabled_steering_datasets", [])]
    result: dict[str, Path] = {}
    max_workers = int(config["experiment"].get("annotation", {}).get("max_workers", 1))
    for name in dict.fromkeys(names):
        items = _limit(
            config,
            [
                CanonicalItem.from_dict(row)
                for row in read_jsonl(data_dir / "processed" / f"{name}.jsonl")
            ],
        )
        long_path = data_dir / "scored" / f"{name}_delean_long.jsonl"
        score_items(
            items,
            dimensions,
            long_path,
            label_fn,
            max_workers=max_workers,
            expected_key_fn=expected_key_fn,
        )
        long_rows = list(read_jsonl(long_path))
        if expected_key_fn is not None:
            expected_keys = {
                expected_key_fn(item, dimension)
                for item in items
                for dimension in dimensions
            }
            long_rows = current_annotation_rows(long_rows, expected_keys)
        wide = annotations_to_wide(items, long_rows, dimensions=dimensions)
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
        rows = list(
            read_jsonl(data_dir / "scored" / f"{dataset}_with_4d_demands.jsonl")
        )
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
    root = outputs_dir / "activations" / capture_artifact_id(config)
    written: list[Path] = []
    indexed: dict[str, list[str]] = {}
    for capability in config["experiment"]["capabilities"]:
        indexed[capability] = []
        rows = list(
            read_jsonl(data_dir / "processed" / "extraction" / f"{capability}.jsonl")
        )
        for row in _limit(config, rows):
            item = CanonicalItem.from_dict(row)
            item_hash = sha256_text(f"{item.prompt}\0{item.gold_answer}")[:12]
            path = (
                root / capability / f"{_safe_id(item.item_id)}__{item_hash}.safetensors"
            )
            indexed[capability].append(str(path.relative_to(root)).replace("\\", "/"))
            if path.exists():
                written.append(path)
                continue
            instruction = answer_instruction(item.answer_type, item.dataset)
            generic_ids = _tensor_input(
                serialize_reasoning_prefill(
                    tokenizer, GENERIC_PROMPT, item.prompt, instruction
                )
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
    atomic_save_json(
        root / "index.json",
        {"capture_artifact_id": capture_artifact_id(config), "shards": indexed},
    )
    _write_manifest(config, outputs_dir, "03_capture_contrasts")
    return written


def extract_vectors(config: dict) -> Path:
    _, outputs_dir = _directories(config)
    capture_id = capture_artifact_id(config)
    activation_root = outputs_dir / "activations" / capture_id
    index = json.loads((activation_root / "index.json").read_text(encoding="utf-8"))
    deltas: dict[str, torch.Tensor] = {}
    for capability in config["experiment"]["capabilities"]:
        paths = [
            activation_root / relative
            for relative in index["shards"].get(capability, [])
        ]
        if not paths:
            raise ValueError(f"no activation shards found for {capability}")
        deltas[capability] = torch.stack(
            [load_file(str(path))["delta"] for path in paths]
        )
    vectors = aggregate_capability_vectors(deltas)
    vector_root = outputs_dir / "vectors" / capture_id
    tensor_path = vector_root / "capability_vectors.safetensors"
    metadata_path = vector_root / "capability_vectors.json"
    save_vector_library(
        tensor_path,
        metadata_path,
        vectors,
        metadata={
            "layer": int(config["experiment"]["target_layer"]),
            "capture_artifact_id": capture_id,
            "scaling": "mean_norm",
            "counts": {name: int(values.shape[0]) for name, values in deltas.items()},
        },
    )
    _write_manifest(config, outputs_dir, "04_extract_vectors")
    return tensor_path


def analyze_similarity(config: dict) -> dict[str, Path]:
    _, outputs_dir = _directories(config)
    capture_id = capture_artifact_id(config)
    vector_root = outputs_dir / "vectors" / capture_id
    activation_root = outputs_dir / "activations" / capture_id
    index = json.loads((activation_root / "index.json").read_text(encoding="utf-8"))
    vectors, _ = load_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
    )
    units = {name: forms["unit"] for name, forms in vectors.items()}
    coherence: dict[str, float] = {}
    for capability in units:
        shards = [
            activation_root / relative
            for relative in index["shards"].get(capability, [])
        ]
        values = torch.stack([load_file(str(path))["delta"] for path in shards])
        coherence[capability] = vector_coherence(values)
    similarity_path = outputs_dir / "metrics" / f"{capture_id}_vector_similarity.json"
    coherence_path = outputs_dir / "metrics" / f"{capture_id}_vector_coherence.json"
    atomic_save_json(similarity_path, cosine_similarity_matrix(units))
    atomic_save_json(coherence_path, coherence)
    _write_manifest(config, outputs_dir, "05_analyze_similarity")
    return {"similarity": similarity_path, "coherence": coherence_path}


def run_steering(config: dict, model: Any, tokenizer: Any) -> Path:
    data_dir, outputs_dir = _directories(config)
    capture_id = capture_artifact_id(config)
    vector_root = outputs_dir / "vectors" / capture_id
    vectors, _ = load_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
    )
    scaling = str(config["experiment"].get("vector_scaling", "mean_norm"))
    form = {"mean_norm": "steering", "unit": "unit", "raw": "raw"}[scaling]
    layer = resolve_decoder_layer(model, int(config["experiment"]["target_layer"]))
    output_path = outputs_dir / "generations" / f"{steering_artifact_id(config)}.jsonl"
    existing = list(read_jsonl(output_path)) if output_path.exists() else []
    completed = {generation_key(row) for row in existing if row.get("status") == "ok"}
    baseline_cache = {
        generation_key(row)[:3]: row["raw_output"]
        for row in existing
        if row.get("status") == "ok" and float(row["alpha"]) == 0.0
    }
    for dataset in config["data"].get("enabled_steering_datasets", []):
        rows = list(
            read_jsonl(data_dir / "processed" / "evaluation" / f"{dataset}.jsonl")
        )
        for row in _limit(config, rows):
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
                    key_record = {
                        "dataset": dataset,
                        "item_id": item.item_id,
                        "prompt": item.prompt,
                        "gold_answer": item.gold_answer,
                        "steering_capability": capability,
                        "alpha": alpha,
                    }
                    key = generation_key(key_record)
                    if key in completed:
                        continue
                    cache_key = key[:3]
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
                            max_new_tokens=int(
                                config["model"].get("max_new_tokens", 2048)
                            ),
                        )
                        if alpha == 0.0:
                            baseline_cache[cache_key] = output
                    predicted = extract_answer(output, item.answer_type)
                    append_jsonl(
                        output_path,
                        {
                            "dataset": dataset,
                            "item_id": item.item_id,
                            "prompt": item.prompt,
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
    run_id = steering_artifact_id(config)
    generation_path = outputs_dir / "generations" / f"{run_id}.jsonl"
    rows = [row for row in read_jsonl(generation_path) if row.get("status") == "ok"]
    report: dict[str, Any] = {"datasets": {}}
    csv_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_report: dict[str, Any] = {
            "by_capability": {},
            "demand_slices": {},
            "population": {},
            "specificity": {},
            "diagonal_dominance": {},
        }
        paired_dataset_rows: list[Mapping[str, Any]] = []
        expected_alphas = [float(alpha) for alpha in config["experiment"]["alphas"]]
        for capability in sorted({row["steering_capability"] for row in dataset_rows}):
            raw_capability_rows = [
                row for row in dataset_rows if row["steering_capability"] == capability
            ]
            capability_rows = paired_alpha_rows(raw_capability_rows, expected_alphas)
            if not capability_rows:
                raise ValueError(
                    f"no complete alpha population for {dataset}/{capability}"
                )
            paired_dataset_rows.extend(capability_rows)
            raw_items = {str(row["item_id"]) for row in raw_capability_rows}
            paired_items = {str(row["item_id"]) for row in capability_rows}
            dataset_report["population"][capability] = {
                "complete_items": len(paired_items),
                "excluded_incomplete_items": len(raw_items - paired_items),
            }
            effects = accuracy_by_alpha(capability_rows)
            dataset_report["by_capability"][capability] = effects
            dataset_report["demand_slices"][capability] = {}
            for demand in config["experiment"]["capabilities"]:
                slices = {
                    slice_name: accuracy_by_demand_slice(
                        capability_rows, demand, slice_name
                    )
                    for slice_name in ("high", "low")
                }
                if any(slices.values()):
                    dataset_report["demand_slices"][capability][demand] = slices
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
            matrix = specificity_matrix(paired_dataset_rows, alpha=alpha)
            dataset_report["specificity"][str(alpha)] = matrix
            if matrix and any(row_name in row for row_name, row in matrix.items()):
                dataset_report["diagonal_dominance"][str(alpha)] = diagonal_dominance(
                    matrix
                )
        report["datasets"][dataset] = dataset_report
    json_path = outputs_dir / "metrics" / f"{run_id}_steering_metrics.json"
    csv_path = outputs_dir / "metrics" / f"{run_id}_steering_metrics.csv"
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
