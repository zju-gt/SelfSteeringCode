"""Composable services for every numbered Self-Steering MVP stage."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

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
    paired_alpha_rows,
    specificity_report,
)
from self_steering.models.generation import generate_batch_with_optional_steering
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
from self_steering.utils.seed import seed_everything
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


def _resolved_model_identity(model: Any = None, tokenizer: Any = None) -> dict[str, str | None]:
    model_config = getattr(model, "config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(tokenizer_kwargs, Mapping):
        tokenizer_kwargs = {}
    return {
        "model_commit": getattr(model_config, "_commit_hash", None),
        "tokenizer_commit": tokenizer_kwargs.get("_commit_hash"),
    }


def _write_manifest(
    config: dict[str, Any],
    outputs_dir: Path,
    stage: str,
    *,
    artifacts: Mapping[str, Path] | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> None:
    fingerprints = {
        name: sha256_file(path)
        for name, path in (artifacts or {}).items()
        if Path(path).is_file()
    }
    rubric_hashes: dict[str, str] = {}
    rubrics_dir = Path(
        config["experiment"].get("paths", {}).get("rubrics_dir", "rubrics")
    )
    for capability in config["experiment"].get("capabilities", []):
        rubric_path = rubrics_dir / f"{capability}.txt"
        if rubric_path.is_file():
            rubric_hashes[str(capability)] = sha256_file(rubric_path)
    prompt_hashes = {
        "generic_prompt": sha256_text(GENERIC_PROMPT),
        "capability_prompts": sha256_text(
            json.dumps(CAPABILITY_PROMPTS, sort_keys=True, ensure_ascii=False)
        ),
    }
    identity = {
        "stage": stage,
        "config": config,
        "fingerprints": fingerprints,
        "rubric_hashes": rubric_hashes,
        "prompt_hashes": prompt_hashes,
    }
    run_id = sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:16]
    manifest = build_manifest(
        config,
        run_id=run_id,
        dataset_fingerprints=fingerprints,
        rubric_hashes=rubric_hashes,
        prompt_hashes=prompt_hashes,
        artifact_hashes=fingerprints,
        model_resolution=_resolved_model_identity(model, tokenizer),
        seed=int(config["experiment"].get("seed", 42)),
    )
    atomic_save_json(outputs_dir / "manifests" / f"{stage}.json", manifest)
    atomic_save_json(outputs_dir / "manifests" / stage / f"{run_id}.json", manifest)


def capture_artifact_id(config: dict[str, Any]) -> str:
    data_dir, _ = _directories(config)
    extraction_hashes: dict[str, str] = {}
    for capability in config["experiment"]["capabilities"]:
        path = data_dir / "processed" / "extraction" / f"{capability}.jsonl"
        if path.is_file():
            extraction_hashes[capability] = sha256_file(path)
    identity = {
        "model": config["model"].get("name"),
        "revision": config["model"].get("revision"),
        "dtype": config["model"].get("dtype"),
        "target_layer": config["experiment"]["target_layer"],
        "generic_prompt": GENERIC_PROMPT,
        "capability_prompts": CAPABILITY_PROMPTS,
        "answer_instructions": {
            "choice": answer_instruction("choice", "mmlu"),
            "math": answer_instruction("math", "math500"),
            "aime": answer_instruction("math", "aime2024"),
        },
        "serialization_contract": "chat-template/continue-final-message/reasoning-prefill-v1",
        "extraction_sha256": extraction_hashes,
    }
    return sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:16]


def _generation_item_identity(row: Mapping[str, Any]) -> str:
    identity = {
        "dataset": row.get("dataset"),
        "item_id": row.get("item_id"),
        "prompt": row.get("prompt"),
        "gold_answer": row.get("gold_answer"),
        "answer_type": row.get("answer_type"),
        "demand_memberships": row.get("demand_memberships", {}),
    }
    return sha256_text(json.dumps(identity, sort_keys=True, ensure_ascii=False))


def generation_key(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    item_identity = str(row.get("item_identity") or _generation_item_identity(row))
    return (
        str(row.get("run_id", "")),
        item_identity,
        str(row["steering_capability"]),
        float(row["alpha"]),
    )


def generation_identity_record(
    *,
    run_id: str,
    dataset: str,
    item: CanonicalItem,
    source_row: Mapping[str, Any],
    steering_capability: str,
    alpha: float,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "dataset": dataset,
        "item_id": item.item_id,
        "prompt": item.prompt,
        "gold_answer": item.gold_answer,
        "answer_type": item.answer_type,
        "choices": item.choices,
        "demand_memberships": source_row.get("demand_memberships", {}),
        "steering_capability": steering_capability,
        "alpha": float(alpha),
    }
    record["item_identity"] = _generation_item_identity(record)
    return record


def canonical_generation_rows(
    rows: list[dict[str, Any]], *, run_id: str | None = None
) -> list[dict[str, Any]]:
    """Keep the latest successful record for each generation identity."""

    latest: dict[tuple[str, str, str, float], tuple[int, dict[str, Any]]] = {}
    for index, source in enumerate(rows):
        if source.get("status") != "ok":
            continue
        if run_id is not None and source.get("run_id") not in {None, run_id}:
            continue
        row = dict(source)
        if run_id is not None:
            row.setdefault("run_id", run_id)
        row.setdefault("item_identity", _generation_item_identity(row))
        latest[generation_key(row)] = (index, row)
    return [row for _, row in sorted(latest.values(), key=lambda pair: pair[0])]


def steering_artifact_id(config: dict[str, Any]) -> str:
    data_dir, outputs_dir = _directories(config)
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
        "do_sample": config["model"].get("do_sample", False),
        "use_cache": config["model"].get("use_cache", True),
        "vector_scaling": config["experiment"].get("vector_scaling"),
        "alphas": config["experiment"].get("alphas"),
        "capabilities": config["experiment"].get("capabilities"),
        "seed": config["experiment"].get("seed", 42),
        "enabled_steering_datasets": config["data"].get(
            "enabled_steering_datasets", []
        ),
        "generic_prompt": GENERIC_PROMPT,
        "answer_instructions": {
            "choice": answer_instruction("choice", "mmlu"),
            "math": answer_instruction("math", "math500"),
            "aime": answer_instruction("math", "aime2024"),
        },
        "evaluation_sha256": {
            dataset: sha256_file(path)
            for dataset in config["data"].get("enabled_steering_datasets", [])
            if (path := data_dir / "processed" / "evaluation" / f"{dataset}.jsonl").is_file()
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
    _write_manifest(config, outputs_dir, "00_prepare_data", artifacts=paths)
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
            completed = len(long_rows)
        else:
            expected_pairs = {
                (item.item_id, dimension) for item in items for dimension in dimensions
            }
            completed_pairs = {
                (str(row.get("item_id")), str(row.get("demand")))
                for row in long_rows
                if row.get("status") == "ok"
            }
            completed = len(expected_pairs & completed_pairs)
        total = len(items) * len(dimensions)
        if completed != total:
            raise RuntimeError(
                f"{name}: annotations are incomplete ({completed}/{total}). "
                f"Progress is saved in {long_path}. Fix the API errors and rerun "
                "scripts/01_score_demands.py; successful annotations will be reused."
            )
        wide = annotations_to_wide(items, long_rows, dimensions=dimensions)
        wide_path = data_dir / "scored" / f"{name}_with_4d_demands.jsonl"
        write_jsonl(wide_path, wide)
        result[f"{name}_long"] = long_path
        result[f"{name}_wide"] = wide_path
    _write_manifest(config, outputs_dir, "01_score_demands", artifacts=result)
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
    _write_manifest(config, outputs_dir, "02_prepare_items", artifacts=result)
    return result


def _tensor_input(token_ids: Any) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        return token_ids.unsqueeze(0) if token_ids.ndim == 1 else token_ids
    return torch.tensor([token_ids], dtype=torch.long)


def _safe_id(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)


def _operation_error_record(error: Exception, **context: Any) -> dict[str, Any]:
    is_cuda_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
        "cuda" in str(error).lower() and "out of memory" in str(error).lower()
    )
    record = {
        **context,
        "status": "error",
        "error_type": "cuda_oom" if is_cuda_oom else type(error).__name__,
        "error": repr(error),
    }
    if is_cuda_oom:
        record["hint"] = (
            "Reduce prompt/max_new_tokens or concurrent GPU work, or use more GPU memory."
        )
    return record


def capture_contrasts(config: dict, model: Any, tokenizer: Any) -> list[Path]:
    data_dir, outputs_dir = _directories(config)
    seed_everything(int(config["experiment"].get("seed", 42)))
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
            if path.exists():
                indexed[capability].append(
                    str(path.relative_to(root)).replace("\\", "/")
                )
                written.append(path)
                continue
            try:
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
                delta = capture_prompt_contrast(
                    model, layer, generic_ids, capability_ids
                )
                atomic_save_tensors(
                    path,
                    {"delta": delta},
                    metadata={"item_id": item.item_id, "capability": capability},
                )
            except Exception as error:
                append_jsonl(
                    root / "errors.jsonl",
                    _operation_error_record(
                        error,
                        stage="capture_contrasts",
                        capture_artifact_id=capture_artifact_id(config),
                        item_id=item.item_id,
                        dataset=item.dataset,
                        capability=capability,
                        prompt_sha256=sha256_text(item.prompt),
                        target_layer=int(config["experiment"]["target_layer"]),
                    ),
                )
                continue
            indexed[capability].append(str(path.relative_to(root)).replace("\\", "/"))
            written.append(path)
    atomic_save_json(
        root / "index.json",
        {"capture_artifact_id": capture_artifact_id(config), "shards": indexed},
    )
    _write_manifest(
        config,
        outputs_dir,
        "03_capture_contrasts",
        artifacts={"index": root / "index.json"},
        model=model,
        tokenizer=tokenizer,
    )
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
    _write_manifest(
        config, outputs_dir, "04_extract_vectors", artifacts={"vectors": tensor_path}
    )
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
    _write_manifest(
        config,
        outputs_dir,
        "05_analyze_similarity",
        artifacts={"similarity": similarity_path, "coherence": coherence_path},
    )
    return {"similarity": similarity_path, "coherence": coherence_path}


def run_steering(config: dict, model: Any, tokenizer: Any) -> Path:
    data_dir, outputs_dir = _directories(config)
    seed_everything(int(config["experiment"].get("seed", 42)))
    capture_id = capture_artifact_id(config)
    vector_root = outputs_dir / "vectors" / capture_id
    vectors, _ = load_vector_library(
        vector_root / "capability_vectors.safetensors",
        vector_root / "capability_vectors.json",
    )
    vector_sha256 = sha256_file(vector_root / "capability_vectors.safetensors")
    scaling = str(config["experiment"].get("vector_scaling", "mean_norm"))
    form = {"mean_norm": "steering", "unit": "unit", "raw": "raw"}[scaling]
    layer = resolve_decoder_layer(model, int(config["experiment"]["target_layer"]))
    run_id = steering_artifact_id(config)
    output_path = outputs_dir / "generations" / f"{run_id}.jsonl"
    existing = list(read_jsonl(output_path)) if output_path.exists() else []
    current_existing = canonical_generation_rows(existing, run_id=run_id)
    completed = {generation_key(row) for row in current_existing}
    baseline_cache = {
        generation_key(row)[:2]: row["raw_output"]
        for row in current_existing
        if float(row["alpha"]) == 0.0
    }
    capabilities = list(config["experiment"]["capabilities"])
    alphas = list(config["experiment"]["alphas"])
    max_new_tokens = int(config["model"].get("max_new_tokens", 2048))
    batch_size = int(
        config["experiment"].get("generation", {}).get("batch_size", 1)
    )
    contexts: list[tuple[str, Mapping[str, Any], CanonicalItem, torch.Tensor]] = []
    for dataset in config["data"].get("enabled_steering_datasets", []):
        rows = _limit(
            config,
            list(
                read_jsonl(data_dir / "processed" / "evaluation" / f"{dataset}.jsonl")
            ),
        )
        for row in rows:
            item = CanonicalItem.from_dict(row)
            contexts.append(
                (
                    dataset,
                    row,
                    item,
                    _tensor_input(
                        serialize_reasoning_prefill(
                            tokenizer,
                            GENERIC_PROMPT,
                            item.prompt,
                            answer_instruction(item.answer_type, item.dataset),
                        )
                    ),
                )
            )

    def record_for(
        context: tuple[str, Mapping[str, Any], CanonicalItem, torch.Tensor],
        capability: str,
        alpha: float,
    ) -> tuple[dict[str, Any], tuple[str, str, str, float]]:
        dataset, source_row, item, _ = context
        record = generation_identity_record(
            run_id=run_id,
            dataset=dataset,
            item=item,
            source_row=source_row,
            steering_capability=capability,
            alpha=alpha,
        )
        record.update(
            {
                "capture_artifact_id": capture_id,
                "vector_sha256": vector_sha256,
                "vector_scaling": scaling,
                "model_revision": config["model"].get("revision"),
                "model_resolution": _resolved_model_identity(model, tokenizer),
                "generation_parameters": {
                    "max_new_tokens": max_new_tokens,
                    "batch_size": batch_size,
                    "do_sample": bool(config["model"].get("do_sample", False)),
                    "use_cache": bool(config["model"].get("use_cache", True)),
                },
            }
        )
        return record, generation_key(record)

    expected = {
        key
        for context in contexts
        for capability in capabilities
        for raw_alpha in alphas
        for _, key in [record_for(context, capability, float(raw_alpha))]
    }
    completed &= expected
    total = len(expected)
    resumed = len(completed)
    failures = 0
    progress = tqdm(total=total, initial=resumed, desc=output_path.stem)

    def persist(
        context: tuple[str, Mapping[str, Any], CanonicalItem, torch.Tensor],
        record: dict[str, Any],
        key: tuple[str, str, str, float],
        *,
        output: str | None = None,
        error: Exception | None = None,
    ) -> None:
        nonlocal failures
        _, _, item, _ = context
        try:
            if error is not None:
                raise error
            assert output is not None
            predicted = extract_answer(output, item.answer_type)
            persisted = {
                **record,
                "raw_output": output,
                "predicted_answer": predicted,
                "correct": is_correct(
                    predicted,
                    item.gold_answer,
                    dataset=item.dataset,
                    answer_type=item.answer_type,
                ),
                "status": "ok",
            }
        except Exception as caught:
            persisted = _operation_error_record(
                caught,
                **record,
                stage="run_steering",
                target_layer=int(config["experiment"]["target_layer"]),
            )
        append_jsonl(output_path, persisted)
        if persisted["status"] == "ok":
            completed.add(key)
        else:
            failures += 1
        progress.update(1)
        progress.set_postfix(resumed=resumed, failed=failures)

    def batches(values: list[Any]) -> list[list[Any]]:
        return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]

    try:
        baseline_tasks: list[
            tuple[
                tuple[str, Mapping[str, Any], CanonicalItem, torch.Tensor],
                list[tuple[dict[str, Any], tuple[str, str, str, float]]],
            ]
        ] = []
        for context in contexts:
            pending = []
            for capability in capabilities:
                record, key = record_for(context, capability, 0.0)
                if key not in completed:
                    pending.append((record, key))
            if not pending:
                continue
            cache_key = pending[0][1][:2]
            if cache_key in baseline_cache:
                for record, key in pending:
                    persist(context, record, key, output=baseline_cache[cache_key])
            else:
                baseline_tasks.append((context, pending))
        for batch in batches(baseline_tasks):
            try:
                outputs = generate_batch_with_optional_steering(
                    model,
                    tokenizer,
                    [context[3] for context, _ in batch],
                    layer,
                    vector=None,
                    alpha=0.0,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as error:
                for context, pending in batch:
                    for record, key in pending:
                        persist(context, record, key, error=error)
                continue
            for (context, pending), output in zip(batch, outputs, strict=True):
                baseline_cache[pending[0][1][:2]] = output
                for record, key in pending:
                    persist(context, record, key, output=output)

        for capability in capabilities:
            for raw_alpha in alphas:
                alpha = float(raw_alpha)
                if alpha == 0.0:
                    continue
                pending = []
                for context in contexts:
                    record, key = record_for(context, capability, alpha)
                    if key not in completed:
                        pending.append((context, record, key))
                for batch in batches(pending):
                    try:
                        outputs = generate_batch_with_optional_steering(
                            model,
                            tokenizer,
                            [context[3] for context, _, _ in batch],
                            layer,
                            vector=vectors[capability][form],
                            alpha=alpha,
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception as error:
                        for context, record, key in batch:
                            persist(context, record, key, error=error)
                        continue
                    for (context, record, key), output in zip(batch, outputs, strict=True):
                        persist(context, record, key, output=output)
    finally:
        progress.close()
    _write_manifest(
        config,
        outputs_dir,
        "06_run_steering",
        artifacts={"generations": output_path},
        model=model,
        tokenizer=tokenizer,
    )
    return output_path


def score_generations(config: dict) -> dict[str, Path]:
    _, outputs_dir = _directories(config)
    run_id = steering_artifact_id(config)
    generation_path = outputs_dir / "generations" / f"{run_id}.jsonl"
    rows = canonical_generation_rows(list(read_jsonl(generation_path)), run_id=run_id)
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
            specificity = specificity_report(
                paired_dataset_rows,
                alpha=alpha,
                capabilities=config["experiment"]["capabilities"],
            )
            dataset_report["specificity"][str(alpha)] = {
                key: specificity[key]
                for key in ("matrix", "counts", "missing_cells")
            }
            dataset_report["diagonal_dominance"][str(alpha)] = specificity[
                "diagonal_dominance"
            ]
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
    _write_manifest(
        config,
        outputs_dir,
        "07_score_generations",
        artifacts={"metrics_json": json_path, "metrics_csv": csv_path},
    )
    return {"json": json_path, "csv": csv_path}
