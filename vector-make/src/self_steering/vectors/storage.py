"""Safetensors persistence for the vector library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from self_steering.utils.io import atomic_save_json, atomic_save_tensors


FORMS = ("raw", "unit", "steering")


def save_vector_library(
    tensor_path: Path | str,
    metadata_path: Path | str,
    vectors: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    metadata: Mapping[str, Any],
) -> None:
    flattened: dict[str, torch.Tensor] = {}
    norms: dict[str, dict[str, float]] = {}
    for capability, forms in vectors.items():
        norms[capability] = {}
        for form in FORMS:
            if form not in forms:
                raise ValueError(f"missing vector form {capability}.{form}")
            tensor = forms[form]
            flattened[f"{capability}__{form}"] = tensor
            norms[capability][form] = float(tensor.detach().float().norm().item())
    atomic_save_tensors(tensor_path, flattened, metadata={"format": "self-steering-v1"})
    payload = dict(metadata)
    payload["norms"] = norms
    payload["capabilities"] = sorted(vectors)
    atomic_save_json(metadata_path, payload)


def load_vector_library(
    tensor_path: Path | str,
    metadata_path: Path | str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    flattened = load_file(str(tensor_path), device="cpu")
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in flattened.items():
        capability, separator, form = key.partition("__")
        if not separator or form not in FORMS:
            raise ValueError(f"invalid vector key: {key}")
        vectors.setdefault(capability, {})[form] = tensor
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return vectors, metadata

