"""Continuous final-position residual addition during prefill and decoding."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@contextmanager
def add_steering_vector(
    layer: nn.Module,
    vector: torch.Tensor,
    alpha: float,
) -> Iterator[None]:
    """Add ``alpha * vector`` to the final sequence position on every call."""

    if vector.ndim != 1:
        raise ValueError("steering vector must be one-dimensional")
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    source_vector = vector.detach()
    converted_vectors: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError(
                "decoder layer output must be a [batch, sequence, hidden] tensor"
            )
        if hidden.shape[-1] != source_vector.numel():
            raise ValueError(
                f"steering vector hidden size {source_vector.numel()} does not match "
                f"layer hidden size {hidden.shape[-1]}"
            )
        cache_key = (hidden.device, hidden.dtype)
        scaled = converted_vectors.get(cache_key)
        if scaled is None:
            scaled = source_vector.to(device=hidden.device, dtype=hidden.dtype) * float(
                alpha
            )
            converted_vectors[cache_key] = scaled
        steered = hidden.clone()
        steered[:, -1, :] = steered[:, -1, :] + scaled
        return _replace_hidden(output, steered)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
