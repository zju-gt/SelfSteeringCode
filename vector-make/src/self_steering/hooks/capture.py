"""Read a decoder block's final-position output residual."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn


def _hidden_from_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("decoder layer output must contain a [batch, sequence, hidden] tensor")
    return hidden


@dataclass
class CaptureBuffer:
    _value: torch.Tensor | None = None

    @property
    def value(self) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("no activation was captured")
        return self._value


@contextmanager
def capture_last_token(layer: nn.Module) -> Iterator[CaptureBuffer]:
    """Capture the first batch element's final-position output residual."""

    buffer = CaptureBuffer()

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_output(output)
        buffer._value = hidden[0, -1, :].detach().to(device="cpu", dtype=torch.float32)
        return output

    handle = layer.register_forward_hook(hook)
    try:
        yield buffer
    finally:
        handle.remove()

