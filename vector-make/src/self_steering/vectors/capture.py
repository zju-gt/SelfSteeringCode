"""No-grad activation capture for same-question prompt contrasts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from self_steering.hooks.capture import capture_last_token
from self_steering.models.generation import model_input_device


def capture_activation(
    model: Any,
    layer: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("capture requires input_ids with shape [1, sequence]")
    ids = input_ids.to(model_input_device(model))
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode(), capture_last_token(layer) as captured:
        model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    return captured.value


def contrast_delta(capability: torch.Tensor, generic: torch.Tensor) -> torch.Tensor:
    if capability.shape != generic.shape:
        raise ValueError("capability and generic activations must have matching shapes")
    return capability.detach().cpu().to(torch.float32) - generic.detach().cpu().to(
        torch.float32
    )


def capture_prompt_contrast(
    model: Any,
    layer: nn.Module,
    generic_input_ids: torch.Tensor,
    capability_input_ids: torch.Tensor,
) -> torch.Tensor:
    generic = capture_activation(model, layer, generic_input_ids)
    capability = capture_activation(model, layer, capability_input_ids)
    return contrast_delta(capability, generic)
