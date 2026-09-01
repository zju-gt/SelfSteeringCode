"""Greedy text generation with optional continuous residual steering."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from self_steering.hooks.intervention import add_steering_vector


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def generate_with_optional_steering(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    layer: nn.Module,
    vector: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
) -> str:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] != 1:
        raise ValueError("MVP generation currently requires batch size one")
    device = model_input_device(model)
    prompt_ids = input_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids)
    context = (
        add_steering_vector(layer, vector, alpha)
        if vector is not None and float(alpha) != 0.0
        else nullcontext()
    )
    kwargs: dict[str, Any] = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    with torch.inference_mode(), context:
        output_ids = model.generate(**kwargs)
    new_tokens = output_ids[0, prompt_ids.shape[1] :].detach().cpu()
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

