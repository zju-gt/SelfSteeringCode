"""Greedy text generation with optional continuous residual steering."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

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
    return generate_batch_with_optional_steering(
        model,
        tokenizer,
        [input_ids],
        layer,
        vector,
        alpha,
        max_new_tokens,
    )[0]


def generate_batch_with_optional_steering(
    model: Any,
    tokenizer: Any,
    input_ids: Sequence[torch.Tensor],
    layer: nn.Module,
    vector: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
) -> list[str]:
    """Generate a left-padded batch under one shared steering intervention."""

    if not input_ids:
        return []
    rows: list[torch.Tensor] = []
    for ids in input_ids:
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("each input_ids tensor must have shape [1, sequence]")
        rows.append(ids[0])
    max_length = max(row.numel() for row in rows)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0
    padded = torch.full(
        (len(rows), max_length),
        int(pad_token_id),
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    attention_mask = torch.zeros_like(padded)
    for index, row in enumerate(rows):
        padded[index, -row.numel() :] = row
        attention_mask[index, -row.numel() :] = 1
    device = model_input_device(model)
    prompt_ids = padded.to(device)
    attention_mask = attention_mask.to(device)
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
    new_tokens = output_ids[:, prompt_ids.shape[1] :].detach().cpu()
    return [
        tokenizer.decode(tokens, skip_special_tokens=True)
        for tokens in new_tokens
    ]
