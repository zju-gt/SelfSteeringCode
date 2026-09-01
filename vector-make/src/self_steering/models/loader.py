"""Lazy Hugging Face checkpoint loading."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(config: dict[str, Any]):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load Qwen") from exc
    name = str(config["name"])
    shared = {
        "revision": config.get("revision", "main"),
        "cache_dir": config.get("cache_dir"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    shared = {key: value for key, value in shared.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(name, **shared)
    dtype_name = str(config.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    model_kwargs = dict(shared)
    model_kwargs.update(
        {
            "torch_dtype": DTYPES[dtype_name],
            "device_map": config.get("device_map", "auto"),
        }
    )
    if config.get("attention_implementation"):
        model_kwargs["attn_implementation"] = config["attention_implementation"]
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def resolve_decoder_layer(model: Any, index: int) -> nn.Module:
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise ValueError("model does not expose model.layers decoder blocks") from exc
    if not isinstance(index, int) or not 0 <= index < len(layers):
        raise ValueError(f"target layer {index!r} is outside [0, {len(layers) - 1}]")
    layer = layers[index]
    if not isinstance(layer, nn.Module):
        return layer
    return layer

