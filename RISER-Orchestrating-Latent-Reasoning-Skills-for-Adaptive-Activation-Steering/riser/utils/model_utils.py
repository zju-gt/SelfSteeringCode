from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model: {model_name}")
    
    torch_dtype = getattr(torch, dtype)
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if device in {"auto", "balanced", "balanced_low_0"}:
        model_kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if device not in {"auto", "balanced", "balanced_low_0"}:
        model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded on {device} with dtype {dtype}")
    
    return model, tokenizer
