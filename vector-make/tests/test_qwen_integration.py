"""No-download integration coverage for real Qwen2 cached decoding."""

import pytest
import torch

from self_steering.models.generation import generate_with_optional_steering


class DecodeOnlyTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def decode(self, tokens, **kwargs) -> str:
        return " ".join(str(token) for token in tokens.tolist())


def _qwen_classes():
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except Exception as error:
        pytest.skip(
            "installed Transformers/Torch environment cannot construct Qwen2: "
            f"{type(error).__name__}: {error}"
        )
    return Qwen2Config, Qwen2ForCausalLM


def test_real_qwen2_generate_uses_cached_single_token_decode_with_steering() -> None:
    Qwen2Config, Qwen2ForCausalLM = _qwen_classes()
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
    )
    try:
        model = Qwen2ForCausalLM(config).eval()
    except Exception as error:
        pytest.skip(
            "installed Transformers/Torch environment cannot initialize Qwen2: "
            f"{type(error).__name__}: {error}"
        )

    layer = model.model.layers[0]
    sequence_lengths: list[int] = []

    def observe_input(module, args):
        sequence_lengths.append(int(args[0].shape[1]))

    handle = layer.register_forward_pre_hook(observe_input)
    try:
        text = generate_with_optional_steering(
            model,
            DecodeOnlyTokenizer(),
            torch.tensor([[1, 2, 3]]),
            layer,
            vector=torch.ones(config.hidden_size),
            alpha=0.25,
            max_new_tokens=2,
        )
    finally:
        handle.remove()

    assert isinstance(text, str)
    assert sequence_lengths[0] == 3
    assert sequence_lengths[1:] == [1]
