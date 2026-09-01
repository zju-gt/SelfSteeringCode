from types import SimpleNamespace

import pytest
import torch
from torch import nn

from self_steering.models.generation import generate_with_optional_steering
from self_steering.models.loader import resolve_decoder_layer


class Nested:
    pass


def test_resolve_decoder_layer_uses_zero_based_index() -> None:
    model = Nested()
    model.model = Nested()
    model.model.layers = [object(), object(), object()]
    assert resolve_decoder_layer(model, 1) is model.model.layers[1]


def test_resolve_decoder_layer_rejects_out_of_bounds() -> None:
    model = Nested()
    model.model = Nested()
    model.model.layers = [object()]
    with pytest.raises(ValueError, match="target layer"):
        resolve_decoder_layer(model, 1)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    def decode(self, tokens, **kwargs):
        self.decoded = tokens
        self.decode_kwargs = kwargs
        return "generated text"


class FakeGenerateModel:
    def __init__(self):
        self.embedding = nn.Embedding(10, 2)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.kwargs = kwargs
        suffix = torch.tensor([[7, 8]], device=kwargs["input_ids"].device)
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


def test_generation_is_greedy_and_decodes_only_new_tokens() -> None:
    model = FakeGenerateModel()
    tokenizer = FakeTokenizer()
    layer = nn.Identity()
    text = generate_with_optional_steering(
        model,
        tokenizer,
        torch.tensor([[1, 2, 3]]),
        layer,
        vector=None,
        alpha=0.0,
        max_new_tokens=10,
    )
    assert text == "generated text"
    assert tokenizer.decoded.tolist() == [7, 8]
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["use_cache"] is True

