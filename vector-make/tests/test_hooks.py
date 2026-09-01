import pytest
import torch
from torch import nn

from self_steering.hooks.capture import capture_last_token
from self_steering.hooks.intervention import add_steering_vector


class TupleBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1, "cache")


class TensorBlock(nn.Module):
    def forward(self, hidden):
        return hidden + 1


def test_capture_reads_last_token_without_changing_output() -> None:
    block = TupleBlock()
    hidden = torch.zeros(1, 3, 2)
    with capture_last_token(block) as captured:
        output = block(hidden)
    assert torch.equal(captured.value, torch.ones(2))
    assert torch.equal(output[0], torch.ones_like(hidden))
    assert output[1] == "cache"


def test_prefill_intervention_changes_only_last_position() -> None:
    block = TupleBlock()
    hidden = torch.zeros(1, 3, 2)
    with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=1.0):
        output = block(hidden)[0]
    assert torch.equal(output[0, 0], torch.ones(2))
    assert torch.equal(output[0, 1], torch.ones(2))
    assert torch.equal(output[0, -1], torch.tensor([3.0, 4.0]))


def test_cached_decode_intervention_changes_single_current_position() -> None:
    block = TupleBlock()
    with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=-1.0):
        output = block(torch.zeros(1, 1, 2))[0]
    assert torch.equal(output[0, 0], torch.tensor([-1.0, -2.0]))


def test_intervention_preserves_tensor_output_type() -> None:
    block = TensorBlock()
    with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=0.5):
        output = block(torch.zeros(1, 2, 2))
    assert isinstance(output, torch.Tensor)
    assert torch.equal(output[0, -1], torch.tensor([2.0, 2.5]))


def test_hook_is_removed_after_context_error() -> None:
    block = TensorBlock()
    with pytest.raises(RuntimeError, match="stop"):
        with add_steering_vector(block, torch.tensor([2.0, 3.0]), alpha=1.0):
            raise RuntimeError("stop")
    assert torch.equal(block(torch.zeros(1, 1, 2)), torch.ones(1, 1, 2))


def test_intervention_rejects_hidden_size_mismatch() -> None:
    block = TensorBlock()
    with add_steering_vector(block, torch.tensor([1.0, 2.0, 3.0]), alpha=1.0):
        with pytest.raises(ValueError, match="hidden size"):
            block(torch.zeros(1, 1, 2))
