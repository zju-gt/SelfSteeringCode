import torch
from torch import nn

from self_steering.vectors.capture import capture_activation, contrast_delta


class TinyCaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        self.layer.weight.data.copy_(torch.eye(2))
        self.embedding = nn.Embedding(10, 2)
        self.embedding.weight.data.copy_(torch.arange(20, dtype=torch.float32).reshape(10, 2))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return self.layer(self.embedding(input_ids))


def test_capture_activation_reads_final_block_output() -> None:
    model = TinyCaptureModel()
    activation = capture_activation(model, model.layer, torch.tensor([[1, 2]]))
    assert torch.equal(activation, torch.tensor([4.0, 5.0]))


def test_contrast_delta_subtracts_generic_from_capability() -> None:
    generic = torch.tensor([1.0, 2.0])
    capability = torch.tensor([4.0, 8.0])
    assert torch.equal(contrast_delta(capability, generic), torch.tensor([3.0, 6.0]))

