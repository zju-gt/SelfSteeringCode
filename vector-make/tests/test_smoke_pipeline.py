from pathlib import Path

import torch
from torch import nn

from self_steering.datasets.registry import DatasetRegistry
from self_steering.datasets.types import CanonicalItem
from self_steering.pipeline import (
    analyze_similarity,
    capture_contrasts,
    extract_vectors,
    prepare_data,
    prepare_items,
    run_steering,
    score_demands,
    score_generations,
)
from self_steering.utils.io import read_jsonl


class ContextLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.cumsum(dim=1)


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ContextLayer()])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 4)
        torch.manual_seed(0)
        nn.init.normal_(self.embedding.weight)
        self.model = TinyDecoder()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.forward(input_ids)
        suffix = torch.tensor([[1]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        return [ord(character) % 127 + 1 for character in text]

    def decode(self, tokens, **kwargs) -> str:
        return "Reasoning: fixture. Final Answer: \\boxed{2}"


def smoke_config(tmp_path: Path) -> dict:
    return {
        "model": {
            "name": "tiny-fixture",
            "revision": "test",
            "num_hidden_layers": 1,
            "max_new_tokens": 1,
        },
        "data": {
            "enabled_steering_datasets": ["math500"],
            "sources": {"mmlu": {}, "math500": {}},
        },
        "experiment": {
            "capabilities": ["QLl", "QLq"],
            "target_layer": 0,
            "high_demand_threshold": 4,
            "low_demand_threshold": 1,
            "vector_scaling": "mean_norm",
            "alphas": [0, 1],
            "annotation": {"max_workers": 2},
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "outputs_dir": str(tmp_path / "outputs"),
            },
        },
    }


def test_offline_pipeline_from_registry_to_metrics(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    registry = DatasetRegistry()
    registry.register(
        "mmlu",
        lambda source: [
            CanonicalItem(
                "m1", "mmlu", "test", "Choose two.", "A", "choice", {"A": "2"}
            ),
            CanonicalItem(
                "m2", "mmlu", "test", "Choose one.", "A", "choice", {"A": "1"}
            ),
        ],
    )
    registry.register(
        "math500",
        lambda source: [
            CanonicalItem("x1", "math500", "test", "Compute 1+1.", "2", "math")
        ],
    )

    prepare_data(config, registry)

    def label(item: CanonicalItem, dimension: str) -> dict:
        return {
            "item_id": item.item_id,
            "demand": dimension,
            "level": 4,
            "status": "ok",
        }

    score_demands(config, label)
    prepare_items(config)
    model = TinyModel()
    tokenizer = TinyTokenizer()
    assert len(capture_contrasts(config, model, tokenizer)) == 4
    assert extract_vectors(config).exists()
    assert all(path.exists() for path in analyze_similarity(config).values())
    generations = run_steering(config, model, tokenizer)
    assert len(list(read_jsonl(generations))) == 4
    metrics = score_generations(config)
    assert all(path.exists() for path in metrics.values())
