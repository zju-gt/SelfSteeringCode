from pathlib import Path

import pytest
import torch

from self_steering.vectors.extract import aggregate_capability_vectors
from self_steering.vectors.similarity import cosine_similarity_matrix, vector_coherence
from self_steering.vectors.storage import load_vector_library, save_vector_library


def test_mean_norm_vectors_share_the_mean_raw_norm() -> None:
    deltas = {
        "QLl": torch.tensor([[3.0, 0.0]]),
        "QLq": torch.tensor([[0.0, 1.0]]),
    }
    vectors = aggregate_capability_vectors(deltas)
    assert torch.allclose(vectors["QLl"]["steering"].norm(), torch.tensor(2.0))
    assert torch.allclose(vectors["QLq"]["steering"].norm(), torch.tensor(2.0))
    assert torch.allclose(vectors["QLl"]["unit"].norm(), torch.tensor(1.0))


def test_zero_norm_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        aggregate_capability_vectors({"QLl": torch.zeros(2, 3)})


def test_cosine_matrix_and_coherence() -> None:
    vectors = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    matrix = cosine_similarity_matrix(vectors)
    assert matrix["a"]["a"] == pytest.approx(1.0)
    assert matrix["a"]["b"] == pytest.approx(0.0)
    assert vector_coherence(torch.tensor([[1.0, 0.0], [2.0, 0.0]])) == pytest.approx(
        1.0
    )


def test_vector_library_round_trip(tmp_path: Path) -> None:
    vectors = aggregate_capability_vectors({"QLl": torch.tensor([[1.0, 2.0]])})
    tensor_path = tmp_path / "vectors.safetensors"
    metadata_path = tmp_path / "vectors.json"
    save_vector_library(tensor_path, metadata_path, vectors, metadata={"layer": 19})
    loaded, metadata = load_vector_library(tensor_path, metadata_path)
    assert torch.equal(loaded["QLl"]["raw"], torch.tensor([1.0, 2.0]))
    assert metadata["layer"] == 19
