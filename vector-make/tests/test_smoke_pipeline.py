import torch

from self_steering.evaluation.metrics import accuracy_by_alpha
from self_steering.vectors.extract import aggregate_capability_vectors


def test_fixture_vector_to_metric_smoke() -> None:
    vectors = aggregate_capability_vectors(
        {
            "QLl": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            "QLq": torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        }
    )
    assert set(vectors) == {"QLl", "QLq"}
    metrics = accuracy_by_alpha(
        [{"alpha": 0.0, "correct": False}, {"alpha": 1.0, "correct": True}]
    )
    assert metrics[1.0]["delta"] == 1.0

