"""Cosine separation and within-capability coherence metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional


def cosine_similarity_matrix(
    vectors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    names = list(vectors)
    if not names:
        return {}
    stacked = torch.stack(
        [vectors[name].detach().cpu().to(torch.float32) for name in names]
    )
    norms = stacked.norm(dim=1)
    if torch.any(norms == 0) or not torch.all(torch.isfinite(norms)):
        raise ValueError("cosine similarity requires finite non-zero vectors")
    normalized = functional.normalize(stacked, dim=1)
    values = normalized @ normalized.T
    return {
        left: {right: float(values[i, j].item()) for j, right in enumerate(names)}
        for i, left in enumerate(names)
    }


def vector_coherence(deltas: torch.Tensor) -> float:
    if deltas.ndim != 2 or deltas.shape[0] < 2:
        raise ValueError("coherence requires at least two item deltas")
    values = deltas.detach().cpu().to(torch.float32)
    norms = values.norm(dim=1)
    if torch.any(norms == 0) or not torch.all(torch.isfinite(norms)):
        raise ValueError("coherence requires finite non-zero item deltas")
    normalized = functional.normalize(values, dim=1)
    similarities = normalized @ normalized.T
    mask = ~torch.eye(values.shape[0], dtype=torch.bool)
    return float(similarities[mask].mean().item())
