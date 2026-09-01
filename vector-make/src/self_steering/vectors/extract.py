"""Mean-difference vector aggregation and CAA-style norm calibration."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def aggregate_capability_vectors(
    deltas: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    if not deltas:
        raise ValueError("at least one capability delta tensor is required")
    raw_vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, torch.Tensor] = {}
    hidden_size: int | None = None
    for capability, values in deltas.items():
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(
                f"{capability} deltas must have shape [items, hidden_size] with items > 0"
            )
        if hidden_size is None:
            hidden_size = values.shape[1]
        elif values.shape[1] != hidden_size:
            raise ValueError("all capabilities must use the same hidden size")
        raw = values.detach().to(device="cpu", dtype=torch.float32).mean(dim=0)
        norm = raw.norm()
        if not torch.isfinite(norm) or norm.item() == 0.0:
            raise ValueError(f"{capability} produced a zero norm or non-finite vector")
        raw_vectors[capability] = raw
        norms[capability] = norm

    mean_norm = torch.stack(list(norms.values())).mean()
    return {
        capability: {
            "raw": raw,
            "unit": raw / norms[capability],
            "steering": raw / norms[capability] * mean_norm,
        }
        for capability, raw in raw_vectors.items()
    }
