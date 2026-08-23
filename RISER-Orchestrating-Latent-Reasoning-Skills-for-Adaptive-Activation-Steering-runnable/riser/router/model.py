"""Router network used to select and compose RISER primitive vectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RouterConfig:
    """Serializable configuration for :class:`Router`."""

    hidden_size: int
    num_primitives: int
    bottleneck_dim: int = 1024
    selection_threshold: float = 0.7
    max_strength: float = 2.0
    strength_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_primitives", "bottleneck_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be between 0 and 1")
        if self.max_strength <= 0.0:
            raise ValueError("max_strength must be positive")
        if self.strength_temperature <= 0.0:
            raise ValueError("strength_temperature must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RouterConfig":
        return cls(**dict(values))


class Router(nn.Module):
    """Predict primitive selection probabilities and intervention strengths."""

    def __init__(self, config: Union[RouterConfig, Mapping[str, Any]]) -> None:
        super().__init__()
        if isinstance(config, Mapping):
            config = RouterConfig.from_dict(config)
        if not isinstance(config, RouterConfig):
            raise TypeError("config must be a RouterConfig or mapping")

        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(config.bottleneck_dim, config.bottleneck_dim),
            nn.SiLU(),
        )
        self.selection_head = nn.Linear(config.bottleneck_dim, config.num_primitives)
        self.strength_head = nn.Linear(config.bottleneck_dim, config.num_primitives)

    def _prepare_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden_state, torch.Tensor):
            raise TypeError("hidden_state must be a torch.Tensor")
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must have shape [hidden] or [batch, hidden]")
        if hidden_state.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, "
                f"got {hidden_state.shape[-1]}"
            )
        return hidden_state

    def forward(
        self,
        hidden_state: torch.Tensor,
        hard: bool = False,
    ):
        hidden_state = self._prepare_hidden_state(hidden_state)
        features = self.feature_extractor(hidden_state)
        selection_logits = self.selection_head(features)
        selection_probs = torch.sigmoid(selection_logits)
        strength_logits = self.strength_head(features)
        strength = torch.sigmoid(
            strength_logits / self.config.strength_temperature
        ) * self.config.max_strength

        if hard:
            selection_mask = (
                selection_probs >= self.config.selection_threshold
            ).to(selection_probs.dtype)
        else:
            selection_mask = selection_probs

        return (
            selection_mask,
            strength,
            selection_probs,
            selection_logits,
            features,
        )

    @staticmethod
    def _library_tensor(
        primitive_library: Union[torch.Tensor, Mapping[int, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(primitive_library, Mapping):
            if not primitive_library:
                raise ValueError("primitive_library cannot be empty")
            values = [primitive_library[key] for key in sorted(primitive_library)]
            primitive_library = torch.stack(values)
        if not isinstance(primitive_library, torch.Tensor):
            raise TypeError("primitive_library must be a tensor or mapping")
        if primitive_library.ndim != 2:
            raise ValueError("primitive_library must have shape [num_primitives, hidden]")
        return primitive_library.to(device=device, dtype=dtype)

    @torch.no_grad()
    def route(
        self,
        hidden_state: torch.Tensor,
        primitive_library: Union[torch.Tensor, Mapping[int, torch.Tensor]],
        hard: bool = False,
    ):
        prepared_hidden = self._prepare_hidden_state(hidden_state)
        library = self._library_tensor(
            primitive_library,
            device=prepared_hidden.device,
            dtype=prepared_hidden.dtype,
        )
        if library.shape[0] != self.config.num_primitives:
            raise ValueError(
                f"Expected {self.config.num_primitives} primitive vectors, "
                f"got {library.shape[0]}"
            )
        if library.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"Expected primitive hidden size {self.config.hidden_size}, "
                f"got {library.shape[1]}"
            )

        selection_mask, strength, selection_probs, selection_logits, features = self(
            prepared_hidden,
            hard=hard,
        )
        injection = (selection_mask * strength) @ library
        selected_primitives = [
            torch.where(row > 0.5)[0].detach().cpu().tolist()
            for row in selection_mask
        ]
        info = {
            "selected_primitives": selected_primitives,
            "selected_strengths": [
                strength[i, indices].detach().cpu().tolist()
                for i, indices in enumerate(
                    [row for row in selected_primitives]
                )
            ],
            "selection_mask": selection_mask.detach().cpu().tolist(),
            "strengths": strength.detach().cpu().tolist(),
            "selection_probs": selection_probs.detach().cpu().tolist(),
            "selection_logits": selection_logits.detach().cpu().tolist(),
            "features": features.detach().cpu().tolist(),
        }
        return injection, info


class FixedRouter(Router):
    """A deterministic Router for experiments without a Router checkpoint.

    The fixed route is independent of the hidden state.  It keeps the same
    ``forward`` and ``route`` contract as :class:`Router`, so it can be used
    by the existing inference and generation wrappers.
    """

    def __init__(
        self,
        config: Union[RouterConfig, Mapping[str, Any]],
        primitive_ids: Sequence[int] = (),
        strengths: Sequence[float] = (),
    ) -> None:
        nn.Module.__init__(self)
        if isinstance(config, Mapping):
            config = RouterConfig.from_dict(config)
        if not isinstance(config, RouterConfig):
            raise TypeError("config must be a RouterConfig or mapping")
        if len(primitive_ids) != len(strengths):
            raise ValueError("primitive_ids and strengths must have the same length")

        selection = torch.zeros(config.num_primitives, dtype=torch.float32)
        fixed_strengths = torch.zeros(config.num_primitives, dtype=torch.float32)
        for primitive_id, strength in zip(primitive_ids, strengths):
            if not isinstance(primitive_id, int) or isinstance(primitive_id, bool):
                raise TypeError("primitive_ids must contain integers")
            if not 0 <= primitive_id < config.num_primitives:
                raise ValueError(
                    f"primitive id {primitive_id} is outside [0, {config.num_primitives})"
                )
            if not 0.0 <= float(strength) <= config.max_strength:
                raise ValueError(
                    f"fixed strengths must be between 0 and {config.max_strength}"
                )
            if selection[primitive_id] > 0:
                raise ValueError(f"primitive id {primitive_id} was specified twice")
            selection[primitive_id] = 1.0
            fixed_strengths[primitive_id] = float(strength)

        self.config = config
        self.register_buffer("fixed_selection", selection)
        self.register_buffer("fixed_strengths", fixed_strengths)

    def forward(self, hidden_state: torch.Tensor, hard: bool = False):
        hidden_state = self._prepare_hidden_state(hidden_state)
        batch_size = hidden_state.shape[0]
        selection_mask = self.fixed_selection.to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        ).expand(batch_size, -1)
        strength = self.fixed_strengths.to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        ).expand(batch_size, -1)
        selection_probs = selection_mask
        selection_logits = torch.where(
            selection_mask > 0.5,
            torch.full_like(selection_mask, 20.0),
            torch.full_like(selection_mask, -20.0),
        )
        features = torch.zeros(
            batch_size,
            self.config.bottleneck_dim,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        return (
            selection_mask,
            strength,
            selection_probs,
            selection_logits,
            features,
        )


__all__ = ["Router", "FixedRouter", "RouterConfig"]
