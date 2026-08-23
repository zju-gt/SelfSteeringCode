"""Generation wrapper that applies Router-selected activation steering."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .hooks import ActivationInjectionHook
from ..router.inference import RouterInference


class SteeredModel(nn.Module):
    """Wrap a causal model and inject a composed vector at one target layer."""

    def __init__(
        self,
        base_model: nn.Module,
        router_inference: RouterInference,
        cache_routing: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.router_inference = router_inference
        self.target_layer = router_inference.target_layer
        self.cache_routing = cache_routing
        self._cached_vector: Optional[torch.Tensor] = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._active_hook: Optional[ActivationInjectionHook] = None

        primitive_library = router_inference.primitive_library
        expected_hidden = router_inference.router.config.hidden_size
        if primitive_library.ndim != 2:
            raise ValueError("primitive_library must have shape [num_primitives, hidden]")
        if primitive_library.shape[1] != expected_hidden:
            raise ValueError(
                f"Primitive hidden size {primitive_library.shape[1]} does not match "
                f"Router hidden size {expected_hidden}"
            )

    def _inject(self, hidden_state: torch.Tensor):
        if hidden_state.ndim != 2:
            raise ValueError("injection hidden state must have shape [batch, hidden]")

        if self.cache_routing and self._cached_vector is not None:
            cached = self._cached_vector.to(
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )
            if cached.shape[0] == 1 and hidden_state.shape[0] > 1:
                cached = cached.expand(hidden_state.shape[0], -1)
            if cached.shape != hidden_state.shape:
                raise ValueError(
                    "Cached routing vector shape does not match the current batch: "
                    f"{tuple(cached.shape)} vs {tuple(hidden_state.shape)}"
                )
            return hidden_state + cached, self._last_info or {}

        injected, info = self.router_inference.inject_activation(hidden_state)
        if injected.shape != hidden_state.shape:
            raise ValueError(
                "Router returned an invalid injected state shape: "
                f"{tuple(injected.shape)} vs {tuple(hidden_state.shape)}"
            )
        self._last_info = info
        if self.cache_routing:
            self._cached_vector = (injected - hidden_state).detach()
        return injected, info

    def _register_hook(self) -> ActivationInjectionHook:
        if self._active_hook is not None:
            self._active_hook.remove()
        hook = ActivationInjectionHook(self._inject, self.target_layer)
        hook.register(self.base_model)
        self._active_hook = hook
        return hook

    def _remove_hook(self) -> None:
        if self._active_hook is not None:
            self._active_hook.remove()
            self._active_hook = None

    def generate(self, *args, **kwargs):
        self.clear_route_cache()
        self._register_hook()
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self._remove_hook()

    def forward(self, *args, **kwargs):
        self.clear_route_cache()
        self._register_hook()
        try:
            return self.base_model(*args, **kwargs)
        finally:
            self._remove_hook()

    def clear_route_cache(self) -> None:
        self._cached_vector = None
        self._last_info = None

    def get_last_routing_info(self) -> Optional[Dict[str, Any]]:
        return self._last_info


__all__ = ["SteeredModel"]
