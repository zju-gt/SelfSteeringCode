from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING
import torch
import logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel

from .model import FixedRouter, Router, RouterConfig

logger = logging.getLogger(__name__)


class RouterInference:
    
    def __init__(
        self,
        router: Router,
        primitive_library: torch.Tensor,
        target_layer: int,
        device: str = "cuda",
    ):
        self.router = router
        self.primitive_library = primitive_library.to(device)
        self.target_layer = target_layer
        self.device = device
        
        self.router.eval()
        self.router.to(device)
        
        logger.info(f"RouterInference initialized: target_layer={target_layer}, "
                   f"num_primitives={primitive_library.shape[0]}")
    
    @torch.no_grad()
    def inject_activation(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        hidden_state = hidden_state.to(self.device)
        
        v_inject, info = self.router.route(
            hidden_state,
            self.primitive_library,
            hard=True,
        )
        
        injected_state = hidden_state + v_inject
        
        return injected_state, info
    
    def get_routing_info(
        self,
        hidden_state: torch.Tensor,
    ) -> Dict[str, Any]:
        hidden_state = hidden_state.to(self.device)
        
        with torch.no_grad():
            selection_mask, strength, selection_probs, _, _ = self.router.forward(
                hidden_state,
                hard=True,
            )
        
        selected_indices = torch.where(selection_mask[0] > 0.5)[0].cpu().tolist()
        selected_strengths = strength[0][selected_indices].cpu().tolist()
        
        return {
            "selected_primitives": selected_indices,
            "strengths": selected_strengths,
            "selection_probs": selection_probs[0].cpu().tolist(),
            "num_selected": len(selected_indices),
        }
    
    @classmethod
    def from_pretrained(
        cls,
        router_path: Optional[str],
        primitive_library_path: str,
        target_layer: int,
        device: str = "cuda",
        fixed_primitives: Optional[Sequence[int]] = None,
        fixed_strengths: Optional[Sequence[float]] = None,
        fixed_max_strength: float = 2.0,
    ) -> "RouterInference":
        primitive_library = cls._load_primitive_library(primitive_library_path)

        if router_path is None:
            return cls.from_fixed_library(
                primitive_library=primitive_library,
                target_layer=target_layer,
                device=device,
                fixed_primitives=fixed_primitives,
                fixed_strengths=fixed_strengths,
                fixed_max_strength=fixed_max_strength,
            )

        checkpoint = torch.load(router_path, map_location="cpu")
        
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Router checkpoint missing config")
        
        router = Router(config)
        router.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(
            router=router,
            primitive_library=primitive_library,
            target_layer=target_layer,
            device=device,
        )

    @staticmethod
    def _load_primitive_library(primitive_library_path: str) -> torch.Tensor:
        library_data = torch.load(primitive_library_path, map_location="cpu")
        if isinstance(library_data, dict) and "primitives" in library_data:
            primitive_library = library_data["primitives"]
        else:
            primitive_library = library_data

        if isinstance(primitive_library, dict):
            if not primitive_library:
                raise ValueError("Primitive library cannot be empty")
            sorted_ids = sorted(primitive_library.keys())
            primitive_library = torch.stack([primitive_library[i] for i in sorted_ids])
        if not isinstance(primitive_library, torch.Tensor):
            raise TypeError("Primitive library must contain a tensor or tensor mapping")
        if primitive_library.ndim != 2 or primitive_library.shape[0] == 0:
            raise ValueError(
                "Primitive library must have shape [num_primitives, hidden] and be non-empty"
            )
        return primitive_library.float()

    @classmethod
    def from_fixed_library(
        cls,
        primitive_library: torch.Tensor,
        target_layer: int,
        device: str = "cuda",
        fixed_primitives: Optional[Sequence[int]] = None,
        fixed_strengths: Optional[Sequence[float]] = None,
        fixed_max_strength: float = 2.0,
    ) -> "RouterInference":
        """Create deterministic routing from primitive IDs and strengths.

        An empty selection is valid and produces no activation injection.  If
        strengths are omitted, every selected primitive receives strength 1.0.
        """
        if not isinstance(primitive_library, torch.Tensor):
            raise TypeError("primitive_library must be a tensor")
        if primitive_library.ndim != 2 or primitive_library.shape[0] == 0:
            raise ValueError(
                "primitive_library must have shape [num_primitives, hidden] and be non-empty"
            )
        if fixed_max_strength <= 0.0:
            raise ValueError("fixed_max_strength must be positive")
        primitive_ids = [] if fixed_primitives is None else list(fixed_primitives)
        if fixed_strengths is None:
            strengths = [1.0] * len(primitive_ids)
        else:
            strengths = list(fixed_strengths)
        if len(primitive_ids) != len(strengths):
            raise ValueError(
                "fixed_primitives and fixed_strengths must have the same length"
            )

        config = RouterConfig(
            hidden_size=int(primitive_library.shape[1]),
            num_primitives=int(primitive_library.shape[0]),
            bottleneck_dim=1,
            selection_threshold=0.5,
            max_strength=float(fixed_max_strength),
        )
        router = FixedRouter(
            config=config,
            primitive_ids=primitive_ids,
            strengths=[float(value) for value in strengths],
        )
        return cls(
            router=router,
            primitive_library=primitive_library,
            target_layer=target_layer,
            device=device,
        )

    @classmethod
    def from_fixed(
        cls,
        primitive_library_path: str,
        target_layer: int,
        device: str = "cuda",
        fixed_primitives: Optional[Sequence[int]] = None,
        fixed_strengths: Optional[Sequence[float]] = None,
        fixed_max_strength: float = 2.0,
    ) -> "RouterInference":
        """Create deterministic routing directly from a saved library."""
        return cls.from_fixed_library(
            primitive_library=cls._load_primitive_library(primitive_library_path),
            target_layer=target_layer,
            device=device,
            fixed_primitives=fixed_primitives,
            fixed_strengths=fixed_strengths,
            fixed_max_strength=fixed_max_strength,
        )
