from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import logging

from .model import Router, RouterConfig

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
        router_path: str,
        primitive_library_path: str,
        target_layer: int,
        device: str = "cuda",
    ) -> "RouterInference":
        checkpoint = torch.load(router_path, map_location="cpu")
        
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Router checkpoint missing config")
        
        router = Router(config)
        router.load_state_dict(checkpoint["model_state_dict"])
        
        library_data = torch.load(primitive_library_path, map_location="cpu")
        primitive_library = library_data["primitives"]
        
        if isinstance(primitive_library, dict):
            sorted_ids = sorted(primitive_library.keys())
            primitive_library = torch.stack([primitive_library[i] for i in sorted_ids])
        
        return cls(
            router=router,
            primitive_library=primitive_library,
            target_layer=target_layer,
            device=device,
        )
