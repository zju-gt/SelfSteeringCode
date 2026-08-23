from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ActivationInjectionHook:
    
    def __init__(
        self,
        injection_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, dict]],
        target_layer: int,
    ):
        self.injection_fn = injection_fn
        self.target_layer = target_layer
        self.hook_handle = None
        self.last_info = None
        
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        last_token_hidden = hidden_states[:, -1:, :]
        
        injected_hidden, info = self.injection_fn(last_token_hidden[:, 0, :])
        self.last_info = info
        
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = injected_hidden
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    def register(self, model: nn.Module):
        if hasattr(model, 'model'):
            layers = model.model.layers
        elif hasattr(model, 'transformer'):
            layers = model.transformer.h
        else:
            raise ValueError("Unknown model architecture")
        
        target = layers[self.target_layer]
        self.hook_handle = target.register_forward_hook(self._hook_fn)
        
        logger.info(f"Registered injection hook at layer {self.target_layer}")
    
    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("Removed injection hook")
    
    def get_last_info(self) -> Optional[dict]:
        return self.last_info
