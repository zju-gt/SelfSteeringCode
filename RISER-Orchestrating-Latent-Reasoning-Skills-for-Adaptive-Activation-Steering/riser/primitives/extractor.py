from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationPair:
    positive_activation: torch.Tensor
    negative_activation: torch.Tensor
    positive_text: str
    negative_text: str
    task: str
    task_id: str
    layer: int
    
    @property
    def difference(self) -> torch.Tensor:
        return self.positive_activation - self.negative_activation


class ContrastivePromptGenerator:
    
    def __init__(
        self,
        positive_templates: List[str],
        negative_templates: List[str]
    ):
        self.positive_templates = positive_templates
        self.negative_templates = negative_templates
        
    def generate_pair(
        self,
        task_description: str,
        task_id: str = "",
        **format_kwargs
    ) -> Tuple[str, str, str]:
        import random
        
        pos_template = random.choice(self.positive_templates)
        neg_template = random.choice(self.negative_templates)
        
        format_vars = {"task_description": task_description, **format_kwargs}
        positive_prompt = pos_template.format(**format_vars)
        negative_prompt = neg_template.format(**format_vars)
        
        return positive_prompt, negative_prompt, task_id


class ActivationExtractor:
    
    def __init__(
        self,
        model_name: str,
        target_layers: List[int] = None,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype)
        
        if target_layers is None:
            self.target_layers = None
        elif isinstance(target_layers, int):
            self.target_layers = [target_layers]
        else:
            self.target_layers = list(target_layers)
        
        logger.info(f"Loading model: {model_name}")
        model_kwargs = {
            "torch_dtype": self.dtype,
        }
        if device in {"auto", "balanced", "balanced_low_0"}:
            model_kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if device not in {"auto", "balanced", "balanced_low_0"}:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        if self.target_layers is None:
            num_layers = self._get_num_layers()
            middle_layer = num_layers // 2
            self.target_layers = [middle_layer]
            logger.info(f"No target layers specified, using middle layer: {middle_layer}")
        
        logger.info(f"Target layers for extraction: {self.target_layers}")
        
        self.activations = {}
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model, 'model'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer'):
            return len(self.model.transformer.h)
        else:
            raise ValueError("Unknown model architecture, cannot determine number of layers")
        
    def _hook_fn(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[name] = hidden_states.detach()
        return hook
    
    def _register_hooks(self):
        hooks = []
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")
        
        for layer_idx in self.target_layers:
            target = layers[layer_idx]
            hook = target.register_forward_hook(self._hook_fn(f"layer_{layer_idx}"))
            hooks.append(hook)
        
        return hooks
    
    def extract_activation(
        self,
        prompt: str,
        max_length: int = 512,
        layer_aggregation: str = "last",
    ) -> torch.Tensor:
        self.activations = {}
        hooks = self._register_hooks()
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            layer_activations = []
            for layer_idx in self.target_layers:
                activation = self.activations[f"layer_{layer_idx}"]
                last_token_activation = activation[0, -1, :]
                layer_activations.append(last_token_activation)
            
            if layer_aggregation == "last":
                result = layer_activations[-1]
            elif layer_aggregation == "concat":
                result = torch.cat(layer_activations, dim=0)
            elif layer_aggregation == "mean":
                result = torch.stack(layer_activations).mean(dim=0)
            else:
                raise ValueError(f"Unknown layer_aggregation: {layer_aggregation}")
            
            return result.cpu()
            
        finally:
            for hook in hooks:
                hook.remove()
    
    def extract_pair(
        self,
        positive_prompt: str,
        negative_prompt: str,
        task: str,
        task_id: str,
        max_length: int = 512,
        layer_aggregation: str = "last",
    ) -> ActivationPair:
        logger.debug(f"Extracting pair for task {task_id}")
        
        pos_activation = self.extract_activation(positive_prompt, max_length, layer_aggregation)
        neg_activation = self.extract_activation(negative_prompt, max_length, layer_aggregation)
        
        return ActivationPair(
            positive_activation=pos_activation,
            negative_activation=neg_activation,
            positive_text=positive_prompt,
            negative_text=negative_prompt,
            task=task,
            task_id=task_id,
            layer=self.target_layers[-1] if len(self.target_layers) == 1 else self.target_layers,
        )
    
    def extract_batch(
        self,
        prompt_pairs: List[Tuple[str, str, str, str]],
        max_length: int = 512,
        batch_size: int = 4,
        layer_aggregation: str = "last",
    ) -> List[ActivationPair]:
        pairs = []
        
        for i in range(0, len(prompt_pairs), batch_size):
            batch = prompt_pairs[i:i+batch_size]
            
            for pos_prompt, neg_prompt, task, task_id in batch:
                pair = self.extract_pair(
                    pos_prompt, neg_prompt, task, task_id, max_length, layer_aggregation
                )
                pairs.append(pair)
                
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i+len(batch)}/{len(prompt_pairs)} pairs")
        
        return pairs
