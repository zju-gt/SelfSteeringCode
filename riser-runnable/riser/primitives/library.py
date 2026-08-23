from typing import Dict, List, Optional
import torch
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PrimitiveLibrary:
    
    def __init__(
        self,
        primitives: Optional[Dict[int, torch.Tensor]] = None,
        primitive_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.primitives = primitives or {}
        self.primitive_names = primitive_names or []
        self.metadata = metadata or {}
        
    @property
    def num_primitives(self) -> int:
        return len(self.primitives)
    
    @property
    def hidden_dim(self) -> Optional[int]:
        if self.primitives:
            first_key = next(iter(self.primitives))
            return self.primitives[first_key].shape[0]
        return None
    
    def add_primitive(
        self,
        primitive_id: int,
        vector: torch.Tensor,
        name: Optional[str] = None,
    ):
        self.primitives[primitive_id] = vector
        
        if name:
            while len(self.primitive_names) <= primitive_id:
                self.primitive_names.append(f"primitive_{len(self.primitive_names)}")
            self.primitive_names[primitive_id] = name
        
        logger.debug(f"Added primitive {primitive_id}: {name or 'unnamed'}")
    
    def get_primitive(self, primitive_id: int) -> torch.Tensor:
        if primitive_id not in self.primitives:
            raise KeyError(f"Primitive {primitive_id} not found in library")
        return self.primitives[primitive_id]
    
    def get_name(self, primitive_id: int) -> str:
        if primitive_id < len(self.primitive_names):
            return self.primitive_names[primitive_id]
        return f"primitive_{primitive_id}"
    
    def get_all_vectors(self) -> torch.Tensor:
        if not self.primitives:
            raise ValueError("Library is empty")
        
        sorted_ids = sorted(self.primitives.keys())
        vectors = [self.primitives[i].float() for i in sorted_ids]
        
        return torch.stack(vectors)
    
    def save(self, save_path: str, metadata_path: Optional[str] = None):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "primitives": self.primitives,
            "primitive_names": self.primitive_names,
            "num_primitives": self.num_primitives,
            "hidden_dim": self.hidden_dim,
        }
        
        torch.save(save_data, save_path)
        logger.info(f"Saved {self.num_primitives} primitives to {save_path}")
        
        if metadata_path:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            full_metadata = {
                "num_primitives": self.num_primitives,
                "hidden_dim": self.hidden_dim,
                "primitive_names": self.primitive_names,
                **self.metadata,
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, save_path: str, metadata_path: Optional[str] = None) -> "PrimitiveLibrary":
        save_path = Path(save_path)
        if not save_path.exists():
            raise FileNotFoundError(f"Primitive library not found: {save_path}")
        
        data = torch.load(save_path, map_location='cpu')
        
        primitives = data["primitives"]
        primitive_names = data.get("primitive_names", [])
        
        metadata = {}
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
        
        logger.info(
            f"Loaded {len(primitives)} primitives from {save_path} "
            f"(hidden_dim={data.get('hidden_dim')})"
        )
        
        return cls(
            primitives=primitives,
            primitive_names=primitive_names,
            metadata=metadata,
        )
    
    def __repr__(self) -> str:
        return (
            f"PrimitiveLibrary(num_primitives={self.num_primitives}, "
            f"hidden_dim={self.hidden_dim})"
        )
    
    def __len__(self) -> int:
        return self.num_primitives
    
    def summary(self) -> str:
        lines = [
            f"Primitive Library Summary",
            f"=" * 40,
            f"Number of primitives: {self.num_primitives}",
            f"Hidden dimension: {self.hidden_dim}",
            f"",
            f"Primitives:",
        ]
        
        for i in sorted(self.primitives.keys()):
            name = self.get_name(i)
            vector = self.primitives[i]
            norm = torch.norm(vector).item()
            lines.append(f"  [{i}] {name}: norm={norm:.4f}")
        
        return "\n".join(lines)
