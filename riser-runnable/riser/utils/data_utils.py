from typing import List, Dict, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_tasks(
    task_sources: List[Dict[str, Any]],
    cache_dir: str = "cache",
) -> List[Dict[str, str]]:
    all_tasks = []
    
    for source_config in task_sources:
        source_name = source_config["name"]
        num_samples = source_config.get("num_samples", 100)
        
        logger.info(f"Loading {num_samples} samples from {source_name}")
        
        try:
            tasks = _load_from_source(source_name, source_config, cache_dir)
            if len(tasks) > num_samples:
                import random
                tasks = random.sample(tasks, num_samples)
            
            all_tasks.extend(tasks)
            logger.info(f"Loaded {len(tasks)} tasks from {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to load from {source_name}: {e}")
    
    logger.info(f"Total tasks loaded: {len(all_tasks)}")
    return all_tasks


def _load_from_source(
    source_name: str,
    config: Dict[str, Any],
    cache_dir: str,
) -> List[Dict[str, str]]:
    if source_name == "arc_challenge":
        return _load_arc_challenge(cache_dir)
    elif source_name == "mmlu":
        return _load_mmlu(config, cache_dir)
    elif source_name == "gpqa":
        return _load_gpqa(cache_dir)
    elif source_name == "truthful_qa":
        return _load_truthful_qa(cache_dir)
    elif source_name == "gsm8k":
        return _load_gsm8k(cache_dir)
    elif source_name == "math":
        return _load_math(cache_dir)
    else:
        raise ValueError(f"Unknown source: {source_name}")


def _load_arc_challenge(cache_dir: str) -> List[Dict[str, str]]:
    from datasets import load_dataset
    
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    
    tasks = []
    for i, item in enumerate(dataset):
        question = item["question"]
        choices = item["choices"]["text"]
        choice_str = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
        
        task_text = f"{question}\n\nChoices:\n{choice_str}"
        
        tasks.append({
            "task_id": f"arc_{i}",
            "task": task_text,
            "source": "arc_challenge",
        })
    
    return tasks


def _load_mmlu(config: Dict[str, Any], cache_dir: str) -> List[Dict[str, str]]:
    from datasets import load_dataset
    
    subjects = config.get("subjects", ["abstract_algebra"])
    tasks = []
    
    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="dev")
        
        for i, item in enumerate(dataset):
            question = item["question"]
            choices = item["choices"]
            choice_str = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
            
            task_text = f"{question}\n\nChoices:\n{choice_str}"
            
            tasks.append({
                "task_id": f"mmlu_{subject}_{i}",
                "task": task_text,
                "source": "mmlu",
            })
    
    return tasks


def _load_gpqa(cache_dir: str) -> List[Dict[str, str]]:
    logger.warning("GPQA loading not fully implemented, using dummy data")
    return [
        {
            "task_id": f"gpqa_{i}",
            "task": f"GPQA placeholder question {i}",
            "source": "gpqa",
        }
        for i in range(10)
    ]


def _load_truthful_qa(cache_dir: str) -> List[Dict[str, str]]:
    from datasets import load_dataset
    
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    tasks = []
    for i, item in enumerate(dataset):
        question = item["question"]
        
        tasks.append({
            "task_id": f"truthful_qa_{i}",
            "task": question,
            "source": "truthful_qa",
        })
    
    return tasks


def _load_gsm8k(cache_dir: str) -> List[Dict[str, str]]:
    from datasets import load_dataset
    
    dataset = load_dataset("gsm8k", "main", split="train")
    
    tasks = []
    for i, item in enumerate(dataset):
        question = item["question"]
        
        tasks.append({
            "task_id": f"gsm8k_{i}",
            "task": question,
            "source": "gsm8k",
        })
    
    return tasks


def _load_math(cache_dir: str) -> List[Dict[str, str]]:
    logger.warning("MATH loading not fully implemented, using dummy data")
    return [
        {
            "task_id": f"math_{i}",
            "task": f"MATH placeholder problem {i}",
            "source": "math",
        }
        for i in range(10)
    ]


def save_results(
    results: Any,
    output_path: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    return config
