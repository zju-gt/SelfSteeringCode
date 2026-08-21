from typing import List, Dict, Optional, Tuple
import os
import json
import time
import logging
from dataclasses import dataclass
import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .extractor import ActivationPair

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    score: float
    reasoning: str
    generation: str


class LLMJudgeFilter:
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key_env: str = "ANTHROPIC_API_KEY",
        eval_prompt_template: str = "",
        positive_threshold: float = 80.0,
        negative_threshold: float = 20.0,
        positive_eval_config: Optional[Dict] = None,
        negative_eval_config: Optional[Dict] = None,
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit_delay: float = 1.0,
    ):
        self.provider = provider
        self.model = model
        self.eval_prompt_template = eval_prompt_template
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.positive_eval_config = positive_eval_config or {}
        self.negative_eval_config = negative_eval_config or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM Judge with {provider}/{model}")
    
    def _call_judge_api(
        self,
        generation: str,
        task: str,
        expected_behavior: str,
    ) -> QualityScore:
        eval_prompt = self.eval_prompt_template.format(
            generation=generation,
            task=task,
            expected_behavior=expected_behavior,
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": eval_prompt
                    }],
                    timeout=self.timeout,
                )
                
                response_text = response.content[0].text.strip()
                
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                
                score = float(result.get("score", 0))
                reasoning = result.get("reasoning", "")
                
                time.sleep(self.rate_limit_delay)
                
                return QualityScore(
                    score=score,
                    reasoning=reasoning,
                    generation=generation,
                )
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from API response (attempt {attempt+1}): {e}")
                logger.debug(f"Response text: {response_text}")
                if attempt == self.max_retries - 1:
                    return QualityScore(score=0.0, reasoning="Parse error", generation=generation)
                time.sleep(self.rate_limit_delay * 2)
                
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.rate_limit_delay * 2)
        
        return QualityScore(score=0.0, reasoning="Max retries exceeded", generation=generation)
    
    def generate_with_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def evaluate_pair(
        self,
        pair: ActivationPair,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> Tuple[QualityScore, QualityScore, bool]:
        logger.debug(f"Generating text for task {pair.task_id}")
        
        pos_generation = self.generate_with_model(
            model, tokenizer, pair.positive_text, max_new_tokens, device
        )
        neg_generation = self.generate_with_model(
            model, tokenizer, pair.negative_text, max_new_tokens, device
        )
        
        pos_score = self._call_judge_api(
            generation=pos_generation,
            task=pair.task,
            expected_behavior=self.positive_eval_config.get(
                "expected_behavior",
                "rigorous step-by-step reasoning"
            ),
        )
        
        neg_score = self._call_judge_api(
            generation=neg_generation,
            task=pair.task,
            expected_behavior=self.negative_eval_config.get(
                "expected_behavior",
                "minimal or no reasoning"
            ),
        )
        
        is_valid = (
            pos_score.score >= self.positive_threshold and
            neg_score.score <= self.negative_threshold
        )
        
        logger.debug(
            f"Task {pair.task_id}: pos={pos_score.score:.1f}, "
            f"neg={neg_score.score:.1f}, valid={is_valid}"
        )
        
        return pos_score, neg_score, is_valid
    
    def filter_pairs(
        self,
        pairs: List[ActivationPair],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> Tuple[List[ActivationPair], List[Dict]]:
        filtered_pairs = []
        evaluation_results = []
        
        logger.info(f"Filtering {len(pairs)} activation pairs...")
        
        for i, pair in enumerate(pairs):
            try:
                pos_score, neg_score, is_valid = self.evaluate_pair(
                    pair, model, tokenizer, max_new_tokens, device
                )
                
                result = {
                    "task_id": pair.task_id,
                    "task": pair.task,
                    "positive_score": pos_score.score,
                    "positive_reasoning": pos_score.reasoning,
                    "negative_score": neg_score.score,
                    "negative_reasoning": neg_score.reasoning,
                    "is_valid": is_valid,
                }
                evaluation_results.append(result)
                
                if is_valid:
                    filtered_pairs.append(pair)
                
                if (i + 1) % 10 == 0:
                    valid_count = len(filtered_pairs)
                    logger.info(
                        f"Processed {i+1}/{len(pairs)} pairs, "
                        f"valid: {valid_count} ({100*valid_count/(i+1):.1f}%)"
                    )
                    
            except Exception as e:
                logger.error(f"Error evaluating pair {pair.task_id}: {e}")
                evaluation_results.append({
                    "task_id": pair.task_id,
                    "task": pair.task,
                    "error": str(e),
                    "is_valid": False,
                })
        
        logger.info(
            f"Filtering complete: {len(filtered_pairs)}/{len(pairs)} pairs passed "
            f"({100*len(filtered_pairs)/len(pairs):.1f}%)"
        )
        
        return filtered_pairs, evaluation_results
