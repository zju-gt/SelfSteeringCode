"""Baseline-versus-steered generation runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional

import torch

from .records import EvaluationExample, EvaluationResult


Metric = Callable[[str, Optional[str]], Optional[float]]


class EvaluationRunner:
    def __init__(self, baseline_model, steered_model, tokenizer, device=None):
        self.baseline_model = baseline_model
        self.steered_model = steered_model
        self.tokenizer = tokenizer
        self.device = device

    def _model_device(self, model):
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    @staticmethod
    def _move_inputs(encoded, device):
        if device is None:
            return encoded
        if hasattr(encoded, "to"):
            return encoded.to(device)
        if isinstance(encoded, Mapping):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        return encoded

    @staticmethod
    def _sequences(output):
        sequences = getattr(output, "sequences", output)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise TypeError("model.generate must return a [batch, sequence] tensor")
        return sequences

    def _generate(self, model, prompt, generation_kwargs):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded.input_ids
        device = self._model_device(model)
        encoded = self._move_inputs(encoded, device)
        input_length = int(input_ids.shape[-1])

        start = time.perf_counter()
        output = model.generate(**encoded, **generation_kwargs)
        elapsed = time.perf_counter() - start
        sequences = self._sequences(output)
        generated_ids = sequences[0, input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_length = int(generated_ids.shape[-1])
        return {
            "text": text,
            "input_tokens": input_length,
            "output_tokens": generated_length,
            "total_tokens": input_length + generated_length,
            "latency_seconds": elapsed,
        }

    def _iter_results(
        self,
        examples: Iterable[EvaluationExample],
        generation_kwargs: Dict,
        metrics: Mapping[str, Metric],
    ) -> Iterator[EvaluationResult]:
        for example in examples:
            baseline = self._generate(
                self.baseline_model,
                example.prompt,
                generation_kwargs,
            )
            steered = self._generate(
                self.steered_model,
                example.prompt,
                generation_kwargs,
            )
            metric_values = {}
            for name, metric in metrics.items():
                metric_values[name] = {
                    "baseline": metric(baseline["text"], example.reference),
                    "steered": metric(steered["text"], example.reference),
                }
            get_routing_info = getattr(
                self.steered_model,
                "get_last_routing_info",
                None,
            )
            routing = get_routing_info() if get_routing_info else None
            yield EvaluationResult(
                example_id=example.example_id,
                prompt=example.prompt,
                reference=example.reference,
                baseline_output=baseline["text"],
                steered_output=steered["text"],
                baseline_input_tokens=baseline["input_tokens"],
                baseline_output_tokens=baseline["output_tokens"],
                baseline_total_tokens=baseline["total_tokens"],
                steered_input_tokens=steered["input_tokens"],
                steered_output_tokens=steered["output_tokens"],
                steered_total_tokens=steered["total_tokens"],
                baseline_latency_seconds=baseline["latency_seconds"],
                steered_latency_seconds=steered["latency_seconds"],
                metrics=metric_values,
                routing=routing,
                metadata=example.metadata,
            )

    def run(
        self,
        examples: Iterable[EvaluationExample],
        generation_kwargs: Optional[Dict] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
    ) -> List[EvaluationResult]:
        return list(
            self._iter_results(
                examples,
                generation_kwargs=dict(generation_kwargs or {}),
                metrics=dict(metrics or {}),
            )
        )

    def run_to_jsonl(
        self,
        examples: Iterable[EvaluationExample],
        path,
        generation_kwargs: Optional[Dict] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
    ) -> int:
        """Evaluate and flush each completed baseline/steered pair to JSONL."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for result in self._iter_results(
                examples,
                generation_kwargs=dict(generation_kwargs or {}),
                metrics=dict(metrics or {}),
            ):
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                count += 1
        return count

    @staticmethod
    def write_jsonl(results: Iterable[EvaluationResult], path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
