"""Baseline-versus-steered generation runner."""

from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

import torch
from tqdm.auto import tqdm

from riser.utils.chat import format_chat_prompt, has_chat_template

from .records import EvaluationExample, EvaluationResult


Metric = Callable[[str, Optional[str]], Optional[float]]


class EvaluationRunner:
    def __init__(
        self,
        baseline_model,
        steered_model,
        tokenizer,
        device=None,
        use_chat_template: Optional[bool] = None,
    ):
        self.baseline_model = baseline_model
        self.steered_model = steered_model
        self.tokenizer = tokenizer
        self.device = device
        self.use_chat_template = (
            has_chat_template(tokenizer)
            if use_chat_template is None
            else use_chat_template
        )

    def _format_prompt(self, prompt: str) -> str:
        return format_chat_prompt(
            self.tokenizer,
            prompt,
            require_chat_template=self.use_chat_template,
        )

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
        formatted_prompt = self._format_prompt(prompt)
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
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

    def _generate_batch(self, model, prompts, generation_kwargs):
        """Generate a padded batch and return one accounting record per prompt."""

        prompts = list(prompts)
        if not prompts:
            return []
        if hasattr(self.tokenizer, "padding_side"):
            # Decoder-only generation must use left padding so the final input
            # position is the last real prompt token for every example.
            self.tokenizer.padding_side = "left"
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        encoded = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("tokenizer must return a [batch, sequence] input_ids tensor")

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            input_lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()
        else:
            input_lengths = [int(input_ids.shape[-1])] * len(prompts)

        device = self._model_device(model)
        encoded = self._move_inputs(encoded, device)
        start = time.perf_counter()
        output = model.generate(**encoded, **generation_kwargs)
        elapsed = time.perf_counter() - start
        sequences = self._sequences(output)
        if sequences.shape[0] != len(prompts):
            raise ValueError(
                "model.generate returned a different batch size: "
                f"expected {len(prompts)}, got {sequences.shape[0]}"
            )
        padded_length = int(input_ids.shape[-1])
        if sequences.shape[-1] < padded_length:
            raise ValueError("model.generate returned sequences shorter than the input batch")

        generated_ids = sequences[:, padded_length:]
        records = []
        for index, input_length in enumerate(input_lengths):
            generated = generated_ids[index]
            records.append(
                {
                    "text": self.tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                    ),
                    "input_tokens": int(input_length),
                    "output_tokens": int(generated.shape[-1]),
                    "total_tokens": int(input_length + generated.shape[-1]),
                    # This is the elapsed time for the batch that produced
                    # this example; per-example GPU timing needs synchronization.
                    "latency_seconds": elapsed,
                }
            )
        return records

    @staticmethod
    def _routing_for_example(routing, index: int, batch_size: int):
        if not isinstance(routing, Mapping):
            return routing
        selected = {}
        for key, value in routing.items():
            if isinstance(value, list) and len(value) == batch_size:
                selected[key] = value[index]
            else:
                selected[key] = value
        return selected

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    @staticmethod
    def _progress_total(examples: Iterable[EvaluationExample]) -> Optional[int]:
        try:
            return len(examples)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return None

    def _iter_results_with_progress(
        self,
        results: Iterator[EvaluationResult],
        examples: Iterable[EvaluationExample],
        batch_size: int,
        show_progress: bool,
        progress_desc: str,
    ) -> Iterator[EvaluationResult]:
        if not show_progress:
            yield from results
            return

        progress = tqdm(
            total=self._progress_total(examples),
            desc=progress_desc,
            unit="sample",
        )
        pending = 0
        try:
            for result in results:
                yield result
                pending += 1
                if pending >= batch_size:
                    progress.update(pending)
                    pending = 0
        finally:
            if pending:
                progress.update(pending)
            progress.close()

    @staticmethod
    def _result_from_parts(
        example: EvaluationExample,
        baseline,
        steered,
        metrics: Mapping[str, Metric],
        routing,
        result_metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationResult:
        metric_values = {}
        for name, metric in metrics.items():
            metric_values[name] = {
                "baseline": metric(baseline["text"], example.reference),
                "steered": metric(steered["text"], example.reference),
            }
        metadata = dict(example.metadata)
        metadata.update(dict(result_metadata or {}))
        return EvaluationResult(
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
            metadata=metadata,
        )

    def _iter_results(
        self,
        examples: Iterable[EvaluationExample],
        generation_kwargs: Dict,
        metrics: Mapping[str, Metric],
        batch_size: int = 1,
        result_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[EvaluationResult]:
        self._validate_batch_size(batch_size)
        get_routing_info = getattr(
            self.steered_model,
            "get_last_routing_info",
            None,
        )

        if batch_size == 1:
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
                routing = get_routing_info() if get_routing_info else None
                yield self._result_from_parts(
                    example,
                    baseline,
                    steered,
                    metrics,
                    routing,
                    result_metadata,
                )
            return

        iterator = iter(examples)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return
            prompts = [example.prompt for example in batch]
            baselines = self._generate_batch(
                self.baseline_model,
                prompts,
                generation_kwargs,
            )
            steereds = self._generate_batch(
                self.steered_model,
                prompts,
                generation_kwargs,
            )
            routing = get_routing_info() if get_routing_info else None
            for index, (example, baseline, steered) in enumerate(
                zip(batch, baselines, steereds)
            ):
                yield self._result_from_parts(
                    example,
                    baseline,
                    steered,
                    metrics,
                    self._routing_for_example(routing, index, len(batch)),
                    result_metadata,
                )

    def run(
        self,
        examples: Iterable[EvaluationExample],
        generation_kwargs: Optional[Dict] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        batch_size: int = 1,
        result_metadata: Optional[Mapping[str, Any]] = None,
        show_progress: bool = False,
        progress_desc: str = "Benchmark inference",
    ) -> List[EvaluationResult]:
        return list(
            self._iter_results_with_progress(
                self._iter_results(
                    examples,
                    generation_kwargs=dict(generation_kwargs or {}),
                    metrics=dict(metrics or {}),
                    batch_size=batch_size,
                    result_metadata=result_metadata,
                ),
                examples,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
        )

    def run_to_jsonl(
        self,
        examples: Iterable[EvaluationExample],
        path,
        generation_kwargs: Optional[Dict] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        batch_size: int = 1,
        result_metadata: Optional[Mapping[str, Any]] = None,
        show_progress: bool = False,
        progress_desc: str = "Benchmark inference",
    ) -> int:
        """Evaluate and flush each completed baseline/steered pair to JSONL."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for result in self._iter_results_with_progress(
                self._iter_results(
                    examples,
                    generation_kwargs=dict(generation_kwargs or {}),
                    metrics=dict(metrics or {}),
                    batch_size=batch_size,
                    result_metadata=result_metadata,
                ),
                examples,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_desc=progress_desc,
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
