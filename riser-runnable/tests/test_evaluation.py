import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from riser.evaluation import (
    EvaluationExample,
    EvaluationRunner,
    exact_match,
    substring_match,
)


class TinyTokenizer:
    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist()
        if values == [9, 10]:
            return "Correct"
        return "Correct with extra explanation"


class BatchTokenizer:
    def __init__(self):
        self.padding_side = "right"

    def __call__(self, prompts, return_tensors=None, padding=False):
        if not isinstance(prompts, list) or not padding:
            raise AssertionError("batch tokenization must receive a padded list")
        lengths = [2 if "short" in prompt else 3 for prompt in prompts]
        max_length = max(lengths)
        input_ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            start = max_length - length
            input_ids[row, start:] = torch.arange(1, length + 1)
            attention_mask[row, start:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist()
        return {9: "A", 10: "B", 19: "S", 20: "T"}.get(values[0], "unknown")


class TinyModel:
    def __init__(self, generated_ids):
        self.generated_ids = torch.tensor([generated_ids])

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, self.generated_ids], dim=1)


class TinySteeredModel(TinyModel):
    def get_last_routing_info(self):
        return {"selected_primitives": [[1]], "strengths": [[0.8]]}


class FailingAfterFirstModel(TinyModel):
    def __init__(self, generated_ids):
        super().__init__(generated_ids)
        self.calls = 0

    def generate(self, input_ids, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("generation failed on second example")
        return super().generate(input_ids, **kwargs)


class BatchModel:
    def __init__(self, token_start):
        self.token_start = token_start
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append(tuple(input_ids.shape))
        generated = torch.arange(
            self.token_start,
            self.token_start + input_ids.shape[0],
            dtype=input_ids.dtype,
        ).unsqueeze(1)
        return torch.cat([input_ids, generated], dim=1)


class BatchSteeredModel(BatchModel):
    def __init__(self):
        super().__init__(19)
        self._routing = None

    def generate(self, input_ids, **kwargs):
        result = super().generate(input_ids, **kwargs)
        self._routing = {
            "selected_primitives": [
                [index] for index in range(input_ids.shape[0])
            ],
            "strengths": [[0.5 + index] for index in range(input_ids.shape[0])],
        }
        return result

    def get_last_routing_info(self):
        return self._routing


class EvaluationTests(unittest.TestCase):
    def test_reference_metrics_normalize_text(self):
        self.assertEqual(exact_match("  The Answer ", "the answer"), 1.0)
        self.assertEqual(substring_match("The answer is five.", "five"), 1.0)
        self.assertEqual(exact_match("wrong", "right"), 0.0)

    def test_runner_records_paired_outputs_and_token_usage(self):
        runner = EvaluationRunner(
            baseline_model=TinyModel([9, 10]),
            steered_model=TinySteeredModel([9, 11]),
            tokenizer=TinyTokenizer(),
        )
        examples = [EvaluationExample("q1", "Question", reference="correct")]

        results = runner.run(examples, metrics={"exact": exact_match})

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.baseline_output, "Correct")
        self.assertEqual(result.steered_output, "Correct with extra explanation")
        self.assertEqual(result.baseline_input_tokens, 2)
        self.assertEqual(result.baseline_output_tokens, 2)
        self.assertEqual(result.baseline_total_tokens, 4)
        self.assertEqual(result.steered_total_tokens, 4)
        self.assertEqual(result.metrics["exact"]["baseline"], 1.0)
        self.assertEqual(result.routing["selected_primitives"], [[1]])

    def test_runner_writes_one_json_object_per_line(self):
        runner = EvaluationRunner(
            baseline_model=TinyModel([9, 10]),
            steered_model=TinySteeredModel([9, 11]),
            tokenizer=TinyTokenizer(),
        )
        results = runner.run([EvaluationExample("q1", "Question")])

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.jsonl"
            runner.write_jsonl(results, output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["example_id"], "q1")

    def test_streaming_runner_flushes_completed_pairs_before_later_failure(self):
        runner = EvaluationRunner(
            baseline_model=FailingAfterFirstModel([9, 10]),
            steered_model=TinySteeredModel([9, 11]),
            tokenizer=TinyTokenizer(),
        )
        examples = [
            EvaluationExample("q1", "Question 1"),
            EvaluationExample("q2", "Question 2"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "streamed.jsonl"
            with self.assertRaisesRegex(RuntimeError, "second example"):
                runner.run_to_jsonl(examples, output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["example_id"], "q1")

    def test_streaming_runner_reports_progress_after_each_batch(self):
        runner = EvaluationRunner(
            baseline_model=BatchModel(9),
            steered_model=BatchSteeredModel(),
            tokenizer=BatchTokenizer(),
        )
        examples = [
            EvaluationExample("q1", "short one"),
            EvaluationExample("q2", "long prompt"),
            EvaluationExample("q3", "short three"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "progress.jsonl"
            with patch("riser.evaluation.runner.tqdm") as progress_factory:
                runner.run_to_jsonl(
                    examples,
                    output_path,
                    batch_size=2,
                    show_progress=True,
                    progress_desc="MMLU inference",
                )

        progress_factory.assert_called_once_with(
            total=3,
            desc="MMLU inference",
            unit="sample",
        )
        progress = progress_factory.return_value
        self.assertEqual(progress.update.call_args_list[0].args, (2,))
        self.assertEqual(progress.update.call_args_list[1].args, (1,))
        progress.close.assert_called_once_with()

    def test_runner_batches_generation_preserves_order_and_per_example_metadata(self):
        tokenizer = BatchTokenizer()
        baseline = BatchModel(9)
        steered = BatchSteeredModel()
        runner = EvaluationRunner(
            baseline_model=baseline,
            steered_model=steered,
            tokenizer=tokenizer,
        )
        examples = [
            EvaluationExample("q1", "short one", reference="A"),
            EvaluationExample("q2", "long prompt", reference="B"),
            EvaluationExample("q3", "short three", reference="A"),
        ]

        results = runner.run(
            examples,
            batch_size=2,
            metrics={"exact": exact_match},
        )

        self.assertEqual([result.example_id for result in results], ["q1", "q2", "q3"])
        self.assertEqual(baseline.calls, [(2, 3), (1, 2)])
        self.assertEqual(steered.calls, [(2, 3), (1, 2)])
        self.assertEqual([result.baseline_input_tokens for result in results], [2, 3, 2])
        self.assertEqual([result.baseline_output_tokens for result in results], [1, 1, 1])
        self.assertEqual([result.baseline_output for result in results], ["A", "B", "A"])
        self.assertEqual([result.steered_output for result in results], ["S", "T", "S"])
        self.assertEqual(results[0].routing["selected_primitives"], [0])
        self.assertEqual(results[1].routing["selected_primitives"], [1])
        self.assertEqual(results[2].routing["selected_primitives"], [0])
        self.assertEqual(tokenizer.padding_side, "left")

    def test_runner_merges_run_metadata_into_each_result(self):
        runner = EvaluationRunner(
            baseline_model=TinyModel([9, 10]),
            steered_model=TinySteeredModel([9, 11]),
            tokenizer=TinyTokenizer(),
        )

        result = runner.run(
            [EvaluationExample("q1", "Question", metadata={"subject": "math"})],
            result_metadata={
                "batch_size": 2,
                "max_new_tokens": 2048,
                "layer": 20,
                "primitive": [0],
                "strength": [1.0],
                "generated_at": "2025-08-25T08:15:00",
            },
        )[0]

        self.assertEqual(result.metadata["subject"], "math")
        self.assertEqual(result.metadata["batch_size"], 2)
        self.assertEqual(result.metadata["max_new_tokens"], 2048)
        self.assertEqual(result.metadata["generated_at"], "2025-08-25T08:15:00")


if __name__ == "__main__":
    unittest.main()
