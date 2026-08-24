import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
