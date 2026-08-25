import json
import tempfile
import unittest
from pathlib import Path

from riser.evaluation.cli import build_parser, load_examples
from examples.collect_mmlu_math_vectors import make_prompt_pair


class ExampleCliTests(unittest.TestCase):
    def test_collector_uses_riser_vector_elicitation_prompts(self):
        task, positive, negative = make_prompt_pair(
            "abstract_algebra",
            "Which option?",
            ["one", "two", "three", "four"],
        )

        self.assertEqual(task, "abstract_algebra\n\nWhich option?\n\nChoices:\nA. one\nB. two\nC. three\nD. four")
        self.assertIn(
            "Role: You are a meticulous logician focused on absolute precision.",
            positive,
        )
        self.assertIn(
            "Task: Derive the answer to the following question using proof-level rigor.",
            positive,
        )
        self.assertIn(
            "• Verify: Check each intermediate calculation or logic jump for errors.",
            positive,
        )
        self.assertTrue(positive.endswith("Question:\n" + task + "\nRigorous Derivation:"))

        self.assertIn(
            'Role: You are a fluent conversationalist acting on "autopilot".',
            negative,
        )
        self.assertIn(
            "Task: Provide a plausibly sounding answer based on surface-level associations.",
            negative,
        )
        self.assertIn(
            '• Approximate: Do not perform actual calculations or verification. Use "ballpark" figures.',
            negative,
        )
        self.assertTrue(negative.endswith("Question:\n" + task + "\nPlausible Response:"))

    def test_evaluation_parser_accepts_core_arguments(self):
        args = build_parser().parse_args(
            [
                "--model",
                "tiny-model",
                "--router",
                "router.pt",
                "--library",
                "library.pt",
                "--layer",
                "4",
                "--input",
                "examples.jsonl",
                "--output",
                "results.jsonl",
                "--max-new-tokens",
                "12",
                "--device",
                "cpu",
                "--no-cache-routing",
            ]
        )

        self.assertEqual(args.layer, 4)
        self.assertEqual(args.max_new_tokens, 12)
        self.assertTrue(args.no_cache_routing)

    def test_load_examples_reads_jsonl_and_missing_file_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(
                json.dumps({"id": "q1", "prompt": "Question", "reference": "Answer"})
                + "\n",
                encoding="utf-8",
            )
            examples = load_examples(path)

        self.assertEqual(examples[0].example_id, "q1")
        self.assertEqual(examples[0].reference, "Answer")
        with self.assertRaises(FileNotFoundError):
            load_examples("does-not-exist.jsonl")


if __name__ == "__main__":
    unittest.main()
