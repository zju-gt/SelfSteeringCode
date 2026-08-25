import json
import tempfile
import unittest
from pathlib import Path


from scripts.prepare_mmlu_eval import (
    answer_to_letter,
    build_neutral_prompt,
    convert_rows,
    extract_last_choice,
    mmlu_choice_match,
)
from scripts.evaluate_mmlu import build_parser as build_evaluator_parser


class MMLUPreparationTests(unittest.TestCase):
    def test_answer_to_letter_accepts_integer_and_letter_answers(self):
        self.assertEqual(answer_to_letter(0), "A")
        self.assertEqual(answer_to_letter(3), "D")
        self.assertEqual(answer_to_letter("b"), "B")

    def test_answer_to_letter_rejects_invalid_values(self):
        for value in (-1, 4, "E", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    answer_to_letter(value)

    def test_extract_last_choice_uses_the_final_standalone_option(self):
        text = "Option A is considered first. The final answer is (c)."
        self.assertEqual(extract_last_choice(text), "C")
        self.assertIsNone(extract_last_choice("No option was selected."))

    def test_mmlu_choice_match_compares_last_choice(self):
        self.assertEqual(mmlu_choice_match("The answer is A.", "A"), 1.0)
        self.assertEqual(mmlu_choice_match("A was rejected; final answer B", "B"), 1.0)
        self.assertEqual(mmlu_choice_match("The answer is A.", "B"), 0.0)
        self.assertIsNone(mmlu_choice_match("The answer is A.", None))

    def test_build_neutral_prompt_contains_only_question_choices_and_answer_slot(self):
        prompt = build_neutral_prompt(
            "Which option?", ["one", "two", "three", "four"]
        )

        self.assertEqual(
            prompt,
            "Question:\n"
            "Which option?\n\n"
            "Choices:\n"
            "A. one\n"
            "B. two\n"
            "C. three\n"
            "D. four\n\n"
            "Answer:",
        )
        self.assertNotIn("reason", prompt.lower())
        self.assertNotIn("explain", prompt.lower())

    def test_convert_rows_writes_prompt_reference_and_metadata(self):
        rows = [
            {
                "id": "mmlu_0",
                "task_id": "mmlu_0",
                "subject": "abstract_algebra",
                "question": "Which option?",
                "choices": ["one", "two", "three", "four"],
                "answer": 1,
                "positive_prompt": "Reason carefully. Answer:",
                "negative_prompt": "Only answer. Answer:",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "pairs.jsonl"
            output_path = Path(directory) / "evaluation.jsonl"
            input_path.write_text(
                json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8"
            )

            count = convert_rows(input_path, output_path)

            self.assertEqual(count, 1)
            converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(converted["id"], "mmlu_0")
        self.assertEqual(
            converted["prompt"],
            "Question:\n"
            "Which option?\n\n"
            "Choices:\n"
            "A. one\n"
            "B. two\n"
            "C. three\n"
            "D. four\n\n"
            "Answer:",
        )
        self.assertEqual(converted["reference"], "B")
        self.assertEqual(converted["metadata"]["subject"], "abstract_algebra")
        self.assertEqual(converted["metadata"]["choices"][1], "two")


class MMLUEvaluatorTests(unittest.TestCase):
    def test_parser_defaults_to_2048_new_tokens(self):
        args = build_evaluator_parser().parse_args(
            [
                "--model",
                "Qwen/model",
                "--library",
                "primitives.pt",
                "--layer",
                "20",
                "--input",
                "evaluation.jsonl",
                "--output",
                "results.jsonl",
            ]
        )

        self.assertEqual(args.max_new_tokens, 2048)
        self.assertEqual(args.batch_size, 1)

    def test_parser_accepts_batch_size(self):
        args = build_evaluator_parser().parse_args(
            [
                "--model",
                "Qwen/model",
                "--library",
                "primitives.pt",
                "--layer",
                "20",
                "--input",
                "evaluation.jsonl",
                "--output",
                "results.jsonl",
                "--batch-size",
                "4",
            ]
        )

        self.assertEqual(args.batch_size, 4)

    def test_parser_accepts_fixed_route_arguments(self):
        args = build_evaluator_parser().parse_args(
            [
                "--model",
                "Qwen/model",
                "--library",
                "primitives.pt",
                "--layer",
                "20",
                "--input",
                "evaluation.jsonl",
                "--output",
                "results.jsonl",
                "--fixed-primitives",
                "0",
                "2",
                "--fixed-strengths",
                "0.5",
                "1.0",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--max-new-tokens",
                "32",
            ]
        )

        self.assertEqual(args.layer, 20)
        self.assertEqual(args.fixed_primitives, [0, 2])
        self.assertEqual(args.fixed_strengths, [0.5, 1.0])
        self.assertEqual(args.max_new_tokens, 32)


if __name__ == "__main__":
    unittest.main()
