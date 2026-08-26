import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_results import load_results, summarize_results


class ResultAnalysisTests(unittest.TestCase):
    def test_summarize_results_reports_metrics_and_paired_outcomes(self):
        records = [
            {
                "example_id": "q1",
                "metrics": {
                    "mmlu_accuracy": {"baseline": 1.0, "steered": 1.0},
                    "exact_match": {"baseline": 0.0, "steered": 1.0},
                },
                "metadata": {"subject": "algebra"},
            },
            {
                "example_id": "q2",
                "metrics": {
                    "mmlu_accuracy": {"baseline": 0.0, "steered": 1.0},
                    "exact_match": {"baseline": 0.0, "steered": 0.0},
                },
                "metadata": {"subject": "algebra"},
            },
            {
                "example_id": "q3",
                "metrics": {
                    "mmlu_accuracy": {"baseline": 1.0, "steered": 0.0},
                },
                "metadata": {"subject": "geometry"},
            },
        ]

        summary = summarize_results(records)

        self.assertEqual(summary["count"], 3)
        mmlu = summary["metrics"]["mmlu_accuracy"]
        self.assertEqual(mmlu["baseline_count"], 3)
        self.assertEqual(mmlu["steered_count"], 3)
        self.assertAlmostEqual(mmlu["baseline_mean"], 2 / 3)
        self.assertAlmostEqual(mmlu["steered_mean"], 2 / 3)
        self.assertEqual(mmlu["both_correct"], 1)
        self.assertEqual(mmlu["both_wrong"], 0)
        self.assertEqual(mmlu["steered_gain"], 1)
        self.assertEqual(mmlu["steered_loss"], 1)

        algebra = summary["subjects"]["algebra"]["mmlu_accuracy"]
        self.assertEqual(algebra["baseline_count"], 2)
        self.assertAlmostEqual(algebra["baseline_mean"], 0.5)
        self.assertAlmostEqual(algebra["steered_mean"], 1.0)

    def test_load_results_skips_blank_lines_and_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "\n" + json.dumps({"metrics": {}}) + "\n", encoding="utf-8"
            )
            self.assertEqual(len(load_results(path)), 1)

            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_results(path)


if __name__ == "__main__":
    unittest.main()
