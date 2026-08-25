import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from riser.evaluation.paths import resolve_results_path


class ResultPathTests(unittest.TestCase):
    def test_default_path_contains_timestamp_in_requested_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            path = resolve_results_path(
                output=None,
                output_dir="artifacts/mmlu_preliminary",
                now=datetime(2025, 8, 25, 8, 15),
            )

        self.assertEqual(
            path,
            Path("artifacts/mmlu_preliminary/results_250825_0815.jsonl"),
        )

    def test_explicit_output_and_environment_override_default(self):
        explicit = resolve_results_path(
            output="custom/results.jsonl",
            output_dir="artifacts/mmlu_preliminary",
            now=datetime(2025, 8, 25, 8, 15),
        )
        self.assertEqual(explicit, Path("custom/results.jsonl"))

        with patch.dict(os.environ, {"RESULTS_OUTPUT": "env/results.jsonl"}):
            environment = resolve_results_path(
                output=None,
                output_dir="artifacts/mmlu_preliminary",
                now=datetime(2025, 8, 25, 8, 15),
            )
        self.assertEqual(environment, Path("env/results.jsonl"))


if __name__ == "__main__":
    unittest.main()
