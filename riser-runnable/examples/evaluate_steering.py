"""Run the initial baseline-versus-steered JSONL evaluation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riser.evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
