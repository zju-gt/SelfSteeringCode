import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_prepare_data.py",
    "01_score_demands.py",
    "02_prepare_items.py",
    "03_capture_contrasts.py",
    "04_extract_vectors.py",
    "05_analyze_similarity.py",
    "06_run_steering.py",
    "07_score_generations.py",
]


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    source = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source, env.get("PYTHONPATH", "")) if part
    )
    return env


def test_every_cli_provides_help_without_loading_model() -> None:
    for name in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--help"],
            capture_output=True,
            text=True,
            env=subprocess_env(),
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "--config" in result.stdout
        assert "--override" in result.stdout


def test_prepare_data_cli_runs_with_local_fixtures(tmp_path: Path) -> None:
    mmlu = ROOT / "tests" / "fixtures" / "mmlu_items.jsonl"
    math500 = ROOT / "tests" / "fixtures" / "math_items.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "00_prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "model.yaml"),
            "--config",
            str(ROOT / "configs" / "data.yaml"),
            "--config",
            str(ROOT / "configs" / "experiment.yaml"),
            "--override",
            f"data.sources.mmlu.local_path={mmlu.as_posix()}",
            "--override",
            f"data.sources.math500.local_path={math500.as_posix()}",
            "--override",
            f"experiment.paths.data_dir={tmp_path.as_posix()}/data",
            "--override",
            f"experiment.paths.outputs_dir={tmp_path.as_posix()}/outputs",
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    rows = (
        (tmp_path / "data" / "processed" / "math500.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 1
