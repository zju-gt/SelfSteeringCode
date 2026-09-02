import os
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _common import build_parser, resolved_config


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
        assert "replaces the default" in result.stdout
        assert "--override" in result.stdout


def test_prepare_data_cli_uses_default_configs_from_any_working_directory(
    tmp_path: Path,
) -> None:
    mmlu = ROOT / "tests" / "fixtures" / "mmlu_items.jsonl"
    math500 = ROOT / "tests" / "fixtures" / "math_items.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "00_prepare_data.py"),
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
        cwd=tmp_path,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    rows = (
        (tmp_path / "data" / "processed" / "math500.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 1


def test_explicit_config_replaces_default_configs(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
model:
  name: custom-model
  revision: custom-tag
  num_hidden_layers: 1
data:
  enabled_steering_datasets: []
experiment:
  capabilities: [QLl, QLq, CL, MCr]
  target_layer: 0
  high_demand_threshold: 4
  low_demand_threshold: 1
  vector_scaling: raw
  alphas: [0.0]
""".strip(),
        encoding="utf-8",
    )

    args = build_parser("test").parse_args(["--config", str(config_path)])
    config = resolved_config(args)

    assert config["model"]["name"] == "custom-model"
    assert "cache_dir" not in config["model"]


def test_score_demands_builds_metamind_client() -> None:
    script = runpy.run_path(str(SCRIPTS_DIR / "01_score_demands.py"))
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    script["build_annotation_client"](FakeOpenAI)

    assert captured["base_url"] == "https://newapi.metamind.work/v1"
    assert captured["api_key"].startswith("sk-")
