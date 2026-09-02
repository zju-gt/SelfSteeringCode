# Default CLI Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow every numbered experiment script to use the three repository YAML files automatically when `--config` is omitted, while explicit `--config` arguments replace those defaults.

**Architecture:** Keep configuration selection in the shared `scripts/_common.py` CLI layer. Resolve repository-rooted default paths only when argparse produced no explicit config list, then delegate merging and validation to the existing `load_config()` function.

**Tech Stack:** Python 3.11+, argparse, pathlib, pytest, Markdown

---

### Task 1: Default and explicit configuration selection

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `scripts/_common.py`

- [x] **Step 1: Write failing tests for default loading and explicit replacement**

Update the stage 00 subprocess test so it omits all `--config` arguments and uses `cwd=tmp_path`. Add this focused parser/resolution test:

```python
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
```

Import `build_parser` and `resolved_config` from the scripts directory at the top of the test module.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_cli.py -q
```

Expected: the no-config subprocess case fails because argparse currently requires `--config`; the explicit-config test passes or remains independent of that failure.

- [x] **Step 3: Implement minimal default selection**

In `scripts/_common.py`, define repository-rooted defaults and make the flag optional:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "model.yaml",
    PROJECT_ROOT / "configs" / "data.yaml",
    PROJECT_ROOT / "configs" / "experiment.yaml",
)
```

Configure `--config` without `required=True`, and resolve paths with:

```python
config_paths = args.config if args.config is not None else DEFAULT_CONFIG_PATHS
config = load_config(config_paths, args.override)
```

Update the help text to state that the three repository configs are used when the flag is omitted and that repeated explicit flags replace the defaults.

- [x] **Step 4: Run focused and full tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_cli.py -q
PYTHONPATH=src pytest -q -rs
```

Expected: CLI tests pass; full suite reports all available tests passing with only the known local Qwen integration skip if PyTorch remains unavailable.

- [x] **Step 5: Commit CLI behavior**

```bash
git add scripts/_common.py tests/test_cli.py
git commit -m "feat: default experiment CLI configs"
```

### Task 2: Simplify user-facing commands

**Files:**
- Modify: `README.md`
- Modify: `docs/self_steering_experiment_guide.md`

- [x] **Step 1: Update README commands**

Remove repeated default `--config` arguments from the override example and all eight pipeline commands. Add one explicit custom-config example and state that supplying any `--config` replaces the three defaults.

- [x] **Step 2: Update the Chinese experiment guide**

Delete the `BASE_CONFIG` array, replace every `"${BASE_CONFIG[@]}"` occurrence with direct script invocation, and add a concise example explaining complete replacement:

```bash
python scripts/00_prepare_data.py --config custom.yaml
```

Keep `EVALS` override arrays and all formal MATH500/AIME workflows otherwise unchanged.

- [x] **Step 3: Verify documentation and repository behavior**

Run:

```bash
rg -n "BASE_CONFIG|--config configs/model.yaml" README.md docs/self_steering_experiment_guide.md
PYTHONPATH=src pytest -q -rs
python -m compileall -q src scripts tests
git diff --check
```

Expected: the search has no matches; tests and compilation succeed; `git diff --check` emits no errors.

- [x] **Step 4: Commit documentation**

```bash
git add README.md docs/self_steering_experiment_guide.md docs/superpowers/plans/2026-09-02-default-cli-configs.md
git commit -m "docs: simplify experiment CLI commands"
```
