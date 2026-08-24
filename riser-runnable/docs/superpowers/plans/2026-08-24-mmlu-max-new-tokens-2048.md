# MMLU max-new-tokens 2048 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MMLU 初步实验的默认生成上限统一调整为 2048，并保留现有结果文件不变。

**Architecture:** Shell runner 继续通过 `MAX_NEW_TOKENS` 环境变量向评测 CLI 传递生成上限；评测 CLI 的 argparse 默认值与 shell 默认值保持一致。中文文档记录该默认值和覆盖方式，测试锁定 CLI 默认值。

**Tech Stack:** Bash, Python argparse, unittest, Markdown。

---

### Task 1: Update defaults and add regression coverage

**Files:**
- Modify: `riser-runnable/scripts/run_mmlu_preliminary.sh:21`
- Modify: `riser-runnable/scripts/evaluate_mmlu.py:35`
- Test: `riser-runnable/tests/test_mmlu_scripts.py`

- [ ] **Step 1: Add a failing parser-default test**

Add this method to `MMLUEvaluatorTests`:

```python
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
```

- [ ] **Step 2: Run the focused test and verify the current default fails**

Run from `riser-runnable`:

```bash
python -m unittest tests.test_mmlu_scripts.MMLUEvaluatorTests.test_parser_defaults_to_2048_new_tokens -v
```

Expected before implementation: `FAIL` because the current argparse default is `256`.

- [ ] **Step 3: Change both runtime defaults**

Change the shell default:

```bash
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
```

Change the Python CLI default:

```python
parser.add_argument("--max-new-tokens", type=int, default=2048)
```

Do not alter explicit CLI or environment-variable overrides.

- [ ] **Step 4: Run the focused test and verify it passes**

```bash
python -m unittest tests.test_mmlu_scripts.MMLUEvaluatorTests.test_parser_defaults_to_2048_new_tokens -v
```

Expected: `OK`.

### Task 2: Document the new default without changing artifacts

**Files:**
- Modify: `riser-runnable/docs/mmlu_preliminary_experiment_zh.md`

- [ ] **Step 1: Update the configuration section**

State that `MAX_NEW_TOKENS` defaults to `2048`, that it is intended to prevent reasoning truncation, and that it can still be overridden, for example:

```bash
MAX_NEW_TOKENS=512 bash scripts/run_mmlu_preliminary.sh
```

Explicitly state that changing the default does not regenerate the existing files under `artifacts/mmlu_preliminary/` until the runner is executed again.

### Task 3: Verify and commit the scoped change

**Files:**
- No additional files; do not modify `riser-runnable/artifacts/mmlu_preliminary/`.

- [ ] **Step 1: Run the complete test suite**

```bash
python -m unittest discover -s tests -v
```

Expected: all applicable tests pass; the existing optional Judge test may remain skipped.

- [ ] **Step 2: Run compile, CLI, and whitespace checks**

```bash
python -m compileall -q riser scripts tests
python scripts/evaluate_mmlu.py --help
git diff --check
```

Expected: all commands exit successfully and help output shows `--max-new-tokens`.

- [ ] **Step 3: Confirm scope and commit**

```bash
git status --short
git diff --stat
git add -- riser-runnable/scripts/run_mmlu_preliminary.sh riser-runnable/scripts/evaluate_mmlu.py riser-runnable/tests/test_mmlu_scripts.py riser-runnable/docs/mmlu_preliminary_experiment_zh.md
git commit -m "chore: raise MMLU generation token default"
```

Expected: only the shell script, evaluator, regression test, and Chinese documentation are committed; existing artifacts remain unchanged.
