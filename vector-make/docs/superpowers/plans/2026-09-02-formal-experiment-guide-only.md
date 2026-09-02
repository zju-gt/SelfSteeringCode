# Formal-Only Self-Steering Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the smoke-test workflow from the Chinese experiment guide while retaining real-data formal runs and the factual `--limit` CLI reference.

**Architecture:** Modify only `docs/self_steering_experiment_guide.md`. Delete smoke-specific text and commands, then renumber later sections; do not change experiment code, YAML, or the historical guide implementation plan.

**Tech Stack:** Markdown, ripgrep, pytest.

---

### Task 1: Remove the smoke workflow

**Files:**
- Modify: `docs/self_steering_experiment_guide.md`

- [x] **Step 1: Verify the old smoke content is present**

Run:

```bash
rg -n -i "smoke|data_smoke|outputs_smoke" docs/self_steering_experiment_guide.md
```

Expected: matches in the introduction, `max_new_tokens` advice, and the complete smoke-test section.

- [x] **Step 2: Make the minimal document edit**

Apply exactly these content changes:

- change the introduction from “small-scale check then formal experiment” to direct formal experiment execution;
- remove the complete `## 6. 小规模 smoke test` section, including `SMOKE` overrides and reduced-data commands;
- keep the neutral explanation of which stages read `--limit`;
- change the `max_new_tokens` advice from a smoke-specific range to a general task/output-length recommendation;
- renumber the former sections 7–10 to sections 6–9;
- keep both formal command sets: default MATH500 and MATH500 plus AIME 2024/2025/2026.

- [x] **Step 3: Verify smoke-specific content is absent**

Run:

```bash
rg -n -i "smoke|data_smoke|outputs_smoke" docs/self_steering_experiment_guide.md
```

Expected: no output and exit status 1.

### Task 2: Verify formal-run content and repository health

**Files:**
- Verify: `docs/self_steering_experiment_guide.md`

- [x] **Step 1: Verify both formal workflows remain**

Run:

```bash
rg -n "默认：只评测 MATH500|扩展到 MATH500 和 AIME|00_prepare_data.py|07_score_generations.py" docs/self_steering_experiment_guide.md
```

Expected: both formal headings and the first/last pipeline commands are present.

- [x] **Step 2: Verify Markdown diff and project tests**

Run:

```bash
git diff --check
pytest -q
```

Expected: no whitespace errors and no test failures; the real Qwen test may be explicitly skipped on the current workstation.

- [x] **Step 3: Commit the change**

```bash
git add docs/self_steering_experiment_guide.md docs/superpowers/plans/2026-09-02-formal-experiment-guide-only.md
git commit -m "docs: focus experiment guide on formal runs"
```
