# MMLU preliminary experiment: default max new tokens 2048

## Goal

将 MMLU 初步实验的默认生成上限从 256 调整为 2048，以避免数学题的 reasoning 输出在最终答案前被截断。

## Scope

- 修改 `scripts/run_mmlu_preliminary.sh` 中的 `MAX_NEW_TOKENS` 默认值为 `2048`。
- 修改 `scripts/evaluate_mmlu.py` 的 `--max-new-tokens` CLI 默认值为 `2048`。
- 在 `docs/mmlu_preliminary_experiment_zh.md` 中说明默认值及其作用。
- 保留环境变量和 CLI 参数覆盖能力；用户仍可显式设置其他 token 上限。

## Non-goals

- 不立即重新运行实验。
- 不修改或覆盖现有 `artifacts/mmlu_preliminary/` 结果。
- 不改变生成、评测、steering 或 JSONL 保存逻辑。

## Verification

- 检查 shell 脚本和 CLI 默认值均为 2048。
- 运行现有单元测试、Python 编译检查和 `git diff --check`。
- 确认工作区只包含本次配置、文档及设计说明改动。
