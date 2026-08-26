#!/usr/bin/env bash

set -euo pipefail

# User-editable defaults. Environment variables override these values.
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_FILE="${RESULTS_FILE:-artifacts/mmlu_preliminary/results_260826_0709.jsonl}"
METRICS_RAW="${METRICS_RAW:-mmlu_accuracy exact_match substring_match}"

if [[ "$#" -gt 1 ]]; then
    echo "Usage: bash scripts/analyze_results.sh [path/to/results.jsonl]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$#" -eq 1 ]]; then
    RESULTS_FILE="$1"
fi

METRIC_ARGS=()
if [[ -n "${METRICS_RAW}" ]]; then
    read -r -a METRICS <<< "${METRICS_RAW}"
    for metric in "${METRICS[@]}"; do
        METRIC_ARGS+=(--metric "${metric}")
    done
fi

exec "${PYTHON_BIN}" scripts/analyze_results.py \
    --input "${RESULTS_FILE}" \
    "${METRIC_ARGS[@]}"
