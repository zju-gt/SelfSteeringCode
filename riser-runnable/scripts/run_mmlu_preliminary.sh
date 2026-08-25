#!/usr/bin/env bash

set -euo pipefail
export CUDA_VISIBLE_DEVICES=2

# Edit these values for the server, or override them as environment variables.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/sda/llm_weights/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
LAYERS_RAW="${LAYERS_RAW:-20}"
INJECT_LAYER="${INJECT_LAYER:-20}"
DATASET_NAME="${DATASET_NAME:-cais/mmlu}"
SUBJECTS_RAW="${SUBJECTS_RAW:-abstract_algebra college_mathematics elementary_mathematics high_school_mathematics}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
CLUSTERS="${CLUSTERS:-6}"
MAX_LENGTH="${MAX_LENGTH:-512}"
AGGREGATION="${AGGREGATION:-last}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FIXED_PRIMITIVES_RAW="${FIXED_PRIMITIVES_RAW:-0}"
FIXED_STRENGTHS_RAW="${FIXED_STRENGTHS_RAW:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

read -r -a LAYERS <<< "${LAYERS_RAW}"
read -r -a SUBJECTS <<< "${SUBJECTS_RAW}"
read -r -a FIXED_PRIMITIVES <<< "${FIXED_PRIMITIVES_RAW}"
read -r -a FIXED_STRENGTHS <<< "${FIXED_STRENGTHS_RAW}"

if [[ "${MODEL_PATH}" == /path/to/* || -z "${MODEL_PATH}" ]]; then
    echo "请先设置 MODEL_PATH 为 Qwen 模型目录或 Hugging Face 模型名。" >&2
    exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "找不到 Python 命令: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ "${#FIXED_PRIMITIVES[@]}" -ne "${#FIXED_STRENGTHS[@]}" ]]; then
    echo "FIXED_PRIMITIVES_RAW 和 FIXED_STRENGTHS_RAW 的数量必须相同。" >&2
    exit 2
fi

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE 必须是正整数。" >&2
    exit 2
fi

if [[ "${DEVICE}" == cuda* ]]; then
    "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
        echo "DEVICE=${DEVICE} 但 PyTorch 未检测到 CUDA。" >&2
        exit 2
    }
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/artifacts/mmlu_preliminary"
fi
mkdir -p "${OUTPUT_DIR}"

PROMPT_PAIRS_OUTPUT="${OUTPUT_DIR}/prompt_pairs.jsonl"
VECTORS_OUTPUT="${OUTPUT_DIR}/vectors.pt"
LIBRARY_OUTPUT="${OUTPUT_DIR}/primitives.pt"
METADATA_OUTPUT="${OUTPUT_DIR}/primitives.json"
EVALUATION_INPUT="${OUTPUT_DIR}/evaluation.jsonl"
RESULTS_OUTPUT="${OUTPUT_DIR}/results.jsonl"

# echo "[1/3] Collecting MMLU prompt pairs and activation vectors..."
# "${PYTHON_BIN}" examples/collect_mmlu_math_vectors.py \
#     --model "${MODEL_PATH}" \
#     --dataset-name "${DATASET_NAME}" \
#     --subjects "${SUBJECTS[@]}" \
#     --split "${SPLIT}" \
#     --num-samples "${NUM_SAMPLES}" \
#     --seed "${SEED}" \
#     --layers "${LAYERS[@]}" \
#     --device "${DEVICE}" \
#     --dtype "${DTYPE}" \
#     --max-length "${MAX_LENGTH}" \
#     --aggregation "${AGGREGATION}" \
#     --clusters "${CLUSTERS}" \
#     --prompt-pairs-output "${PROMPT_PAIRS_OUTPUT}" \
#     --vectors-output "${VECTORS_OUTPUT}" \
#     --library-output "${LIBRARY_OUTPUT}" \
#     --metadata-output "${METADATA_OUTPUT}"

# echo "[2/3] Preparing MMLU evaluation inputs..."
# "${PYTHON_BIN}" scripts/prepare_mmlu_eval.py \
#     --input "${PROMPT_PAIRS_OUTPUT}" \
#     --output "${EVALUATION_INPUT}"

echo "[3/3] Running baseline versus fixed-vector steering evaluation..."
"${PYTHON_BIN}" scripts/evaluate_mmlu.py \
    --model "${MODEL_PATH}" \
    --library "${LIBRARY_OUTPUT}" \
    --layer "${INJECT_LAYER}" \
    --input "${EVALUATION_INPUT}" \
    --output "${RESULTS_OUTPUT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --fixed-primitives "${FIXED_PRIMITIVES[@]}" \
    --fixed-strengths "${FIXED_STRENGTHS[@]}"

echo "实验完成，输出文件："
printf '  %s\n' \
    "${PROMPT_PAIRS_OUTPUT}" \
    "${VECTORS_OUTPUT}" \
    "${LIBRARY_OUTPUT}" \
    "${METADATA_OUTPUT}" \
    "${EVALUATION_INPUT}" \
    "${RESULTS_OUTPUT}"
