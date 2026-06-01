#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Inference for the winning seed of VARIANT on one CSV. Reads
# <REPL_OUTPUT_DIR>/winners.json to find the winning finetune checkpoint dir,
# then runs inference_dnabert2.py.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR
#   VARIANT
#   INPUT_CSV          path to the CSV to predict on
#   OUTPUT_FILENAME    name for the predictions CSV (e.g. test_predictions.csv)
#   MAX_LENGTH         max BPE token length for this window
# Optional env:
#   BATCH_SIZE (16), CONDA_ENV (dna)

set -euo pipefail

echo "=== inference ${VARIANT}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
conda activate "${CONDA_ENV:-dna}" 2>/dev/null || source activate "${CONDA_ENV:-dna}" 2>/dev/null || true
export PYTHONNOUSERSITE=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/lindseylm/.cache/huggingface

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-512}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then
    echo "ERROR: ${WINNERS_JSON} not found (select_best_model must run first)"; exit 1
fi

# Extract winner fields. print_winner_exports.py emits shlex-quoted exports.
eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"

echo "  winner seed:   ${WINNER_SEED}"
echo "  winner path:   ${WINNER_PATH}"
echo "  base model:    ${BASE_MODEL}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

# The winning checkpoint dir holds the saved tokenizer; --tokenizer_path falls
# back to the base model automatically inside inference_dnabert2.py if missing.
python inference_dnabert2.py \
    --input_csv "${INPUT_CSV}" \
    --model_path "${WINNER_PATH}" \
    --tokenizer_path "${WINNER_PATH}" \
    --output_csv "${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --save_metrics

echo "Done: $(date)"
