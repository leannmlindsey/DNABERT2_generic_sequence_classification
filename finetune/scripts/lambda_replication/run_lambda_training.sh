#!/bin/bash
#
# DNABERT-2 LAMBDA_v1 replication — STAGE 1: fire off all training jobs.
#
# For each segment length in SEGMENT_LENGTHS, submits one finetune sbatch job
# per (variant, seed). All jobs run in parallel (no --dependency chaining).
# Once they all complete, run run_lambda_inference.sh to pick the best seed and
# run inference + embedding analysis.
#
# DNABERT-2 supports 2k and 4k only; any 8k entry in SEGMENT_LENGTHS is skipped.
#
# Usage:
#   1. Edit lambda_replication.conf — confirm LAMBDA_BASE and OUTPUT_DIR.
#   2. bash finetune/scripts/lambda_replication/run_lambda_training.sh
#   3. Wait for jobs: squeue -u $USER
#   4. bash finetune/scripts/lambda_replication/run_lambda_inference.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# REPO_ROOT is the finetune/ dir (holds train.py, inference_dnabert2.py,
# embedding_analysis_dnabert2.py, scripts/analyze_genome_wide_results.py).
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# --- validate -----------------------------------------------------------------

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to placeholder"
    exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"
    exit 1
}
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# DNABERT-2 supports 2k/4k only — drop 8k (and warn) before doing anything.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    if [ "${LEN}" = "8k" ]; then
        echo "WARNING: DNABERT-2 does not support 8k context — skipping 8k."
        continue
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
if [ -z "${RUN_LENGTHS}" ]; then
    echo "ERROR: no runnable lengths after dropping 8k (DNABERT-2 supports 2k/4k)"; exit 1
fi

# Validate per-length input dirs exist before submitting anything.
for LEN in ${RUN_LENGTHS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    [ -d "${LDIR}" ] || { echo "ERROR: ${LDIR} not found"; exit 1; }
    for f in train.csv test.csv; do
        [ -f "${LDIR}/${f}" ] || { echo "ERROR: ${LDIR}/${f} not found"; exit 1; }
    done
    if [ ! -f "${LDIR}/dev.csv" ] && [ ! -f "${LDIR}/val.csv" ]; then
        echo "ERROR: ${LDIR} must contain dev.csv or val.csv"; exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "DNABERT-2 LAMBDA replication — Stage 1: training"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  REPO_ROOT:       ${REPO_ROOT}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  BASE_MODEL:      ${BASE_MODEL}"
echo "  SEEDS:           ${SEEDS}"
echo "  FT params:       lr=${LR} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS} fp16=${USE_FP16}"
echo "============================================================"

# --- common sbatch flags ------------------------------------------------------

FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)

# REPO_ROOT is propagated to every job so they can cd to the real repo — SLURM
# stages each job script to /var/spool/slurm/... where BASH_SOURCE[0] can't
# recover the original location.
FT_ENV_BASE="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},BASE_MODEL=${BASE_MODEL},LR=${LR},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},NUM_EPOCHS=${NUM_EPOCHS},USE_FP16=${USE_FP16}"

NUM_JOBS=0

for LEN in ${RUN_LENGTHS}; do
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    mkdir -p "${REPL_LEN_DIR}"

    # Resolve per-window max token length (MAX_LENGTH_<LEN>).
    ml_var="MAX_LENGTH_${LEN}"
    MAX_LENGTH="${!ml_var:-512}"

    echo ""
    echo "--- length: ${LEN} (max_length=${MAX_LENGTH}) ---"
    echo "    lambda dir:   ${LAMBDA_DIR}"
    echo "    output dir:   ${REPL_LEN_DIR}"

    for VARIANT in ${VARIANTS}; do
        for SEED in ${SEEDS}; do
            JOB="ft_${LEN}_${VARIANT}_s${SEED}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${FT_FLAGS[@]}" \
                --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${FT_ENV_BASE},VARIANT=${VARIANT},SEED=${SEED},LEN=${LEN},MAX_LENGTH=${MAX_LENGTH}" \
                "${SCRIPT_DIR}/lambda_finetune_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done
    done
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
