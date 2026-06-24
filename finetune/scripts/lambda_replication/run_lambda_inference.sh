#!/bin/bash
#
# DNABERT-2 LAMBDA_v1 replication — STAGE 2: pick the best seed per variant and
# submit all inference + embedding jobs.
#
# For each segment length in SEGMENT_LENGTHS (8k skipped — unsupported):
#   1. Run select_best_model.py to pick the per-variant winning seed by test-set
#      MCC across finetune seeds; writes winners.json.
#   2. Submit the embedding analysis job (Surface D) per variant.
#   3. Submit one diagnostic-inference job per (variant, dataset):
#        - test       train_val_test/<LEN>/test.csv            (Surface A)
#        - fpr        fpr_test/<LEN>/bacteria_segments_<LEN>.csv (Surface B)
#        - gc_control shuffled_controls/<LEN>/test_shuffled.csv  (Surface B)
#        - fnr        FNR_<LEN> if set                           (Surface B, optional)
#   4. If GENOME_WIDE_<LEN> is set, submit one genome-wide inference job per CSV
#      per variant (Surface C), each emitting genome_wide_<stem>_predictions.csv.
#      No aggregate analysis job — the central harvest does any clustering.
#
# Re-running is safe: each inference job overwrites its own predictions CSV.
#
# Usage (after run_lambda_training.sh has finished — verify with `squeue`):
#   bash finetune/scripts/lambda_replication/run_lambda_inference.sh


# This lambda_replication dir. The driver runs on a login node (not SLURM-staged),
# so deriving from BASH_SOURCE is safe and avoids a hardcoded path that drifts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# DNABERT-2 supports 2k/4k only — drop 8k.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    if [ "${LEN}" = "8k" ]; then
        echo "WARNING: DNABERT-2 does not support 8k context — skipping 8k."
        continue
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
[ -n "${RUN_LENGTHS}" ] || { echo "ERROR: no runnable lengths (2k/4k)"; exit 1; }

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# Sanity: every length must have completed training before inference.
for LEN in ${RUN_LENGTHS}; do
    LDIR="${OUTPUT_DIR}/${LEN}"
    if [ ! -d "${LDIR}/finetune" ]; then
        echo "ERROR: ${LDIR}/finetune missing — run run_lambda_training.sh first and"
        echo "       wait for all jobs to finish (squeue -u \$USER)."
        exit 1
    fi
done

# --- common sbatch flags ------------------------------------------------------

INF_FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)
EMB_FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${EMB_MEM}" --time="${EMB_TIME}" --cpus-per-task=8)

EMB_ENV_BASE="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},BASE_MODEL=${BASE_MODEL},POOLING=${POOLING},EMB_SEED=${EMB_SEED},NN_EPOCHS=${NN_EPOCHS},NN_LR=${NN_LR},BATCH_SIZE=${INF_BATCH_SIZE},INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}"

echo "============================================================"
echo "DNABERT-2 LAMBDA replication — Stage 2: winners + inference"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "============================================================"

NUM_JOBS=0
cd "${REPO_ROOT}"

for LEN in ${RUN_LENGTHS}; do
    echo ""
    echo "--- length: ${LEN} ---"

    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"

    ml_var="MAX_LENGTH_${LEN}"
    MAX_LENGTH="${!ml_var:-512}"

    # --- select winners (login-node; reads JSON only) ---
    echo "  selecting best seed per variant..."
    ALLOW_PARTIAL_FLAG=""
    if [ "${ALLOW_PARTIAL_TRAINING:-false}" = "true" ]; then
        ALLOW_PARTIAL_FLAG="--allow-partial"
    fi
    python "${SCRIPT_DIR}/select_best_model.py" \
        --output_dir "${REPL_LEN_DIR}" \
        --variants ${VARIANTS} \
        --base_model "${BASE_MODEL}" \
        ${ALLOW_PARTIAL_FLAG}

    # --- assemble diagnostic dataset list (name -> path) ---
    declare -a DIAG_NAMES DIAG_PATHS
    DIAG_NAMES=(test fpr gc_control)
    DIAG_PATHS=(
        "${LAMBDA_BASE}/train_val_test/${LEN}/test.csv"
        "${LAMBDA_BASE}/fpr_test/${LEN}/bacteria_segments_${LEN}.csv"
        "${LAMBDA_BASE}/shuffled_controls/${LEN}/test_shuffled.csv"
    )

    # Optional FNR — indirect lookup on FNR_<LEN>.
    fnr_var="FNR_${LEN}"
    FNR_PATH="${!fnr_var:-}"
    if [ -n "${FNR_PATH}" ]; then
        if [ -f "${FNR_PATH}" ]; then
            DIAG_NAMES+=(fnr)
            DIAG_PATHS+=("${FNR_PATH}")
        else
            echo "  WARNING: ${fnr_var}=${FNR_PATH} not found — skipping fnr for ${LEN}"
        fi
    fi

    # Validate built-in diagnostics exist before submitting.
    for i in "${!DIAG_NAMES[@]}"; do
        if [ ! -f "${DIAG_PATHS[$i]}" ]; then
            echo "ERROR: diagnostic '${DIAG_NAMES[$i]}' missing: ${DIAG_PATHS[$i]}"; exit 1
        fi
    done

    # --- assemble genome-wide CSV list (may be empty, one file, or many) ---
    declare -a GW_CSVS=()
    gw_var="GENOME_WIDE_${LEN}"
    GW_PATH="${!gw_var:-}"
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_CSVS=("${GW_PATH}")
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            for csv in "${GW_PATH}"/*.csv; do
                GW_CSVS+=("${csv}")
            done
            shopt -u nullglob
            [ "${#GW_CSVS[@]}" -eq 0 ] && \
                echo "  WARNING: ${gw_var}=${GW_PATH} has no *.csv — skipping genome-wide for ${LEN}"
        else
            echo "  WARNING: ${gw_var}=${GW_PATH} is not a file or directory — skipping genome-wide for ${LEN}"
        fi
        [ "${#GW_CSVS[@]}" -gt 0 ] && echo "  genome-wide CSVs for ${LEN}: ${#GW_CSVS[@]} file(s)"
    fi

    # Which variants actually have winners for this length.
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"
    HAVE_VARIANTS=$(python -c "import json; print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))")

    for VARIANT in ${VARIANTS}; do
        # --- embedding analysis (Surface D) — independent of winners ---
        EMB_JOB="emb_${LEN}_${VARIANT}"
        echo "    submitting ${EMB_JOB}..."
        sbatch \
            --job-name="${EMB_JOB}" \
            --output="${LOGDIR}/${EMB_JOB}_%j.out" \
            --error="${LOGDIR}/${EMB_JOB}_%j.err" \
            "${EMB_FLAGS[@]}" \
            --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${EMB_ENV_BASE},VARIANT=${VARIANT},LEN=${LEN},MAX_LENGTH=${MAX_LENGTH}" \
            "${SCRIPT_DIR}/lambda_embedding_job.sh"
        NUM_JOBS=$((NUM_JOBS + 1))

        # Skip prediction surfaces if no winning seed for this variant.
        if [[ " ${HAVE_VARIANTS} " != *" ${VARIANT} "* ]]; then
            echo "    skip ${VARIANT} predictions: no winner (training incomplete?)"
            continue
        fi

        INF_ENV="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${INF_BATCH_SIZE}"

        # Diagnostic inference (Surfaces A + B)
        for i in "${!DIAG_NAMES[@]}"; do
            NAME="${DIAG_NAMES[$i]}"
            CSV="${DIAG_PATHS[$i]}"
            JOB="inf_${LEN}_${VARIANT}_${NAME}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,${INF_ENV},INPUT_CSV=${CSV},OUTPUT_FILENAME=${NAME}_predictions.csv" \
                "${SCRIPT_DIR}/lambda_inference_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done

        # Genome-wide inference (Surface C) — one job per CSV, canonical
        # genome_wide_<stem>_predictions.csv name. No aggregate analysis job:
        # the central harvest does any clustering (and Delta-AI has no CPU
        # partition for it), so these jobs are independent (no afterok chain).
        if [ "${#GW_CSVS[@]}" -gt 0 ]; then
            for csv in "${GW_CSVS[@]}"; do
                stem=$(basename "${csv}" .csv)
                JOB="gwinf_${LEN}_${VARIANT}_${stem}"
                echo "    submitting ${JOB} -> genome_wide_${stem}_predictions.csv ..."
                sbatch \
                    --job-name="${JOB}" \
                    --output="${LOGDIR}/${JOB}_%j.out" \
                    --error="${LOGDIR}/${JOB}_%j.err" \
                    "${INF_FLAGS[@]}" \
                    --export="ALL,${INF_ENV},INPUT_CSV=${csv},OUTPUT_FILENAME=genome_wide_${stem}_predictions.csv" \
                    "${SCRIPT_DIR}/lambda_inference_job.sh"
                NUM_JOBS=$((NUM_JOBS + 1))
            done
        fi
    done

    unset DIAG_NAMES DIAG_PATHS GW_CSVS
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}/<LEN>/inference/, embedding/"
