#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of DNABERT-2 LAMBDA replication: finetune ONE (variant, seed).
# Submitted by run_lambda_training.sh. All paths/resources come via --export.
#
# Required env:
#   REPO_ROOT          finetune/ dir (holds train.py)
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val(dev)/test CSV directory
#   VARIANT            dnabert2
#   SEED               integer
#   MAX_LENGTH         max BPE token length for this window
# Optional env:
#   BASE_MODEL (zhihan1996/DNABERT-2-117M), LR (3e-5), BATCH_SIZE (8),
#   EVAL_BATCH_SIZE (16), NUM_EPOCHS (3), USE_FP16 (1|0), CONDA_ENV (dna)


echo "=== finetune ${VARIANT} seed=${SEED} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (source conda.sh; conda activate dna).
module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-dna}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-dna}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-dna}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1

# Stay offline so the Biowulf HTTPS proxy can't 503 us mid-run. Cache must be
# pre-warmed from a login node (see lambda_replication/README.md).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/lindseylm/.cache/huggingface

# REPO_ROOT is supplied by the launcher via --export (the batch script is staged
# to the SLURM spool dir, so its own path can't be used to find the repo).
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-zhihan1996/DNABERT-2-117M}
LR=${LR:-3e-5}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-3}
USE_FP16=${USE_FP16:-1}
MAX_LENGTH=${MAX_LENGTH:-512}

OUTPUT_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}"
RUN_NAME="ft_${VARIANT}_${LEN:-x}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:   ${BASE_MODEL}"
echo "  lambda dir:   ${LAMBDA_DIR}"
echo "  output:       ${OUTPUT_DIR}"
echo "  lr=${LR}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}  epochs=${NUM_EPOCHS}"

FP16_FLAG=""
[ "${USE_FP16}" = "1" ] && FP16_FLAG="--fp16"

# --save_model writes the (best, since load_best_model_at_end) model to
# OUTPUT_DIR so inference can load it later. --eval_and_save_results writes the
# test-set metrics to OUTPUT_DIR/results/<run_name>/eval_results.json.
python train.py \
    --model_name_or_path "${BASE_MODEL}" \
    --data_path "${LAMBDA_DIR}" \
    --kmer -1 \
    --run_name "${RUN_NAME}" \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCHS} \
    ${FP16_FLAG} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --warmup_steps 50 \
    --logging_steps 100000 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --save_model True \
    --eval_and_save_results True \
    --seed ${SEED}

# select_best_model.py expects <seed_dir>/test_results.json. train.py writes the
# test-set metrics under results/<run_name>/eval_results.json — surface it.
EVAL_JSON="${OUTPUT_DIR}/results/${RUN_NAME}/eval_results.json"
if [ -f "${EVAL_JSON}" ]; then
    cp "${EVAL_JSON}" "${OUTPUT_DIR}/test_results.json"
    echo "  wrote ${OUTPUT_DIR}/test_results.json"
else
    echo "  WARNING: ${EVAL_JSON} not found — test_results.json not written"
fi

echo "Done: $(date)"
