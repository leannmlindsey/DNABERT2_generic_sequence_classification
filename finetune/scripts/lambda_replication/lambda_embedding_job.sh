#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Surface D: pretrained-embedding analysis for ONE variant at one window.
# Trains a linear probe + 3-layer NN on frozen DNABERT-2 embeddings.
# Uses the pretrained base model (no finetune checkpoint needed), so it can run
# any time after the data is in place.
#
# Required env:
#   REPO_ROOT, REPL_OUTPUT_DIR, LAMBDA_DIR, VARIANT, MAX_LENGTH
# Optional env:
#   BASE_MODEL (zhihan1996/DNABERT-2-117M), BATCH_SIZE (16), POOLING (mean),
#   EMB_SEED (42), NN_EPOCHS (100), NN_LR (0.001),
#   INCLUDE_RANDOM_BASELINE (false), CONDA_ENV (dna)


echo "=== embedding analysis ${VARIANT} len=${LEN:-?} ==="
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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/lindseylm/.cache/huggingface

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-zhihan1996/DNABERT-2-117M}
BATCH_SIZE=${BATCH_SIZE:-16}
POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_LR=${NN_LR:-0.001}
MAX_LENGTH=${MAX_LENGTH:-512}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# embedding_analysis_dnabert2.py writes directly into --output_dir (it does NOT
# append the model name), so point it straight at embedding/<variant>.
OUTPUT_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE,,}" = "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

echo "  lambda dir:  ${LAMBDA_DIR}"
echo "  output dir:  ${OUTPUT_DIR}"
echo "  pooling=${POOLING}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}"
echo "  nn_epochs=${NN_EPOCHS}  nn_lr=${NN_LR}  random_baseline=${INCLUDE_RANDOM_BASELINE}"

python embedding_analysis_dnabert2.py \
    --csv_dir "${LAMBDA_DIR}" \
    --model_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --pooling "${POOLING}" \
    --seed ${EMB_SEED} \
    --nn_epochs ${NN_EPOCHS} \
    --nn_lr ${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)"
