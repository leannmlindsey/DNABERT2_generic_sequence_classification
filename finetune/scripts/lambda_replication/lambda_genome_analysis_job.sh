#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Genome-wide threshold + clustering summary for ONE variant. Runs after all
# genome-wide inference jobs for that variant have written
# genome_wide_<asm>_predictions.csv (+ _metrics.json) into inference/<variant>/.
#
# analyze_genome_wide_results.py globs *.json in its input dir, so we first
# stage ONLY the genome_wide_* prediction/metrics files into a clean directory.
# That keeps the diagnostic surfaces (test/fpr/fnr/gc_control) out of the scan
# while leaving the canonical genome_wide_*_predictions.csv files in place under
# inference/<variant>/ for the harvest pipeline.
#
# Required env:
#   REPO_ROOT, REPL_OUTPUT_DIR, VARIANT
# Optional env:
#   MERGE_GAP (5000), MIN_CLUSTER_SIZE (5000), SMOOTH_WINDOW (5), CONDA_ENV (dna)


echo "=== genome analysis ${VARIANT} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (source conda.sh; conda activate dna). CPU-only: no CUDA module.
source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-dna}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-dna}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-dna}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MERGE_GAP=${MERGE_GAP:-5000}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-5000}
SMOOTH_WINDOW=${SMOOTH_WINDOW:-5}

PREDICTIONS_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
ANALYSIS_DIR="${REPL_OUTPUT_DIR}/genome_wide_analysis/${VARIANT}"
STAGE_DIR="${ANALYSIS_DIR}/input"
mkdir -p "${STAGE_DIR}"

echo "  predictions dir: ${PREDICTIONS_DIR}"
echo "  analysis dir:    ${ANALYSIS_DIR}"

# Stage only genome-wide files into a clean scan dir.
shopt -s nullglob
n=0
for f in "${PREDICTIONS_DIR}"/genome_wide_*_predictions.csv \
         "${PREDICTIONS_DIR}"/genome_wide_*_predictions_metrics.json; do
    cp -f "${f}" "${STAGE_DIR}/"
    n=$((n + 1))
done
shopt -u nullglob

if [ "${n}" -eq 0 ]; then
    echo "  WARNING: no genome_wide_*_predictions files in ${PREDICTIONS_DIR} — nothing to analyze"
    exit 0
fi
echo "  staged ${n} genome-wide file(s) into ${STAGE_DIR}"

python scripts/analyze_genome_wide_results.py \
    -d "${STAGE_DIR}" \
    -m "${VARIANT}" \
    -r "${ANALYSIS_DIR}" \
    --merge-gap ${MERGE_GAP} \
    --min-cluster-size ${MIN_CLUSTER_SIZE} \
    --window-size ${SMOOTH_WINDOW} \
    --verbose

echo "Done: $(date)"
