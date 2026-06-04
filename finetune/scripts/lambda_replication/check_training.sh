#!/bin/bash
#
# DNABERT-2 LAMBDA replication — check that all STAGE 1 training jobs finished.
#
# Reads the same lambda_replication.conf as the launcher, then for every
# (length, variant, seed) cell reports:
#   RESULTS  test_results.json present (training + eval finished)
#   MODEL    saved model present (pytorch_model.bin or model.safetensors)
#   MCC      test-set Matthews correlation straight from test_results.json
#   LOG      whether the matching SLURM .out log ended with "Done:"
# and lists any non-empty .err files (potential failures).
#
# Usage:
#   bash finetune/scripts/lambda_replication/check_training.sh
#
# Run this after run_lambda_training.sh and before run_lambda_inference.sh.

# Absolute path to this lambda_replication dir on Biowulf (hardcoded so it is
# correct no matter what directory the script is launched/submitted from).
SCRIPT_DIR="/vf/users/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/DNABERT2_generic_sequence_classification/finetune/scripts/lambda_replication"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is empty (check ${CONFIG})"; exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR not found: ${OUTPUT_DIR}"; exit 1
fi

# DNABERT-2 supports 2k/4k only — mirror the launcher and drop any 8k entry.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    [ "${LEN}" = "8k" ] && continue
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"

LOGDIR="${OUTPUT_DIR}/logs"

echo "============================================================"
echo "DNABERT-2 LAMBDA replication — training check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  SEEDS:           ${SEEDS}"
echo "============================================================"
echo ""

TOTAL=0
OK=0

printf "%-4s %-9s %-5s  %-8s  %-8s  %-8s  %s\n" LEN VARIANT SEED RESULTS MODEL MCC LOG
for LEN in ${RUN_LENGTHS}; do
    for VARIANT in ${VARIANTS}; do
        for SEED in ${SEEDS}; do
            TOTAL=$((TOTAL + 1))
            D="${OUTPUT_DIR}/${LEN}/finetune/${VARIANT}/seed-${SEED}"

            if [ -f "${D}/test_results.json" ]; then R=ok; else R=MISSING; fi

            if [ -f "${D}/pytorch_model.bin" ] || [ -f "${D}/model.safetensors" ]; then
                M=ok
            else
                M=MISSING
            fi

            MCC=$(python - "${D}/test_results.json" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    v = d.get('eval_matthews_correlation', d.get('matthews_correlation'))
    print(f"{v:.4f}" if isinstance(v, (int, float)) else "?")
except Exception:
    print("-")
PY
)

            LOG=$(ls -t "${LOGDIR}/ft_${LEN}_${VARIANT}_s${SEED}_"*.out 2>/dev/null | head -1)
            if [ -n "${LOG}" ] && grep -q "^Done:" "${LOG}"; then
                L=done
            elif [ -n "${LOG}" ]; then
                L="NO 'Done:'"
            else
                L="no .out"
            fi

            [ "${R}" = ok ] && [ "${M}" = ok ] && [ "${L}" = done ] && OK=$((OK + 1))

            printf "%-4s %-9s %-5s  %-8s  %-8s  %-8s  %s\n" \
                "${LEN}" "${VARIANT}" "${SEED}" "${R}" "${M}" "${MCC}" "${L}"
        done
    done
done

echo ""
echo "Healthy: ${OK} / ${TOTAL}"

echo ""
echo "=== non-empty .err files (potential failures) ==="
ERRS=$(find "${LOGDIR}" -name "ft_*.err" -size +0c -printf "%s  %p\n" 2>/dev/null | sort -rn)
if [ -n "${ERRS}" ]; then
    echo "${ERRS}"
else
    echo "  (none)"
fi
