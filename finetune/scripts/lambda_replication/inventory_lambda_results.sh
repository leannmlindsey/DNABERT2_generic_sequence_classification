#!/bin/bash
#
# Completeness inventory for LAMBDA results across one or more model OUTPUT_DIRs.
# For each model/window it reports the three surfaces so you can see what's
# missing before transferring:
#   embD          embedding_analysis_results.json  (OK / pretrained-only / MISSING)
#   diag          test / fpr / gc_control / fnr prediction CSVs  ('-' = missing)
#   genome_wide   count of genome_wide_*_predictions.csv
#
# Usage:
#   bash inventory_lambda_results.sh <OUTPUT_DIR> [<OUTPUT_DIR> ...]
# Tip: grab each model's OUTPUT_DIR from its conf:
#   grep '^export OUTPUT_DIR' <repo>/.../lambda_replication.conf

for OUT in "$@"; do
    echo "============================================================"
    echo "OUTPUT: ${OUT}"
    echo "============================================================"
    if [ ! -d "${OUT}" ]; then echo "  (directory not found)"; continue; fi

    windows=$(ls "${OUT}" 2>/dev/null | grep -E '^[0-9]+k$' | sort)
    [ -z "${windows}" ] && { echo "  (no <N>k window dirs found)"; continue; }

    for W in ${windows}; do
        WD="${OUT}/${W}"

        emb=$(find "${WD}/embedding" -name embedding_analysis_results.json 2>/dev/null | head -1)
        if [ -n "${emb}" ]; then
            if grep -q '"random_' "${emb}" 2>/dev/null || grep -q 'embedding_power' "${emb}" 2>/dev/null; then
                embstat="OK(+random)"
            else
                embstat="pretrained-only!"
            fi
        else
            embstat="MISSING"
        fi

        diag=""
        for d in test fpr gc_control fnr; do
            n=$(find "${WD}/inference" -name "${d}_predictions.csv" 2>/dev/null | wc -l)
            if [ "${n}" -gt 0 ]; then diag="${diag} ${d}"; else diag="${diag} -${d}"; fi
        done

        gw=$(find "${WD}/inference" -name 'genome_wide_*_predictions.csv' 2>/dev/null | wc -l)

        printf "  %-4s  embD=%-16s diag:%-28s genome_wide=%s\n" "${W}" "${embstat}" "${diag}" "${gw}"
    done
done
