#!/bin/bash
#
# DNABERT-2 LAMBDA replication — verify the RANDOM-EMBEDDING baseline actually ran
# (don't trust INCLUDE_RANDOM_BASELINE=true alone). Per the Delta retrain plan
# step 4a: each embedding cell must have produced BOTH the random-embedding
# artifact AND the random metrics, with random MCC well BELOW pretrained MCC.
#
# DNABERT-2 specifics (differ from megaDNA's example key names):
#   - the embedding analysis is PER-LENGTH (2k, 4k), using the PRETRAINED model
#     (not per finetune seed), and runs in the inference stage.
#   - artifacts live in:  <OUTPUT_DIR>/<LEN>/embedding/<variant>/
#       embeddings_random.npz              (random embeddings cache)
#       embedding_analysis_results.json    with keys:
#         linear_probe.mcc / three_layer_nn.mcc          (pretrained)
#         random_baseline_linear.mcc / random_baseline_nn.mcc   (random)
#   - DNABERT-2 does NOT write an embedding_power key; it is pretrained − random,
#     computed here for display.
#
# Usage:
#   bash finetune/scripts/lambda_replication/check_random_baseline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

[ -n "${OUTPUT_DIR}" ] || { echo "ERROR: OUTPUT_DIR empty (check ${CONFIG})"; exit 1; }

# DNABERT-2 supports 2k/4k only — mirror the drivers and drop any 8k entry.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    [ "${LEN}" = "8k" ] && continue
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"

echo "============================================================"
echo "DNABERT-2 LAMBDA — random-baseline completeness check"
echo "  OUTPUT_DIR:              ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS:         ${RUN_LENGTHS}"
echo "  VARIANTS:                ${VARIANTS}"
echo "  INCLUDE_RANDOM_BASELINE: ${INCLUDE_RANDOM_BASELINE:-false}"
echo "============================================================"

if [ "${INCLUDE_RANDOM_BASELINE:-false}" != "true" ]; then
    echo "NOTE: INCLUDE_RANDOM_BASELINE is not 'true' — the random arm was not requested."
fi

TOTAL=0
COMPLETE=0

# Print one cell's status. Reads the results JSON with python (already in env).
report_cell() {
    local len="$1" variant="$2" emb_dir="$3"
    local npz="${emb_dir}/embeddings_random.npz"
    local json="${emb_dir}/embedding_analysis_results.json"

    local npz_ok="MISSING"; [ -f "${npz}" ] && npz_ok="ok"

    python - "$json" "$len" "$variant" "$npz_ok" <<'PY'
import json, sys
json_path, length, variant, npz_ok = sys.argv[1:5]
def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "?"
try:
    d = json.load(open(json_path))
except Exception:
    print(f"  {length:<3} {variant:<10} json=MISSING  npz={npz_ok}  -> INCOMPLETE")
    sys.exit(3)

rl = (d.get("random_baseline_linear") or {}).get("mcc")
rn = (d.get("random_baseline_nn") or {}).get("mcc")
pl = (d.get("linear_probe") or {}).get("mcc")
pn = (d.get("three_layer_nn") or {}).get("mcc")

have_random = isinstance(rl, (int, float)) and isinstance(rn, (int, float))
status = "COMPLETE" if (have_random and npz_ok == "ok") else "INCOMPLETE"

# embedding power = pretrained - random (sanity: should be >= 0-ish; random is weak)
def power(p, r):
    return f"{p - r:+.4f}" if isinstance(p, (int, float)) and isinstance(r, (int, float)) else "?"

print(f"  {length:<3} {variant:<10} npz={npz_ok:<7} "
      f"LP[pre={fmt(pl)} rand={fmt(rl)} pow={power(pl, rl)}] "
      f"NN[pre={fmt(pn)} rand={fmt(rn)} pow={power(pn, rn)}] -> {status}")

# warn if random isn't clearly below pretrained (random init should be the weak baseline)
for name, p, r in (("LP", pl, rl), ("NN", pn, rn)):
    if isinstance(p, (int, float)) and isinstance(r, (int, float)) and r >= p:
        print(f"      WARN: random {name} MCC ({r:.4f}) >= pretrained ({p:.4f}) — unexpected; inspect this cell")

sys.exit(0 if status == "COMPLETE" else 3)
PY
    return $?
}

for LEN in ${RUN_LENGTHS}; do
    for VARIANT in ${VARIANTS}; do
        EMB_DIR="${OUTPUT_DIR}/${LEN}/embedding/${VARIANT}"
        TOTAL=$((TOTAL + 1))
        if [ ! -d "${EMB_DIR}" ]; then
            echo "  ${LEN} ${VARIANT}  embedding dir MISSING (${EMB_DIR}) -> INCOMPLETE"
            continue
        fi
        if report_cell "${LEN}" "${VARIANT}" "${EMB_DIR}"; then
            COMPLETE=$((COMPLETE + 1))
        fi
    done
done

echo "------------------------------------------------------------"
echo "random baseline complete: ${COMPLETE} / ${TOTAL} cells"
[ "${COMPLETE}" -eq "${TOTAL}" ] && echo "ALL CELLS OK" || echo "RE-RUN the INCOMPLETE cells (embedding job for that length)."
