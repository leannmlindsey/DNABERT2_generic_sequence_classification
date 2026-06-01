# DNABERT-2 LAMBDA_v1 replication

Driver scripts that reproduce the LAMBDA paper surfaces for DNABERT-2, mirroring
the ProkBERT `lambda_replication/` pipeline. Two commands run the whole thing:
one submits all training jobs, the other picks the best seed per variant and
submits every inference + embedding job.

DNABERT-2 has a single architecture (one "variant", `dnabert2`) and supports
**2k and 4k contexts only** — `8k` is skipped automatically.

## Layout

```
finetune/
  train.py                       # finetune entry point (patched: accepts val.csv or dev.csv)
  inference_dnabert2.py          # prediction entry point
  embedding_analysis_dnabert2.py # Surface D (linear probe + 3-layer NN on frozen embeddings)
  scripts/
    analyze_genome_wide_results.py
    lambda_replication/          # <-- this directory
      lambda_replication.conf    # all paths + hyperparameters (edit this)
      run_lambda_training.sh     # STAGE 1 launcher
      run_lambda_inference.sh    # STAGE 2 launcher
      lambda_finetune_job.sh     # sbatch body: one (variant, seed) finetune
      lambda_embedding_job.sh    # sbatch body: Surface D
      lambda_inference_job.sh    # sbatch body: one prediction surface
      lambda_genome_analysis_job.sh  # sbatch body: genome-wide summary
      select_best_model.py       # best-of-N seed by test MCC -> winners.json
      print_winner_exports.py    # winners.json -> shell exports (used by inference job)
```

## Outputs

Everything lands under `OUTPUT_DIR` from the config:

```
/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/DNABERT2/DNABERT2_generic_sequence_classification/outputs/
  <ws>/                                  # 2k, 4k
    finetune/<variant>/seed-<N>/         # saved model + test_results.json
    embedding/<variant>/                 # embedding_analysis_results.json + .npz + .pkl
    winners.json                         # best seed per variant
    inference/<variant>/                 # test_predictions.csv, fpr_predictions.csv,
                                         #   gc_control_predictions.csv, fnr_predictions.csv,
                                         #   genome_wide_<asm>_predictions.csv (+ _metrics.json)
    genome_wide_analysis/<variant>/      # <variant>_individual.csv / _summary.csv / _phage_predictions.csv
  logs/                                  # all SLURM stdout/stderr
```

## One-time: pre-warm the HF cache (avoids 503s on the Biowulf proxy)

Every job runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and
`HF_HOME=/data/lindseylm/.cache/huggingface`. Populate that cache once from a
login node (which has outbound HTTPS):

```bash
export HF_HOME=/data/lindseylm/.cache/huggingface
module load conda 2>/dev/null || true
conda activate dna
python - <<'PY'
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
m = "zhihan1996/DNABERT-2-117M"
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
AutoModel.from_pretrained(m, trust_remote_code=True)
AutoModelForSequenceClassification.from_pretrained(m, trust_remote_code=True, num_labels=2)
print("cached", m)
PY
```

## Smoke test (run this ONE thing first)

Train a single seed for the single variant at 2k, then confirm the output
structure. Run from the repo root on Biowulf:

```bash
cd /path/to/DNABERT2_generic_sequence_classification/finetune

REPO_ROOT="$(pwd)"
LAMBDA_DIR="/vf/users/Irp-jiang/share/lindseylm/LAMBDA_v1/train_val_test/2k"
OUT="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/DNABERT2/DNABERT2_generic_sequence_classification/outputs/2k"

sbatch --job-name=smoke_ft_2k_dnabert2_s1 \
  --partition=gpu --gres=gpu:a100:1 --mem=64g --time=8:00:00 --cpus-per-task=8 \
  --output="${OUT}/../logs/smoke_%j.out" --error="${OUT}/../logs/smoke_%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=dna,REPL_OUTPUT_DIR=${OUT},LAMBDA_DIR=${LAMBDA_DIR},VARIANT=dnabert2,SEED=1,LEN=2k,MAX_LENGTH=512,BASE_MODEL=zhihan1996/DNABERT-2-117M,LR=3e-5,BATCH_SIZE=8,EVAL_BATCH_SIZE=16,NUM_EPOCHS=3,USE_FP16=1" \
  scripts/lambda_replication/lambda_finetune_job.sh
```

When it finishes, expect:

```
outputs/2k/finetune/dnabert2/seed-1/test_results.json          # has eval_matthews_correlation
outputs/2k/finetune/dnabert2/seed-1/config.json + model weights # loadable by inference
```

(`mkdir -p "${OUT}/../logs"` first if the logs dir doesn't exist yet.)

## Full sweep

```bash
cd /path/to/DNABERT2_generic_sequence_classification

# 1. edit finetune/scripts/lambda_replication/lambda_replication.conf if needed
# 2. submit all training jobs (variant x seed x {2k,4k})
bash finetune/scripts/lambda_replication/run_lambda_training.sh

# 3. wait until every job is done
squeue -u $USER

# 4. pick winners + submit all inference + embedding jobs
bash finetune/scripts/lambda_replication/run_lambda_inference.sh
```

## Notes / assumptions

- **`MAX_LENGTH_2k` / `MAX_LENGTH_4k`** (config) are DNABERT-2 BPE token caps
  (~0.25 × bp). Defaults 512 / 1024. Bump them if you see truncation, lower
  them on OOM.
- **`FNR_<LEN>` and `GENOME_WIDE_<LEN>`** are left empty — set them once the
  LAMBDA_v1 phage-only and genome-wide segment paths are confirmed. The core
  surfaces (test, fpr, gc_control, embedding) run without them.
- **Genome-wide analysis** stages only `genome_wide_*` files into a clean scan
  dir before running `analyze_genome_wide_results.py` (which globs `*.json`), so
  the diagnostic surfaces don't pollute it. The canonical
  `genome_wide_<asm>_predictions.csv` files stay in `inference/<variant>/` for
  the harvest pipeline.
- The only Python change made was in `finetune/train.py`: the validation split
  now falls back to `val.csv` when `dev.csv` is absent (LAMBDA_v1 ships
  `val.csv`). Algorithms and model code are untouched.
