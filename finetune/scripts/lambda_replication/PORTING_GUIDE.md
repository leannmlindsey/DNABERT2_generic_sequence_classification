# Porting a genomic language model into the LAMBDA replication pipeline

This documents everything done to wire **DNABERT-2** into the LAMBDA_v1
replication pipeline on Biowulf, written as a checklist so the same work can be
repeated for other models (GENA-LM variants, NT, HyenaDNA, etc.). It separates
**the generic pipeline shape** (same for every model) from **the per-model
changes** (what you actually edit for a new model).

The pipeline mirrors the ProkBERT / GENA-LM `lambda_replication` layout: a single
config file drives a two-stage SLURM workflow — **train all seeds**, then
**pick the winner and run all inference** — and every model repo carries its own
copy of these scripts adapted to that model.

---

## 0. Working rules (do not skip)

- **The agent edits files only.** The user checks all code and makes **all**
  commits, pushes, and runs (training/eval/sbatch). Biowulf is unreachable from
  the agent environment and `git push` fails there too, so nothing is committed,
  pushed, or executed automatically — the agent prepares edits and hands over
  exact commands.
- **Laptop → GitHub → Biowulf.** Edit on the laptop, commit/push, then `git pull`
  on Biowulf before running. Config changes have no effect until pulled there.
- **Dataset root:** `LAMBDA_BASE=/vf/users/Irp-jiang/share/lindseylm/LAMBDA_v1`.
- **Outputs:** under `/data/lindseylm/...` — **never** `/gpfs/gsfs12/users/lindseylm/...`
  (that path no longer exists for this account).

---

## 1. The two-stage workflow (same for every model)

```bash
# 0. (one time) pre-warm the HF cache from a LOGIN node — jobs run offline.
# 1. Edit lambda_replication.conf — confirm LAMBDA_BASE + OUTPUT_DIR, set paths.
bash .../lambda_replication/run_lambda_training.sh    # finetune × N seeds × windows
# 2. wait — squeue -u $USER
bash .../lambda_replication/check_training.sh          # confirm all seeds healthy
bash .../lambda_replication/run_lambda_inference.sh    # pick winner + all inference
# 3. wait — squeue -u $USER
bash .../lambda_replication/check_inference.sh         # confirm all outputs landed
```

**Files in `lambda_replication/`:**

| File | Role |
|------|------|
| `lambda_replication.conf` | the only file you normally edit — all paths + hyperparameters |
| `run_lambda_training.sh` | submit one finetune job per (variant × window × seed) |
| `lambda_finetune_job.sh` | sbatch body: one finetune run |
| `select_best_model.py` | pick best-of-N seed per (variant × window) by **test-set MCC** → `winners.json` |
| `run_lambda_inference.sh` | run winner selection, embedding analysis, and all diagnostic + genome-wide inference |
| `lambda_inference_job.sh` | sbatch body: one inference run (predictions CSV + metrics JSON) |
| `lambda_embedding_job.sh` | sbatch body: pretrained-embedding analysis (Surface D) |
| `lambda_genome_analysis_job.sh` | sbatch body: threshold/clustering summary over genome-wide predictions |
| `print_winner_exports.py` | emit shell exports for the winning checkpoint |
| `check_training.sh` / `check_inference.sh` | post-hoc verification helpers (see §6) |

---

## 2. Model-selection logic (important — two levels)

There are **two independent selection steps**; keep them straight when porting:

1. **Per-seed checkpoint** (inside one finetune run): chosen by the HF Trainer
   default `metric_for_best_model=eval_loss` with `load_best_model_at_end=True`.
   This is upstream behavior — **leave it alone**, do not switch it to MCC.
2. **Cross-seed winner** (downstream, across the N seeds): `select_best_model.py`
   reads each `seed-<N>/test_results.json` and picks the max **test-set MCC**
   (`eval_matthews_correlation`), writing `winners.json`. This is the
   LAMBDA-specific step (the LAMBDA paper reports MCC).

`train.py`'s `compute_metrics` emits `accuracy`, `f1`, `matthews_correlation`;
`trainer.evaluate` prefixes them with `eval_`. `lambda_finetune_job.sh` copies
`results/<run_name>/eval_results.json` to `seed-<N>/test_results.json` so
`select_best_model.py` can find it (it accepts the key candidates
`eval_matthews_correlation`, `eval_mcc`, `matthews_correlation`, `mcc`).

---

## 3. Dataset + output layout (same for every model)

**Inputs** (auto-derived from `LAMBDA_BASE`):
```
LAMBDA_BASE/
├── train_val_test/<W>/{train,val,test}.csv     finetune + embedding + test diagnostic (Surface A)
├── fpr_test/<W>/bacteria_segments_<W>.csv       FPR control (Surface B)  — auto-derived
├── shuffled_controls/<W>/test_shuffled.csv      GC control  (Surface B)  — auto-derived
├── fnr_test/<W>/...                             FNR control (Surface B)  — set via FNR_<W>
└── genome_wide/<W>/                             genome-wide (Surface C)  — set via GENOME_WIDE_<W>
```
CSVs have a `sequence` column and a `label` column (0/1). FPR is bacteria-only
(all label 0); FNR is phage-only (all label 1). **Verified LAMBDA_v1 paths:**
`fnr_test/2k/phage_annotated_segments_2k.csv` (phrog-annotated subset, by design),
`fnr_test/4k/phage_segments_4k_2k.csv`, and `genome_wide/{2k,4k}/` (directories).

**Outputs** (`OUTPUT_DIR/<W>/...`): `finetune/<variant>/seed-<N>/` (test_results.json
+ saved model), `embedding/<variant>/`, `winners.json`, `inference/<variant>/`
(`<diag>_predictions.csv` + `_predictions_metrics.json`), `genome_wide_analysis/<variant>/`,
and a shared `OUTPUT_DIR/logs/`.

**Inference metrics JSON** carries `accuracy, precision, recall, f1, mcc, auc,
sensitivity, specificity, {true,false}_{positives,negatives}, num_samples`.
Inference is **uniform across all datasets** — interpret per dataset downstream:
MCC for the mixed **test** set; **FPR = 1 − accuracy** for fpr; **FNR = 1 − accuracy**
for fnr (MCC/AUC are meaningless on the single-class control sets).

### 3a. Ready-to-paste config path block

The dataset paths are identical across model repos (same `LAMBDA_BASE`). Copy
this block into `lambda_replication.conf` verbatim; only change `OUTPUT_DIR`
(and `SEGMENT_LENGTHS` if the model supports different windows):

```bash
# --- paths (portable across model repos) -------------------------------------
export LAMBDA_BASE="/vf/users/Irp-jiang/share/lindseylm/LAMBDA_v1"
export OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/<MODEL>_generic_sequence_classification/outputs"
export HF_HOME="/data/lindseylm/.cache/huggingface"

# Windows to run (DNABERT-2 = 2k 4k; add 8k only if the model supports it).
export SEGMENT_LENGTHS="2k 4k"

# Per-window max token length — tokenizer-dependent (DNABERT-2 BPE ≈ 0.25 × bp).
export MAX_LENGTH_2k="512"
export MAX_LENGTH_4k="1024"

# FPR (bacteria-only) and GC control (shuffled) are AUTO-DERIVED by the inference
# driver from LAMBDA_BASE — NOT set here:
#   ${LAMBDA_BASE}/fpr_test/<W>/bacteria_segments_<W>.csv
#   ${LAMBDA_BASE}/shuffled_controls/<W>/test_shuffled.csv

# FNR (phage-only) — set explicitly per window (VERIFIED to exist):
export FNR_2k="${LAMBDA_BASE}/fnr_test/2k/phage_annotated_segments_2k.csv"
export FNR_4k="${LAMBDA_BASE}/fnr_test/4k/phage_segments_4k_2k.csv"

# Genome-wide — directory of *.csv per window (one inference job per CSV):
export GENOME_WIDE_2k="${LAMBDA_BASE}/genome_wide/2k"
export GENOME_WIDE_4k="${LAMBDA_BASE}/genome_wide/4k"
```

Note `FNR_2k` uses the **phrog-annotated** subset by design (smaller/different
from the 4k sliding-window set; it drives the paper/website figure). For an 8k
model, add `MAX_LENGTH_8k`, `FNR_8k=${LAMBDA_BASE}/fnr_test/8k/phage_segments_8k_4k.csv`,
and `GENOME_WIDE_8k=${LAMBDA_BASE}/genome_wide/8k` (per the GENA-LM config).

---

## 4. ⚙️ What to change per model (the actual porting work)

Start from a working model's `lambda_replication/` dir (e.g. this one or GENA-LM)
and change only the following:

### 4a. Config (`lambda_replication.conf`)
- `OUTPUT_DIR` → `/data/lindseylm/.../<MODEL>_generic_sequence_classification/outputs`
- `BASE_MODEL` / `VARIANTS` → the model's HF id(s). Multi-variant models (GENA-LM)
  define per-variant presets (model_name + LR + WD + scheduler) in the launcher's
  `variant_preset()` block; single-architecture models (DNABERT-2) use one variant.
- **`SEGMENT_LENGTHS` / `WINDOWS`** → only the context lengths the model supports.
  DNABERT-2 supports **2k/4k only — 8k is dropped automatically** by the launcher.
  Models with ≤512-token caps (e.g. BERT-base) truncate at 4k/8k — decide whether
  to run those windows.
- Per-window max token length (`MAX_LENGTH_<W>`) → match the tokenizer. DNABERT-2
  uses **BPE (~4 bp/token)** with ALiBi, so `512` (2k) / `1024` (4k), ≈ 0.25 × bp.
  k-mer or char tokenizers need different values.
- Finetune hyperparameters (`LR`, `BATCH_SIZE`, `NUM_EPOCHS`, `*_FP16/PRECISION`,
  `EVAL_STEPS`, etc.) → the model's documented finetune recipe.
- Diagnostic paths `FNR_<W>` / `GENOME_WIDE_<W>` → the LAMBDA_v1 paths in §3
  (same `LAMBDA_BASE`, so portable across model repos verbatim).
- `CONDA_ENV` → the model's env (DNABERT-2 = `dna`; GENA-LM = `gena_lm`).
- `HF_HOME` → `/data/lindseylm/.cache/huggingface`.

### 4b. Job scripts (`lambda_*_job.sh`)
- Swap the **conda env** and any model module loads (e.g. `module load CUDA/12.8`
  for GPU finetune; the genome-analysis job is CPU-only and omits CUDA).
- `trust_remote_code=True` if the model needs it (DNABERT-2 / GENA-LM do).
- Keep the **offline-cache env** in GPU jobs: `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/data/lindseylm/.cache/huggingface`.

### 4c. `train.py` / model code
- This fork's **only** code change to upstream `train.py` is a `val.csv`
  fallback when `dev.csv` is absent (LAMBDA_v1 ships `val.csv`). Apply the same
  to any model whose train script assumes `dev.csv`.
- Confirm the train script's `compute_metrics` emits `matthews_correlation`
  (→ `eval_matthews_correlation`) so `select_best_model.py` finds the winner key.

---

## 5. 🐛 Biowulf gotchas (cost real debugging — apply to every model)

- **`conda activate` under `set -e` silently kills the job.** In a non-interactive
  batch shell, `conda activate` falls through to legacy `source activate`, which
  sources a script; under `set -euo pipefail` a non-zero line inside it aborts the
  whole job **before** any `|| true` guard runs. Symptom: empty `.err`, output
  stops right after the last echo. **Fix:** keep job scripts bare — **no**
  `set -euo pipefail`, **no** `set +e/-e`, **no** `2>/dev/null` masking. Just:
  ```
  module load conda
  module load CUDA/12.8        # GPU jobs only
  source activate "${CONDA_ENV}"
  ```
  Debug silent failures by resubmitting with `--wrap='bash -x <jobscript>'`.
- **Run offline.** The Biowulf HTTPS proxy 503s mid-run; pre-warm the HF cache
  once from a **login node**, then jobs run with `HF_HUB_OFFLINE=1`.
- **Propagate `REPO_ROOT` via `--export`.** SLURM stages job scripts to
  `/var/spool/slurm/...`, so `BASH_SOURCE[0]` can't recover the repo location;
  the launcher passes `REPO_ROOT` explicitly.

---

## 6. Verification helpers

- `check_training.sh` — per (window × variant × seed): `test_results.json` present,
  saved model present, test-set MCC from the JSON, and whether the `.out` ended
  with `Done:`; plus a `Healthy: N/TOTAL` tally and any non-empty `.err`.
- `check_inference.sh` — per (window × variant): winner present, embedding results
  present, each diagnostic's predictions+metrics with `acc=`/`mcc=`, genome-wide
  predictions count vs expected, and analysis CSVs; plus non-empty `.err`.

Both source `lambda_replication.conf`, so they auto-use the real paths/seeds —
copy them into each model's `lambda_replication/` dir unchanged.

---

## 7. Per-model porting checklist (tl;dr)

- [ ] Copy `lambda_replication/` from a working model repo
- [ ] `OUTPUT_DIR`, `BASE_MODEL`/`VARIANTS`, `CONDA_ENV` in the conf
- [ ] `SEGMENT_LENGTHS`/`WINDOWS` = the model's supported context lengths
- [ ] `MAX_LENGTH_<W>` = tokenizer-appropriate token counts
- [ ] Finetune hyperparameters = model's recipe
- [ ] `FNR_<W>` / `GENOME_WIDE_<W>` = LAMBDA_v1 paths (§3)
- [ ] Job scripts: conda env, module loads, `trust_remote_code`, offline env
- [ ] `train.py`: `val.csv` fallback + `matthews_correlation` in `compute_metrics`
- [ ] No `set -e` / `2>/dev/null` in job scripts (§5)
- [ ] Pre-warm HF cache from a login node
- [ ] Run training → `check_training.sh` → inference → `check_inference.sh`
