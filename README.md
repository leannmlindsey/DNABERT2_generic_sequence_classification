# DNABERT-2 Generic Sequence Classification

> **Fork of [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2)** — adds generic CSV-based binary classification scripts, used to benchmark DNABERT-2 on the [LAMBDA prophage-detection benchmark](https://github.com/leannmlindsey/LAMBDA).
>
> Original docs preserved verbatim in [`UPSTREAM_README.md`](./UPSTREAM_README.md).

---

## Relationship to the upstream training code

The fine-tune entry point in this fork (`finetune/train.py`) is upstream's own
training script — a thin layer over `transformers.Trainer` with
`AutoModelForSequenceClassification`, driven by `HfArgumentParser`. This fork
does not replace it; it preserves the script and drives it from SLURM wrappers.
The only code change is that the validation split now falls back to `val.csv`
when `dev.csv` is absent (LAMBDA_v1 ships `val.csv`). The defaults below come
from upstream's documented finetune example; the LAMBDA wrappers set the
LAMBDA-specific values and pass every flag through so they can be overridden:

| Parameter | Default (this fork) | Source / rationale |
|-----------|---------------------|--------------------|
| `model_name_or_path` | `zhihan1996/DNABERT-2-117M` | upstream |
| `kmer` | -1 (BPE, no k-mer) | upstream — DNABERT-2 replaces k-mers with BPE |
| `learning_rate` | 3e-5 | upstream finetune example |
| `per_device_train_batch_size` | 8 | upstream finetune example |
| `per_device_eval_batch_size` | 16 | upstream finetune example |
| `gradient_accumulation_steps` | 1 | upstream finetune example |
| `num_train_epochs` | 3 | this fork — LAMBDA wrapper (upstream example uses 5) |
| `model_max_length` | 512 (2k) / 1024 (4k) | this fork — ~0.25 × bp length (BPE), per LAMBDA window |
| `warmup_steps` | 50 | upstream finetune example |
| `eval_steps` / `save_steps` | 200 | upstream finetune example |
| `save_total_limit` | 1 | this fork |
| `load_best_model_at_end` | True | this fork |
| `metric_for_best_model` | `eval_loss` | upstream HF default for per-seed checkpoint selection (the cross-seed LAMBDA winner is chosen downstream by test-set MCC — see below) |
| `fp16` | on (A100) | upstream finetune example |
| `seed` | 1–5 (LAMBDA sweep); 42 otherwise | this fork / HF convention |

DNABERT-2 has a single architecture, so the LAMBDA pipeline sweeps seeds and
window lengths rather than architectures. Per-seed model selection uses the HF
default `eval_loss`; the LAMBDA winner across seeds is chosen by test-set MCC in
`select_best_model.py`. The upstream GUE scripts (`scripts/run_dnabert2.sh`,
`run_dnabert1.sh`, `run_nt.sh`) are unchanged; use them for the original
benchmark.

## What this fork adds

| File | Purpose |
|------|---------|
| `finetune/train.py` | Fine-tune a DNABERT-2 checkpoint on a binary CSV dataset (`train.csv` / `dev.csv` (or `val.csv`) / `test.csv` with `sequence,label` columns). Upstream's script; this fork adds the `val.csv` fallback. |
| `finetune/inference_dnabert2.py` | Inference with a locally-stored fine-tuned checkpoint; writes per-class probabilities + predicted label, and an optional metrics JSON when labels are present. |
| `finetune/embedding_analysis_dnabert2.py` | Extract pretrained embeddings; train a linear probe + 3-layer NN; compute silhouette score, PCA, and (optionally) a random-init baseline and embedding power. |
| `finetune/scripts/analyze_genome_wide_results.py` | Threshold + clustering filter for genome-wide windowed predictions; per-genome and aggregate metrics. |
| `finetune/scripts/analyze_phage_only_results.py` | FNR analysis over phage-only prediction CSVs. |
| `finetune/scripts/check_embedding_sizes.py` | Sanity-check extracted embedding dimensions. |
| `finetune/scripts/wrapper_run_embedding_analysis.sh` | SLURM submission wrapper for embedding analysis on a CSV directory. |
| `finetune/scripts/wrapper_run_inference.sh` | SLURM submission wrapper for single embedding-head inference. |
| `finetune/scripts/wrapper_run_batch_inference.sh` | SLURM submission wrapper for batch inference (one job per CSV in an `INPUT_LIST`). |
| `finetune/scripts/lambda_replication/lambda_replication.conf` | Config file for the LAMBDA-replication pipeline. |
| `finetune/scripts/lambda_replication/run_lambda_training.sh` | Submit all finetune jobs (seed × window). |
| `finetune/scripts/lambda_replication/run_lambda_inference.sh` | Pick the best seed, then submit embedding analysis + all diagnostic and genome-wide inference. |
| `finetune/scripts/lambda_replication/lambda_*_job.sh` | sbatch bodies for one finetune / embedding / inference job. (`lambda_genome_analysis_job.sh` is deprecated/not submitted — genome-wide clustering is done centrally by the LAMBDA harvest.) |
| `finetune/scripts/lambda_replication/select_best_model.py` | Pick the best of N finetune seeds per variant by test-set MCC. |
| `finetune/scripts/lambda_replication/print_winner_exports.py` | Emit shell exports for the winning checkpoint (read by the inference job). |

## Installation

### Biowulf / x86 + A100 (the pinned recipe)

Create the conda env and install the pinned dependencies:

```bash
conda create -n dnabert2_env python=3.8
conda activate dnabert2_env
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch matching your toolkit:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Modern CUDA PyTorch needs the Triton fix too.** Any torch ≥2.x pulls Triton
> 3.x, which breaks DNABERT-2's bundled flash-attention kernel (`dot() got an
> unexpected keyword argument 'trans_b'` at the first training step). Run
> `pip uninstall -y triton` after installing torch — see the Triton note in the
> Delta-AI recipe below. (The original `torch==1.13.1` pin ships an old Triton
> that is unaffected.)

### Delta-AI / NCSA (GH200, aarch64)

The pins above **do not** work on a GH200: it is aarch64 + Hopper (sm_90), and
neither Python 3.8 nor `torch==1.13.1` has an aarch64 CUDA wheel. Use Python 3.11
and a modern CUDA torch instead (confirmed 2026-06-24 on a GH200 node).

> The `/work/hdd/bfzj/llindsey1/...` paths below are the NCSA Delta-AI account
> used for the paper — change the prefixes to your own filesystem (and keep the
> env + HF cache off small/quota'd home directories).

```bash
conda create -y -p /work/hdd/bfzj/llindsey1/conda/envs/dnabert2_env python=3.11
conda activate /work/hdd/bfzj/llindsey1/conda/envs/dnabert2_env
pip install --no-cache-dir torch          # -> torch 2.12.1+cu130 on py3.11 aarch64
pip install --no-cache-dir -r requirements-delta.txt
pip uninstall -y triton                   # REQUIRED: forces DNABERT-2 to its PyTorch attention path
# verify CUDA is live before training:
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> **Why `pip uninstall -y triton`:** DNABERT-2's remote model code ships a Triton
> flash-attention kernel that calls `tl.dot(..., trans_b=True)`, which Triton 3.x
> (bundled with torch 2.12) rejects — training dies at step 0 with
> `dot() got an unexpected keyword argument 'trans_b'`. The model auto-falls back
> to an exact pure-PyTorch attention when Triton can't be imported, so removing
> Triton is the fix. It isn't needed for eager training/inference. See
> [`requirements-delta.txt`](./requirements-delta.txt) for detail.

See [`requirements-delta.txt`](./requirements-delta.txt) for the full rationale and
the pinned/unpinned split.

**Delta config changes** (already applied in
`finetune/scripts/lambda_replication/lambda_replication.conf`; adjust the `/work/hdd/...`
prefixes for your account):

- `LAMBDA_BASE` / `OUTPUT_DIR` / `HF_HOME` under `/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/...`
- `CONDA_ENV` = the full env path (`/work/.../conda/envs/dnabert2_env`)
- SBATCH flags in the drivers: `--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1`
- `FNR_2k` repointed to the in-dataset fixed-window file
  `${LAMBDA_BASE}/fnr_test/2k/phage_segments_2k_1k.csv` (was a stale Biowulf `/home/...` path)
- `PHROG_2k` added → `${LAMBDA_BASE}/fnr_test/2k/phage_annotated_segments_2k.csv` (PHROG table)
- `INCLUDE_RANDOM_BASELINE=true` (random-embedding baseline for the LP/NN tables)
- in-job conda activation disabled (jobs inherit the env via `sbatch --export=ALL`)

The optional flash-attention path and the original environment notes are in
[`UPSTREAM_README.md`](./UPSTREAM_README.md#3-setup-environment).

## Using the fork

Two supported workflows:

| If you want to... | Go to |
|---|---|
| Use DNABERT-2 on **your own** binary classification CSV (finetune, evaluate embeddings, predict) | [Generic classification](#generic-classification) |
| **Replicate** the LAMBDA phage paper — train all seeds per window, pick the best, run all diagnostic + genome-wide inference | [LAMBDA replication](#lambda-replication) |

### Generic classification

**Inputs:** a directory containing `train.csv`, `dev.csv` (or `val.csv`),
`test.csv`. Each CSV must have a `sequence` column and a `label` column (0/1).

Three sub-steps, each a separate SLURM submission:

```bash
# 1. Embedding analysis — linear probe + 3-layer NN on pretrained embeddings
#    (edit the CSV_DIR / MODEL_PATH config block at the top, then run)
bash finetune/scripts/wrapper_run_embedding_analysis.sh

# 2. Fine-tuning — full encoder fine-tune (edit DATA_PATH / hyperparams)
cd finetune
python train.py --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path /path/to/csv_dir --kmer -1 --model_max_length 512 \
    --per_device_train_batch_size 8 --learning_rate 3e-5 \
    --num_train_epochs 3 --fp16 --output_dir output/run --save_model True

# 3. Inference — predictions + metrics on a CSV (or a list of CSVs):
bash finetune/scripts/wrapper_run_inference.sh         # single CSV
bash finetune/scripts/wrapper_run_batch_inference.sh   # INPUT_LIST: one CSV path per line, one job each
```

`INPUT_LIST` in the batch-inference wrapper is a text file with one CSV path per
line; one SLURM job per input.

For the full flag list for any script, run `python <script>.py --help`.

### LAMBDA replication

A two-step workflow over a single config file. The pipeline loops over the
LAMBDA_v1 segment lengths (**2k / 4k** — DNABERT-2 has no 8k context, so 8k is
skipped automatically) and for each length submits: finetune × N seeds,
best-seed selection (by test-set MCC), pretrained embedding analysis (linear
probe + 3-layer NN, optional random baseline), inference on the matching-length
diagnostic CSVs, and genome-wide inference (one predictions CSV per input).
Genome-wide threshold/clustering is **not** done per-repo — the central LAMBDA
harvest aggregates the canonical `genome_wide_*_predictions.csv` files across all
models.

```bash
# 1. Edit the config — LAMBDA_BASE and OUTPUT_DIR are required;
#    SEEDS, MAX_LENGTH_<LEN>, FNR_<LEN>, GENOME_WIDE_<LEN> are optional.
$EDITOR finetune/scripts/lambda_replication/lambda_replication.conf

# 2. Launch all training (finetune × N seeds per window, in parallel —
#    no dependency chaining)
bash finetune/scripts/lambda_replication/run_lambda_training.sh

# 3. Wait — squeue -u $USER

# 4. Launch all inference (per length: pick winner by test-MCC; run embedding
#    analysis; run inference on test, fpr, gc_control, fnr, phrog (2k); one
#    genome-wide inference job per input CSV)
bash finetune/scripts/lambda_replication/run_lambda_inference.sh
```

After each stage, verify completeness with the bundled checkers:

```bash
bash finetune/scripts/lambda_replication/check_training.sh         # test_results.json per seed
bash finetune/scripts/lambda_replication/check_inference.sh        # winners, diagnostics, genome counts
bash finetune/scripts/lambda_replication/check_random_baseline.sh  # random arm actually ran (per length)
```

`check_random_baseline.sh` confirms each embedding cell produced **both**
`embeddings_random.npz` **and** the `random_baseline_linear` / `random_baseline_nn`
MCC keys (not just that `INCLUDE_RANDOM_BASELINE=true`), and prints pretrained vs
random MCC + embedding power per length — random should sit well below pretrained.

One-time, before the first run: pre-warm the HuggingFace cache from a login node
(the jobs run offline to avoid proxy 503s) — see the
[lambda_replication README](./finetune/scripts/lambda_replication/README.md).

**Expected LAMBDA_v1 layout** (auto-derived from `LAMBDA_BASE`):

```
LAMBDA_BASE/
├── train_val_test/<LEN>/{train,val,test}.csv     finetune + embedding + test diagnostic
├── fpr_test/<LEN>/bacteria_segments_<LEN>.csv    fpr diagnostic
└── shuffled_controls/<LEN>/test_shuffled.csv     gc_control diagnostic
```

FNR, PHROG, and genome-wide inputs are not part of the core LAMBDA_v1 split;
provide them via the optional `FNR_<LEN>`, `PHROG_<LEN>`, and `GENOME_WIDE_<LEN>`
config variables. `FNR_<LEN>` is the sliding-window phage set
(`phage_segments_<LEN>_*.csv`); `PHROG_2k` is the phage-annotated subset
(`phage_annotated_segments_2k.csv`, 2k only) that feeds the paper's PHROG table —
the two are different files. The PHROG job writes the canonical
`${PHROG_MODEL_TAG}_phage_annotated_segments_2k_predictions.csv` (the exact name
the central PHROG table reads) and carries the input's `phrog_category` /
`phrog_db_category` / `label` columns through alongside `pred_label`.
`GENOME_WIDE_<LEN>` can be a single CSV or a directory of CSVs (each becomes its
own inference job).

**Output layout:**

```
<OUTPUT_DIR>/
├── <LEN>/                              one subdir per SEGMENT_LENGTHS entry (2k, 4k)
│   ├── finetune/<variant>/seed-<N>/    test_results.json, saved model
│   ├── embedding/<variant>/            embedding_analysis_results.json, .npz, classifiers
│   ├── winners.json                    picked by run_lambda_inference.sh
│   └── inference/<variant>/            <dataset>_predictions.csv (+ _metrics.json),
│                                        incl. genome_wide_<stem>_predictions.csv
└── logs/                               SLURM stdout/stderr per job (shared)
```

## Available models

| Model | Parameters | Tokenizer | Context | HuggingFace |
| --- | --- | --- | --- | --- |
| DNABERT-2-117M | 117M | BPE | ALiBi (no fixed max; LAMBDA uses 512 / 1024 tokens for 2k / 4k) | [zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M) |

## Citation

If you use DNABERT-2 itself, cite the original paper:

```bibtex
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

If you use this fork as part of the LAMBDA prophage-detection benchmark, also
cite the LAMBDA paper:

```bibtex
@article{LAMBDA2026,
  author  = {Lindsey, LeAnn M. and Pershing, Nicole L. and Dufault-Thompson, Keith and Gwak, Ho-jin and Habib, Anisa and Schindler, Aaron and Rakheja, Arjun and Round, June and Stephens, W. Zac and Blaschke, Anne J. and Sundar, Hari and Jiang, Xiaofang},
  title   = {{LAMBDA}: A Prophage Detection Benchmark for Genomic Language Models},
  year    = {2026},
  doi     = {10.64898/2026.03.26.714501},
  url     = {https://doi.org/10.64898/2026.03.26.714501}
}
```
