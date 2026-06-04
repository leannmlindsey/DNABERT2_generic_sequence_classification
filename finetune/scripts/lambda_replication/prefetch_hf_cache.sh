#!/bin/bash
#
# Prefetch the HuggingFace model FILES into the local cache so the OFFLINE SLURM
# jobs can find them. RUN ON A LOGIN NODE — Biowulf compute nodes have no internet
# and the jobs run with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.
#
# This only DOWNLOADS the repo files (snapshot_download); it does NOT instantiate
# the model, so a model/transformers version mismatch won't abort the prefetch.
#
# Reads BASE_MODEL, HF_HOME, CONDA_ENV from lambda_replication.conf. Optionally
# pin a model revision by exporting HF_REVISION=<commit-or-tag> before running.
#
# Usage:
#   bash finetune/scripts/lambda_replication/prefetch_hf_cache.sh

# Absolute path to this lambda_replication dir on Biowulf (hardcoded so it is
# correct no matter what directory the script is launched/submitted from).
SCRIPT_DIR="/vf/users/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/DNABERT2_generic_sequence_classification/finetune/scripts/lambda_replication"
source "${SCRIPT_DIR}/lambda_replication.conf"

source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-/data/lindseylm/.cache/huggingface}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE     # must be ONLINE to download

echo "Prefetching files for '${BASE_MODEL}' (revision='${HF_REVISION:-main}')"
echo "  into HF_HOME=${HF_HOME}"
echo "  conda env=${CONDA_DEFAULT_ENV}   python=$(command -v python)"

BASE_MODEL="${BASE_MODEL}" HF_REVISION="${HF_REVISION:-}" python -c "
import os
from huggingface_hub import snapshot_download
m = os.environ['BASE_MODEL']
rev = os.environ.get('HF_REVISION') or None
path = snapshot_download(repo_id=m, revision=rev)
print('PREFETCH OK:', m, '(rev=%s)' % (rev or 'main'), '->', path)
"
