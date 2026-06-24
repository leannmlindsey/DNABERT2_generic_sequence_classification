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

# This lambda_replication dir. Runs on a login node (not SLURM-staged), so
# deriving from BASH_SOURCE is safe and avoids a hardcoded path that drifts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lambda_replication.conf"

# BIOWULF ONLY (disabled for Delta): activate the conda env on the login node
# BEFORE running this script; it then runs in your already-active env. Re-enable
# on a cluster that does not inherit your shell environment.
# source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
# conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/hf_cache}"
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
