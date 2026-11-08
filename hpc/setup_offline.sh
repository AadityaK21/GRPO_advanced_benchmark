#!/bin/bash
# Run once on the HPC login node (it has internet through the IIT Delhi proxy; compute nodes do not).
# Creates .venv, installs the pinned versions, downloads the models and GSM8K into $HF_HOME, and runs
# the tests. Then every job can run with HF_HUB_OFFLINE=1.
#     bash hpc/setup_offline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/hf_cache}

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.14.1 transformers==5.19.0 trl==1.15.0 "datasets>=5.1" peft pyyaml pytest numpy matplotlib scikit-learn scipy

python - <<'PY'
from datasets import load_dataset
from huggingface_hub import snapshot_download
for name in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
    print("model", name, snapshot_download(name))
for split in ("train", "test"):
    print("gsm8k", split, len(load_dataset("openai/gsm8k", "main", split=split)))
PY

git status --short --untracked-files=no
git log --oneline -1
python -m pytest -q
echo "Ready. On a GPU node, also run: python -m pytest -q tests/test_against_trl.py (the Triton check)"
