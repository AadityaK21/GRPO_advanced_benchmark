#!/bin/bash
# The same grid on a Slurm cluster, as an array: task K runs job K of the block (skipped if done).
#     N=$(python scripts/grid.py list --block tier1 | wc -l)
#     sbatch --array=0-$((N-1))%8 --export=ALL,BLOCK=tier1 hpc/grid_array.slurm
#SBATCH --job-name=grpo-grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
BLOCK=${BLOCK:?pass the block with --export=ALL,BLOCK=tier1}
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python scripts/grid.py run-one --block "$BLOCK" --index "$SLURM_ARRAY_TASK_ID"
