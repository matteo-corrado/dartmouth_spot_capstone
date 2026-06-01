#!/bin/bash -l
# SLURM CPU-only job: the wall-clock-heavy stages of wake-word training —
# generate (Piper VITS positives) + augment (RIR/noise/gain + mel feature
# extraction over ~18 GB). Runs on a DEDICATED full 64-core Discovery `standard`
# node, NOT the GPU node (where you'd get only a ~8-core fair share of the shared
# 8-GPU l40sx8 box). The GPU is the tiny `train` stage only — see sbatch_gpu_train.sh.
#
# Pairs with sbatch_gpu_train.sh via the shared DartFS scratch: this writes
# features to data_dir on scratch; the GPU job (afterok dependency) trains over
# them. No GPU held during the CPU hours, and CPU jobs don't count against the
# 2-GPU QOS cap. `-l` login shell REQUIRED (sbatch doesn't source .bashrc).
#
# Prereqs (LOGIN node, see README.md): conda env `lkww`, configs validated,
# `livekit-wakeword setup` already downloaded corpora to data_dir on scratch.
# `logs/` must exist at submit cwd: cd <scratch clone> && mkdir -p logs.
#
#SBATCH --job-name=wake-prep
#SBATCH --account=free
#SBATCH --partition=standard          # CPU-only; full dedicated node (no --gres)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64            # whole node — parallel TTS synth + mel augment
#SBATCH --mem=256G                    # 18 GB audio + feature buffers + page cache
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-2%1                 # 3 candidates; %1 = sequential (safe under the
                                      # unpublished free CPU cap). Raise to %2/%3 if
                                      # `sacctmgr show qos` shows TRES headroom.
#SBATCH --output=logs/wake-prep_%A_%a.out
#SBATCH --error=logs/wake-prep_%A_%a.err
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<NETID>@dartmouth.edu
set -euo pipefail
export PYTHONUNBUFFERED=1

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate lkww
cd "/dartfs-hpc/scratch/$USER/spot-capstone"

CONFIGS=(configs/wake/wakeword_hey_spot.yaml \
         configs/wake/wakeword_hi_spot.yaml \
         configs/wake/wakeword_big_yellow.yaml)
idx="${SLURM_ARRAY_TASK_ID:-0}"
(( idx >= 0 && idx < ${#CONFIGS[@]} )) \
  || { echo "FATAL: SLURM_ARRAY_TASK_ID=$idx out of range 0..$(( ${#CONFIGS[@]} - 1 ))" >&2; exit 4; }
CONFIG=${CONFIGS[$idx]}

python -c "from livekit.wakeword import load_config; load_config('$CONFIG')" \
  || { echo "FATAL: $CONFIG invalid — validate on the login node first (README step 2)" >&2; exit 2; }

echo "[prep] task=$idx config=$CONFIG host=$(hostname) cores=$(nproc) $(date -Iseconds)"
livekit-wakeword generate "$CONFIG"
livekit-wakeword augment  "$CONFIG"      # folds in feature extraction (the 'extract' stage)
echo "[prep] DONE task=$idx -> features written under data_dir on scratch"
