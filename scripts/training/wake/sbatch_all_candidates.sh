#!/bin/bash -l
# SLURM ARRAY job: train all 3 wake-word candidates on Dartmouth Discovery,
# one candidate per array task (hey_spot, hi_spot, big_yellow).
#
# `-l` (login shell) is REQUIRED — sbatch does NOT source .bashrc, so `conda
# activate` fails without it.
#
# Same prereqs as sbatch_hey_spot.sh (see README.md). `livekit-wakeword setup`
# runs ONCE on the login node — all 3 configs share data_dir, so the ~18 GB
# download is reused across array tasks. Compute nodes are OFFLINE.
# `logs/` must exist at the SUBMIT cwd: `cd <scratch clone> && mkdir -p logs`.
#
#SBATCH --job-name=wake-train
#SBATCH --account=free               # Discovery free/public QOS
#SBATCH --partition=gpuq             # GPU partition; GPU TYPE chosen via --gres
#SBATCH --gres=gpu:l40s:1            # one L40S per task. For H200: --gres=gpu:h200:1
#SBATCH --time=06:00:00              # MANDATORY — Discovery's 1 h default would kill the train
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --array=0-2%2                # 3 candidates; %2 caps CONCURRENT tasks at 2
                                     # (free-account QOS limit QOSMaxGRESPerUser=2).
                                     # A CLI --array override MUST re-state %2 or it drops the cap.
#SBATCH --output=logs/wake-train_%A_%a.out
#SBATCH --error=logs/wake-train_%A_%a.err
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
  || { echo "FATAL: SLURM_ARRAY_TASK_ID=$idx out of range 0..$(( ${#CONFIGS[@]} - 1 )); use --array=0-2%2" >&2; exit 4; }
CONFIG=${CONFIGS[$idx]}

# preflight — fail loud BEFORE the long run:
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA not available on $(hostname)" >&2; exit 3; }
python -c "from livekit.wakeword import load_config; load_config('$CONFIG')" \
  || { echo "FATAL: $CONFIG invalid — validate on the login node first (README step 2)" >&2; exit 2; }

echo "task=$idx config=$CONFIG host=$(hostname) $(date -Iseconds)"; nvidia-smi -L
livekit-wakeword run "$CONFIG"      # generate -> augment -> train -> export -> eval, OFFLINE
echo "DONE task=$idx -> output/$(basename "${CONFIG%.yaml}" | sed 's/wakeword_//')/"
