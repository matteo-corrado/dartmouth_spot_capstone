#!/bin/bash -l
# SLURM GPU job: the SHORT stage — train the classifier head + export + eval, over
# the features sbatch_cpu_prep.sh already wrote to shared DartFS scratch. The model
# is tiny (<2 GB VRAM), so a fair ~8-core share of the shared 8-GPU l40sx8 node is
# plenty; we are NOT here to do the CPU-heavy generate/augment (that was the prep job).
# `-l` login shell REQUIRED (sbatch doesn't source .bashrc).
#
# Submit AFTER sbatch_cpu_prep.sh, gated on it completing OK:
#   prep=$(sbatch --parsable scripts/training/wake/sbatch_cpu_prep.sh)
#   sbatch --dependency=afterok:$prep scripts/training/wake/sbatch_gpu_train.sh
# (afterok on the prep ARRAY waits for all candidates' features; use aftercorr to
#  pair train-task i to prep-task i instead.)
#
#SBATCH --job-name=wake-train
#SBATCH --account=free
#SBATCH --partition=gpu_preempt             # free GPU; --gres=gpu:l40s:1 lands on l40sx8 (8xL40S)
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8            # fair share of the 8-GPU node; train barely uses CPU
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-2%2                # 3 candidates; %2 = free GPU concurrency cap (QOSMaxGRESPerUser)
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
  || { echo "FATAL: SLURM_ARRAY_TASK_ID=$idx out of range 0..$(( ${#CONFIGS[@]} - 1 ))" >&2; exit 4; }
CONFIG=${CONFIGS[$idx]}

python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA not available on $(hostname)" >&2; exit 3; }

echo "[train] task=$idx config=$CONFIG host=$(hostname) $(date -Iseconds)"; nvidia-smi -L
livekit-wakeword train  "$CONFIG"        # consumes the prep job's features from scratch
livekit-wakeword export "$CONFIG"
livekit-wakeword eval   "$CONFIG"
echo "[train] DONE task=$idx -> output/$(basename "${CONFIG%.yaml}" | sed 's/wakeword_//')/"
