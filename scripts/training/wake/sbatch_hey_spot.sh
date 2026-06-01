#!/bin/bash -l
# SLURM batch job: train the "hey spot" wake-word classifier on Dartmouth Discovery.
#
# `-l` (login shell) is REQUIRED — sbatch does NOT source .bashrc, so `conda
# activate` fails without it.
#
# Prereqs (run ONCE on the LOGIN node — see README.md in this dir):
#   1. conda env `lkww` (livekit-wakeword[train,eval,export]==0.2.1, prebuilt wheels)
#   2. config schema-validated
#   3. `livekit-wakeword setup --config <cfg>` has downloaded the ~18 GB corpora to
#      data_dir on scratch. Compute nodes are OFFLINE — every download must already
#      be done on the login node.
#   4. `logs/` must exist at the SUBMIT cwd (sbatch won't create it). Submit from the
#      scratch clone: `cd /dartfs-hpc/scratch/$USER/spot-capstone && mkdir -p logs`.
#
#SBATCH --job-name=wake-hey-spot
#SBATCH --account=free               # Discovery free/public QOS
#SBATCH --partition=gpuq             # GPU partition; the GPU TYPE is chosen via --gres
#SBATCH --gres=gpu:l40s:1            # one L40S. For an H200: --gres=gpu:h200:1
#SBATCH --time=06:00:00              # MANDATORY — Discovery's 1 h default would kill the train
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread         # physical cores, no hyperthread siblings
#SBATCH --output=logs/wake-hey-spot_%j.out
#SBATCH --error=logs/wake-hey-spot_%j.err
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<NETID>@dartmouth.edu    # uncomment + set NetID to get emailed
set -euo pipefail
export PYTHONUNBUFFERED=1            # live, unbuffered logs in the .out file

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate lkww
cd "/dartfs-hpc/scratch/$USER/spot-capstone"

# preflight — fail loud BEFORE the (long) run, with the exact fix in the message:
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FATAL: CUDA not available on $(hostname)" >&2; exit 3; }
python -c "from livekit.wakeword import load_config; load_config('configs/wake/wakeword_hey_spot.yaml')" \
  || { echo "FATAL: config invalid — validate on the login node first (README step 2)" >&2; exit 2; }

echo "host=$(hostname) $(date -Iseconds)"; nvidia-smi -L
# run = generate -> augment -> train -> export -> eval, in one process, OFFLINE.
livekit-wakeword run configs/wake/wakeword_hey_spot.yaml
echo "DONE -> output/hey_spot/hey_spot.onnx + output/hey_spot/hey_spot_eval.json"
