#!/bin/bash -l
# Does livekit-wakeword's Piper/VITS `generate` parallelize across CPU cores?
#
# generate is ALREADY batched (tts_batch_size=50). The only no-code lever is the number of
# BLAS/torch intra-op threads the VITS forward uses. This times `generate` at several thread
# counts so you can see which regime you're in:
#   - speedup grows with threads  -> forward is compute-bound, more cores help (pin them, fewer
#     concurrent candidates x bigger core count).
#   - speedup ~1 across threads   -> phonemization/Python-bound, threads useless; only running
#     the candidates concurrently (sbatch --array=...%3) buys you anything.
#
# Run on the login node for a quick read, or (cleaner, uncontended cores) an interactive node:
#   srun --account=free --partition=standard --cpus-per-task=16 --time=00:30:00 --pty bash -l
#
# Usage:  scripts/training/wake/bench_generate_threads.sh [N_CLIPS] [THREADS...]
#   bench_generate_threads.sh                # 1000 clips, threads 1 4 16
#   bench_generate_threads.sh 2000 1 8 32
set -euo pipefail

N=${1:-1000}; shift || true
THREADS=("$@"); [ ${#THREADS[@]} -eq 0 ] && THREADS=(1 4 16)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DATA="/dartfs-hpc/scratch/$USER/lkww-data"
CFG="$ROOT/configs/wake/wakeword_hey_spot.yaml"
CKPT="$DATA/piper/en-us-libritts-high.pt"
[ -f "$CKPT" ] || { echo "FATAL: VITS checkpoint missing: $CKPT  (run setup, or fix data_dir)"; exit 1; }

# Bench config: absolute data_dir baked in (no $VAR-in-file surprises), N positives, nothing
# else, so the wall time is dominated by the VITS positive synth we want to measure.
BENCH=$(mktemp); OUT=$(mktemp -d)
trap 'rm -f "$BENCH"; rm -rf "$OUT"' EXIT
python - "$DATA" "$OUT" "$N" "$CFG" "$BENCH" <<'PY'
import sys, yaml
data, out, n, cfg, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
c = yaml.safe_load(open(cfg))
c.update(data_dir=data, output_dir=out, n_samples=n,
         n_samples_val=0, n_background_samples=0, n_background_samples_val=0)
yaml.safe_dump(c, open(dst, "w"))
PY
python -c "from livekit.wakeword import load_config; print('config loads OK, data_dir', load_config('$BENCH').data_dir)"

echo "=== generate $N clips @ threads ${THREADS[*]} ==="
base=""
for n in "${THREADS[@]}"; do
  rm -rf "${OUT:?}"/*
  start=$SECONDS
  OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n \
    livekit-wakeword generate "$BENCH" >"/tmp/wwbench_$n.log" 2>&1 || true
  wall=$((SECONDS - start)); [ "$wall" -lt 1 ] && wall=1
  made=$(find "$OUT" -name 'clip_*.wav' 2>/dev/null | wc -l)
  if [ "$made" -lt $((N * 9 / 10)) ]; then
    echo "threads=$n  FAILED/incomplete (clips=$made) — tail:"; tail -4 "/tmp/wwbench_$n.log"; break
  fi
  [ -z "$base" ] && base=$wall
  awk -v t="$n" -v w="$wall" -v c="$made" -v b="$base" \
    'BEGIN{printf "threads=%-3s wall=%4ds  clips=%d  %.3fs/clip  speedup=%.2fx\n", t, w, c, w/c, b/w}'
done
echo "--- speedup ~1 across threads => phonemization/Python-bound (only --array %3 helps)."
echo "--- speedup grows with threads => VITS forward parallelizes (give candidates more cores)."
