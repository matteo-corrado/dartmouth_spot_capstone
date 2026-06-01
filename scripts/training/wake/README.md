# Wake-word training on Dartmouth Discovery

SLURM batch scripts to train the custom livekit-wakeword classifier(s) off-box on
**Dartmouth Discovery** (SLURM + conda, x86_64), then ship the ONNX to the Jetson.
Design rationale + on-device inference: `docs/superpowers/plans/2026-05-30-stage2f-p3-wake-training.md`.

Two ways to run — pick one:

**Combined (simple, one job).** generate→augment→train→export→eval all on the GPU node:
| File | What it does |
|------|--------------|
| `sbatch_hey_spot.sh` | Train **one** candidate (`hey_spot`). Start here. |
| `sbatch_all_candidates.sh` | SLURM **array** — train all 3 (`hey_spot`, `hi_spot`, `big_yellow`), one per task. |

The catch: the GPU node `l40sx8` is a *shared 8-GPU* box, so your fair CPU share is
only ~8 cores — and generate+augment (Piper TTS + mel over ~18 GB) is **CPU-bound**,
so it crawls on 8 cores while the GPU sits idle.

**Split (efficient — recommended for the real run).** CPU bulk on a dedicated 64-core
node, then a short GPU slice for the tiny train:
| File | What it does |
|------|--------------|
| `sbatch_cpu_prep.sh` | `standard` partition, **full 64-core** node: `generate`+`augment` → features on scratch. No GPU (doesn't touch the 2-GPU cap). |
| `sbatch_gpu_train.sh` | `gpuq`+`gpu:l40s:1`: `train`+`export`+`eval` over those features. Gate on the prep job with `--dependency=afterok`. |

Configs live in `configs/wake/wakeword_{hey_spot,hi_spot,big_yellow}.yaml`. Both flows
share the same conda env, data download, and config validation (steps ①–③ below).

## NOTHING COMPILES HERE

Discovery is **x86_64**, so every dependency installs as a prebuilt binary — **no
build nodes (andes/polaris) needed:**
- system libs (espeak-ng, ffmpeg, libsndfile, sox) → conda-forge prebuilt binaries
- Python deps (torch+CUDA, onnxruntime, scipy, soundfile, piper-phonemize) → PyPI
  manylinux wheels (cp311/x86_64)
- livekit-wakeword 0.2.1 → pure-Python wheel (`py3-none-any`)

The `pip --only-binary=:all:` flag below makes pip **refuse to compile** — if a wheel
is ever missing it fails loudly (naming the package) instead of silently building.

---

## Run order (all `①②③` on the Discovery LOGIN node — it has internet; compute nodes are offline)

### ① Build the conda env + clone the repo (one-time)
```bash
SCRATCH=/dartfs-hpc/scratch/$USER
mkdir -p "$SCRATCH" ~/.conda/pkgs/cache ~/.conda/envs
git clone -b tour_guide_upgrade_matteo \
  git@github.com:matteo-corrado/dartmouth_spot_capstone.git "$SCRATCH/spot-capstone"
cd "$SCRATCH/spot-capstone"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh   # enables `conda` (no module to load)
conda create -n lkww python=3.11 -y
conda activate lkww
conda install -n lkww -c conda-forge espeak-ng libsndfile ffmpeg sox -y   # prebuilt system libs

# Pure-Python wheel + wheel-only deps. --only-binary=:all: => pip never compiles.
pip install --only-binary=:all: "livekit-wakeword[train,eval,export]==0.2.1"
livekit-wakeword --help    # sanity: lists setup generate augment train export eval run
```
If that pip line errors naming one package as "no matching distribution / would build",
get just that one from conda-forge (`conda install -c conda-forge <pkg>`) then re-run the
pip line — do NOT drop `--only-binary`.

### ② Point data_dir at scratch + validate every config (the gate — no GPU, seconds)
```bash
sed -i "s|^data_dir:.*|data_dir: $SCRATCH/lkww-data|" configs/wake/wakeword_*.yaml
for c in configs/wake/wakeword_*.yaml; do
  python -c "from livekit.wakeword import load_config; print('OK', load_config('$c').model_name)"
done
```
If a config throws on an unknown field (pydantic `extra='forbid'`), dump the schema and
delete the offending key(s) — suspects are the `augmentation.*` / `batch_n_per_class` blocks:
```bash
python -c "from livekit.wakeword.config import WakeWordConfig, AugmentationConfig; import json; \
print(json.dumps(WakeWordConfig.model_json_schema()['properties'], indent=2, default=str)); \
print('--- AUG ---'); print(json.dumps(AugmentationConfig.model_json_schema()['properties'], indent=2, default=str))"
```

### ③ Download all corpora ONCE (~18 GB → scratch; reused by every candidate)
```bash
livekit-wakeword setup --config configs/wake/wakeword_hey_spot.yaml
# smoke-test alt (skip the 18 GB ACAV negatives, validation-only ~176 MB): add --skip-acav
```

> **One-time check:** read your real free-tier limits before submitting —
> `sacctmgr show qos format=Name,MaxTRESPU%30,MaxJobsPU,MaxWall` (CPU/job cap, not
> published) and `sinfo -o "%P %G"` (partitions + gres). The GPU cap is 2 concurrent
> (hence `%2`); if `l40s_nova`/`adanova01` is `down`, `gpuq`→`l40sx8` is your free L40S.

### ④ Submit — pick ONE flow. Submit from the scratch clone so `logs/` resolves:
```bash
cd /dartfs-hpc/scratch/$USER/spot-capstone
mkdir -p logs                               # sbatch won't create it; --output/--error need it
```

**④a — Combined (simple):**
```bash
sbatch scripts/training/wake/sbatch_hey_spot.sh          # one candidate
# or:  sbatch scripts/training/wake/sbatch_all_candidates.sh   # all 3 (2 run at a time)
squeue -u $USER
tail -f logs/wake-hey-spot_*.out            # watch (Ctrl-C stops the tail, not the job)
```

**④b — Split (recommended): CPU prep on a dedicated 64-core node, then GPU train, gated:**
```bash
prep=$(sbatch --parsable scripts/training/wake/sbatch_cpu_prep.sh)   # generate+augment, standard, 64c
sbatch --dependency=afterok:$prep scripts/training/wake/sbatch_gpu_train.sh   # train+export+eval, l40s
squeue -u $USER
tail -f logs/wake-prep_*.out                # then logs/wake-train_*.out
```
Both arrays default to all 3 candidates; for just `hey_spot` add `--array=0` to each
`sbatch`. Prep defaults to `%1` (sequential, safe under the unknown CPU cap) — raise to
`%2/%3` if `sacctmgr` shows headroom. The GPU job reads the prep's features from the same
scratch (no copy).

Optional — watch the FIRST run live on an interactive L40S before trusting the batch:
```bash
srun --account=free --partition=gpuq --gres=gpu:l40s:1 --time=01:00:00 --mem=64G --cpus-per-task=8 --pty bash -l
source /optnfs/common/miniconda3/etc/profile.d/conda.sh && conda activate lkww
cd /dartfs-hpc/scratch/$USER/spot-capstone && livekit-wakeword run configs/wake/wakeword_hey_spot.yaml
```

### ⑤ After the job — sanity-check + ship the 3-stage chain to the Jetson
```bash
python -c "import json;d=json.load(open('output/hey_spot/hey_spot_eval.json'));print({k:d.get(k) for k in ('recall','fpph','optimal_threshold','optimal_recall','optimal_fpph','validation_hours')})"

mkdir -p wake-artifacts && cp output/hey_spot/hey_spot.onnx wake-artifacts/
# the 2 frozen front-end ONNX from THIS pinned pkg — byte-exact pairing with the trained head:
python -c "from livekit.wakeword.resources import get_mel_model_path, get_embedding_model_path; import shutil; \
shutil.copy(get_mel_model_path(),'wake-artifacts/melspectrogram.onnx'); \
shutil.copy(get_embedding_model_path(),'wake-artifacts/embedding_model.onnx')"
sha256sum wake-artifacts/*.onnx | tee wake-artifacts/sha256.txt

ssh spotdog@<jetson> 'mkdir -p /mnt/ssd/wake-models'
scp wake-artifacts/*.onnx wake-artifacts/sha256.txt spotdog@<jetson>:/mnt/ssd/wake-models/
ssh spotdog@<jetson> 'cd /mnt/ssd/wake-models && sha256sum -c sha256.txt && ln -sf hey_spot.onnx active.onnx'
```

On the Jetson the real operating threshold comes from the Phase C DET sweep
(`tests/audio/eval_wake_det.py`, ≥3 h real negatives); until then it runs at 0.5.
