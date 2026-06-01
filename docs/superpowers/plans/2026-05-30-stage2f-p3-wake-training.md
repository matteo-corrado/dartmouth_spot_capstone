# Stage 2F P3 — Custom "Hey Spot" Wake Word Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a custom `livekit-wakeword` "hey spot" classifier off-box on Dartmouth Discovery (SLURM + conda), deploy its ONNX to the Jetson, run inference through spot-env's `onnxruntime-gpu` (no `livekit` package at runtime), derive the deploy threshold from real held-out audio, and ship it behind `SPOT_WAKE_BACKEND=livekit` with the sherpa-onnx + push-to-talk fallback intact.

**Architecture:** Three phases. **(A) Cluster training** — operational runbook the operator runs on **Dartmouth Discovery** (SLURM + conda; L40S or H200 GPU partition; Python 3.11; spot-env is untouched). **(B) On-device inference** — TDD code in spot-env (Python 3.10): the exported `hey_spot.onnx` is only the *classifier head* (`embeddings (B,16,96) → score (B,1)`), so we hand-roll the full 3-stage ONNX chain (`melspectrogram.onnx → embedding_model.onnx → hey_spot.onnx`) directly via `onnxruntime-gpu`, because the `livekit` pip package requires Python 3.11 and cannot import in spot-env. **(C) Eval/threshold/deploy** — derive the operating point from real audio (TPR on the existing `*_lab.wav` positives, FAR on ≥3h of newly-recorded negatives) and flip the backend only if the real held-out TPR clears the gate.

**Tech Stack:** livekit-wakeword (pinned commit `1ec7f680`, Apache-2.0, conv-attention head); Piper VITS synthetic positives; Dartmouth Discovery (SLURM + conda, L40S/H200); onnxruntime-gpu 1.23.0 (CUDA EP) on aarch64; numpy. Verified model-currency (2026-05-30): livekit-wakeword is openWakeWord's direct successor and the current best-in-class for custom-phrase, synthetic-trained, ONNX-exported KWS; sherpa-onnx KWS stays as the rollback.

**Key facts (verified at livekit commit `1ec7f680` / v0.2.1):**
- CLI is a Typer app `livekit-wakeword`. `setup` uses `-c/--config`; every other subcommand (`generate`, `augment`, `train`, `export`, `eval`) takes the config path as a **positional** arg. `run <config>` does generate→augment→extract→train→export→eval in one process.
- `augment` folds feature-extraction in (no separate `extract` subcommand).
- Positives are 100% synthesized from the config's `target_phrases` via Piper VITS (904-speaker SLERP). **ElevenLabs is not in the pipeline** (and is ToS-blocked for training — V10). **There is no custom-positive-dir field (V1 gap): real recordings can only be an external eval set, never training input.**
- Real room realism is injected through `augmentation.background_paths` and `augmentation.rir_paths` (lists of dirs; 16 kHz mono WAV; globbed `**/*.wav`).
- Exported classifier I/O: input `embeddings` `(B,16,96)` f32; output `score` `(B,1)` f32; opset 18.
- Front-end (frozen, shared across all wake words, bundled in the package): `melspectrogram.onnx` (~1.06 MB) and `embedding_model.onnx` (~1.33 MB). Fetchable from the pinned commit without installing:
  - `https://raw.githubusercontent.com/livekit/livekit-wakeword/1ec7f680df30ff4ca0ebae6b5983441e94b10980/src/livekit/wakeword/resources/melspectrogram.onnx`
  - `https://raw.githubusercontent.com/livekit/livekit-wakeword/1ec7f680df30ff4ca0ebae6b5983441e94b10980/src/livekit/wakeword/resources/embedding_model.onnx`
- Inference chain math (from `inference/model.py`): scale int16 → float32 by `/32768.0`; mel over the whole 2 s (32000-sample) window → `(1, T, 32)`, then **`mel = mel/10.0 + 2.0`**; slide a **76-mel-frame window, stride 8** → one `(76,32,1)` → 96-d embedding each; stack the **last 16** embeddings → `(1,16,96)` → classifier → score in [0,1]. Read every ONNX input name dynamically (`session.get_inputs()[0].name`); only the exported classifier is guaranteed-named.

---

## File Structure

| File | Responsibility | Phase |
|------|----------------|-------|
| `scripts/training/setup_livekit_wakeword.sh` (LEGACY) | Jetson-era (`/mnt/ssd`, spot-env) — **superseded** by the inline Discovery conda flow (A1); not run on the cluster | A |
| `scripts/training/sbatch_wake_train.sh` (CREATE) | Discovery SLURM array job: `run` the 3 candidate configs on L40S/H200 | A |
| `configs/wake/wakeword_hey_spot.yaml` (MODIFY) | Production base config, schema-aligned to the pinned commit | A |
| `configs/wake/wakeword_hi_spot.yaml` (CREATE) | Candidate B | A |
| `configs/wake/wakeword_big_yellow.yaml` (CREATE) | Candidate C | A |
| `src/voice_control/wake/livekit.py` (REWRITE) | Hand-rolled 3-stage ONNX chain via onnxruntime-gpu; no `livekit` import | B |
| `tests/voice_control/wake/test_livekit_windows.py` (CREATE) | Unit test for the pure windowing helper | B |
| `tests/audio/eval_wake_det.py` (CREATE) | On-device DET sweep → threshold at FAR ≤ target | C |
| `scripts/training/train_wake_hey_spot.sh` (LEGACY) | Jetson-era — **superseded** by the sbatch flow (A4); not run on the cluster | A |

Unchanged and relied upon: `src/voice_control/wake/__init__.py` (`make_wake_detector` already routes `SPOT_WAKE_BACKEND=livekit`), `src/voice_control/wake/sherpa_onnx.py` (fallback), `scripts/record_corpus.py` (corpus recorder), `tests/audio/eval_wake.py` (P2 harness).

**Success criteria (whole plan):**
1. Cluster produces `hey_spot.onnx`, `hi_spot.onnx`, `big_yellow.onnx` + their `_eval.json` (with `optimal_threshold`).
2. On the Jetson, `SPOT_WAKE_BACKEND=livekit spot-env/bin/python -m tests.audio.eval_wake` runs the 3-stage chain with no `livekit` import and reports a real TPR.
3. `tests/audio/eval_wake_det.py` emits a TPR-vs-FAR sweep over real audio and prints the threshold meeting FAR ≤ 1/hr (or reports it is unmet → keep sherpa + PTT).
4. The unit suite stays green; the default backend stays `sherpa_onnx` until the gate in Task C3 is cleared.

---

## Phase A — Cluster training on Dartmouth Discovery (runbook; verification = artifacts, not unit tests)

> Runs on **Dartmouth Discovery** (SLURM + conda; docs: rc.dartmouth.edu). Discovery rules baked in below: conda comes from `source /optnfs/common/miniconda3/etc/profile.d/conda.sh` (there is no anaconda module to load); the sbatch script MUST begin `#!/bin/bash -l` (sbatch does NOT source `.bashrc`, so `conda activate` fails otherwise); GPU jobs go to an **L40S** (`l40s_nova`, free/public) or **H200** partition; **`--time` is mandatory** (Discovery's 1 h default would kill the train); data + clone live on **`/dartfs-hpc/scratch/<NETID>/`** (home is only 50 GB); and because compute-node egress is not guaranteed, **all downloads (conda, git, dataset) happen on the LOGIN node** and the sbatch job trains offline. spot-env on the Jetson is never touched here. Replace `<NETID>` with your Dartmouth NetID throughout.

### Task A1: Discovery conda env + repo on scratch (LOGIN node — has internet)

**Files:** none (environment + working tree on Discovery scratch).

- [ ] **Step 1: Clone this repo + build the conda env on the LOGIN node**

```bash
# Discovery LOGIN node. Use scratch — home is only 50 GB; the env + 18 GB data won't fit.
SCRATCH=/dartfs-hpc/scratch/$USER
mkdir -p "$SCRATCH" ~/.conda/pkgs/cache ~/.conda/envs        # last two avoid a first-run conda cache error
git clone -b tour_guide_upgrade_matteo <your-repo-url> "$SCRATCH/spot-capstone"
cd "$SCRATCH/spot-capstone"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh      # enables `conda` (no module to load)
conda create -n lkww python=3.11 -y
conda activate lkww
conda install -n lkww -c conda-forge espeak-ng libsndfile ffmpeg sox -y   # TTS/audio deps (no apt on HPC)
pip install "livekit-wakeword[train,eval,export] @ git+https://github.com/livekit/livekit-wakeword@1ec7f680df30ff4ca0ebae6b5983441e94b10980"
```

- [ ] **Step 2: Verify the CLI + CUDA torch on an L40S node**

```bash
# short interactive GPU slice (-l login shell so conda works):
srun --partition=l40s_nova --gres=gpu:1 --time=00:15:00 --pty bash -l
source /optnfs/common/miniconda3/etc/profile.d/conda.sh && conda activate lkww
livekit-wakeword --help
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
exit
```
Expected: the Typer help lists `setup generate augment train export eval run`; torch prints a `2.5+` version and `True` (CUDA visible on the L40S node).

- [ ] **Step 3: Find your SLURM account (paid/private partitions only)**

```bash
sacctmgr show associations where user=$USER
```
`l40s_nova` is free/public and usually needs no `--account`. If you instead use an **H200** partition (`h200` / `h200_preemptable`) or your output shows a required account/QOS, note the account string and uncomment `--account` in the sbatch script (A4). The legacy `scripts/training/{setup_livekit_wakeword,train_wake_hey_spot}.sh` are Jetson-era (`/mnt/ssd`, spot-env) and are **superseded by this Discovery flow** — do not run them on the cluster.

### Task A2: Schema-align and harden the base config

**Files:**
- Modify: `configs/wake/wakeword_hey_spot.yaml`

- [ ] **Step 1: Dump the pinned config schema (the existing YAML predates this commit — verify every key)**

Run on the cluster venv:
```bash
python -c "from livekit.wakeword.config import WakeWordConfig; import json; print(json.dumps(WakeWordConfig.model_json_schema()['properties'], indent=2, default=str))" | tee /tmp/lkww_schema.json
python -c "from livekit.wakeword.config import AugmentationConfig; import json; print(json.dumps(AugmentationConfig.model_json_schema()['properties'], indent=2, default=str))" | tee /tmp/lkww_aug_schema.json
# Fallback if the config is not pydantic (no model_json_schema) — dump its source:
python -c "from livekit.wakeword import config; import inspect; print(inspect.getsource(config))" | tee /tmp/lkww_config_src.py
```
Expected: a list of valid top-level fields (incl. `target_phrases`, `custom_negative_phrases`, `n_samples`, `steps`, `target_fp_per_hour`, `model`) and augmentation fields (incl. `background_paths`, `rir_paths`). **Note which of the existing YAML's augmentation keys (`background_snr_min_db`, `rir_prob`, `gain_min_db`, `pitch_shift_semitones`, `time_stretch_min/max`) actually exist in this schema.** pydantic with `extra='forbid'` will reject unknown keys; if it ignores them, they are silently dead. Either way, drop keys not in the schema.

- [ ] **Step 2: Apply the production training deltas + drop any schema-invalid keys**

Edit `configs/wake/wakeword_hey_spot.yaml`: set `model.model_size: large`, `n_samples: 50000`, `steps: 80000`, `target_fp_per_hour: 1.0` (train for high recall; the *deploy* threshold is DET-derived later — do not hardcode a low threshold). **Repoint `data_dir`** from the Jetson path `/mnt/ssd/livekit-wakeword-data` to Discovery scratch `/dartfs-hpc/scratch/<NETID>/lkww-data` — `data_dir` is used ONLY during cluster training; the Jetson never reads this config. Keep `target_phrases` and `custom_negative_phrases` as-is. Remove augmentation keys that Step 1 proved invalid; keep only schema-valid fields.

```yaml
# (only the changed lines shown — match existing file layout)
data_dir: /dartfs-hpc/scratch/<NETID>/lkww-data   # was /mnt/ssd/... (Jetson); Discovery scratch
n_samples: 50000
target_fp_per_hour: 1.0
steps: 80000
model:
  model_type: conv_attention
  model_size: large
```

- [ ] **Step 3: Validate the config loads**

Run: `python -c "from livekit.wakeword.config import WakeWordConfig; WakeWordConfig.from_yaml('configs/wake/wakeword_hey_spot.yaml'); print('config OK')"`
(If the loader function differs, use the one named in the schema dump — e.g. `WakeWordConfig.model_validate(yaml.safe_load(...))`.)
Expected: `config OK` with no validation error.

- [ ] **Step 4: Commit**

```bash
git add configs/wake/wakeword_hey_spot.yaml
git commit -m "stage2f p3: hey_spot config -> large/50k/80k, schema-aligned to pinned commit"
```

### Task A3: Candidate B and C configs

**Files:**
- Create: `configs/wake/wakeword_hi_spot.yaml`, `configs/wake/wakeword_big_yellow.yaml`

- [ ] **Step 1: Derive the two alternates from the base**

Copy `wakeword_hey_spot.yaml` to each, changing only `model_name`, `target_phrases`, and `custom_negative_phrases` (hard-negatives per the spec's confusable table). `model_name` also names the output dir and the exported `.onnx`.

`wakeword_hi_spot.yaml` (diff from base):
```yaml
model_name: hi_spot
target_phrases:
  - "hi spot"
  - "hi, spot"
custom_negative_phrases:
  - "hi scott"
  - "hipster"
  - "high spot"
  - "hi sport"
  - "hide spot"
```

`wakeword_big_yellow.yaml` (diff from base):
```yaml
model_name: big_yellow
target_phrases:
  - "big yellow"
custom_negative_phrases:
  - "big fellow"
  - "big mellow"
  - "pig yellow"
  - "big yell"
  - "big jello"
```

- [ ] **Step 2: Validate both load** (same command as A2 Step 3, per file). Expected: `config OK` for each.

- [ ] **Step 3: Commit**

```bash
git add configs/wake/wakeword_hi_spot.yaml configs/wake/wakeword_big_yellow.yaml
git commit -m "stage2f p3: hi_spot + big_yellow candidate configs"
```

### Task A4: SLURM array job for the 3 candidates

**Files:**
- Create: `scripts/training/sbatch_wake_train.sh`

- [ ] **Step 1: Write the sbatch script**

```bash
#!/bin/bash -l
# -l (login shell) is REQUIRED on Discovery: sbatch does NOT source .bashrc, so
# without it `conda activate` fails. (rc.dartmouth.edu/hpc/sbatch)
#SBATCH --job-name=wake-train
#SBATCH --partition=l40s_nova       # free/public L40S (cap 2 GPUs/node, 3-day). Alt: h200 / h200_preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00             # MANDATORY — Discovery default is 1h and would kill the train
#SBATCH --array=0-2
#SBATCH --output=%x-%a-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<NETID>@dartmouth.edu
## --account only for H200/paid partitions; uncomment with the string from `sacctmgr` (A1 Step 3):
##SBATCH --account=<SLURM_ACCOUNT>
# One candidate per array task: generate->augment->train->export->eval, OFFLINE.
# Prereq: A4 Step 2 ran `setup` once on the LOGIN node (~18 GB -> data_dir on scratch).
# --time is a ceiling: smoke-test per-step time with a `--steps 1000` run first; large/80k
# must finish inside the wall-time or the job is killed before `export` and no ONNX is written.
set -euo pipefail

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate lkww
cd "/dartfs-hpc/scratch/$USER/spot-capstone"     # the repo clone from A1 (has configs/)

CONFIGS=(configs/wake/wakeword_hey_spot.yaml \
         configs/wake/wakeword_hi_spot.yaml \
         configs/wake/wakeword_big_yellow.yaml)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Training $CONFIG on $(hostname), GPU $CUDA_VISIBLE_DEVICES"; nvidia-smi
livekit-wakeword run "$CONFIG"
```

- [ ] **Step 2: One-time data download, then submit**

```bash
# LOGIN node (has internet) — pre-fetch ALL data so the sbatch job runs offline:
cd "/dartfs-hpc/scratch/$USER/spot-capstone"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh && conda activate lkww
livekit-wakeword setup -c configs/wake/wakeword_hey_spot.yaml   # ~18GB -> data_dir (scratch), once
# (smoke option: append --skip-acav for ~176MB validation-only negatives)
sbatch scripts/training/sbatch_wake_train.sh                    # 3-candidate array on L40S
squeue -u $USER                                                 # watch the array
```
Expected: `setup` populates the scratch `data_dir`; `sbatch` returns a job id; each array task ends with `output/<model_name>/<model_name>.onnx`, `<model_name>_eval.json`, `<model_name>_det.png`. All Piper/ACAV/MUSAN/RIR fetches happen here on the login node, so the compute job needs no internet.

- [ ] **Step 3: Sanity-read each eval JSON**

```bash
for m in hey_spot hi_spot big_yellow; do
  echo "== $m =="; python -c "import json;d=json.load(open('output/$m/${m}_eval.json'));print({k:d[k] for k in ('recall','fpph','optimal_threshold','optimal_recall','optimal_fpph','validation_hours')})"
done
```
Expected: each prints `optimal_threshold` and `optimal_recall`/`optimal_fpph`. **These FPPH numbers are on synthetic negatives — they pick a *starting* threshold only; the real operating point comes from Phase C.**

- [ ] **Step 4: Commit the sbatch script**

```bash
git add scripts/training/sbatch_wake_train.sh
git commit -m "stage2f p3: SLURM array job to train 3 wake candidates"
```

### Task A5: Stage artifacts for the Jetson

**Files:** none (artifact movement).

- [ ] **Step 1: Collect the 3 classifiers + the 2 shared front-end models**

```bash
# LOGIN node, in the repo clone on scratch (where output/ landed):
cd "/dartfs-hpc/scratch/$USER/spot-capstone"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh && conda activate lkww
mkdir -p wake-artifacts
for m in hey_spot hi_spot big_yellow; do cp output/$m/$m.onnx wake-artifacts/; done
# Front-end models are frozen + shared — copy once (from the installed package):
python -c "from livekit.wakeword.resources import get_mel_model_path, get_embedding_model_path; import shutil; shutil.copy(get_mel_model_path(),'wake-artifacts/melspectrogram.onnx'); shutil.copy(get_embedding_model_path(),'wake-artifacts/embedding_model.onnx')"
# Probe the front-end input shapes — confirms the embedding model accepts a BATCH
# dim N>1 (the loader feeds (N,76,32,1)). If batch is static=1, B1 must loop.
python -c "import onnxruntime as o; [print(p, o.InferenceSession('wake-artifacts/'+p).get_inputs()[0].shape) for p in ('melspectrogram.onnx','embedding_model.onnx')]"
sha256sum wake-artifacts/*.onnx | tee wake-artifacts/sha256.txt
ls -lh wake-artifacts/
```
Expected: 5 ONNX files (3 classifiers ~ small, `melspectrogram.onnx` ~1.06 MB, `embedding_model.onnx` ~1.33 MB). The embedding model's input shape shows a symbolic / `-1` batch dim (dynamic — batch inference is fine; this matches upstream's own batched `extract_embeddings`). `sha256.txt` records every hash.

- [ ] **Step 2: Transfer to the Jetson**

```bash
scp wake-artifacts/*.onnx wake-artifacts/sha256.txt spotdog@<jetson>:/mnt/ssd/wake-models/
ssh spotdog@<jetson> "cd /mnt/ssd/wake-models && sha256sum -c sha256.txt"
```
Expected: every file reports `OK`; `/mnt/ssd/wake-models/` holds the 5 ONNX (+ `sha256.txt`). No `active.onnx` yet — Task C3 creates that symlink.

---

## Phase B — On-device inference (TDD, spot-env Python 3.10, onnxruntime-gpu)

### Task B1: Rewrite the livekit backend as a hand-rolled 3-stage ONNX chain

**Files:**
- Rewrite: `src/voice_control/wake/livekit.py`
- Test: `tests/voice_control/wake/test_livekit_windows.py`

The current file does `from livekit.wakeword import WakeWordModel` and `self.model.predict(window)` — that import **fails on spot-env's Python 3.10** (`StrEnum`), so `SPOT_WAKE_BACKEND=livekit` can never work as written. Replace it with three `onnxruntime` sessions. Extract the mel-frame windowing into a pure function so it is unit-testable without any model.

- [ ] **Step 1: Write the failing test for the windowing helper**

```python
# tests/voice_control/wake/test_livekit_windows.py
import numpy as np
from src.voice_control.wake.livekit import embedding_windows, EMBEDDING_WINDOW, EMBEDDING_STRIDE


def test_window_count_and_shape():
    # floor((T-76)/8)+1 windows of (76,32), count derived from the constants
    T = 200
    mel = np.zeros((T, 32), dtype=np.float32)
    win = embedding_windows(mel)
    expected = (T - EMBEDDING_WINDOW) // EMBEDDING_STRIDE + 1
    assert win.shape == (expected, EMBEDDING_WINDOW, 32)


def test_too_few_frames_returns_empty():
    mel = np.zeros((EMBEDDING_WINDOW - 1, 32), dtype=np.float32)
    assert embedding_windows(mel).shape[0] == 0


def test_windows_advance_by_stride():
    mel = np.arange(200 * 32, dtype=np.float32).reshape(200, 32)
    win = embedding_windows(mel)
    # second window starts EMBEDDING_STRIDE frames after the first
    assert np.array_equal(win[1, 0], mel[EMBEDDING_STRIDE])
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `spot-env/bin/python -m pytest tests/voice_control/wake/test_livekit_windows.py -v`
Expected: FAIL — `ImportError: cannot import name 'embedding_windows'` (the rewrite does not exist yet).

- [ ] **Step 3: Rewrite `src/voice_control/wake/livekit.py`**

```python
"""LiveKit-Wakeword inference for the Spot voice loop (Stage 2F P3).

spot-env is Python 3.10; the livekit-wakeword pip package needs 3.11 (StrEnum),
so we do NOT import it at runtime. We run its 3-stage ONNX chain directly through
onnxruntime: melspectrogram.onnx -> embedding_model.onnx -> hey_spot.onnx (the
trained classifier head). The two front-end models are frozen and shared across
all wake words; only the classifier is custom-trained.

Pinned to livekit-wakeword commit 1ec7f680. API mirrors the sherpa-onnx path so
client_mic.py swaps backends via SPOT_WAKE_BACKEND with no other change:
    is_available() -> bool
    process_frame(pcm16_bytes) -> bool
    reset() -> None
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

WAKE_DIR = Path(os.environ.get("LIVEKIT_WAKE_DIR", "/mnt/ssd/wake-models"))
MEL_MODEL = WAKE_DIR / "melspectrogram.onnx"
EMB_MODEL = WAKE_DIR / "embedding_model.onnx"
# Default = the deploy symlink (active.onnx -> winning candidate) so swapping
# candidates needs no env change (spec Component 1 deploy path).
CLF_MODEL = Path(os.environ.get("LIVEKIT_WAKE_MODEL", str(WAKE_DIR / "active.onnx")))
THRESHOLD_FILE = WAKE_DIR / "active_threshold.json"

SAMPLE_RATE = 16000
# 2 s + one extra 10 ms hop guarantees >= 200 mel frames -> >= 16 embedding
# windows (floor((T-76)/8)+1 >= 16 needs T >= 196), avoiding the boundary case
# where a 2.000 s window emits 199 frames -> 15 windows -> a silent 0.0 score.
WINDOW_SAMPLES = SAMPLE_RATE * 2 + 160
STEP_SAMPLES = int(SAMPLE_RATE * 0.08)    # re-score every 80 ms
EMBEDDING_WINDOW = 76                      # mel frames per embedding
EMBEDDING_STRIDE = 8                       # mel-frame hop
MIN_EMBEDDINGS = 16                        # classifier consumes the last 16


def _load_threshold() -> float:
    """Precedence: env var > deployed threshold file > 0.5 (logged loud).

    Falling back to the default means the Phase C DET sweep was not completed
    for this model — we warn rather than silently ship a guessed threshold.
    """
    env = os.environ.get("LIVEKIT_WAKE_THRESHOLD")
    if env is not None:
        return float(env)
    if THRESHOLD_FILE.exists():
        return float(json.loads(THRESHOLD_FILE.read_text())["threshold"])
    print("[wake livekit] WARNING: no LIVEKIT_WAKE_THRESHOLD and no "
          f"{THRESHOLD_FILE.name}; using default 0.5 (run the Phase C DET sweep)")
    return 0.5


def embedding_windows(mel: np.ndarray) -> np.ndarray:
    """Slide a 76-frame / stride-8 window over mel frames.

    mel: (T, 32). Returns (N, 76, 32); N == 0 when T < 76.
    """
    n_frames = mel.shape[0]
    starts = range(0, n_frames - EMBEDDING_WINDOW + 1, EMBEDDING_STRIDE)
    wins = [mel[s:s + EMBEDDING_WINDOW] for s in starts]
    if not wins:
        return np.empty((0, EMBEDDING_WINDOW, mel.shape[1]), dtype=mel.dtype)
    return np.stack(wins, axis=0)


class LiveKitWakeWord:
    name = "livekit"

    def __init__(self) -> None:
        self._available = False
        self.last_score = 0.0
        for p in (MEL_MODEL, EMB_MODEL, CLF_MODEL):
            if not p.exists():
                print(f"[wake livekit] model missing: {p}")
                print("[wake livekit] train on the cluster + scp the 3 ONNX to /mnt/ssd/wake-models")
                return
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._mel = ort.InferenceSession(str(MEL_MODEL), providers=providers)
        self._emb = ort.InferenceSession(str(EMB_MODEL), providers=providers)
        self._clf = ort.InferenceSession(str(CLF_MODEL), providers=providers)
        self._mel_in = self._mel.get_inputs()[0].name
        self._emb_in = self._emb.get_inputs()[0].name
        self._clf_in = self._clf.get_inputs()[0].name
        self.threshold = _load_threshold()
        self.buffer = np.zeros(0, dtype=np.int16)
        self._pending = 0
        self._available = True
        print(f"[wake livekit] 3-stage chain loaded, threshold={self.threshold}")

    def is_available(self) -> bool:
        return self._available

    def reset(self) -> None:
        if self._available:
            self.buffer = np.zeros(0, dtype=np.int16)
            self._pending = 0
            self.last_score = 0.0

    def _score(self, window_i16: np.ndarray) -> float:
        audio = window_i16.astype(np.float32) / 32768.0
        mel = self._mel.run(None, {self._mel_in: audio[np.newaxis, :]})[0]
        if mel.ndim == 4:
            mel = mel[:, 0, :, :]                         # (1,1,T,32) -> (1,T,32)
        mel = mel[0]                                      # (T, 32)
        assert mel.ndim == 2 and mel.shape[1] == 32, f"unexpected mel shape {mel.shape}"
        mel = mel / 10.0 + 2.0                            # Google front-end calibration
        wins = embedding_windows(mel)
        if wins.shape[0] < MIN_EMBEDDINGS:
            return 0.0
        batch = wins[..., np.newaxis].astype(np.float32)  # (N, 76, 32, 1)
        emb = self._emb.run(None, {self._emb_in: batch})[0].squeeze(axis=(1, 2))  # (N, 96)
        seq = emb[-MIN_EMBEDDINGS:][np.newaxis, :, :].astype(np.float32)          # (1, 16, 96)
        return float(self._clf.run(None, {self._clf_in: seq})[0][0, 0])

    def process_frame(self, pcm16_bytes: bytes) -> bool:
        if not self._available:
            return False
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.buffer = np.concatenate([self.buffer, samples])[-WINDOW_SAMPLES:]
        self._pending += len(samples)
        if len(self.buffer) < WINDOW_SAMPLES or self._pending < STEP_SAMPLES:
            return False
        self._pending = 0
        self.last_score = self._score(self.buffer)
        return self.last_score >= self.threshold
```

- [ ] **Step 4: Run the windowing test to confirm it passes**

Run: `spot-env/bin/python -m pytest tests/voice_control/wake/test_livekit_windows.py -v`
Expected: 3 passed.

- [ ] **Step 5: Confirm no `livekit` import remains and the module imports under 3.10**

Run: `spot-env/bin/python -c "import src.voice_control.wake.livekit as m; print('import OK', m.LiveKitWakeWord.name)"`
Expected: `import OK livekit` (no `StrEnum`/`livekit` ImportError; onnxruntime import is lazy inside `__init__`).

- [ ] **Step 6: Commit**

```bash
git add src/voice_control/wake/livekit.py tests/voice_control/wake/test_livekit_windows.py
git commit -m "stage2f p3: hand-roll livekit 3-stage ONNX wake chain (no py3.11 dep)"
```

### Task B2: On-device DET sweep + threshold derivation over real audio

**Files:**
- Create: `tests/audio/eval_wake_det.py`

`eval_wake.py` fires at a single fixed threshold. For deployment we need a TPR-vs-FAR sweep over real audio and the threshold that meets FAR ≤ target. The rewritten detector now exposes `last_score`, so we sweep that.

- [ ] **Step 1: Write the DET sweep**

```python
# tests/audio/eval_wake_det.py
"""Stage 2F P3 wake DET sweep over the real lab corpus.

For each wav, take the MAX livekit classifier score across its 80ms-step
windows, then sweep thresholds: TPR over positives, false-accepts/hour over the
negative audio. Prints the highest-recall threshold with FAR CI-upper <= target.
Positives: hey_spot_*_lab.wav. Negatives: adv_*_lab.wav + idle_*_lab.wav.
"""
import glob
import json
import math
import pathlib
import sys
import wave

import numpy as np

from src.voice_control.wake.livekit import WINDOW_SAMPLES

CORPUS = pathlib.Path(__file__).resolve().parent
POSITIVE_GLOB = "hey_spot_*_lab.wav"
NEGATIVE_GLOBS = ["adv_*_lab.wav", "idle_*_lab.wav"]
SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 30 // 1000 * 2     # 30 ms PCM16 frames
TARGET_FAR_PER_HOUR = 1.0


def _read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: expected 16kHz"
        return w.readframes(w.getnframes())


def _frames(pcm):
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


def _max_score(det, pcm):
    det.reset()
    best = 0.0
    # Prepend 2 s+ of silence so even sub-2s clips fill the detector's window and
    # the clip is scored within a full window (matches how the live loop fills).
    padded = bytes(WINDOW_SAMPLES * 2) + pcm
    for f in _frames(padded):
        det.process_frame(f)          # updates det.last_score (reset() cleared it)
        best = max(best, det.last_score)
    return best


def _ci_upper_per_hour(false_accepts, hours):
    if hours <= 0:
        return float("inf")
    if false_accepts == 0:
        return 3.0 / hours
    # Conservative closed-form upper bound (no scipy dep); deliberately looser
    # than a normal approx at low counts so the FAR<=1/hr gate is not passed on
    # a single borderline false-accept.
    return (false_accepts + 1 + 1.96 * math.sqrt(false_accepts + 1)) / hours


def main():
    from src.voice_control.wake.livekit import LiveKitWakeWord
    det = LiveKitWakeWord()
    if not det.is_available():
        raise SystemExit("[eval_wake_det] livekit detector unavailable — deploy the 3 ONNX first")

    pos = sorted(glob.glob(str(CORPUS / POSITIVE_GLOB)))
    neg = sorted(sum([glob.glob(str(CORPUS / g)) for g in NEGATIVE_GLOBS], []))
    pos_scores = [_max_score(det, _read_pcm16(p)) for p in pos]
    neg_seconds, neg_scores = 0.0, []
    for n in neg:
        pcm = _read_pcm16(n)
        neg_seconds += len(pcm) / 2 / SAMPLE_RATE
        neg_scores.append(_max_score(det, pcm))
    hours = neg_seconds / 3600.0

    best = None
    for thr in [round(t, 3) for t in np.linspace(0.05, 0.95, 181)]:
        tpr = sum(s >= thr for s in pos_scores) / len(pos_scores) if pos_scores else 0.0
        fa = sum(s >= thr for s in neg_scores)
        far_ci = _ci_upper_per_hour(fa, hours)
        if far_ci <= TARGET_FAR_PER_HOUR and (best is None or tpr > best["tpr"]):
            best = {"threshold": thr, "tpr": tpr, "false_accepts": fa, "far_ci_upper": round(far_ci, 3)}

    out = {
        "positives": len(pos_scores),
        "negative_hours": round(hours, 4),
        "best_at_far_target": best,
        "note": "negative_hours < 3 => FAR<=1/hr UNMEASURABLE (spec C3); record more idle audio",
    }
    print(json.dumps(out, indent=2))
    if hours < 3.0:
        print(f"[eval_wake_det] WARNING: only {hours:.3f}h negatives — FAR<=1/hr not yet provable.")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run it (models must be deployed from Phase A)**

Run: `SPOT_WAKE_BACKEND=livekit spot-env/bin/python -m tests.audio.eval_wake_det`
Expected: JSON with `best_at_far_target` (a `{threshold, tpr, ...}` or `null` if unmet) and the `negative_hours` warning if < 3 h. (If models are not yet deployed it exits with the "unavailable" message — that is the correct guard, not a failure of this task.)

- [ ] **Step 3: Commit**

```bash
git add tests/audio/eval_wake_det.py
git commit -m "stage2f p3: on-device DET sweep -> deploy threshold at FAR<=1/hr"
```

---

## Phase C — Eval, threshold, deploy

### Task C1: Record ≥3 h of real negatives (FAR measurability)

**Files:** none (recording; writes `tests/audio/idle_*_lab.wav`).

The committed `*_lab.wav` corpus already gives real positives (`hey_spot_*_lab.wav`) and short negatives, but is minutes long — FAR ≤ 1/hr is unmeasurable on it (spec C3 needs ≥3 h zero-fire negatives). `scripts/record_corpus.py` records 16 kHz mono ch0 (the production path).

- [ ] **Step 1: Capture long idle/background sessions**

Run repeatedly across realistic rooms/times (the recorder's idle clips are 5 s each; for bulk negatives, also just leave a long ambient capture running and split it, or re-run `--only idle` many times). Aim for ≥3 h total of `idle_*_lab.wav` (and optionally more `adv_*_lab.wav` confusables):
```bash
# Option A — short labelled idle clips via the recorder (uses XVF3800 ch0):
spot-env/bin/python scripts/record_corpus.py --room lab --only idle
# Option B (recommended for >=3h) — capture long ambient blocks on the same mic,
# then split into 5s wavs. Repeat the capture across rooms/times to reach >=3h:
arecord -D plughw:reSpeaker -f S16_LE -r 16000 -c 1 -d 1800 tests/audio/idle_long_lab.wav
ffmpeg -i tests/audio/idle_long_lab.wav -f segment -segment_time 5 -c copy tests/audio/idle_seg_%03d_lab.wav
rm tests/audio/idle_long_lab.wav
```
Expected: additional `tests/audio/idle_*_lab.wav` files; verify total negative duration:
```bash
spot-env/bin/python -c "import glob,wave; s=sum(wave.open(f).getnframes() for f in glob.glob('tests/audio/idle_*_lab.wav')+glob.glob('tests/audio/adv_*_lab.wav')); print(f'{s/16000/3600:.2f} h negatives')"
```

- [ ] **Step 2: (optional) record more real positives for a tighter TPR** via `--only wake` if the existing positive count is too small to trust TPR.

- [ ] **Step 3: Commit the expanded corpus** (corpus is tracked per the P2 decision)

```bash
git add tests/audio/idle_*_lab.wav tests/audio/adv_*_lab.wav
git commit -m "stage2f p3: expand real negative corpus to >=3h for FAR measurement"
```

### Task C2: Pick the winning candidate + deploy threshold

**Files:** none (evaluation; updates `tests/audio/baseline.json` if a gate is cleared).

- [ ] **Step 1: Run the DET sweep per candidate**

```bash
for m in hey_spot hi_spot big_yellow; do
  echo "== $m =="
  SPOT_WAKE_BACKEND=livekit LIVEKIT_WAKE_MODEL=/mnt/ssd/wake-models/$m.onnx \
    spot-env/bin/python -m tests.audio.eval_wake_det | tee tests/audio/det_$m.json
done
```
Expected: a `tests/audio/det_<m>.json` per candidate holding `best_at_far_target`. Choose the candidate with the highest `tpr` at FAR ≤ 1/hr; note its `threshold` (persisted in the file, so it survives terminal scroll / sequential subagents).

- [ ] **Step 2: Decide the gate** (spec Component 1 fallback rule)

If the winner's real held-out `tpr` ≥ the agreed target (spec: 0.9, or whatever the DET curve yields at FAR ≤ 1/hr) → proceed to C3. **Else stop: keep `SPOT_WAKE_BACKEND=sherpa_onnx` + push-to-talk** and record the decision. Do not ship a flaky low-threshold wake.

- [ ] **Step 3: If proceeding, pin the threshold + winner in `baseline.json`**

Add a `wake_livekit` block to `baseline.json` (do not overwrite the existing P2 keys) recording the winner model, the derived `threshold`, the real `tpr`, and `negative_hours` — the regression anchor. Also write the runtime threshold file the loader reads (precedence: env > this file > 0.5):

```bash
printf '{"threshold": %s, "model": "%s"}\n' "<derived>" "<winner>" > /mnt/ssd/wake-models/active_threshold.json
git add tests/audio/baseline.json tests/audio/det_*.json
git commit -m "stage2f p3: record livekit wake winner + DET-derived threshold"
```

### Task C3: Deploy

**Files:** none (runtime config; the default-backend decision is documented, not code-forced).

- [ ] **Step 1: Point the active model + threshold at the winner**

```bash
# active.onnx is the deploy symlink the loader defaults to (no env churn per
# candidate; a symlink also avoids the cp-onto-itself hazard if winner==hey_spot):
ln -sf /mnt/ssd/wake-models/<winner>.onnx /mnt/ssd/wake-models/active.onnx
# threshold comes from active_threshold.json (written in C2). Set the backend in
# the launcher (scripts/run_voice_control.py or the service env):
#   SPOT_WAKE_BACKEND=livekit      (LIVEKIT_WAKE_THRESHOLD env optional — overrides the file)
```

- [ ] **Step 2: End-to-end on-device confirmation**

Run: `SPOT_WAKE_BACKEND=livekit LIVEKIT_WAKE_THRESHOLD=<derived> spot-env/bin/python -m tests.audio.eval_wake`
Expected: real TPR matches the C2 winner; the chain runs with no `livekit` import. Then a live smoke test: say "hey spot" → wake fires; rollback verified by `SPOT_WAKE_BACKEND=sherpa_onnx` still working.

- [ ] **Step 3: Commit any launcher/env change**

```bash
git add scripts/run_voice_control.py   # only if the env wiring changed there
git commit -m "stage2f p3: default to livekit wake backend (rollback: SPOT_WAKE_BACKEND=sherpa_onnx)"
```

---

## Out of scope (explicitly deferred)
- Forking livekit-wakeword to ingest real positives (V1 gap) — only if synthetic-trained recall is inadequate after C2.
- The P4 gate-liveness-bound threshold loader (lowering the deploy threshold toward over-fire once speaker-ID is live) — that is P4, not P3.
- ReDimNet2-B3 / WeSpeaker speaker-ID (P4), Personal-VAD (P6/P8) — separate plans.

## Self-review notes
- **Spec coverage:** Component 1 (3 candidates, large/50k/80k, DET-derived threshold, Piper synthetic positives, sherpa+PTT fallback, `/mnt/ssd/wake-models/`, `SPOT_WAKE_BACKEND` rollback) — covered by A2–A4, B1–B2, C2–C3. V1 gap (real positives = eval-only) — Phase C. V10 (no ElevenLabs) — pipeline is Piper-only; the dead `scripts/synth_wake_elevenlabs.py` retirement is tracked separately (latent-issue list), not in this plan.
- **Placeholder scan:** thresholds are derived (C2), not hardcoded; the one genuinely runtime-discovered item (augmentation schema field names) is a deliberate verification step (A2 Step 1), not a placeholder.
- **Type consistency:** `embedding_windows`, `last_score`, `_score`, `WINDOW_SAMPLES`/`STEP_SAMPLES`/`EMBEDDING_WINDOW`/`EMBEDDING_STRIDE`/`MIN_EMBEDDINGS`, `is_available/process_frame/reset` are consistent across B1 and the B2/C consumers.
