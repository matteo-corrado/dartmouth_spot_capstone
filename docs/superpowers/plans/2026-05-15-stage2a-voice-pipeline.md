# Stage 2A — Voice Pipeline Implementation Plan (rev 2026-05-21)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock the voice loop end-to-end on `tour_guide_upgrade_matteo` branch.

1. Brain runtime: Ollama → llama.cpp (Gemma 4 E4B via NVIDIA Jetson container) with **native Gemma 4 tool-calling** (`--jinja`), **single unified GBNF grammar** (actions array + response string in one shape — no separate intent router), and **streaming SSE → TTS chunker** for low-latency freeform.
2. ASR: broken Riva bridge → **Nemotron Speech Streaming 0.6B** (purpose-built for voice agents, 24ms median TTFT, cache-aware streaming, sherpa-onnx int8 ONNX).
3. VAD: webrtcvad → Silero VAD via sherpa-onnx.
4. Wake word: sherpa-onnx KWS → **LiveKit-Wakeword** custom-trained "hey spot" (accent-agnostic English using Piper VITS full 904-speaker SLERP pool, heavy RIR augmentation for far-field, conv-attention classifier, adversarial negatives for "spot"-confusables). 2-stage architecture: wake model gates Nemotron Streaming (commercial pattern, all local).

Exit tag: `stage2a-voice-complete`.

**Why rev 2:** original plan written before research into 2026 SOTA. Key changes:
- Ollama bug #15260 was fixed Apr 21 2026 (PR #15678); llama.cpp swap is now justified by streaming SSE + native Gemma 4 parser, not bug-bypass.
- Confirmed `unsloth/gemma-4-e4b-it-GGUF` (lowercase) does NOT exist. Correct path: `ggml-org/gemma-4-E4B-it-GGUF`.
- Parakeet TDT 0.6B is batch-oriented (waits for endpoint). Nemotron Speech Streaming is purpose-built for real-time voice agents — better fit.
- Custom intent router (regex) is brittle on compound utterances. Replaced with single GBNF that lets Gemma 4 decide per turn.
- Off-the-shelf LiveKit "hey spot" model does not exist on the hub. Custom training is required regardless; doing it accent-agnostic + far-field-augmented from the start solves the British-accent + distance recognition problem reported in field testing.
- Original plan was missing streaming output for freeform path — biggest UX miss.

**Architecture:**

```
Mic (16kHz mono)
  ↓
LiveKit-Wakeword "hey spot" (Stage 1 — always-on, ~50KB ONNX, conv-attention, CPU)
  ↓ wake fires
Silero VAD (already buffering audio; ~50ms tail captured pre-wake to avoid clipping first command syllable)
  ↓ speech segment
Nemotron Speech Streaming ASR (Stage 2 — sherpa-onnx CUDA, 560ms chunk int8)
  ↓ transcript stream
LLMBrain (llama.cpp /v1/chat/completions stream:true)
  ↓ token stream
TTS chunker (split on sentence-end punctuation; first chunk goes to TTS as soon as available)
  ↓
sherpa-onnx Kokoro TTS → ALSA out
```

**Tech Stack:** llama.cpp (NVIDIA Jetson Orin container `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`), `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` + matching mmproj, sherpa-onnx >= 1.12.40 (for Nemotron streaming + Silero VAD), `livekit-wakeword` (training pipeline + ONNX runtime inference), ONNX Runtime 1.23.0 (CUDA EP on Orin), Python 3.10, existing Boston Dynamics SDK 5.0.1.1.

**Spec:** `docs/superpowers/specs/2026-05-15-stage2-design.md` §2A.

**Branch:** `tour_guide_upgrade_matteo` (continues from `stage1.5-complete` at 966a96d).

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `scripts/preflight_2a.py` | T0 VRAM + baseline-latency audit script |
| Create | `scripts/setup_llamacpp_jetson.sh` | T1 pull NVIDIA Jetson llama.cpp container, smoke-test |
| Create | `scripts/setup_llamacpp_models.py` | T2 download Gemma 4 E4B GGUFs to `/mnt/ssd/llamacpp-models/` |
| Create | `src/voice_control/grammar/spot_action.gbnf` | T3 single unified GBNF: `{"actions":[...],"response":"..."}` |
| Create | `systemd/llama-server.service` | T4 long-lived llama-server daemon on port 11435 with `--jinja` + streaming |
| Create | `src/voice_control/brain/__init__.py` | T5 backend registry |
| Create | `src/voice_control/brain/llamacpp_backend.py` | T5 streaming HTTP client for llama-server |
| Create | `src/voice_control/brain/ollama_backend.py` | T5 extracted Ollama path (rollback) |
| Modify | `src/voice_control/llm_brain.py` | T5 dispatch via Backend interface, drop regex intent router |
| Create | `tests/audio/*.wav` | T6 audio corpus — diverse speakers/accents where reachable |
| Create | `tests/utterances/action_corpus.yaml` | T6 expected (utterance, action, params) tuples |
| Create | `scripts/setup_nemotron.sh` | T7 download Nemotron Streaming sherpa-onnx archive |
| Create | `src/voice_control/asr/__init__.py` | T7 backend registry |
| Create | `src/voice_control/asr/nemotron.py` | T7 Nemotron Streaming backend (default) |
| Create | `src/voice_control/asr/parakeet.py` | T7 Parakeet TDT v3 rollback backend |
| Modify | `src/voice_control/server.py` | T7 dispatch via Backend interface based on `SPOT_ASR_BACKEND` env |
| Modify | `src/voice_control/client_mic.py` | T8 swap webrtcvad → Silero; T10 wake-gated capture |
| Create | `scripts/setup_livekit_wakeword.sh` | T9 install livekit-wakeword + datasets |
| Create | `configs/wakeword_hey_spot.yaml` | T9 training config: accent-agnostic English, RIR-augmented |
| Create | `scripts/train_wake_hey_spot.sh` | T9 training pipeline wrapper |
| Create | `src/voice_control/wake_word_livekit.py` | T9 inference wrapper (conv-attention ONNX) |
| Modify | `src/voice_control/wake_word.py` | T9 dispatch sherpa_onnx vs livekit via `SPOT_WAKE_BACKEND` |
| Modify | `src/voice_control/spot_tts.py` | T11 add `enqueue_streaming(token_iter)` for SSE-driven chunking |
| Create | `scripts/mic_verify.py` | T12 end-to-end mic verify (wake + ASR + action) |
| Modify | `requirements.txt` | T7 add sherpa-onnx>=1.12.40 deps; T9 add livekit-wakeword |
| Modify | `docs/project/stage1-rollback.md` | T12 mark qwen pair removable after mic-verify pass |
| Create | `docs/project/stage2-rollback.md` | T13 Stage 2 rollback runbook (5 layers + env-var flips) |

**Deferred to post-2B (NOT in this plan):**
- Delete RivaBackend — keep on disk until 2B end-to-end on real Spot passes. Code retention cost ≈ 0.
- E2B speculative drafter / gemma4_assistant MTP head — gate on `>50%` accept-rate benchmark; defer if unproven.

---

## Pre-conditions

- `git status` shows clean working tree.
- `git tag` lists `stage1.5-complete` at `966a96d`.
- `/mnt/ssd` mounted with ≥ 30 GB free (`df -h /mnt/ssd`).
- Docker daemon running (`systemctl is-active docker`). Pre-built Jetson llama.cpp container needs Docker (alternative: native build, see T1 step 5).
- `nvidia-smi` shows AGX Orin available and CUDA driver loaded.
- Mic + speaker plugged in (T6, T12 need them).
- ElevenLabs API key NOT required for 2A. Stage 2E.1 only.

---

## Task 0 — Pre-flight + VRAM budget audit + baseline latency capture

**Files:** Create: `scripts/preflight_2a.py`

This task captures the "before" state so we can quantify improvement after the swap. Without this, the mic-verify gates in T12 have no comparison point.

- [ ] **Step 1: Write the audit script**

```python
#!/usr/bin/env python3
"""Stage 2A pre-flight audit.

Captures baseline VRAM headroom + current Ollama-path latency so the
post-swap mic-verify can quantify improvement.

Outputs:
  /tmp/stage2a_preflight_<ts>.json   — structured snapshot
  /tmp/stage2a_preflight_<ts>.log    — human-readable log
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("SPOT_OLLAMA_MODEL", "gemma4:e4b")
N_WARMUP = 1
N_TRIALS = 5

PROMPTS = [
    "stand up",
    "walk forward two meters",
    "what do you see",
    "tell me about Dartmouth Hall",
    "battery status",
]


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def tegrastats_snapshot() -> dict:
    try:
        line = subprocess.check_output(
            ["tegrastats", "--interval", "200"], text=True, timeout=1
        ).strip().split("\n")[0]
    except subprocess.TimeoutExpired as e:
        line = (e.output or "").decode("utf-8", "ignore").strip().split("\n")[0]
    return {"raw": line}


def df_ssd() -> dict:
    out = run(["df", "-B1", "/mnt/ssd"])
    parts = out.splitlines()[1].split()
    return {"used_bytes": int(parts[2]), "avail_bytes": int(parts[3])}


def ollama_chat(prompt: str, fmt_json: bool) -> tuple[str, float]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if fmt_json:
        payload["format"] = "json"
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json()["message"]["content"], elapsed


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(f"/tmp/stage2a_preflight_{ts}.json")
    out_log = Path(f"/tmp/stage2a_preflight_{ts}.log")
    snap: dict = {"timestamp": ts}

    print("=== disk ===")
    snap["disk"] = df_ssd()
    print(f"  /mnt/ssd avail: {snap['disk']['avail_bytes'] / 1e9:.2f} GB")

    print("=== tegrastats ===")
    snap["tegrastats_idle"] = tegrastats_snapshot()
    print(f"  {snap['tegrastats_idle']['raw'][:120]}")

    print("=== ollama warmup ===")
    for _ in range(N_WARMUP):
        ollama_chat("hi", fmt_json=False)

    print("=== ollama freeform latency ===")
    freeform = []
    for p in PROMPTS:
        durations = []
        for _ in range(N_TRIALS):
            _, dt = ollama_chat(p, fmt_json=False)
            durations.append(dt)
        durations.sort()
        median = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
        freeform.append({"prompt": p, "median_s": median, "p95_s": p95})
        print(f"  freeform '{p[:30]:30s}' median={median:.2f}s p95={p95:.2f}s")
    snap["ollama_freeform"] = freeform

    print("=== ollama format:json latency ===")
    json_latencies = []
    for p in PROMPTS[:3]:  # action-shaped prompts only
        durations = []
        for _ in range(N_TRIALS):
            _, dt = ollama_chat(p, fmt_json=True)
            durations.append(dt)
        durations.sort()
        median = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
        json_latencies.append({"prompt": p, "median_s": median, "p95_s": p95})
        print(f"  json     '{p[:30]:30s}' median={median:.2f}s p95={p95:.2f}s")
    snap["ollama_format_json"] = json_latencies

    print("=== tegrastats under load ===")
    snap["tegrastats_loaded"] = tegrastats_snapshot()
    print(f"  {snap['tegrastats_loaded']['raw'][:120]}")

    out_json.write_text(json.dumps(snap, indent=2))
    out_log.write_text(
        f"Stage 2A pre-flight {ts}\n"
        f"disk avail: {snap['disk']['avail_bytes'] / 1e9:.2f} GB\n"
        f"freeform medians: {[(r['prompt'][:20], round(r['median_s'], 2)) for r in freeform]}\n"
        f"format:json medians: {[(r['prompt'][:20], round(r['median_s'], 2)) for r in json_latencies]}\n"
    )
    print(f"\nsaved: {out_json}")
    print(f"saved: {out_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the audit (Ollama must already be live with gemma4:e4b)**

```bash
curl -fs http://localhost:11434/api/tags | grep -q gemma4 || { echo "gemma4:e4b not in Ollama; pull first"; exit 1; }
spot-env/bin/python scripts/preflight_2a.py
```

Expected: JSON snapshot saved under `/tmp/stage2a_preflight_*.json`. Median freeform latency typically 3–6 s on Orin AGX with Ollama; format:json typically 5–8 s due to historical bug #15260 patterns (even after Ollama fix the format-path is slower).

- [ ] **Step 3: Record the snapshot path**

```bash
ls -t /tmp/stage2a_preflight_*.json | head -1
```

Save this filename in your notes — the mic-verify run in T12 will reference it for before/after comparison.

- [ ] **Step 4: Commit the audit script**

```bash
git add scripts/preflight_2a.py
git commit -m "stage 2a t0: pre-flight audit script for baseline latency capture"
git push origin tour_guide_upgrade_matteo
```

---

## Task 1 — llama.cpp via NVIDIA Jetson container

**Files:** Create: `scripts/setup_llamacpp_jetson.sh`

We use NVIDIA's pre-built Jetson container instead of building from source. It is maintained by NVIDIA-AI-IOT, CUDA-linked for AGX Orin (`sm_87`), and avoids a 10–15 min compile every time we want to pull a llama.cpp update.

- [ ] **Step 1: Write the setup script**

```bash
#!/bin/bash
# scripts/setup_llamacpp_jetson.sh — pull and smoke-test NVIDIA Jetson llama.cpp container
set -euo pipefail

IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin"

echo "=== pulling $IMAGE (one-time, ~2 GB) ==="
sudo docker pull "$IMAGE"

echo "=== container smoke test ==="
sudo docker run --rm --runtime=nvidia "$IMAGE" llama-server --help | head -10

echo "=== check NVIDIA runtime ==="
sudo docker run --rm --runtime=nvidia --gpus all "$IMAGE" nvidia-smi || {
  echo "WARNING: nvidia-smi did not run inside container; verify NVIDIA Container Toolkit is installed"
}

echo "=== done ==="
echo "Image cached. T4 will run the daemon."
```

- [ ] **Step 2: Make executable and run**

```bash
chmod +x scripts/setup_llamacpp_jetson.sh
sudo ./scripts/setup_llamacpp_jetson.sh
```

Expected: `llama-server --help` prints usage text; `nvidia-smi` inside the container shows the Orin GPU.

- [ ] **Step 3: Verify image size**

```bash
sudo docker images | grep llama_cpp
```

Expected: image present, ~2 GB on disk. If `/var/lib/docker` is on eMMC, consider moving Docker root to `/mnt/ssd/docker` (see `/etc/docker/daemon.json`) — out of scope for this task, log only if needed.

- [ ] **Step 4: (Optional fallback) Native build**

If the pre-built container is unavailable or you prefer a host install:

```bash
# Only run if T1 step 2 fails
LLAMA_ROOT=/mnt/ssd/llamacpp
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_ROOT"
cd "$LLAMA_ROOT"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

Build time ~10–15 min on AGX Orin. T4 systemd unit has both container and native ExecStart variants; pick one before committing.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_llamacpp_jetson.sh
git commit -m "stage 2a t1: NVIDIA Jetson llama.cpp container setup"
git push origin tour_guide_upgrade_matteo
```

---

## Task 2 — Download Gemma 4 E4B GGUF + mmproj

**Files:** Create: `scripts/setup_llamacpp_models.py`

The correct HF repos (verified May 2026):
- Text + multimodal projector: `ggml-org/gemma-4-E4B-it-GGUF`
- File names use **capital E** in `E4B` — `unsloth/gemma-4-e4b-it-GGUF` (lowercase) does NOT exist.

- [ ] **Step 1: Write the model-download script**

```python
#!/usr/bin/env python3
"""Download Gemma 4 E4B GGUF + mmproj to /mnt/ssd/llamacpp-models/.

Repos verified May 2026:
  ggml-org/gemma-4-E4B-it-GGUF       → text decoder Q4_K_M (~5 GB)
                                     → mmproj-gemma4-e4b-f16.gguf (~1 GB, vision + audio projector)
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path("/mnt/ssd/llamacpp-models")

MODELS = [
    {
        # Q6_K: ~7 GB, perplexity within ~1-2% of BF16 vs Q4_K_M's ~5-10%.
        # AGX Orin 64 GB has ample headroom; Q4_K_M is the current Ollama baseline
        # and we want a measurable quality bump over that, not parity.
        "repo_id": "ggml-org/gemma-4-E4B-it-GGUF",
        "filename": "gemma-4-E4B-it-Q6_K.gguf",
        "local_name": "gemma-4-E4B-Q6_K.gguf",
    },
    {
        "repo_id": "ggml-org/gemma-4-E4B-it-GGUF",
        "filename": "mmproj-gemma4-e4b-f16.gguf",
        "local_name": "mmproj-gemma4-e4b-f16.gguf",
    },
]


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        target = MODELS_DIR / m["local_name"]
        if target.exists():
            print(f"[skip] {m['local_name']} ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"[pull] {m['repo_id']}/{m['filename']} -> {target}")
        downloaded = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=str(MODELS_DIR),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.name != m["local_name"]:
            downloaded_path.rename(target)
        print(f"  done: {target.stat().st_size / 1e9:.2f} GB")

    print("\n=== llamacpp models ===")
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        print(f"  {p.name:50s}  {p.stat().st_size / 1e9:6.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify huggingface-hub installed and ≥ 1.0**

```bash
spot-env/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Expected: `>= 1.0`. If not, `spot-env/bin/pip install --no-deps 'huggingface-hub>=1.0,<2.0'`.

- [ ] **Step 3: Run the download (~6 GB total, ~5–10 min)**

```bash
spot-env/bin/python scripts/setup_llamacpp_models.py
```

Expected:
```
=== llamacpp models ===
  gemma-4-E4B-Q6_K.gguf                              7.00 GB
  mmproj-gemma4-e4b-f16.gguf                         0.94 GB
```

If either file is missing, the repo path is wrong — re-verify on HF before continuing.

- [ ] **Step 4: Verify disk reclaim is intact**

```bash
df -h /mnt/ssd /
```

Expected: `/mnt/ssd` use bumped ~6 GB; `/` unchanged (eMMC stays free).

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_llamacpp_models.py
git commit -m "stage 2a t2: download Gemma 4 E4B GGUF + mmproj to SSD"
git push origin tour_guide_upgrade_matteo
```

---

## Task 3 — Single unified GBNF grammar

**Files:** Create: `src/voice_control/grammar/spot_action.gbnf`

We use ONE grammar for every turn. The model emits `{"actions":[...],"response":"..."}` regardless of intent:
- Action utterance: `actions` populated, `response` is a short confirmation.
- Freeform utterance: `actions` is `[]`, `response` carries the full reply text.

This eliminates the regex intent router entirely. Gemma 4 with grammar decides per-call.

- [ ] **Step 1: Write the grammar**

```gbnf
# spot_action.gbnf — single grammar for every Stage 2A brain turn.
# Output: {"actions": [Action, ...], "response": "<string>"}
# Action set mirrors src/voice_control/spot_dispatch.py.

root          ::= "{" ws "\"actions\":" ws actions-list "," ws "\"response\":" ws string ws "}"

actions-list  ::= "[" ws ( action ( "," ws action )* )? ws "]"

action        ::= "{" ws "\"action\":" ws action-name ( "," ws "\"params\":" ws params )? ws "}"

action-name   ::= "\"stop\"" | "\"freeze\"" | "\"estop\""
              | "\"stand\"" | "\"sit\"" | "\"selfright\""
              | "\"walk\"" | "\"strafe\"" | "\"turn\""
              | "\"body_height\"" | "\"set_speed\"" | "\"set_volume\""
              | "\"go_to\"" | "\"tour\"" | "\"patrol\""
              | "\"come_back\"" | "\"save_location\"" | "\"list_locations\""
              | "\"load_map\"" | "\"list_maps\""
              | "\"go_to_object\"" | "\"follow_me\""
              | "\"describe\"" | "\"check_obstacles\""
              | "\"battery_status\"" | "\"status\"" | "\"power_off\""
              | "\"set_persona\""   # Stage 2E.1 forward-compat; dispatcher handler lands in 2E.1 T16

params        ::= "{" ws ( param ( "," ws param )* )? ws "}"
param         ::= string ":" ws value

# JSON value subset
value         ::= string | number | "true" | "false" | "null" | object | array
object        ::= "{" ws ( pair ( "," ws pair )* )? ws "}"
pair          ::= string ":" ws value
array         ::= "[" ws ( value ( "," ws value )* )? ws "]"

string        ::= "\"" ( [^"\\\x00-\x1f] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""

number        ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?

ws            ::= ( " " | "\t" | "\n" | "\r" )*
```

- [ ] **Step 2: Validate the grammar parses**

```bash
sudo docker run --rm -v "$(pwd)/src/voice_control/grammar:/grammar" \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-gbnf-validator /grammar/spot_action.gbnf "$(cat <<'EOF'
{"actions": [{"action": "stand"}], "response": "Standing up."}
EOF
)" || echo "GBNF parser failed — fix grammar"
```

Expected: validator reports the sample document is accepted. (Tool name may be `llama-gbnf-validator` or `gbnf-validator`; check container.)

- [ ] **Step 3: Confirm action enum matches dispatcher**

```bash
grep -oE '^[a-z_]+:' src/voice_control/llm_brain.py | grep -oE '^[a-z_]+' | sort -u > /tmp/dispatcher_actions.txt
# Strip set_persona from grammar side: dispatcher handler is added in Stage 2E.1 T16,
# but the grammar carries it now so llama-server won't reject the token at 2E.1 ship.
grep -oE '"[a-z_]+"' src/voice_control/grammar/spot_action.gbnf | tr -d '"' | grep -v '^set_persona$' | sort -u > /tmp/grammar_actions.txt
diff /tmp/dispatcher_actions.txt /tmp/grammar_actions.txt && echo "OK" || echo "DIFF — reconcile"
```

Expected: empty diff (set_persona is excluded from the grammar side until 2E.1 T16 lands the dispatcher handler). If any other diff appears, patch the grammar first — never silently drop an action from the grammar.

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/grammar/spot_action.gbnf
git commit -m "stage 2a t3: single unified GBNF for actions + response"
git push origin tour_guide_upgrade_matteo
```

---

## Task 4 — llama-server systemd unit (with --jinja, --flash-attn, streaming)

**Files:** Create: `systemd/llama-server.service`

This unit runs the NVIDIA Jetson container as a long-lived daemon on port 11435.

- [ ] **Step 1: Write the unit file**

```ini
# systemd/llama-server.service
# Install:
#   sudo cp systemd/llama-server.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now llama-server.service
[Unit]
Description=llama.cpp brain server for Spot (Gemma 4 E4B)
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
Restart=on-failure
RestartSec=5
ExecStartPre=-/usr/bin/docker stop llama-server
ExecStartPre=-/usr/bin/docker rm llama-server
ExecStart=/usr/bin/docker run --rm --name llama-server \
    --runtime=nvidia --gpus all \
    -v /mnt/ssd/llamacpp-models:/models:ro \
    -v /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/grammar:/grammar:ro \
    -p 127.0.0.1:11435:8080 \
    ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
    llama-server \
      -m /models/gemma-4-E4B-Q6_K.gguf \
      --mmproj /models/mmproj-gemma4-e4b-f16.gguf \
      -c 8192 \
      -ngl 99 \
      --flash-attn on \
      --jinja \
      --host 0.0.0.0 --port 8080 \
      --log-format json \
      --metrics \
      --image-min-tokens 70 --image-max-tokens 140 \
      --batch-size 512 --ubatch-size 512
ExecStop=/usr/bin/docker stop llama-server

[Install]
WantedBy=multi-user.target
```

Key flags:
- `--jinja` enables the Gemma 4 native chat template + tool-calling parser (llama.cpp PR #21418).
- `--flash-attn on` cuts KV-cache memory.
- `-ngl 99` puts all layers on GPU.
- `-c 8192` context window. Tune up later if Stage 2D conversation log grows long.
- mmproj enables vision + audio (we will not use audio in 2A, but loading it lets us spike Gemma 4 native ASR in a follow-up).

- [ ] **Step 2: Install + start the unit**

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service
```

- [ ] **Step 3: Wait for cold start + verify health**

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 4
  if curl -fs http://127.0.0.1:11435/health 2>/dev/null; then
    echo "[ok] llama-server live"
    break
  fi
  echo "[$i] waiting..."
done
```

Expected: `{"status":"ok"}` after 20–40 s cold start.

- [ ] **Step 4: Verify metrics endpoint**

```bash
curl -s http://127.0.0.1:11435/v1/metrics | head -20
```

Expected: Prometheus-format lines such as `llamacpp:tokens_predicted_seconds_total`.

- [ ] **Step 5: Smoke test — grammar-constrained action JSON**

```bash
GBNF=$(python3 -c 'import json,sys; print(json.dumps(open("src/voice_control/grammar/spot_action.gbnf").read()))')
curl -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are Spot. Output JSON only.\"},
      {\"role\": \"user\", \"content\": \"Stand up\"}
    ],
    \"grammar\": $GBNF,
    \"max_tokens\": 200,
    \"temperature\": 0.2,
    \"cache_prompt\": true
  }" | python3 -m json.tool
```

Expected: response with `choices[0].message.content` parsing to a JSON object with `actions: [{action: "stand"}]` and a non-empty `response`. Latency < 3 s after warmup.

- [ ] **Step 6: Smoke test — streaming SSE (THE critical UX feature)**

```bash
curl -N -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Briefly describe what a tour guide robot at Dartmouth would talk about."}],
    "stream": true,
    "max_tokens": 200,
    "temperature": 0.7
  }' | head -40
```

Expected: a stream of `data: {...}` SSE frames. The first frame should arrive in well under 1 s. Each frame contains `choices[0].delta.content` with one or a few tokens. This is what T11 will pipe into the TTS chunker.

- [ ] **Step 7: Verify under load — token throughput**

```bash
curl -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 100, comma-separated."}],"max_tokens":300,"temperature":0.1}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("tokens:", d["usage"]["completion_tokens"], "time:", d.get("timings", {}).get("predicted_ms"))'
```

Expected: throughput ≥ 15 tok/s on AGX Orin AGX with E4B Q4_K_M. If under 10 tok/s, check `nvidia-smi` shows the container has GPU access.

- [ ] **Step 8: Commit**

```bash
git add systemd/llama-server.service
git commit -m "stage 2a t4: llama-server systemd unit with jinja + streaming"
git push origin tour_guide_upgrade_matteo
```

---

## Task 5 — Brain backend interface (no intent router)

**Files:**
- Create: `src/voice_control/brain/__init__.py`
- Create: `src/voice_control/brain/llamacpp_backend.py`
- Create: `src/voice_control/brain/ollama_backend.py`
- Modify: `src/voice_control/llm_brain.py`

Two backends behind `SPOT_BRAIN_BACKEND=llamacpp|ollama`. **No intent router.** Every turn uses the single GBNF. The model decides whether to emit actions or pure response.

- [ ] **Step 1: Backend registry**

```python
# src/voice_control/brain/__init__.py
"""Brain runtime backends. Selected via SPOT_BRAIN_BACKEND env var."""
import os

_DEFAULT = "llamacpp"


def get_backend():
    name = os.environ.get("SPOT_BRAIN_BACKEND", _DEFAULT).lower()
    if name == "llamacpp":
        from .llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend()
    if name == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"Unknown SPOT_BRAIN_BACKEND={name!r}; valid: llamacpp|ollama")
```

- [ ] **Step 2: llama.cpp backend (streaming-first)**

```python
# src/voice_control/brain/llamacpp_backend.py
"""llama.cpp HTTP backend.

Single API: chat(system, messages, on_token=None) -> str
- If on_token is None: blocking, returns full response.
- If on_token is callable: streams tokens via SSE; on_token(text_delta) is called
  for each delta, and the full concatenated text is returned at the end.

Always sends the unified GBNF. Brain post-parses the JSON to extract actions vs response.
"""
from __future__ import annotations
import base64
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

LLAMACPP_URL = os.environ.get("SPOT_LLAMACPP_URL", "http://127.0.0.1:11435")
GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "spot_action.gbnf"

# Three sampling profiles. chat() picks action|freeform via _sampling_hint;
# vlm_describe() pins profile="vlm". The hint only affects sampling, never
# the code path or grammar — if the hint is wrong, GBNF still produces a
# valid {actions, response} JSON; only the temperature/top_p is suboptimal.
# Profile values follow the Gemma 4 prompting guide (top_k=64, top_p=0.95)
# for freeform; tighter for action determinism; middle for VLM captions.
# Stage 2E.1 will read+override these per-persona (e.g. pirate may want
# higher vlm temp than butler).
SAMPLING_PROFILES: Dict[str, Dict] = {
    "action":   {"temperature": 0.3, "top_p": 0.9,  "top_k": 40, "repeat_penalty": 1.0},
    "freeform": {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.05},
    "vlm":      {"temperature": 0.5, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.0},
}

_GRAMMAR_CACHE: Optional[str] = None


def _grammar() -> str:
    global _GRAMMAR_CACHE
    if _GRAMMAR_CACHE is None:
        _GRAMMAR_CACHE = GRAMMAR_PATH.read_text(encoding="utf-8")
    return _GRAMMAR_CACHE


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, url: str = LLAMACPP_URL):
        self.url = url

    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 60.0,
        profile: str = "action",
        sampling_overrides: Optional[Dict] = None,
        max_tokens: int = 512,
    ) -> str:
        # 2E.1 personas pass sampling_overrides to vary temp/top_p per persona
        # (vlm temp is the canonical example). Default (None) preserves 2A behavior.
        sampling = {**SAMPLING_PROFILES[profile], **(sampling_overrides or {})}
        payload: Dict = {
            "messages": [{"role": "system", "content": system}] + messages,
            "grammar": _grammar(),
            "max_tokens": max_tokens,
            "cache_prompt": True,
            **sampling,
        }
        if on_token is None:
            r = requests.post(
                f"{self.url}/v1/chat/completions", json=payload, timeout=timeout
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        # Streaming SSE path
        payload["stream"] = True
        full = []
        with requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=True,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    full.append(delta)
                    on_token(delta)
        return "".join(full)

    def vlm_describe(
        self, system: str, prompt: str, image_path: Path, timeout: float = 60.0
    ) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 400,
            "cache_prompt": True,
            **SAMPLING_PROFILES["vlm"],
        }
        r = requests.post(
            f"{self.url}/v1/chat/completions", json=payload, timeout=timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 3: Ollama backend (rollback path)**

```python
# src/voice_control/brain/ollama_backend.py
"""Ollama HTTP backend — Stage 1 path, kept as rollback.

Ollama bug #15260 (format=json + thinking) was fixed in PR #15678 (Apr 2026).
This backend assumes Ollama >= 0.27 with the fix; older versions need
think=False to be explicitly set elsewhere.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

OLLAMA_URL = os.environ.get("SPOT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("SPOT_OLLAMA_MODEL", "gemma4:e4b")

# Mirror SAMPLING_PROFILES from llamacpp_backend so rollback to Ollama
# preserves the action / freeform / vlm sampling distinction. Stage 2E.1
# will override these per-persona on top of either backend.
SAMPLING_PROFILES: Dict[str, Dict] = {
    "action":   {"temperature": 0.3, "top_p": 0.9,  "top_k": 40, "repeat_penalty": 1.0},
    "freeform": {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.05},
    "vlm":      {"temperature": 0.5, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.0},
}


class OllamaBackend:
    name = "ollama"

    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 60.0,
        profile: str = "action",
        sampling_overrides: Optional[Dict] = None,
        max_tokens: int = 512,
    ) -> str:
        sampling = {**SAMPLING_PROFILES[profile], **(sampling_overrides or {})}
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": on_token is not None,
            "format": "json",
            "options": {**sampling, "num_predict": max_tokens},
        }
        if on_token is None:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]

        full = []
        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout, stream=True
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("message", {}).get("content")
                if delta:
                    full.append(delta)
                    on_token(delta)
                if obj.get("done"):
                    break
        return "".join(full)

    def vlm_describe(
        self, system: str, prompt: str, image_path: Path, timeout: float = 60.0
    ) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            "stream": False,
            "options": {**SAMPLING_PROFILES["vlm"], "num_predict": 400},
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
```

- [ ] **Step 4: Rewire llm_brain.py — rename class, drop the intent router**

In `src/voice_control/llm_brain.py`:

a. **Rename the brain class.** Change `class SpotBrain:` → `class LLMBrain:`. Update the standalone helper `process_with_brain()` body if it constructs `SpotBrain()` directly. Update the import + construction in `src/voice_control/client_mic.py` (currently `from llm_brain import SpotBrain, DEFAULT_MODEL` and `brain = SpotBrain(model=DEFAULT_MODEL)`). Stage 2E.1 plans assume the new name; 2A is the consolidation point.

b. **Replace the existing `process()` body.** Locate the current `process()` method (Ollama HTTP path with regex intent router). Replace with:

```python
import re
from src.voice_control.brain import get_backend

_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = get_backend()
    return _BACKEND


# Cheap regex to pick the sampling profile. Hint only — GBNF still enforces
# the {actions, response} shape regardless of the profile chosen. If the hint
# is wrong, output is still valid; the temperature/top_p just may be
# suboptimal for that turn.
_ACTION_HINT_RE = re.compile(
    r"\b(stand|sit|walk|turn|go|come|follow|stop|freeze|estop|strafe|"
    r"tour|patrol|save|load|list|find|describe|check|battery|status|"
    r"power|height|speed|volume|self ?right)\b",
    re.IGNORECASE,
)


def _sampling_hint(transcript: str) -> str:
    return "action" if _ACTION_HINT_RE.search(transcript) else "freeform"


def process(self, transcript: str, state=None) -> dict:
    """Single-path brain call. Always returns {actions, response}.

    If on_token_callback is set on self, tokens stream as they arrive.
    """
    messages = self._build_messages(transcript, state or {})
    system = messages[0]["content"]
    user_history = messages[1:]

    profile = _sampling_hint(transcript)
    backend = _backend()
    t0 = time.time()
    raw = backend.chat(
        system,
        user_history,
        on_token=getattr(self, "on_token_callback", None),
        profile=profile,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    try:
        parsed = json.loads(raw)
        actions = parsed.get("actions", [])
        response = parsed.get("response", "").strip()
    except json.JSONDecodeError:
        # Grammar guarantees valid JSON — should never hit. Defensive.
        actions = []
        response = raw.strip()

    # Update conversation history. For describe actions the LLM "response"
    # is often hallucinated (model can't see the camera until the dispatcher
    # runs the VLM call), so sanitize the stored assistant turn — otherwise
    # the fake description sticks for MAX_HISTORY turns and biases later
    # replies. The dispatcher handles the spoken side independently.
    self.history.append({"role": "user", "content": transcript})
    if any(a.get("action") == "describe" for a in actions):
        sanitized = json.dumps({"actions": actions, "response": "Taking a look..."})
        self.history.append({"role": "assistant", "content": sanitized})
    else:
        self.history.append({"role": "assistant", "content": raw})
    # history is a deque(maxlen=MAX_HISTORY) — old turns auto-evicted.

    print(
        f"[Brain-timing] backend={backend.name} profile={profile} "
        f"actions={len(actions)} response_chars={len(response)} "
        f"total_ms={elapsed_ms}"
    )
    return {"actions": actions, "response": response}
```

- [ ] **Step 5: Update module docstring**

Replace the top of `llm_brain.py` (lines 1–15) with:

```python
"""LLM Brain for Spot — conversational robot control via single GBNF grammar.

Architecture (Stage 2A):
- Backend: llama.cpp (default) via SPOT_BRAIN_BACKEND=llamacpp, Ollama fallback.
- Single grammar (src/voice_control/grammar/spot_action.gbnf) on every turn.
- Output shape: {"actions": [...], "response": "..."}.
  Action utterances populate actions[]; freeform utterances leave actions=[].
- No regex intent router. Model decides per turn under grammar.
- Streaming: set self.on_token_callback to receive token deltas (T11 wiring).

Architecture inspired by Boston Dynamics' Robots That Can Chat.
"""
```

- [ ] **Step 6: Test both backends manually**

```bash
# llamacpp (default; requires T4 systemd unit live)
spot-env/bin/python -c "
from src.voice_control.brain import get_backend
b = get_backend()
print('backend:', b.name)
print('--- action ---')
print(b.chat('You are Spot. Output JSON only.',
             [{'role':'user','content':'stand up'}]))
print('--- freeform ---')
print(b.chat('You are Spot. Output JSON only.',
             [{'role':'user','content':'tell me about Dartmouth'}]))
"
```

Expected: `backend: llamacpp`; both prints show valid JSON with `actions` either populated (stand) or empty (Dartmouth).

```bash
# Streaming
spot-env/bin/python -c "
from src.voice_control.brain import get_backend
import sys
b = get_backend()
print('--- streaming freeform ---')
def emit(t):
    sys.stdout.write(t); sys.stdout.flush()
b.chat('You are Spot. Output JSON only.',
       [{'role':'user','content':'tell me about Dartmouth'}],
       on_token=emit)
print()
"
```

Expected: tokens print one-by-one as they arrive; first token visible within ~500 ms.

```bash
# Rollback path
SPOT_BRAIN_BACKEND=ollama spot-env/bin/python -c "
from src.voice_control.brain import get_backend
b = get_backend()
print('backend:', b.name)
print(b.chat('You are Spot. Output JSON only.',
             [{'role':'user','content':'sit down'}]))
"
```

Expected: `backend: ollama`; valid JSON with `actions: [{action: "sit"}]`.

- [ ] **Step 7: Verify history append survives the rewrite**

```bash
spot-env/bin/python -c "
from src.voice_control.llm_brain import LLMBrain
b = LLMBrain()
n0 = len(b.history)
b.process('stand up', {'battery_percent': 85})
n1 = len(b.history)
assert n1 == n0 + 2, f'expected +2 turns (user+assistant), got +{n1 - n0}'
print(f'history grew {n0} -> {n1} OK')
"
```

Expected: `history grew 0 -> 2 OK`. If history did not grow, the process() rewrite dropped the append block — fix before commit.

- [ ] **Step 8: Commit**

```bash
git add src/voice_control/brain/ src/voice_control/llm_brain.py src/voice_control/client_mic.py
git commit -m "stage 2a t5: brain backend interface, drop intent router, single GBNF, rename SpotBrain -> LLMBrain"
git push origin tour_guide_upgrade_matteo
```

---

## Task 6 — Audio corpus capture (diverse speakers)

**Files:** Create: `tests/audio/*.wav`, `tests/utterances/action_corpus.yaml`

Manual recording. The corpus serves three purposes:
- Mic-verify smoke test (T12).
- LiveKit-Wakeword false-acceptance evaluation (T9, T10).
- Stage 2B regression matrix.

To avoid overfitting the wake model + mic-verify to a single voice, capture across at least 3 speakers if reachable (lab mates, housemates). If you can only record your own voice, do so — accent agnosticism comes from training data diversity (T9), not from your single recordings.

- [ ] **Step 1: Set up recording**

```bash
mkdir -p tests/audio
arecord -l    # confirm capture device; note card/device numbers
```

- [ ] **Step 2: Record clean utterances (16 kHz mono PCM16, ~3 s each)**

For each speaker (label files `<utt>_<speakerid>.wav`):

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/stand_up_s1.wav
```

Required utterances (each speaker says all of them; rotate phrasing slightly to add variation):

- `hey_spot.wav` — wake phrase, isolated
- `hey_spot_stand_up.wav` — wake + command in one breath
- `hey_spot_walk_forward.wav`
- `stand_up.wav`
- `sit_down.wav`
- `walk_forward_1m.wav`
- `walk_forward_two_meters.wav`
- `turn_left_90.wav`
- `turn_around.wav`
- `go_to_the_kitchen.wav`
- `come_here.wav`
- `follow_me.wav`
- `stop.wav`
- `battery_status.wav`
- `what_do_you_see.wav`
- `describe_surroundings.wav`
- `tell_me_about_dartmouth_hall.wav`
- `find_a_chair.wav`

- [ ] **Step 3: Record motor-idle / room-noise clips**

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 tests/audio/idle_01.wav
# repeat for idle_02.wav .. idle_05.wav
```

- [ ] **Step 4: Record "Spot"-confusable clips for wake adversarial evaluation**

These phrases trip up naive wake models. Used in T9 evaluation and T10 false-acceptance verification.

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/adv_spot_the_difference.wav   # "spot the difference"
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/adv_on_the_spot.wav           # "on the spot"
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/adv_x_marks_the_spot.wav      # "X marks the spot"
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/adv_hey_dog.wav               # "hey dog"
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/adv_hey_stop.wav              # "hey stop"
```

- [ ] **Step 5: Write the action corpus YAML**

```yaml
# tests/utterances/action_corpus.yaml
# 15–20 representative tour-guide utterances + expected dispatcher outcomes.

utterances:
  - wav: stand_up.wav
    action: stand
    params: {}

  - wav: sit_down.wav
    action: sit
    params: {}

  - wav: walk_forward_1m.wav
    action: walk
    params: {direction: forward, distance: 1.0}

  - wav: walk_forward_two_meters.wav
    action: walk
    params: {direction: forward, distance: 2.0}

  - wav: turn_left_90.wav
    action: turn
    params: {dir: left, deg: 90}

  - wav: turn_around.wav
    action: turn
    params: {deg: 180}

  - wav: go_to_the_kitchen.wav
    action: go_to
    params: {location: kitchen}

  - wav: come_here.wav
    action: come_back
    params: {}

  - wav: follow_me.wav
    action: follow_me
    params: {}

  - wav: stop.wav
    action: stop
    params: {}

  - wav: battery_status.wav
    action: battery_status
    params: {}

  - wav: what_do_you_see.wav
    action: null       # freeform — actions[] expected to be empty
    params: null

  - wav: describe_surroundings.wav
    action: null
    params: null

  - wav: tell_me_about_dartmouth_hall.wav
    action: null
    params: null

  - wav: find_a_chair.wav
    action: go_to_object
    params: {description: chair}
```

- [ ] **Step 6: Commit**

```bash
du -sh tests/audio/
# If < 5 MB total, commit directly:
git add tests/audio/ tests/utterances/action_corpus.yaml
git commit -m "stage 2a t6: audio corpus (utterances + idle + adversarial)"
git push origin tour_guide_upgrade_matteo
```

If `tests/audio/` exceeds 5 MB, move to SSD and symlink (see existing pattern for logs):

```bash
mkdir -p /mnt/ssd/spot-logs/test-audio
mv tests/audio/*.wav /mnt/ssd/spot-logs/test-audio/
ln -sfn /mnt/ssd/spot-logs/test-audio tests/audio
echo "tests/audio" >> .gitignore
git add .gitignore tests/utterances/action_corpus.yaml
git commit -m "stage 2a t6: gitignore tests/audio (corpus on SSD)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 7 — Nemotron Speech Streaming ASR backend

**Files:**
- Create: `scripts/setup_nemotron.sh`
- Create: `src/voice_control/asr/__init__.py`
- Create: `src/voice_control/asr/nemotron.py`
- Create: `src/voice_control/asr/parakeet.py`
- Modify: `src/voice_control/server.py`
- Modify: `requirements.txt`

Nemotron Speech Streaming is purpose-built for voice agents: cache-aware FastConformer + RNNT, configurable 80 / 160 / 560 / 1120 ms chunks at inference time, 24 ms median TTFT on H100 (expect ~200–300 ms on AGX Orin via CUDA EP). sherpa-onnx ships a pre-built INT8 ONNX archive (Apr 25 2026).

Parakeet TDT 0.6B v3 stays as rollback in case Nemotron has unforeseen issues on Orin.

- [ ] **Step 1: Pin deps in requirements.txt**

Open `requirements.txt`. After existing `sherpa-onnx>=1.12.0`, bump to `>=1.12.40` (Nemotron streaming added in 1.12.40):

```
sherpa-onnx>=1.12.40
# huggingface-hub bumped for Parakeet rollback path
huggingface-hub>=1.0,<2.0
# Optional rollback ASR backend
onnx-asr==0.11.0
```

Install:

```bash
spot-env/bin/pip install --no-deps -U 'sherpa-onnx>=1.12.40'
spot-env/bin/pip install --no-deps onnx-asr==0.11.0
spot-env/bin/pip install --no-deps 'huggingface-hub>=1.0,<2.0'
spot-env/bin/pip check
```

Expected: `pip check` reports no broken requirements. If broken, run pin-guardian agent before proceeding.

- [ ] **Step 2: Download Nemotron Streaming (sherpa-onnx pre-built INT8)**

```bash
#!/bin/bash
# scripts/setup_nemotron.sh
set -euo pipefail

NEMO_DIR=/mnt/ssd/nemotron-models
NEMO_ARCHIVE=sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25
NEMO_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMO_ARCHIVE}.tar.bz2

mkdir -p "$NEMO_DIR"
cd "$NEMO_DIR"

if [ -d "$NEMO_ARCHIVE" ]; then
  echo "[skip] already extracted at $NEMO_DIR/$NEMO_ARCHIVE"
  exit 0
fi

wget -nc "$NEMO_URL"
tar xvf "${NEMO_ARCHIVE}.tar.bz2"
rm "${NEMO_ARCHIVE}.tar.bz2"

ls -lh "$NEMO_DIR/$NEMO_ARCHIVE"
```

Run:

```bash
chmod +x scripts/setup_nemotron.sh
./scripts/setup_nemotron.sh
```

Expected: ~880 MB extracted under `/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/` containing `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`.

- [ ] **Step 3: ASR backend registry**

```python
# src/voice_control/asr/__init__.py
"""ASR backends. Selected via SPOT_ASR_BACKEND env var (default: nemotron)."""
import os

_DEFAULT = "nemotron"


def get_backend():
    name = os.environ.get("SPOT_ASR_BACKEND", _DEFAULT).lower()
    if name == "nemotron":
        from .nemotron import NemotronBackend
        return NemotronBackend()
    if name == "parakeet":
        from .parakeet import ParakeetBackend
        return ParakeetBackend()
    raise ValueError(f"Unknown SPOT_ASR_BACKEND={name!r}; valid: nemotron|parakeet")
```

- [ ] **Step 4: Nemotron Streaming backend**

```python
# src/voice_control/asr/nemotron.py
"""Nemotron Speech Streaming 0.6B backend via sherpa-onnx (CUDA EP).

Exposes both:
- transcribe(pcm_bytes) -> str          (offline, for VAD-endpointed turns)
- stream() -> NemotronStream context     (online, words emerge during speech)

For 2A T10 (wake-gated pipeline), the dispatch uses transcribe() once the VAD
segment ends. Streaming is reserved for a Stage 2B/C upgrade where partial
transcripts feed the brain mid-utterance.
"""
from __future__ import annotations
import os
from collections import Counter
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.15

NEMO_DIR = Path(
    os.environ.get(
        "NEMO_DIR",
        "/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
    )
)

HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > threshold


class NemotronBackend:
    name = "nemotron"

    def __init__(self) -> None:
        encoder = str(NEMO_DIR / "encoder.int8.onnx")
        decoder = str(NEMO_DIR / "decoder.int8.onnx")
        joiner = str(NEMO_DIR / "joiner.int8.onnx")
        tokens = str(NEMO_DIR / "tokens.txt")
        for p in (encoder, decoder, joiner, tokens):
            if not Path(p).exists():
                raise FileNotFoundError(f"Nemotron asset missing: {p}")
        print(f"[Nemotron] loading from {NEMO_DIR}")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            provider="cuda",      # fall back to "cpu" if CUDA EP not available
            num_threads=2,
            decoding_method="greedy_search",
        )
        # Warmup
        print("[Nemotron] warming up (1 s silence)")
        warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, warmup)
        while self.recognizer.is_ready(s):
            self.recognizer.decode_stream(s)
        print("[Nemotron] ready")

    def transcribe(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) / 2 / SAMPLE_RATE < MIN_AUDIO_DURATION:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio)
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_stream(s)
        text = self.recognizer.get_result(s).strip()
        if not text:
            return ""
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
```

- [ ] **Step 5: Parakeet rollback backend**

```python
# src/voice_control/asr/parakeet.py
"""Parakeet TDT 0.6B v3 backend via onnx-asr (CUDA EP). Rollback for Nemotron."""
import os
from collections import Counter

import numpy as np
import onnx_asr

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.15
PARAKEET_MODEL_ID = os.environ.get("PARAKEET_MODEL_ID", "nemo-parakeet-tdt-0.6b-v3")
PARAKEET_HF_HOME = os.environ.get("PARAKEET_HF_HOME", "/mnt/ssd/parakeet-models")

HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > threshold


class ParakeetBackend:
    name = "parakeet"

    def __init__(self) -> None:
        os.environ["HF_HOME"] = PARAKEET_HF_HOME
        print(f"[Parakeet] loading {PARAKEET_MODEL_ID} from {PARAKEET_HF_HOME}")
        self.model = onnx_asr.load_model(
            PARAKEET_MODEL_ID,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("[Parakeet] warming up")
        self.model.recognize(np.zeros(SAMPLE_RATE, dtype=np.int16), sample_rate=SAMPLE_RATE)
        print("[Parakeet] ready")

    def transcribe(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) / 2 / SAMPLE_RATE < MIN_AUDIO_DURATION:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        text = self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()
        if not text:
            return ""
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
```

- [ ] **Step 6: Refactor server.py to dispatch via Backend interface**

```python
"""ASR gRPC bridge — dispatches to a configurable Backend.

Backends:
    SPOT_ASR_BACKEND=nemotron  (default) — NVIDIA Nemotron Speech Streaming 0.6B
    SPOT_ASR_BACKEND=parakeet           — NVIDIA Parakeet TDT 0.6B v3 (rollback)

The custom asr.proto (port 50055) is unchanged; client_mic.py uses it as-is.
"""
import os
from concurrent import futures
import grpc

import asr_pb2 as pb
import asr_pb2_grpc as pbg

from src.voice_control.asr import get_backend

SERVER_PORT = 50055
SAMPLE_RATE = 16000


class ASRServicer(pbg.ASRServicer):
    def __init__(self):
        self.backend = get_backend()
        print(f"[Server] backend={self.backend.name}")

    def StreamingRecognize(self, request_iterator, context):
        pcm_chunks = []
        for req in request_iterator:
            if req.HasField("config"):
                continue
            pcm_chunks.append(req.audio.pcm16)

        pcm_bytes = b"".join(pcm_chunks)
        transcript = self.backend.transcribe(pcm_bytes)
        yield pb.StreamingResponse(is_final=True, transcript=transcript, stability=1.0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_ASRServicer_to_server(ASRServicer(), server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    server.start()
    print(f"[Server] listening on :{SERVER_PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
```

- [ ] **Step 7: End-to-end smoke test**

```bash
SPOT_ASR_BACKEND=nemotron spot-env/bin/python -m src.voice_control.server &
SERVER_PID=$!
sleep 12   # Nemotron warmup is heavier than Parakeet
nc -z localhost 50055 || { echo "ASR server did not start"; kill $SERVER_PID; exit 1; }

spot-env/bin/python - <<'EOF'
import grpc, wave
from src.voice_control import asr_pb2 as pb, asr_pb2_grpc as pbg
with wave.open("tests/audio/stand_up.wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
chan = grpc.insecure_channel("localhost:50055")
stub = pbg.ASRStub(chan)
def gen():
    yield pb.StreamingRequest(config=pb.StreamingConfig(language_code="en-US",
        sample_rate_hz=16000, enable_punctuation=True))
    yield pb.StreamingRequest(audio=pb.AudioChunk(pcm16=pcm))
for resp in stub.StreamingRecognize(gen()):
    print("transcript:", resp.transcript)
EOF

kill $SERVER_PID
```

Expected: transcript ≈ `"Stand up."` (case + punctuation may vary).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt scripts/setup_nemotron.sh src/voice_control/asr/ src/voice_control/server.py
git commit -m "stage 2a t7: Nemotron Streaming ASR backend (Parakeet as rollback)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 8 — Silero VAD swap

**Files:** Modify: `src/voice_control/client_mic.py`

`webrtcvad` (2012-era) misclassifies non-speech as speech and triggers idle hallucinations. Silero VAD bundled in sherpa-onnx solves both problems and is already a transitive dependency.

- [ ] **Step 1: Locate webrtcvad usages**

```bash
grep -nE 'webrtcvad|Vad\(|is_speech' src/voice_control/client_mic.py
```

Note line numbers — typically the VAD init + per-frame `vad.is_speech(frame, rate)` calls inside the capture loop.

- [ ] **Step 2: Download Silero VAD ONNX**

```bash
mkdir -p /mnt/ssd/vad-models
cd /mnt/ssd/vad-models
wget -nc https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
ls -lh silero_vad.onnx
```

Expected: ~2 MB file.

- [ ] **Step 3: Replace VAD init in client_mic.py**

Remove `import webrtcvad` and the `vad = webrtcvad.Vad(...)` line. Replace with:

```python
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig

SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512   # Silero default: 512 samples @ 16 kHz = 32 ms


def _make_vad() -> VoiceActivityDetector:
    cfg = VadModelConfig(
        silero_vad=SileroVadModelConfig(
            model=SILERO_VAD_PATH,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.2,
            window_size=VAD_WINDOW_SAMPLES,
        ),
        sample_rate=SAMPLE_RATE,
        debug=False,
    )
    return VoiceActivityDetector(cfg, buffer_size_in_seconds=10)
```

- [ ] **Step 4: Replace per-frame VAD calls**

```python
import numpy as np

def _vad_accept(vad, pcm16_bytes: bytes) -> bool:
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    vad.accept_waveform(samples)
    return vad.is_speech_detected()
```

Replace every `vad.is_speech(frame, SAMPLE_RATE)` with `_vad_accept(vad, frame)`. Frame size must be `VAD_WINDOW_SAMPLES * 2 = 1024 bytes` (int16 mono @ 32 ms). If existing code chunks at 10/20/30 ms, regroup into 32 ms blocks.

- [ ] **Step 5: Drop webrtcvad from requirements.txt**

Open `requirements.txt`. Delete `webrtcvad>=2.0.10`.

- [ ] **Step 6: Smoke test**

```bash
spot-env/bin/python - <<'EOF'
import wave, numpy as np
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig

cfg = VadModelConfig(silero_vad=SileroVadModelConfig(
    model="/mnt/ssd/vad-models/silero_vad.onnx",
    threshold=0.5, min_silence_duration=0.5,
    min_speech_duration=0.2, window_size=512),
    sample_rate=16000, debug=False)
vad = VoiceActivityDetector(cfg, buffer_size_in_seconds=10)

for wav, expect in [("tests/audio/stand_up.wav", "speech"),
                    ("tests/audio/idle_01.wav", "silence")]:
    with wave.open(wav, "rb") as w:
        pcm = w.readframes(w.getnframes())
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    vad.reset()
    vad.accept_waveform(samples)
    segs = 0
    while not vad.empty():
        segs += 1; vad.pop()
    print(f"{wav}: {segs} speech segments (expected {'>=1' if expect == 'speech' else '0'})")
EOF
```

Expected: `stand_up.wav` ≥ 1 segment; `idle_01.wav` 0 segments.

- [ ] **Step 7: Commit**

```bash
git add src/voice_control/client_mic.py requirements.txt
git commit -m "stage 2a t8: Silero VAD via sherpa-onnx, drop webrtcvad"
git push origin tour_guide_upgrade_matteo
```

---

## Task 9 — LiveKit-Wakeword: custom-train "hey spot" (accent-agnostic, far-field)

**Files:**
- Create: `scripts/setup_livekit_wakeword.sh`
- Create: `configs/wakeword_hey_spot.yaml`
- Create: `scripts/train_wake_hey_spot.sh`
- Create: `src/voice_control/wake_word_livekit.py`
- Modify: `src/voice_control/wake_word.py`

This is the most important wake-quality task. The training config produces a wake model that is:
- **Accent-agnostic English**: uses Piper VITS's full 904-speaker SLERP pool — does not weight any single accent. The training data exposes the model to American, British, Australian, Indian, and African Englishes via the diverse synthetic voices.
- **Far-field tolerant**: heavy augmentation with MIT Room Impulse Responses (RIRs) simulates 1–5 m distances and reverb. ACAV100M background noise mixed at varied SNR (5–25 dB) so the model learns to discriminate speech from typical room ambience.
- **Hardened against "spot"-confusables**: adversarial negatives include "spot the difference", "on the spot", "X marks the spot", "stop", "hot spot", "hey dog" — phonetically similar but incorrect phrases that a naive wake model would accept.
- **Conv-attention classifier**: temporal convolutions + multi-head self-attention head (livekit-wakeword default), which delivers ~100× lower false-positives/hour and ~17% higher recall than openWakeWord's flat DNN, per livekit-wakeword's published "hey livekit" benchmark.

- [ ] **Step 1: Install livekit-wakeword and download datasets**

```bash
#!/bin/bash
# scripts/setup_livekit_wakeword.sh
set -euo pipefail

LKWW_ROOT=/mnt/ssd/livekit-wakeword
DATA_DIR=/mnt/ssd/livekit-wakeword-data

if [ ! -d "$LKWW_ROOT" ]; then
  git clone https://github.com/livekit/livekit-wakeword "$LKWW_ROOT"
fi

cd "$LKWW_ROOT"

# livekit-wakeword uses uv (per their README) — install if missing
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Install in our spot-env (separate from system uv to avoid drift)
spot-env/bin/pip install --no-deps -e "$LKWW_ROOT"

# Pre-download shared training data (~20 GB: ACAV100M features + RIRs + Piper)
mkdir -p "$DATA_DIR"
cd "$LKWW_ROOT"
livekit-wakeword setup --config "$REPO/configs/wakeword_hey_spot.yaml"

ls -lh "$DATA_DIR"
```

Run:

```bash
chmod +x scripts/setup_livekit_wakeword.sh
REPO=$(pwd) ./scripts/setup_livekit_wakeword.sh
```

Expected: ACAV100M features (~17 GB), MIT RIR dataset (~10 MB), Piper VITS en-us-libritts-high.pt (~166 MB), background audio (~500 MB–1 GB) downloaded to `/mnt/ssd/livekit-wakeword-data/`.

- [ ] **Step 2: Write the training config (accent-agnostic, far-field)**

```yaml
# configs/wakeword_hey_spot.yaml
# Production-scale "hey spot" wake word.
# Accent-agnostic English: uses Piper VITS's full 904-speaker SLERP pool.
# Far-field tolerant: heavy RIR + background noise augmentation.
# Hardened against confusables: explicit custom_negative_phrases.

data_dir: /mnt/ssd/livekit-wakeword-data

model_name: hey_spot
target_phrases:
  - "hey spot"
  - "hey, spot"
  - "hey spot."

# Adversarial negatives: phonetically or lexically similar phrases that the
# model MUST learn to reject. Without these, far-field background speech
# triggers the wake.
custom_negative_phrases:
  - "spot the difference"
  - "on the spot"
  - "X marks the spot"
  - "hot spot"
  - "stop"
  - "hey stop"
  - "hey dog"
  - "hey scott"
  - "the spot"
  - "spotted"
  - "spotlight"

# TTS backend: Piper VITS uses SLERP across 904 LibriTTS speakers — this is
# what gives accent-agnosticism. NO single-accent weighting.
tts_backend: piper_vits

piper_tts:
  # Use the bundled multi-speaker en-us-libritts-high checkpoint.
  checkpoint_relpath: piper/en-us-libritts-high.pt

# Training sample counts: production scale.
n_samples: 25000           # positives per class
n_samples_val: 5000
n_background_samples: 1500  # ACAV100M negatives per train batch (lots of speech variety)
n_background_samples_val: 500

# Augmentation: critical for far-field robustness.
augmentation:
  # Background noise mixing at varied SNR. 5 dB SNR simulates noisy environments,
  # 25 dB simulates quiet rooms. The model sees both during training.
  background_snr_min_db: 5
  background_snr_max_db: 25
  # Room Impulse Response convolution simulates reverb / distance.
  # The MIT IR dataset bundled with livekit-wakeword covers a wide range of room sizes.
  rir_prob: 0.7        # 70% of positive clips get RIR convolved
  # Gain variation simulates mic placement variability.
  gain_min_db: -10
  gain_max_db: 6
  # Pitch shift simulates speaker variation.
  pitch_shift_semitones: 1.5
  # Time stretch simulates speaking rate variation.
  time_stretch_min: 0.9
  time_stretch_max: 1.1

# Model architecture.
model:
  model_type: conv_attention   # 100x lower FP/hr than DNN, 17% higher recall
  model_size: small            # small is enough for "hey spot"; medium/large for harder phrases

# Training schedule.
steps: 50000                   # 3-phase adaptive: 50k full, 5k recalibration, 5k fine-tune
learning_rate: 0.0001
weight_decay: 0.01
label_smoothing: 0.05
max_negative_weight: 1500.0
target_fp_per_hour: 0.1        # Stricter than default 0.2: we will tune threshold up if recall suffers

batch_n_per_class:
  positive: 50
  adversarial_negative: 50
  ACAV100M_sample: 1024
  background_noise: 50
```

- [ ] **Step 3: Write the training wrapper**

```bash
#!/bin/bash
# scripts/train_wake_hey_spot.sh — run the livekit-wakeword pipeline end-to-end.
set -euo pipefail

CONFIG=configs/wakeword_hey_spot.yaml
WAKE_OUT=/mnt/ssd/wake-models/hey_spot.onnx

mkdir -p "$(dirname "$WAKE_OUT")"

# Full pipeline (generate → augment → extract → train → export)
livekit-wakeword run "$CONFIG"

# Copy the exported model into our SSD wake-models dir
EXPORT_DIR=$(grep -E '^output_dir' "$CONFIG" 2>/dev/null | awk -F'"' '{print $2}' || echo "output")
SRC=$(find "$EXPORT_DIR" -name "hey_spot*.onnx" -print -quit)
[ -n "$SRC" ] && cp "$SRC" "$WAKE_OUT"
ls -lh "$WAKE_OUT"
```

- [ ] **Step 4: Run training**

This takes ~30–90 min on a discrete GPU; on AGX Orin it may run 2–4 h. If the timing is impractical, train on a cloud GPU (RTX 4090 or similar) and `scp` the resulting ONNX to the Jetson.

```bash
chmod +x scripts/train_wake_hey_spot.sh
./scripts/train_wake_hey_spot.sh
```

Expected end state: `/mnt/ssd/wake-models/hey_spot.onnx` (~50–100 KB conv-attention classifier) plus the bundled feature extractors that ship with livekit-wakeword.

- [ ] **Step 5: Evaluate on the recorded corpus**

```bash
spot-env/bin/python - <<'EOF'
from livekit.wakeword import WakeWordModel
import wave
import numpy as np

model = WakeWordModel(models=["/mnt/ssd/wake-models/hey_spot.onnx"])

def score(path):
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16)
    scores = model.predict(audio)
    return scores.get("hey_spot", 0.0)

print("--- POSITIVES (expect high scores) ---")
for f in ["tests/audio/hey_spot.wav",
          "tests/audio/hey_spot_stand_up.wav",
          "tests/audio/hey_spot_walk_forward.wav"]:
    print(f"  {f}: {score(f):.3f}")

print("--- NEGATIVES (expect low scores) ---")
for f in ["tests/audio/idle_01.wav",
          "tests/audio/adv_spot_the_difference.wav",
          "tests/audio/adv_on_the_spot.wav",
          "tests/audio/adv_hey_dog.wav",
          "tests/audio/adv_hey_stop.wav"]:
    print(f"  {f}: {score(f):.3f}")
EOF
```

Expected: positives > 0.7, negatives < 0.3. If a negative scores high (false accept) or a positive scores low (false reject), re-train with more adversarial samples or lower `target_fp_per_hour`. Document the threshold you choose for inference.

- [ ] **Step 6: Write the inference wrapper**

```python
# src/voice_control/wake_word_livekit.py
"""LiveKit-Wakeword inference wrapper for the Spot voice loop.

Provides:
    LiveKitWakeWord(model_path).accept_chunk(pcm16_bytes) -> bool
    LiveKitWakeWord.reset()

Frames are accumulated until a full ~2 s window is available; predict()
returns a score per loaded wake model. We trigger on score >= threshold.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from livekit.wakeword import WakeWordModel

LIVEKIT_MODEL = Path(
    os.environ.get("LIVEKIT_WAKE_MODEL", "/mnt/ssd/wake-models/hey_spot.onnx")
)
SAMPLE_RATE = 16000
WAKE_THRESHOLD = float(os.environ.get("LIVEKIT_WAKE_THRESHOLD", "0.55"))
WINDOW_DURATION_S = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_S)
STEP_SAMPLES = int(SAMPLE_RATE * 0.08)     # 80 ms hop (~12 Hz check rate)


class LiveKitWakeWord:
    name = "livekit"

    def __init__(self) -> None:
        if not LIVEKIT_MODEL.exists():
            raise FileNotFoundError(f"LiveKit wake model missing: {LIVEKIT_MODEL}")
        self.model = WakeWordModel(models=[str(LIVEKIT_MODEL)])
        self.model_key = LIVEKIT_MODEL.stem      # "hey_spot"
        self.buffer = np.zeros(0, dtype=np.int16)
        print(f"[wake livekit] loaded {LIVEKIT_MODEL.name}, threshold={WAKE_THRESHOLD}")

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.int16)

    def accept_chunk(self, pcm16_bytes: bytes) -> bool:
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.buffer = np.concatenate([self.buffer, samples])
        detected = False
        # Slide a 2 s window by 80 ms hops over whatever buffer we have.
        while len(self.buffer) >= WINDOW_SAMPLES:
            window = self.buffer[-WINDOW_SAMPLES:]
            scores = self.model.predict(window)
            score = float(scores.get(self.model_key, 0.0))
            if score >= WAKE_THRESHOLD:
                detected = True
            # Advance the buffer by STEP_SAMPLES
            self.buffer = self.buffer[STEP_SAMPLES:]
        return detected
```

- [ ] **Step 7: SPOT_WAKE_BACKEND env dispatch**

Open `src/voice_control/wake_word.py`. Near the top, add:

```python
import os

_WAKE_BACKEND_NAME = os.environ.get("SPOT_WAKE_BACKEND", "livekit").lower()


def _make_wake_detector():
    if _WAKE_BACKEND_NAME == "livekit":
        from .wake_word_livekit import LiveKitWakeWord
        print("[wake] backend=livekit")
        return LiveKitWakeWord()
    if _WAKE_BACKEND_NAME == "sherpa_onnx":
        # Existing sherpa-onnx KWS path (preserved as rollback)
        print("[wake] backend=sherpa_onnx")
        return _make_wake_detector_sherpa()
    raise ValueError(
        f"Unknown SPOT_WAKE_BACKEND={_WAKE_BACKEND_NAME!r}; valid: livekit|sherpa_onnx"
    )
```

Refactor the existing sherpa-onnx KWS construction in this file into a `_make_wake_detector_sherpa()` helper, exposing the same `accept_chunk(bytes) -> bool` + `reset()` API as `LiveKitWakeWord`. If sherpa-onnx returns a different shape, add a thin adapter that maps `keyword_spotter.is_keyword_detected(...)` to the `accept_chunk` contract.

- [ ] **Step 8: Pin livekit-wakeword in requirements.txt**

Append:

```
# Wake word — livekit-wakeword conv-attention head, custom-trained "hey spot".
# Editable install from /mnt/ssd/livekit-wakeword; do not re-pin here.
livekit-wakeword
```

- [ ] **Step 9: Commit**

```bash
git add scripts/setup_livekit_wakeword.sh configs/wakeword_hey_spot.yaml \
        scripts/train_wake_hey_spot.sh src/voice_control/wake_word_livekit.py \
        src/voice_control/wake_word.py requirements.txt
git commit -m "stage 2a t9: LiveKit-Wakeword hey-spot accent-agnostic far-field training"
git push origin tour_guide_upgrade_matteo
```

---

## Task 10 — 2-stage wake → ASR pipeline integration

**Files:** Modify: `src/voice_control/client_mic.py`

Wire the two stages together so wake fires only on "hey spot" and then routes the following speech to Nemotron Streaming via the existing gRPC server.

- [ ] **Step 1: Locate the existing capture loop**

```bash
grep -nE 'wake|keyword|StreamingRecognize' src/voice_control/client_mic.py
```

- [ ] **Step 2: Wire the 2-stage loop**

Replace the capture loop with the following pattern (preserve existing TTS / dispatch glue around it):

```python
import collections
import time
from src.voice_control.wake_word import _make_wake_detector

# Configurable timings.
PRE_WAKE_BUFFER_S = 0.3       # 300 ms tail kept BEFORE wake fires — captures
                              # the start of the command if user says
                              # "hey spot, sit down" in one breath.
POST_WAKE_TIMEOUT_S = 6.0     # If no speech after wake, return to wake-listen.
VAD_END_OF_TURN_S = 0.6       # Silence to declare turn over.

CHUNK_SAMPLES = 512           # 32 ms @ 16 kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2


def voice_loop(asr_client, brain, tts):
    wake = _make_wake_detector()
    vad = _make_vad()
    pre_wake_ring = collections.deque(
        maxlen=int(PRE_WAKE_BUFFER_S * SAMPLE_RATE / CHUNK_SAMPLES)
    )

    while True:
        wake.reset()
        vad.reset()
        pre_wake_ring.clear()

        # Stage 1: wake-listen. Cheap conv-attention model on every chunk.
        # Silero VAD also runs in parallel so we can peek at any speech that
        # arrives during wake-listen and dump it into the ring buffer.
        while True:
            chunk = read_mic(CHUNK_BYTES)
            pre_wake_ring.append(chunk)
            if wake.accept_chunk(chunk):
                print("[wake] hey spot detected")
                break

        # Stage 2: ASR-listen. Drain the pre-wake ring (already captured) plus
        # ongoing audio until VAD declares the turn over.
        pcm_chunks = list(pre_wake_ring)
        turn_start = time.time()
        silent_for = 0.0
        last_speech = time.time()

        while True:
            chunk = read_mic(CHUNK_BYTES)
            pcm_chunks.append(chunk)
            if _vad_accept(vad, chunk):
                last_speech = time.time()
                silent_for = 0.0
            else:
                silent_for = time.time() - last_speech
            if silent_for > VAD_END_OF_TURN_S:
                break
            if time.time() - turn_start > POST_WAKE_TIMEOUT_S and len(pcm_chunks) < 10:
                # User woke us but said nothing — bail out.
                print("[wake] post-wake timeout; returning to listen")
                break

        pcm_bytes = b"".join(pcm_chunks)
        if len(pcm_bytes) < int(0.3 * SAMPLE_RATE) * 2:
            continue

        # Transcribe via gRPC server
        transcript = asr_client.transcribe(pcm_bytes)
        print(f"[asr] transcript: {transcript!r}")
        if not transcript.strip():
            continue

        # Strip "hey spot" prefix if present (wake bled into ASR)
        clean = _strip_wake_prefix(transcript)

        # Brain + dispatch + TTS happens in T11 (streaming)
        spot_state = get_spot_state()
        result = brain.process(clean, spot_state)
        dispatch(result["actions"])
        if result["response"]:
            tts.enqueue_render(result["response"])


_WAKE_PATTERNS = (
    "hey spot,", "hey spot.", "hey spot ", "hey, spot,", "hey, spot.",
    "hey spot", "hey, spot",
)


def _strip_wake_prefix(text: str) -> str:
    low = text.lower().lstrip()
    for pat in _WAKE_PATTERNS:
        if low.startswith(pat):
            return text[len(pat):].lstrip(" ,.")
    return text
```

- [ ] **Step 3: Smoke test the loop without dispatching**

Add a dry-run flag so you can exercise wake → ASR without firing actions on the robot:

```bash
SPOT_DRY_RUN=1 spot-env/bin/python -m src.voice_control.client_mic --dry-run
```

Speak "hey spot, stand up" and confirm:
- Wake fires within ~200 ms of the phrase.
- Transcript ≈ `"hey spot, stand up"` or `"stand up"` (after prefix strip).
- Brain returns `{"actions":[{"action":"stand"}], "response":"..."}`.

- [ ] **Step 4: Test wake-only false-fire**

Run the loop and play 5 minutes of conversational audio (a podcast or speech recording) into the mic. Count wake fires. Acceptable: 0–2 false fires in 5 min. If more, raise `LIVEKIT_WAKE_THRESHOLD` to 0.65 and re-test.

- [ ] **Step 5: Commit**

```bash
git add src/voice_control/client_mic.py
git commit -m "stage 2a t10: 2-stage wake gated ASR pipeline (livekit + nemotron + silero)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 11 — Streaming SSE → TTS chunker

**Files:** Modify: `src/voice_control/spot_tts.py`, `src/voice_control/client_mic.py`

This is the **biggest perceived-latency win** in Stage 2A. Currently the brain waits for the full response (~3–6 s) before TTS even begins. By piping streaming tokens into a sentence chunker that hands the first sentence to TTS as soon as it lands, the user starts hearing Spot speak within ~500 ms of finishing their utterance.

Only applies to the `response` text. The `actions` array is parsed at the end of the stream (after the full JSON is buffered) because GBNF guarantees JSON shape but emits tokens in document order — `actions` may come before or after `response` depending on grammar.

Plan: stream-parse the JSON incrementally. When we see `"response":"...`, start forwarding subsequent characters into the TTS chunker. Strip the closing quote at end.

- [ ] **Step 1: Add `voice` param to `SpotTTS.speak` + add `TTSChunker` + `enqueue_streaming` to spot_tts.py**

The existing `SpotTTS.speak(text)` (line ~162) already enqueues through `AudioPlayer.enqueue_render` (non-blocking, in submission order). Stage 2A reuses it — the only signature change is an optional `voice` override so Stage 2E.1 personas can swap voice per-call without mutating `self.voice`. All chunks from one streaming turn share a single voice.

Edit `src/voice_control/spot_tts.py`:

a. Extend the `speak` signature (~line 162) to accept `voice`:

```python
def speak(self, text: str, voice: str | None = None):
    """Enqueue text for synthesis. Returns immediately. If voice is None, uses self.voice."""
    if not self.is_available():
        print(f'[TTS] Not available — would say: "{text}"')
        return
    if not text or not text.strip():
        return

    gain = self.volume
    voice = voice or self.voice   # explicit per-call wins; falls back to instance default
    speed = self.speed
    # ... rest of existing _render closure + self._player.enqueue_render() call unchanged ...
```

b. Add the chunker + streaming entry point (same file, after the `SpotTTS` class body):

```python
import re

SENTENCE_END_RE = re.compile(r"[.!?]\s")


class TTSChunker:
    """Buffer streaming tokens; flush a sentence to TTS on each boundary."""

    def __init__(self, tts: "SpotTTS", voice: str | None = None):
        self._tts = tts
        self._voice = voice
        self.buf = []
        self.in_response = False
        self.escape = False
        self.json_buf = []

    def _emit(self, chunk: str) -> None:
        self._tts.speak(chunk, voice=self._voice)

    def accept(self, delta: str) -> None:
        """Feed a token delta from the LLM stream. Parses surrounding JSON
        incrementally; once inside the "response":"..." string value, chars
        flow to the sentence buffer.
        """
        for ch in delta:
            self.json_buf.append(ch)
            if not self.in_response:
                tail = "".join(self.json_buf[-20:])
                if tail.rfind('"response":"') != -1:
                    self.in_response = True
                continue
            if self.escape:
                self.buf.append(ch)
                self.escape = False
            elif ch == "\\":
                self.escape = True
            elif ch == '"':
                self.in_response = False
                rest = "".join(self.buf).strip()
                if rest:
                    self._emit(rest)
                self.buf = []
            else:
                self.buf.append(ch)
                text = "".join(self.buf)
                if SENTENCE_END_RE.search(text):
                    parts = SENTENCE_END_RE.split(text)
                    complete = parts[:-1]
                    tail = parts[-1]
                    if complete:
                        self._emit(" ".join(s.strip() for s in complete if s.strip()))
                    self.buf = list(tail)
```

c. Wire `enqueue_streaming` as a method on `SpotTTS`:

```python
def enqueue_streaming(self, voice: str | None = None) -> "callable":
    """Return a callable that the brain hands token deltas to; sentences
    flush via self.speak(chunk, voice=voice).
    """
    chunker = TTSChunker(self, voice=voice)
    return chunker.accept
```

- [ ] **Step 2: Hook the TTS streamer into the voice loop**

In `src/voice_control/client_mic.py`, replace the brain call in `voice_loop` with:

```python
        on_token = tts.enqueue_streaming()    # 2E.1 will pass voice=… per-persona here
        brain.on_token_callback = on_token
        try:
            result = brain.process(clean, spot_state)
        finally:
            brain.on_token_callback = None

        dispatch(result["actions"])
        # The TTS chunker already emitted the response sentence-by-sentence
        # via tts.speak(chunk); no second tts.speak(result["response"]) call here.
```

Also remove any pre-2A site that called `tts.speak(result["response"])` immediately after `brain.process()` (the streaming chunker now owns the spoken side).

- [ ] **Step 3: End-to-end latency smoke test**

```bash
spot-env/bin/python - <<'EOF'
import time
from src.voice_control.brain import get_backend

b = get_backend()
t_first = None
t0 = time.time()

def on_token(delta):
    global t_first
    if t_first is None:
        t_first = time.time()
        print(f"[stream] first token after {t_first - t0:.2f} s: {delta!r}")
b.chat("You are Spot.",
       [{"role": "user", "content": "tell me about Dartmouth College in a paragraph"}],
       on_token=on_token)
print(f"total: {time.time() - t0:.2f} s")
EOF
```

Expected: `first token after 0.3 – 0.8 s`, `total 3 – 6 s`. The TTS chunker would have started TTS at ~0.8 s — perceived latency cut in half vs the non-streaming path.

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/spot_tts.py src/voice_control/client_mic.py
git commit -m "stage 2a t11: streaming SSE token deltas piped into TTS sentence chunker"
git push origin tour_guide_upgrade_matteo
```

---

## Task 12 — Mic-verify end-to-end (wake + ASR + brain + dispatch)

**Files:** Create: `scripts/mic_verify.py`

End-to-end gate that compares post-swap behaviour against the T0 baseline. On pass we reclaim the qwen Stage 1 rollback layer.

- [ ] **Step 1: Write the verifier**

```python
#!/usr/bin/env python3
"""Stage 2A mic-verify.

Replays tests/audio/*.wav through:
- LiveKit wake word (for wake utterances)
- gRPC ASR (Nemotron default)
- LLMBrain (llama.cpp default, single GBNF)

Reports per-utterance correctness + latency, and compares against the T0
baseline snapshot if SPOT_BASELINE_SNAPSHOT is set.
"""
from __future__ import annotations
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import grpc
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.voice_control import asr_pb2 as pb, asr_pb2_grpc as pbg
from src.voice_control.brain import get_backend
from src.voice_control.wake_word_livekit import LiveKitWakeWord

CORPUS_YAML = ROOT / "tests/utterances/action_corpus.yaml"
AUDIO_DIR = ROOT / "tests/audio"


def transcribe_via_grpc(wav_path: Path) -> str:
    with wave.open(str(wav_path), "rb") as w:
        pcm = w.readframes(w.getnframes())
    chan = grpc.insecure_channel("localhost:50055")
    stub = pbg.ASRStub(chan)
    def gen():
        yield pb.StreamingRequest(config=pb.StreamingConfig(
            language_code="en-US", sample_rate_hz=16000, enable_punctuation=True))
        yield pb.StreamingRequest(audio=pb.AudioChunk(pcm16=pcm))
    for resp in stub.StreamingRecognize(gen()):
        if resp.is_final:
            return resp.transcript.strip()
    return ""


def load_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16)


def main() -> int:
    corpus = yaml.safe_load(CORPUS_YAML.read_text())["utterances"]
    backend = get_backend()
    wake = LiveKitWakeWord()
    print(f"=== mic-verify  brain={backend.name}  corpus={len(corpus)} ===\n")

    results = []
    action_latencies, freeform_latencies = [], []

    for entry in corpus:
        wav = AUDIO_DIR / entry["wav"]
        expected_action = entry["action"]
        expected_params = entry.get("params") or {}

        transcript = transcribe_via_grpc(wav)

        t0 = time.time()
        raw = backend.chat(
            "You are Spot. Output JSON only.",
            [{"role": "user", "content": transcript}],
        )
        elapsed = time.time() - t0
        (action_latencies if expected_action else freeform_latencies).append(elapsed)

        json_parses = False
        action_match = params_match = None
        try:
            parsed = json.loads(raw)
            json_parses = True
            actions = parsed.get("actions", [])
            if expected_action is None:
                action_match = (len(actions) == 0)
                params_match = True
            elif actions:
                actual_action = actions[0].get("action")
                actual_params = actions[0].get("params", {})
                action_match = (actual_action == expected_action)
                params_match = all(
                    actual_params.get(k) == v for k, v in expected_params.items()
                ) if expected_params else True
            else:
                action_match = False
                params_match = False
        except json.JSONDecodeError:
            json_parses = False

        results.append({
            "wav": entry["wav"],
            "transcript": transcript,
            "expected_action": expected_action,
            "json_parses": json_parses,
            "action_match": action_match,
            "params_match": params_match,
            "latency_s": round(elapsed, 2),
        })

    # Wake evaluation
    print("=== wake word evaluation ===")
    pos_files = [AUDIO_DIR / "hey_spot.wav",
                 AUDIO_DIR / "hey_spot_stand_up.wav",
                 AUDIO_DIR / "hey_spot_walk_forward.wav"]
    neg_files = [AUDIO_DIR / f for f in [
        "idle_01.wav", "idle_02.wav", "idle_03.wav",
        "adv_spot_the_difference.wav", "adv_on_the_spot.wav",
        "adv_hey_dog.wav", "adv_hey_stop.wav",
    ]]
    wake_pos_fires = 0
    wake_neg_fires = 0
    for f in pos_files:
        if not f.exists():
            continue
        wake.reset()
        if wake.accept_chunk(load_wav_int16(f).tobytes()):
            wake_pos_fires += 1
    for f in neg_files:
        if not f.exists():
            continue
        wake.reset()
        if wake.accept_chunk(load_wav_int16(f).tobytes()):
            wake_neg_fires += 1

    # Report
    n = len(results)
    action_results = [r for r in results if r["expected_action"]]
    freeform_results = [r for r in results if not r["expected_action"]]

    print("\n=== per-utterance ===")
    for r in results:
        flag = "OK" if (r["json_parses"] and r["action_match"] and r["params_match"]) else "FAIL"
        print(f"  [{flag}] {r['wav']:40s} act={str(r['expected_action']):20s} "
              f"lat={r['latency_s']}s")
        if flag == "FAIL":
            print(f"          transcript={r['transcript']!r}")

    print("\n=== aggregate ===")
    print(f"  wake recall:   {wake_pos_fires}/{len(pos_files)} positives fired")
    print(f"  wake FP:       {wake_neg_fires}/{len(neg_files)} negatives fired")
    action_ok = sum(1 for r in action_results if r["json_parses"] and r["action_match"])
    print(f"  action accuracy: {action_ok}/{len(action_results)}")
    if action_latencies:
        print(f"  action latency p50={statistics.median(action_latencies):.2f}s")
    if freeform_latencies:
        print(f"  freeform latency p50={statistics.median(freeform_latencies):.2f}s")

    gates = {
        "wake_recall_all_pos": wake_pos_fires == len(pos_files),
        "wake_fp_zero": wake_neg_fires == 0,
        "action_90pct": (action_ok / max(1, len(action_results))) >= 0.90,
        "action_p50_lt_2s": (statistics.median(action_latencies) < 2.0) if action_latencies else False,
        "freeform_p50_lt_4s": (statistics.median(freeform_latencies) < 4.0) if freeform_latencies else False,
    }
    print("\n=== gates ===")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Pre-flight — both services live**

```bash
curl -fs http://127.0.0.1:11435/health || { echo "llama-server DOWN"; exit 1; }
SPOT_ASR_BACKEND=nemotron spot-env/bin/python -m src.voice_control.server &
ASR_PID=$!
sleep 12
nc -z localhost 50055 || { echo "ASR server DOWN"; kill $ASR_PID; exit 1; }
```

- [ ] **Step 3: Run mic-verify**

```bash
spot-env/bin/python scripts/mic_verify.py | tee /tmp/mic_verify_$(date +%Y%m%d_%H%M%S).log
echo "Exit: $?"
kill $ASR_PID
```

Expected: exit 0 and all gates PASS. Save the log.

If FAIL, follow the cascade in spec §2A.3:

1. **Wake recall failures**: raise `target_fp_per_hour` to 0.2 and retrain, or drop threshold from 0.55 to 0.45.
2. **Wake FP failures**: add the offending negative to `custom_negative_phrases`, retrain.
3. **Action accuracy**: tighten the GBNF (e.g., add `required` keys); inspect raw response for grammar drift.
4. **Latency**: bump quant (re-pull `Q5_K_M` or `Q6_K` of Gemma 4 E4B), restart `llama-server.service`.
5. **Rollback to Ollama**: `SPOT_BRAIN_BACKEND=ollama spot-env/bin/python scripts/mic_verify.py`.
6. **Rollback to Parakeet**: `SPOT_ASR_BACKEND=parakeet ...`.
7. **Rollback to sherpa-onnx wake**: `SPOT_WAKE_BACKEND=sherpa_onnx ...`.

- [ ] **Step 4: On pass — reclaim the qwen rollback layer**

```bash
ollama list
ollama rm qwen2.5:7b
ollama rm qwen2.5vl:7b
ollama list
df -h /mnt/ssd
```

Expected: ~10.7 GB freed.

- [ ] **Step 5: Update stage1-rollback.md**

Open `docs/project/stage1-rollback.md` and append to "Layer 2 — Brain model rollback":

```markdown
> **Stage 2A.3 update (mic-verify pass):** The qwen2.5:7b + qwen2.5vl:7b pair has been removed via `ollama rm`. This layer is no longer available without re-downloading the pair (~10.7 GB). New rollback ladder lives in `docs/project/stage2-rollback.md`.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/mic_verify.py docs/project/stage1-rollback.md
git commit -m "stage 2a t12: mic-verify end-to-end + qwen pair reclaim on pass"
git push origin tour_guide_upgrade_matteo
```

---

## Task 13 — Write stage2-rollback.md

**Files:** Create: `docs/project/stage2-rollback.md`

- [ ] **Step 1: Write the rollback runbook**

```markdown
# Stage 2 Rollback Runbook

## Stage 2A complete state (tag `stage2a-voice-complete`)

- Brain runtime: llama.cpp via NVIDIA Jetson container, gemma-4-E4B-Q4_K_M.gguf + mmproj on /mnt/ssd/llamacpp-models/, served on port 11435 by `llama-server.service`.
- Grammar: single unified GBNF at src/voice_control/grammar/spot_action.gbnf.
- ASR: Nemotron Speech Streaming 0.6B int8, sherpa-onnx CUDA EP at /mnt/ssd/nemotron-models/, served on port 50055.
- VAD: Silero via sherpa-onnx.
- Wake word: LiveKit-Wakeword conv-attention "hey spot" at /mnt/ssd/wake-models/hey_spot.onnx (custom-trained accent-agnostic English).
- qwen Stage 1 rollback pair removed (see stage1-rollback.md).

## Disk pressure guard

`df -h /mnt/ssd` — must show > 25 GB free. If lower, do not re-acquire qwen pair before reclaiming elsewhere.

## Layer A — Brain runtime rollback (instant)

If llama-server crashes or Gemma 4 E4B regresses on a corner case:

```bash
SPOT_BRAIN_BACKEND=ollama spot-env/bin/python scripts/run_voice_control.py
```

Ollama 0.27+ with gemma4:e4b is still installed. Format-json path latency may
regress slightly; functionality is intact. The Ollama bug #15260 was fixed in
PR #15678 (Apr 2026) so this rollback no longer pays the old "thinking +
format" cost.

## Layer B — ASR backend rollback (instant)

If Nemotron Streaming is unstable (e.g., CUDA EP regression, hallucinations
escaping the blocklist):

```bash
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server
```

Parakeet TDT 0.6B v3 ONNX is on disk at /mnt/ssd/parakeet-models/. Drops
real-time TTFT but is otherwise solid.

## Layer C — Wake word rollback (instant)

If LiveKit-Wakeword false-rejects in some condition (loud room, unfamiliar
speaker accent):

```bash
SPOT_WAKE_BACKEND=sherpa_onnx spot-env/bin/python scripts/run_voice_control.py
```

Sherpa-onnx KWS Zipformer is still installed (unchanged in Stage 2A). May
fall back to the original "Hey Spot" accent issue, but the loop runs.

## Layer D — Wake-by-transcription fallback (manual)

If both wake backends miss far-field utterances, set
`SPOT_WAKE_BACKEND=always_on` (TODO: implement in Stage 2B if Layer C also
fails in production). This streams Nemotron continuously and regex-matches
transcripts for "hey spot" — higher cost, near-perfect recall.

## Layer E — Full Stage 2A revert (destructive)

```bash
sudo systemctl disable --now llama-server.service
git checkout tour_guide_upgrade_matteo
git reset --hard stage1.5-complete
git push origin tour_guide_upgrade_matteo --force-with-lease
```

Re-installs `webrtcvad` + Riva imports (which will fail until Layer B is
configured manually). Use only after consultation.

## Layer F — Brain model quant bump (baseline is Q6_K)

If Gemma 4 E4B Q6_K misbehaves on grammar or output quality regresses:

1. Re-pull a higher quant via setup_llamacpp_models.py with filename swap
   (e.g. `gemma-4-E4B-it-Q8_0.gguf` or BF16). Q8_0 is near-lossless vs BF16
   and fits comfortably in AGX Orin VRAM alongside the rest of the stack.
2. Edit /etc/systemd/system/llama-server.service `ExecStart` `-m` path.
3. `sudo systemctl daemon-reload && sudo systemctl restart llama-server`.
4. Re-run mic-verify to confirm regression resolved.

## Verification after any rollback

Always re-run mic-verify:

```bash
spot-env/bin/python scripts/mic_verify.py
```

Save the log to /tmp/ and compare gates against the T0 baseline snapshot.
```

- [ ] **Step 2: Commit**

```bash
git add docs/project/stage2-rollback.md
git commit -m "stage 2a t13: stage2-rollback runbook 6 layers"
git push origin tour_guide_upgrade_matteo
```

---

## Task 14 — Tag stage2a-voice-complete

- [ ] **Step 1: Confirm all preceding tasks committed**

```bash
git status
git log --oneline -20
```

Expected: clean working tree; recent commits T0–T13.

- [ ] **Step 2: Final smoke test**

```bash
curl -fs http://127.0.0.1:11435/health
SPOT_ASR_BACKEND=nemotron spot-env/bin/python -m src.voice_control.server &
ASR_PID=$!
sleep 12
spot-env/bin/python scripts/mic_verify.py
echo "exit=$?"
kill $ASR_PID
```

Expected: exit 0, all gates PASS.

- [ ] **Step 3: Tag + push**

```bash
git tag -a stage2a-voice-complete -m "Stage 2A: llama.cpp Gemma 4 E4B streaming brain + Nemotron Streaming ASR + Silero VAD + LiveKit-Wakeword hey-spot. Mic-verify passes; qwen pair reclaimed."
git push origin stage2a-voice-complete
```

- [ ] **Step 4: Append memory pointer**

Add a one-line entry under "Linked memories" in
`/home/spotdog/.claude/projects/-home-spotdog-spot-dartmouth-spot-capstone/memory/MEMORY.md`:

```markdown
- [Stage 2A voice pipeline complete](../../...) — llama.cpp Gemma 4 E4B + streaming SSE, Nemotron Streaming ASR, Silero VAD, LiveKit-Wakeword hey-spot (accent-agnostic). Tag: stage2a-voice-complete. qwen pair removed.
```

(The agent reading this plan should write a proper memory file with frontmatter per the auto-memory format, then add this pointer to MEMORY.md.)

---

## Self-Review

**Spec coverage:**
- §2A.0 brain runtime swap → T1, T2, T3, T4, T5 ✅ (rewritten around streaming + native Gemma 4)
- §2A.1 audio corpus → T6 ✅ (extended with adversarial negatives)
- §2A.2 Phase A (ASR backend) → T7 ✅ (Nemotron default, Parakeet rollback)
- §2A.2 Phase B (Silero VAD) → T8 ✅
- §2A.2 Phase C (ASR promotion) → folded into T7
- §2A.3 mic-verify → T12 ✅
- §2A.4 wake-word swap → T9 ✅ (LiveKit with custom-trained "hey spot")
- 2-stage wake → ASR pipeline integration → T10 ✅ (NEW)
- Streaming SSE → TTS chunker → T11 ✅ (NEW; biggest UX win)
- Pre-flight + baseline → T0 ✅ (NEW)
- Stage 2 rollback runbook → T13 ✅
- Final tag → T14 ✅

**Placeholder scan:** the few `...` in code blocks under T5 step 4 are documented as "preserve existing capture-loop glue" — the implementer must edit the existing file, not write the helper from scratch. The wake-word adapter contract in T9 step 7 says "if sherpa-onnx KWS has a different shape, write an adapter" — this is documented intent, not a gap; the adapter signature is fully specified.

**Type consistency:** `chat(system, messages, on_token=None) -> str` signature matches across `LlamaCppBackend` (T5 step 2), `OllamaBackend` (T5 step 3), and the `process()` caller (T5 step 4). `transcribe(pcm_bytes) -> str` matches `NemotronBackend` (T7 step 4), `ParakeetBackend` (T7 step 5), and the `ASRServicer.StreamingRecognize` consumer (T7 step 6). `accept_chunk(pcm16_bytes) -> bool` + `reset()` matches `LiveKitWakeWord` (T9 step 6) and the sherpa-onnx adapter contract (T9 step 7).

**Env var consistency:**
- `SPOT_BRAIN_BACKEND=llamacpp|ollama` (T5, T12, T13)
- `SPOT_ASR_BACKEND=nemotron|parakeet` (T7, T12, T13)
- `SPOT_WAKE_BACKEND=livekit|sherpa_onnx` (T9, T12, T13)
- `LIVEKIT_WAKE_MODEL`, `LIVEKIT_WAKE_THRESHOLD` (T9)
- `NEMO_DIR`, `PARAKEET_HF_HOME`, `PARAKEET_MODEL_ID` (T7)
- `SPOT_LLAMACPP_URL`, `SPOT_OLLAMA_URL`, `SPOT_OLLAMA_MODEL` (T5)

All have documented defaults; rollback layers in T13 reference them by exact name.

**Decisions deferred to post-2B (deliberate):**
- Delete RivaBackend code — wait until 2B end-to-end on real Spot passes.
- E2B speculative drafter / gemma4_assistant MTP — gate on >50% acceptance benchmark; not in Stage 2A.
- Gemma 4 native audio (collapse ASR + brain into one E4B call) — spike post-2B; llama.cpp E4B audio path is flagged unstable as of May 2026 per NVIDIA Jetson AI Lab.
- "Always-on Nemotron + regex wake" — Layer D in rollback runbook; implement only if Layers A–C all fail in field.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Best for catching drift early; each subagent has full task context with zero conversation baggage.

**2. Inline Execution** — execute tasks in the current session using the executing-plans skill, batch checkpoints for review.

Which approach?
