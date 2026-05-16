# Stage 2A — Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock the voice loop end-to-end on `tour_guide_upgrade_matteo` branch — swap Ollama brain runtime to llama.cpp + adaptive thinking, replace broken Riva ASR bridge with Parakeet TDT 0.6B v3, mic-verify gemma4, then swap wake word from sherpa-onnx KWS to LiveKit Wakeword. Exit tag: `stage2a-voice-complete`.

**Architecture:** Per-call GBNF grammar routes action utterances (fast, no thinking) vs free-form utterances (thinking allowed). llama-server runs as a long-lived systemd daemon on port 11435 with gemma4 GGUF + mmproj + E2B speculative drafter. Existing gRPC bridge in `server.py` is refactored to a `Backend` interface with `ParakeetBackend` + `RivaBackend` implementations behind `SPOT_ASR_BACKEND=parakeet|riva` env var. Wake-word swap is behind `SPOT_WAKE_BACKEND=livekit|sherpa_onnx`. All swaps are env-var-flippable for rollback.

**Tech Stack:** llama.cpp (CUDA build, MIT), gemma4 e4b Q4_K_M GGUF + mmproj-BF16.gguf + gemma4 e2b Q4_0 drafter, `onnx-asr==0.11.0`, `huggingface-hub>=1.0,<2.0`, `sherpa-onnx` Silero VAD, LiveKit Wakeword (Apache 2.0, ONNX), Python 3.10, `requests` for HTTP, existing Boston Dynamics SDK 5.0.1.1.

**Spec:** `docs/superpowers/specs/2026-05-15-stage2-design.md` §2A.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `scripts/setup_llamacpp_models.py` | Download + pin gemma4 GGUFs to `/mnt/ssd/llamacpp-models/` |
| Create | `scripts/build_llamacpp.sh` | Build llama.cpp from source with CUDA backend |
| Create | `systemd/llama-server.service` (then install as user unit) | Long-lived llama-server daemon on port 11435 |
| Create | `src/voice_control/action.gbnf` | GBNF grammar for action-dispatch JSON (enumerates valid actions) |
| Create | `src/voice_control/brain/__init__.py` | Backend registry for brain runtimes |
| Create | `src/voice_control/brain/llamacpp_backend.py` | HTTP client for llama-server, per-call grammar selection |
| Create | `src/voice_control/brain/ollama_backend.py` | Extracted Ollama path from `llm_brain.py` |
| Create | `src/voice_control/brain/router.py` | Intent classifier (action vs free-form) |
| Modify | `src/voice_control/llm_brain.py` | Read `SPOT_BRAIN_BACKEND` env, dispatch via Backend interface |
| Create | `scripts/setup_parakeet.py` | Download Parakeet TDT 0.6B v3 to `/mnt/ssd/parakeet-models/` |
| Create | `src/voice_control/asr/__init__.py` | Backend registry for ASR |
| Create | `src/voice_control/asr/parakeet.py` | Parakeet backend (~80 LOC), gRPC-compatible |
| Create | `src/voice_control/asr/riva_backend.py` | Extracted from current `server.py` (kept as rollback only) |
| Modify | `src/voice_control/server.py` | Dispatch via Backend interface based on `SPOT_ASR_BACKEND` env |
| Modify | `src/voice_control/client_mic.py` | Replace `webrtcvad` calls with Silero VAD from sherpa-onnx |
| Create | `tests/audio/*.wav` | 10-15 clean utterances + 5 motor-idle clips |
| Create | `tests/utterances/action_corpus.yaml` | 15-20 expected (utterance, action, location?, object?) tuples |
| Create | `scripts/mic_verify.py` | Replay audio corpus → ASR → brain → diff against expected |
| Modify | `requirements.txt` | Add `onnx-asr==0.11.0`, `huggingface-hub>=1.0,<2.0` |
| Create | `src/voice_control/wake_word_livekit.py` | LiveKit Wakeword ONNX loader |
| Modify | `src/voice_control/wake_word.py` | Read `SPOT_WAKE_BACKEND` env, dispatch sherpa_onnx vs livekit |
| Delete | `src/voice_control/asr/riva_backend.py` | After 2A.4 confirms Parakeet stable |
| Modify | `docs/project/stage1-rollback.md` | Mark qwen pair removable after mic-verify pass |
| Create | `docs/project/stage2-rollback.md` | Stage 2 rollback runbook (env-var flips) |

---

## Task 1 — Build llama.cpp from source (CUDA backend)

**Files:**
- Create: `scripts/build_llamacpp.sh`

- [ ] **Step 1: Write the build script**

```bash
#!/bin/bash
# scripts/build_llamacpp.sh — build llama.cpp from source on Jetson AGX Orin
# JetPack 6.2.1, CUDA 12.6, cuDNN 9.3, aarch64
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:-/mnt/ssd/llamacpp}"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_TAG="${LLAMA_TAG:-master}"   # pin to a tag/sha in production

if [ ! -d "$LLAMA_ROOT" ]; then
  git clone "$LLAMA_REPO" "$LLAMA_ROOT"
fi
cd "$LLAMA_ROOT"
git fetch --all --tags
git checkout "$LLAMA_TAG"

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

echo "=== built binaries ==="
ls -lh build/bin/llama-server build/bin/llama-cli
echo "=== llama-server --version ==="
build/bin/llama-server --version
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/build_llamacpp.sh
./scripts/build_llamacpp.sh
```

Expected: `build/bin/llama-server` exists, `--version` prints a commit sha. Build takes ~5-15 min on AGX Orin.

- [ ] **Step 3: Verify CUDA is linked**

```bash
ldd /mnt/ssd/llamacpp/build/bin/llama-server | grep -iE 'cuda|cublas'
```

Expected: `libcudart.so.12` + `libcublas.so.12` lines present.

- [ ] **Step 4: Smoke run with no model (sanity)**

```bash
/mnt/ssd/llamacpp/build/bin/llama-server --help | head -20
```

Expected: usage text printed, exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_llamacpp.sh
git commit -m "stage 2a: build llama.cpp from source script"
git push origin tour_guide_upgrade_matteo
```

---

## Task 2 — Download gemma4 GGUFs to SSD

**Files:**
- Create: `scripts/setup_llamacpp_models.py`

- [ ] **Step 1: Write the model-download script**

```python
#!/usr/bin/env python3
"""Download + version-pin llama.cpp GGUFs for gemma4 brain to /mnt/ssd/llamacpp-models/.

Mirrors scripts/setup_parakeet.py and scripts/setup_kws.py patterns.

Models pinned for reproducibility:
- gemma-4-e4b-Q4_K_M.gguf (~5 GB, primary brain)
- mmproj-BF16.gguf (~946 MB, vision projector for gemma4)
- gemma-4-e2b-Q4_0.gguf (~1.7 GB, speculative drafter)
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path("/mnt/ssd/llamacpp-models")

# Pinned repo/filename pairs. Update intentionally; do not auto-bump.
MODELS = [
    {
        "repo_id": "unsloth/gemma-4-e4b-it-GGUF",
        "filename": "gemma-4-e4b-it-Q4_K_M.gguf",
        "local_name": "gemma-4-e4b-Q4_K_M.gguf",
    },
    {
        "repo_id": "unsloth/gemma-4-e4b-it-GGUF",
        "filename": "mmproj-BF16.gguf",
        "local_name": "mmproj-BF16.gguf",
    },
    {
        "repo_id": "unsloth/gemma-4-e2b-it-GGUF",
        "filename": "gemma-4-e2b-it-Q4_0.gguf",
        "local_name": "gemma-4-e2b-Q4_0.gguf",
    },
]


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        target = MODELS_DIR / m["local_name"]
        if target.exists():
            print(f"[skip] {m['local_name']} already present ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"[pull] {m['repo_id']} -> {target}")
        downloaded = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        # huggingface_hub may save under the original filename; rename if needed.
        downloaded_path = Path(downloaded)
        if downloaded_path.name != m["local_name"]:
            downloaded_path.rename(target)
        print(f"  done: {target.stat().st_size / 1e9:.2f} GB")

    print()
    print("=== llamacpp models ===")
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        print(f"  {p.name:40s}  {p.stat().st_size / 1e9:6.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify huggingface-hub is installed at >=1.0**

```bash
spot-env/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Expected: `>=1.0`. If not, `spot-env/bin/pip install --no-deps 'huggingface-hub>=1.0,<2.0'`.

- [ ] **Step 3: Run the download (~7.5 GB total, takes ~5-10 min)**

```bash
spot-env/bin/python scripts/setup_llamacpp_models.py
```

Expected:
```
=== llamacpp models ===
  gemma-4-e2b-Q4_0.gguf                       1.70 GB
  gemma-4-e4b-Q4_K_M.gguf                     5.00 GB
  mmproj-BF16.gguf                            0.94 GB
```

- [ ] **Step 4: Verify disk reclaim**

```bash
df -h /mnt/ssd /
```

Expected: `/mnt/ssd` use unchanged or +7.5 GB; `/` use unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_llamacpp_models.py
git commit -m "stage 2a: script to download gemma4 GGUFs to SSD"
git push origin tour_guide_upgrade_matteo
```

---

## Task 3 — Write the action GBNF grammar

**Files:**
- Create: `src/voice_control/action.gbnf`
- Reference: `src/voice_control/llm_brain.py:39-110` (system prompt with action catalog)

The grammar enforces JSON shape AND enumerates valid action names so the model cannot emit nonsense. Action enum mirrors `llm_brain.py:54-82`.

- [ ] **Step 1: Write the grammar file**

```gbnf
# action.gbnf — GBNF for Stage 2A action-dispatch JSON
# Enforces: {"actions": [{...}, ...], "response": "..."}
# Where each action.action is one of the enumerated names.

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

params        ::= "{" ws ( param ( "," ws param )* )? ws "}"
param         ::= string ":" ws value

# Permissive JSON value subset
value         ::= string | number | "true" | "false" | "null" | object | array
object        ::= "{" ws ( pair ( "," ws pair )* )? ws "}"
pair          ::= string ":" ws value
array         ::= "[" ws ( value ( "," ws value )* )? ws "]"

# JSON string — escape sequences allowed
string        ::= "\"" ( [^"\\\x00-\x1f] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""

number        ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?

ws            ::= ( " " | "\t" | "\n" | "\r" )*
```

- [ ] **Step 2: Validate grammar parses**

```bash
/mnt/ssd/llamacpp/build/bin/llama-cli --help-grammar 2>&1 | head -5  # confirm tool understands GBNF
# Quick parse via gbnf tool if available; otherwise test in Task 5 smoke.
```

- [ ] **Step 3: Sanity-check the enum matches dispatcher**

```bash
grep -oE '^[a-z_]+:' src/voice_control/llm_brain.py | grep -oE '^[a-z_]+' | sort -u > /tmp/dispatcher_actions.txt
grep -oE '"[a-z_]+"' src/voice_control/action.gbnf | sort -u > /tmp/grammar_actions.txt
diff /tmp/dispatcher_actions.txt /tmp/grammar_actions.txt || echo "Differences found — fix grammar OR add action to dispatcher"
```

Expected: empty diff (every action in grammar exists in dispatcher; this is best-effort, not exhaustive).

- [ ] **Step 4: Commit**

```bash
git add src/voice_control/action.gbnf
git commit -m "stage 2a: action.gbnf grammar enumerating valid Spot actions"
git push origin tour_guide_upgrade_matteo
```

---

## Task 4 — Install llama-server as a systemd user unit

**Files:**
- Create: `systemd/llama-server.service`

- [ ] **Step 1: Write the unit file**

```ini
# systemd/llama-server.service
# Install via: systemctl --user enable --now $(pwd)/systemd/llama-server.service
[Unit]
Description=llama.cpp brain server for Spot (gemma4 e4b + mmproj + E2B drafter)
After=network.target

[Service]
Type=simple
WorkingDirectory=/mnt/ssd/llamacpp-models
Environment="LLAMA_BIN=/mnt/ssd/llamacpp/build/bin/llama-server"
ExecStart=/bin/bash -c '$LLAMA_BIN \
    -m /mnt/ssd/llamacpp-models/gemma-4-e4b-Q4_K_M.gguf \
    --mmproj /mnt/ssd/llamacpp-models/mmproj-BF16.gguf \
    --model-draft /mnt/ssd/llamacpp-models/gemma-4-e2b-Q4_0.gguf \
    --draft-max 16 \
    -ngl 99 \
    -c 8192 \
    --host 127.0.0.1 \
    --port 11435 \
    --log-format json \
    --metrics'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Install + start the unit**

```bash
mkdir -p ~/.config/systemd/user
cp systemd/llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
```

If user-level systemd is not available on Jetson (no XDG_RUNTIME_DIR), install as system unit instead:
```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service
```

- [ ] **Step 3: Wait for cold start + verify health (10-30 s)**

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 4
  if curl -fs http://127.0.0.1:11435/health 2>/dev/null; then
    echo "[ok] llama-server live"
    break
  fi
  echo "[$i] waiting for llama-server..."
done
```

Expected: `{"status":"ok"}` (or similar — depends on llama.cpp version).

- [ ] **Step 4: Verify metrics endpoint**

```bash
curl -s http://127.0.0.1:11435/v1/metrics | head -20
```

Expected: Prometheus-format lines like `llamacpp:tokens_predicted_seconds_total` or `llamacpp:requests_processing`.

- [ ] **Step 5: Smoke test — action grammar JSON**

```bash
curl -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(cat <<'JSON'
{
  "messages": [
    {"role": "system", "content": "You are Spot. Output JSON only."},
    {"role": "user", "content": "Stand up"}
  ],
  "grammar": "$(cat src/voice_control/action.gbnf | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()))')",
  "max_tokens": 200,
  "temperature": 0.2
}
JSON
)"
```

Expected: JSON response with `choices[0].message.content` containing a valid JSON object with `"actions": [{"action": "stand"}]`. Latency < 5 s (cold or warm).

- [ ] **Step 6: Smoke test — free-form path (no grammar)**

```bash
curl -s -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Briefly describe what a tour guide robot at Dartmouth would talk about."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }' | python3 -m json.tool | head -30
```

Expected: coherent natural-language paragraph in `content`. May include `<think>...</think>` prefix on a thinking model.

- [ ] **Step 7: Commit**

```bash
git add systemd/llama-server.service
git commit -m "stage 2a: llama-server systemd user unit (gemma4 + mmproj + E2B drafter)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 5 — Brain backend interface (refactor llm_brain.py)

**Files:**
- Create: `src/voice_control/brain/__init__.py`
- Create: `src/voice_control/brain/ollama_backend.py`
- Create: `src/voice_control/brain/llamacpp_backend.py`
- Create: `src/voice_control/brain/router.py`
- Modify: `src/voice_control/llm_brain.py`

- [ ] **Step 1: Create the backend registry**

```python
# src/voice_control/brain/__init__.py
"""Brain runtime backends. Selected via SPOT_BRAIN_BACKEND env var."""
import os
from .ollama_backend import OllamaBackend
from .llamacpp_backend import LlamaCppBackend
from .router import classify_intent, IntentKind

_BACKENDS = {
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
}


def get_backend():
    name = os.environ.get("SPOT_BRAIN_BACKEND", "llamacpp").lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown SPOT_BRAIN_BACKEND={name!r}; valid: {list(_BACKENDS)}")
    return cls()
```

- [ ] **Step 2: Create the intent router**

```python
# src/voice_control/brain/router.py
"""Classify utterance intent → action (needs JSON grammar, no thinking)
   vs free-form (no grammar, thinking allowed).
"""
import re
from enum import Enum


class IntentKind(str, Enum):
    ACTION = "action"
    FREEFORM = "freeform"


# Free-form keywords/patterns. Anything matching → FREEFORM. Else ACTION.
_FREEFORM_PATTERNS = [
    r"\bwhat (do|can) you see\b",
    r"\bdescribe\b",
    r"\btell me about\b",
    r"\bwhat is\b",
    r"\bwhat's\b",
    r"\bwho is\b",
    r"\bwho's\b",
    r"\bhow do you\b",
    r"\bhow are you\b",
    r"\bwhy\b",
    r"\bexplain\b",
    r"\bopinion\b",
    r"\bthink about\b",
    r"\bjoke\b",
    r"\bstory\b",
]
_FREEFORM_RE = re.compile("|".join(_FREEFORM_PATTERNS), re.IGNORECASE)


def classify_intent(transcript: str) -> IntentKind:
    """Return ACTION or FREEFORM based on transcript content.

    ACTION = needs a dispatched robot action, GBNF-grammar-constrained JSON output.
    FREEFORM = chat/description/knowledge, plain text output (thinking allowed).
    """
    if not transcript or not transcript.strip():
        return IntentKind.ACTION  # safe default — empty utt → no-op JSON
    if _FREEFORM_RE.search(transcript):
        return IntentKind.FREEFORM
    return IntentKind.ACTION
```

- [ ] **Step 3: Create the Ollama backend (extracted from current llm_brain.py)**

```python
# src/voice_control/brain/ollama_backend.py
"""Ollama HTTP backend — current Stage 1 path. Kept as fallback for SPOT_BRAIN_BACKEND=ollama."""
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests

OLLAMA_URL = "http://localhost:11434"


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "gemma4:e4b", vlm_model: str = "gemma4:e4b"):
        self.model = model
        self.vlm_model = vlm_model

    def chat_json(
        self,
        system: str,
        messages: List[Dict[str, str]],
        timeout: float = 30.0,
    ) -> str:
        """Action-path call: returns raw JSON string. think:on workaround for bug #15260."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "format": "json",
            # NOTE: do NOT add "think": False here — Ollama bug #15260 silently drops format.
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def chat_freeform(
        self,
        system: str,
        messages: List[Dict[str, str]],
        timeout: float = 60.0,
    ) -> str:
        """Free-form path. No format constraint. Thinking allowed."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def vlm_describe(self, system: str, prompt: str, image_path: Path, timeout: float = 60.0) -> str:
        """VLM caption. Free-form path (image describe is always thinking-allowed)."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            "stream": False,
            "think": False,  # VLM path — safe because we don't use format:json here.
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
```

- [ ] **Step 4: Create the llama.cpp HTTP backend**

```python
# src/voice_control/brain/llamacpp_backend.py
"""llama.cpp HTTP backend — talks to llama-server on port 11435.
Per-call GBNF for action path; no grammar for free-form path.
"""
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests

LLAMACPP_URL = "http://127.0.0.1:11435"
ACTION_GBNF_PATH = Path(__file__).resolve().parents[1] / "action.gbnf"


def _read_grammar_text() -> str:
    return ACTION_GBNF_PATH.read_text(encoding="utf-8")


_GRAMMAR_CACHE: Optional[str] = None


def _get_action_grammar() -> str:
    global _GRAMMAR_CACHE
    if _GRAMMAR_CACHE is None:
        _GRAMMAR_CACHE = _read_grammar_text()
    return _GRAMMAR_CACHE


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, url: str = LLAMACPP_URL):
        self.url = url

    def _chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        timeout: float,
        grammar: Optional[str],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload: Dict = {
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        if grammar is not None:
            payload["grammar"] = grammar
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, messages: List[Dict[str, str]], timeout: float = 30.0) -> str:
        """Action path: GBNF-constrained JSON, no thinking allowed by grammar."""
        return self._chat(system, messages, timeout, grammar=_get_action_grammar())

    def chat_freeform(self, system: str, messages: List[Dict[str, str]], timeout: float = 60.0) -> str:
        """Free-form path: no grammar. Thinking allowed."""
        return self._chat(system, messages, timeout, grammar=None, temperature=0.7, max_tokens=400)

    def vlm_describe(self, system: str, prompt: str, image_path: Path, timeout: float = 60.0) -> str:
        """VLM caption. Free-form, image attached via OpenAI vision content shape."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            "temperature": 0.5,
            "max_tokens": 400,
            "cache_prompt": True,
        }
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 5: Rewire llm_brain.py to dispatch via backend**

Open `src/voice_control/llm_brain.py`. Locate `process()` and `query_vlm()`. Replace the direct `requests.post(OLLAMA_URL ...)` calls with:

```python
from src.voice_control.brain import get_backend, classify_intent, IntentKind

_BACKEND = None
def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = get_backend()
    return _BACKEND

# Inside process():
intent = classify_intent(user_transcript)
backend = _backend()
if intent is IntentKind.ACTION:
    raw = backend.chat_json(SYSTEM_PROMPT, messages, timeout=REQUEST_TIMEOUT)
    # parse as JSON, normal dispatch path
else:
    raw = backend.chat_freeform(SYSTEM_PROMPT_FREEFORM, messages, timeout=VLM_TIMEOUT)
    # build a synthetic JSON action=none response with the freeform text in "response"
    raw = json.dumps({"actions": [], "response": raw.strip()})

# Inside query_vlm():
caption = backend.vlm_describe(VLM_SYSTEM_PROMPT, prompt, image_path, timeout=VLM_TIMEOUT)
```

Keep the existing `[Brain-timing]` log lines around the backend calls (so latency telemetry continues to work across both backends). Add `backend.name` and `intent.value` to the log lines for backend-aware analysis:

```python
print(f"[Brain-timing] path={intent.value} backend={backend.name} model={...} total={elapsed_ms}ms ...")
```

Add a `SYSTEM_PROMPT_FREEFORM` constant near the existing `SYSTEM_PROMPT` (line 39-110), with a shorter persona-only system prompt and NO action catalog:

```python
SYSTEM_PROMPT_FREEFORM = """\
You are Spot, a Boston Dynamics quadruped robot at Dartmouth College. \
You are a friendly, helpful general assistant. You are self-aware: you know \
you are a four-legged robot. You have a sense of humor. \
Answer concisely and naturally. Do not output JSON — speak in plain English.
"""
```

- [ ] **Step 6: Add SPOT_BRAIN_BACKEND mention to llm_brain.py module docstring**

Replace the module docstring (lines 1-15) with:

```python
"""LLM Brain for Spot — conversational robot control with adaptive grammar.

Dispatches to one of two runtime backends via SPOT_BRAIN_BACKEND env var:
    llamacpp (default) — llama.cpp llama-server on port 11435, per-call GBNF
                         for action JSON, free-form path for descriptions.
                         Bypasses Ollama bug #15260 entirely.
    ollama — Ollama 0.24.0 on port 11434, format:json with forced thinking
             (rollback path).

Intent classifier in brain/router.py decides per-call whether to send the
action grammar (fast, no thinking) or no grammar (thinking allowed, ~3-6s).

Models (Stage 2A default — gemma4 GGUF on SSD):
    /mnt/ssd/llamacpp-models/gemma-4-e4b-Q4_K_M.gguf
    /mnt/ssd/llamacpp-models/mmproj-BF16.gguf
    /mnt/ssd/llamacpp-models/gemma-4-e2b-Q4_0.gguf (drafter)
"""
```

- [ ] **Step 7: Test both backends manually (Ollama still running from Stage 1.5)**

```bash
# Default backend = llamacpp (requires Task 4 systemd unit live)
spot-env/bin/python -c "
from src.voice_control.brain import get_backend, classify_intent, IntentKind
b = get_backend()
print('backend:', b.name)
print('intent stand:', classify_intent('stand up').value)
print('intent describe:', classify_intent('what do you see').value)
print()
print('--- action call ---')
print(b.chat_json('You are Spot. Output JSON only.', [{'role':'user','content':'stand up'}], timeout=10))
print()
print('--- freeform call ---')
print(b.chat_freeform('You are Spot. Be friendly.', [{'role':'user','content':'hi who are you'}], timeout=20)[:200])
"
```

Expected:
- backend prints `llamacpp`
- intent prints `action` and `freeform`
- action call returns valid JSON `{"actions": [{"action":"stand"}], "response": "..."}`
- freeform call returns natural-language sentences

```bash
# Rollback test
SPOT_BRAIN_BACKEND=ollama spot-env/bin/python -c "
from src.voice_control.brain import get_backend
b = get_backend()
print('backend:', b.name)
print(b.chat_json('You are Spot. Output JSON only.', [{'role':'user','content':'sit down'}]))
"
```

Expected: backend prints `ollama`, returns JSON with action=`sit` (may take 5+ seconds due to thinking).

- [ ] **Step 8: Commit**

```bash
git add src/voice_control/brain/ src/voice_control/llm_brain.py
git commit -m "stage 2a: brain backend interface (llamacpp + ollama) with intent router"
git push origin tour_guide_upgrade_matteo
```

---

## Task 6 — Audio corpus capture

**Files:**
- Create: `tests/audio/` directory with WAV files

This task is **manual recording**. The corpus is needed in 2A.3 mic-verify and 2A.4 wake-word eval.

- [ ] **Step 1: Set up recording**

```bash
mkdir -p tests/audio
arecord -l   # confirm capture device exists; note card/device numbers
```

- [ ] **Step 2: Record 10-15 clean utterances (16 kHz mono PCM16)**

Run this command for each utterance (Ctrl-C to stop, ~2-3 s each):

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 tests/audio/<NAME>.wav
```

Required utterance set (1 WAV each):
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

- [ ] **Step 3: Record 5 motor-idle clips (Spot powered + breathing, NO speech, ~5s each)**

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 tests/audio/idle_01.wav
# repeat for idle_02.wav .. idle_05.wav
```

- [ ] **Step 4: Verify files**

```bash
ls -lh tests/audio/
soxi tests/audio/stand_up.wav 2>/dev/null || file tests/audio/stand_up.wav
```

Expected: 16 kHz, 1 channel, signed 16-bit PCM, sub-100 KB each.

- [ ] **Step 5: Check total size before committing**

```bash
du -sh tests/audio/
```

If < 5 MB total, commit to git. If > 5 MB, move to SSD and gitignore:

```bash
# small case
git add tests/audio/
git commit -m "stage 2a: audio corpus (15 clean utterances + 5 motor-idle clips)"

# large case
mkdir -p /mnt/ssd/spot-logs/test-audio
mv tests/audio/*.wav /mnt/ssd/spot-logs/test-audio/
ln -sfn /mnt/ssd/spot-logs/test-audio tests/audio
echo "tests/audio" >> .gitignore
git add .gitignore
git commit -m "stage 2a: gitignore tests/audio (corpus on SSD)"
```

- [ ] **Step 6: Push**

```bash
git push origin tour_guide_upgrade_matteo
```

---

## Task 7 — Parakeet ASR Backend (Phase A)

**Files:**
- Create: `scripts/setup_parakeet.py`
- Create: `src/voice_control/asr/__init__.py`
- Create: `src/voice_control/asr/parakeet.py`
- Create: `src/voice_control/asr/riva_backend.py`
- Modify: `src/voice_control/server.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Pin new deps in requirements.txt**

Open `requirements.txt`. After the existing `sherpa-onnx>=1.12.0` line, add:

```
# ASR — Parakeet TDT 0.6B v3 via onnx-asr (uses onnxruntime-gpu CUDA EP).
# Install with --no-deps so it doesn't rewrite onnxruntime-gpu / numpy pins:
#     pip install --no-deps onnx-asr==0.11.0
onnx-asr==0.11.0
# huggingface-hub bumped + upper-bounded; needed for Parakeet model download.
huggingface-hub>=1.0,<2.0
```

- [ ] **Step 2: Install deps with --no-deps**

```bash
spot-env/bin/pip install --no-deps onnx-asr==0.11.0
spot-env/bin/pip install --no-deps 'huggingface-hub>=1.0,<2.0'
spot-env/bin/pip check
```

Expected: `pip check` reports no broken requirements. If it does, stop and run the pin-guardian agent.

- [ ] **Step 3: Write the Parakeet model-download script**

```python
#!/usr/bin/env python3
"""Download Parakeet TDT 0.6B v3 ONNX model to /mnt/ssd/parakeet-models/."""
import sys
from pathlib import Path
import onnx_asr

MODELS_DIR = Path("/mnt/ssd/parakeet-models")
MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pull] {MODEL_ID} -> {MODELS_DIR}")
    # onnx-asr caches under MODELS_DIR if HF_HOME points here
    import os
    os.environ["HF_HOME"] = str(MODELS_DIR)
    model = onnx_asr.load_model(MODEL_ID, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("loaded:", model)
    print()
    print(f"=== {MODELS_DIR} contents ===")
    for p in sorted(MODELS_DIR.rglob("*.onnx")):
        print(f"  {p.relative_to(MODELS_DIR)}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the download**

```bash
spot-env/bin/python scripts/setup_parakeet.py
```

Expected: model files under `/mnt/ssd/parakeet-models/`. Total ~600-700 MB.

- [ ] **Step 5: Create the ASR backend registry**

```python
# src/voice_control/asr/__init__.py
"""ASR backends. Selected via SPOT_ASR_BACKEND env var (default: parakeet)."""
import os

_DEFAULT = "parakeet"


def get_backend():
    name = os.environ.get("SPOT_ASR_BACKEND", _DEFAULT).lower()
    if name == "parakeet":
        from .parakeet import ParakeetBackend
        return ParakeetBackend()
    if name == "riva":
        from .riva_backend import RivaBackend
        return RivaBackend()
    raise ValueError(f"Unknown SPOT_ASR_BACKEND={name!r}; valid: parakeet|riva")
```

- [ ] **Step 6: Extract the existing Riva path into riva_backend.py**

Copy lines from `src/voice_control/server.py` covering Riva initialization + `transcribe()` into a new file:

```python
# src/voice_control/asr/riva_backend.py
"""Riva ASR backend — kept as Stage 1 rollback only. Riva server itself is gone
(removed in Stage 1); this code path requires re-acquisition per stage1-rollback.md
Layer 4. DELETE AFTER 2A.4 confirms Parakeet is stable.
"""
import os
from collections import Counter

import numpy as np
import riva.client  # type: ignore[import-not-found]

SAMPLE_RATE = 16000
RIVA_URI = os.environ.get("RIVA_URI", "localhost:50051")
MIN_AUDIO_DURATION = 0.15

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


class RivaBackend:
    name = "riva"

    def __init__(self, uri: str = RIVA_URI):
        print(f"[Riva] Connecting to {uri}")
        self.auth = riva.client.Auth(uri=uri)
        self.asr_service = riva.client.ASRService(self.auth)
        self.config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        warmup = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()
        self.asr_service.offline_recognize(warmup, self.config)
        print("[Riva] ready")

    def transcribe(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) / 2 / SAMPLE_RATE < MIN_AUDIO_DURATION:
            return ""
        resp = self.asr_service.offline_recognize(pcm_bytes, self.config)
        if not resp.results:
            return ""
        text = resp.results[0].alternatives[0].transcript.strip()
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
```

- [ ] **Step 7: Write the Parakeet backend**

```python
# src/voice_control/asr/parakeet.py
"""Parakeet TDT 0.6B v3 ASR backend via onnx-asr (CUDA EP).
Drop-in replacement for the existing Riva backend behind the gRPC server.
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnx_asr

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.15
PARAKEET_MODEL_ID = os.environ.get("PARAKEET_MODEL_ID", "nemo-parakeet-tdt-0.6b-v3")
PARAKEET_HF_HOME = os.environ.get("PARAKEET_HF_HOME", "/mnt/ssd/parakeet-models")

# Reuse the same hallucination guards as the Riva backend.
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
        # Warmup — first call compiles ONNX graphs on GPU
        print("[Parakeet] warming up (1 s silence)")
        warmup = np.zeros(SAMPLE_RATE, dtype=np.int16)
        self.model.recognize(warmup, sample_rate=SAMPLE_RATE)
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

- [ ] **Step 8: Refactor server.py to use the Backend interface**

Replace the body of `src/voice_control/server.py` (keep gRPC service shell, swap the ASR core):

```python
"""ASR gRPC bridge — dispatches to a configurable Backend.

Backends:
    SPOT_ASR_BACKEND=parakeet (default) — NeMo Parakeet TDT 0.6B v3 (GPU)
    SPOT_ASR_BACKEND=riva — NVIDIA Riva (Stage 1 rollback, code-only — Riva server is gone)

The custom asr.proto (port 50055) is unchanged; client_mic.py uses it as-is.
"""
import os
from concurrent import futures
import grpc
import numpy as np

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

- [ ] **Step 9: Smoke-test Parakeet end-to-end**

```bash
# Start server (default parakeet backend) in one terminal
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server &
SERVER_PID=$!
sleep 8   # let Parakeet warm up

# In another shell, transcribe a recorded WAV via the gRPC client path:
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

Expected: transcript like `"Stand up."` printed.

- [ ] **Step 10: Commit Phase A**

```bash
git add requirements.txt scripts/setup_parakeet.py src/voice_control/asr/ src/voice_control/server.py
git commit -m "stage 2a phase A: Parakeet backend + Riva-rollback split behind SPOT_ASR_BACKEND"
git push origin tour_guide_upgrade_matteo
```

---

## Task 8 — Silero VAD swap (Phase B)

**Files:**
- Modify: `src/voice_control/client_mic.py` (replace webrtcvad calls)

`webrtcvad` is 2012-era and misclassifies pure noise. Silero VAD is bundled in `sherpa-onnx`, ~2 MB ONNX, sub-millisecond per frame on Orin CPU.

- [ ] **Step 1: Find current webrtcvad usages**

```bash
grep -nE 'webrtcvad|Vad\(|is_speech' src/voice_control/client_mic.py
```

Note the line numbers — likely VAD init + per-frame `vad.is_speech(frame, rate)` calls inside the recording loop.

- [ ] **Step 2: Verify sherpa-onnx VAD is available**

```bash
spot-env/bin/python -c "
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig
print('sherpa-onnx Silero VAD: OK')
"
```

If `VoiceActivityDetector` is missing, bump sherpa-onnx: `spot-env/bin/pip install --no-deps -U 'sherpa-onnx>=1.12.0'`.

- [ ] **Step 3: Download Silero VAD ONNX (~2 MB)**

```bash
mkdir -p /mnt/ssd/vad-models
cd /mnt/ssd/vad-models
wget -nc https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
ls -lh silero_vad.onnx
```

Expected: ~2 MB file.

- [ ] **Step 4: Replace VAD initialization in client_mic.py**

Find the `import webrtcvad` line and the `vad = webrtcvad.Vad(...)` call. Replace with:

```python
# Top of file (remove the webrtcvad import)
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig

# Where vad was initialized:
SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512   # Silero recommends 512 samples @ 16kHz (32 ms)

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

- [ ] **Step 5: Replace per-frame VAD calls**

Find every `vad.is_speech(frame_bytes, SAMPLE_RATE)` call. Silero VAD operates on float32 samples in a sliding window. Replace with:

```python
import numpy as np

def _vad_is_speech(vad: VoiceActivityDetector, pcm16_bytes: bytes) -> bool:
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    vad.accept_waveform(samples)
    # Silero emits boolean speech state on each window; query current speech flag.
    return vad.is_speech_detected()
```

Replace `vad.is_speech(frame, SAMPLE_RATE)` with `_vad_is_speech(vad, frame)` everywhere. Frame size must be `VAD_WINDOW_SAMPLES * 2` bytes (32 ms @ 16 kHz @ int16).

If the existing code chunks at 10/20/30 ms (webrtcvad sizes), bump to 32 ms (512 samples → 1024 bytes).

- [ ] **Step 6: Remove webrtcvad from requirements.txt**

Open `requirements.txt`. Delete the line `webrtcvad>=2.0.10`.

- [ ] **Step 7: Smoke test VAD on a known clip**

```bash
spot-env/bin/python - <<'EOF'
import wave, numpy as np
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig
cfg = VadModelConfig(silero_vad=SileroVadModelConfig(
    model="/mnt/ssd/vad-models/silero_vad.onnx", threshold=0.5,
    min_silence_duration=0.5, min_speech_duration=0.2, window_size=512),
    sample_rate=16000, debug=False)
vad = VoiceActivityDetector(cfg, buffer_size_in_seconds=10)
with wave.open("tests/audio/stand_up.wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
vad.accept_waveform(samples)
# Drain detected speech segments
segs = []
while not vad.empty():
    segs.append(vad.front)
    vad.pop()
print(f"speech segments: {len(segs)}")
for s in segs:
    print(f"  start={s.start/16000:.2f}s  dur={len(s.samples)/16000:.2f}s")
EOF
```

Expected: ≥1 speech segment detected on `stand_up.wav`.

```bash
spot-env/bin/python - <<'EOF'
# Idle clip: NO speech expected
import wave, numpy as np
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig
cfg = VadModelConfig(silero_vad=SileroVadModelConfig(
    model="/mnt/ssd/vad-models/silero_vad.onnx", threshold=0.5,
    min_silence_duration=0.5, min_speech_duration=0.2, window_size=512),
    sample_rate=16000, debug=False)
vad = VoiceActivityDetector(cfg, buffer_size_in_seconds=10)
with wave.open("tests/audio/idle_01.wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
vad.accept_waveform(samples)
n = 0
while not vad.empty():
    n += 1
    vad.pop()
print(f"speech segments on idle: {n} (expected 0)")
EOF
```

Expected: 0 segments.

- [ ] **Step 8: Commit Phase B**

```bash
git add src/voice_control/client_mic.py requirements.txt
git commit -m "stage 2a phase B: Silero VAD via sherpa-onnx, drops webrtcvad"
git push origin tour_guide_upgrade_matteo
```

---

## Task 9 — Promote Parakeet as default (Phase C)

**Files:**
- Modify: `scripts/run_voice_control.py` (set default env or document the env flip)

- [ ] **Step 1: Confirm Parakeet is default in asr/__init__.py**

```bash
grep -n '_DEFAULT' src/voice_control/asr/__init__.py
```

Expected: `_DEFAULT = "parakeet"` — already set in Task 7.

- [ ] **Step 2: Document the env flip in run_voice_control.py**

Add a comment block near the top of `scripts/run_voice_control.py`:

```python
# Stage 2A Phase C: Parakeet is the default ASR backend.
# Rollback to Riva (code-only; server itself is gone):
#     SPOT_ASR_BACKEND=riva python scripts/run_voice_control.py
# See docs/superpowers/specs/2026-05-15-stage2-design.md §2A.2 for details.
```

- [ ] **Step 3: Capture WER notes vs Riva baseline**

This is a manual evaluation. With no Riva server live, you can't do A/B in real time — instead, run Parakeet over the audio corpus and record observations.

```bash
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server &
SERVER_PID=$!
sleep 8

for wav in tests/audio/*.wav; do
  echo -n "$(basename $wav): "
  spot-env/bin/python - <<EOF
import grpc, wave
from src.voice_control import asr_pb2 as pb, asr_pb2_grpc as pbg
with wave.open("$wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
chan = grpc.insecure_channel("localhost:50055")
stub = pbg.ASRStub(chan)
def gen():
    yield pb.StreamingRequest(config=pb.StreamingConfig(language_code="en-US",
        sample_rate_hz=16000, enable_punctuation=True))
    yield pb.StreamingRequest(audio=pb.AudioChunk(pcm16=pcm))
for resp in stub.StreamingRecognize(gen()):
    print(repr(resp.transcript))
EOF
done | tee /tmp/parakeet_wer_eval.txt
kill $SERVER_PID
```

- [ ] **Step 4: Verify motor-idle hallucination count**

```bash
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server &
SERVER_PID=$!
sleep 8

for wav in tests/audio/idle_*.wav; do
  echo -n "$(basename $wav): "
  spot-env/bin/python - <<EOF
import grpc, wave
from src.voice_control import asr_pb2 as pb, asr_pb2_grpc as pbg
with wave.open("$wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
chan = grpc.insecure_channel("localhost:50055")
stub = pbg.ASRStub(chan)
def gen():
    yield pb.StreamingRequest(config=pb.StreamingConfig(language_code="en-US",
        sample_rate_hz=16000, enable_punctuation=True))
    yield pb.StreamingRequest(audio=pb.AudioChunk(pcm16=pcm))
for resp in stub.StreamingRecognize(gen()):
    print(repr(resp.transcript))
EOF
done | tee /tmp/parakeet_idle_hallucination.txt
kill $SERVER_PID
```

Expected: all transcripts empty `''`. Any non-empty output = a hallucination escape past the blocklist; add to `HALLUCINATION_PHRASES` in `parakeet.py` and re-test.

- [ ] **Step 5: Commit observations as a memory + push**

```bash
git add scripts/run_voice_control.py
git commit -m "stage 2a phase C: Parakeet default; eval notes captured in /tmp"
git push origin tour_guide_upgrade_matteo
```

Save the two `/tmp/` files contents into the conversations.jsonl-equivalent if useful; they don't need to be committed.

---

## Task 10 — Mic-verify gemma4 via llama.cpp + Parakeet

**Files:**
- Create: `tests/utterances/action_corpus.yaml`
- Create: `scripts/mic_verify.py`

- [ ] **Step 1: Define the expected-tuples corpus**

```yaml
# tests/utterances/action_corpus.yaml
# 15-20 representative tour-guide utterances + expected dispatcher outcomes.
# Used by scripts/mic_verify.py to gate qwen pair removal.

utterances:
  - wav: stand_up.wav
    intent: action
    action: stand
    params: {}

  - wav: sit_down.wav
    intent: action
    action: sit
    params: {}

  - wav: walk_forward_1m.wav
    intent: action
    action: walk
    params: {direction: forward, distance: 1.0}

  - wav: walk_forward_two_meters.wav
    intent: action
    action: walk
    params: {direction: forward, distance: 2.0}

  - wav: turn_left_90.wav
    intent: action
    action: turn
    params: {dir: left, deg: 90}

  - wav: turn_around.wav
    intent: action
    action: turn
    params: {deg: 180}

  - wav: go_to_the_kitchen.wav
    intent: action
    action: go_to
    params: {location: kitchen}

  - wav: come_here.wav
    intent: action
    action: come_back  # come_back action = return to last position
    params: {}

  - wav: follow_me.wav
    intent: action
    action: follow_me
    params: {}

  - wav: stop.wav
    intent: action
    action: stop
    params: {}

  - wav: battery_status.wav
    intent: action
    action: battery_status
    params: {}

  - wav: what_do_you_see.wav
    intent: freeform   # routed to free-form path; no JSON action required
    action: null
    params: null

  - wav: describe_surroundings.wav
    intent: freeform
    action: null
    params: null

  - wav: tell_me_about_dartmouth_hall.wav
    intent: freeform
    action: null
    params: null

  - wav: find_a_chair.wav
    intent: action
    action: go_to_object
    params: {description: chair}
```

- [ ] **Step 2: Write the verifier script**

```python
#!/usr/bin/env python3
"""Mic-verify the Stage 2A voice loop end-to-end.

Replays tests/audio/*.wav through the live Parakeet ASR server + the llm_brain
backend (default: llamacpp). Compares the dispatcher output to expected tuples
in tests/utterances/action_corpus.yaml.

Reports:
- ASR transcript per utterance
- Router intent classification (action vs freeform)
- Action / params extracted from brain response
- Diffs vs expected
- Latency p50/p95 per route
- Final pass/fail per Stage 2A.3 gates

Run with:
    spot-env/bin/python scripts/mic_verify.py
"""
from __future__ import annotations
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import grpc
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.voice_control import asr_pb2 as pb, asr_pb2_grpc as pbg
from src.voice_control.brain import get_backend, classify_intent, IntentKind

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


def main() -> int:
    corpus = yaml.safe_load(CORPUS_YAML.read_text())["utterances"]
    backend = get_backend()
    print(f"=== mic-verify  backend={backend.name}  corpus={len(corpus)} ===\n")

    results = []
    action_latencies, freeform_latencies = [], []
    router_correct = 0

    for entry in corpus:
        wav = AUDIO_DIR / entry["wav"]
        expected_intent = entry["intent"]
        expected_action = entry["action"]
        expected_params = entry.get("params") or {}

        # Stage 1: ASR
        transcript = transcribe_via_grpc(wav)

        # Stage 2: route
        intent = classify_intent(transcript)
        intent_str = intent.value
        if intent_str == expected_intent:
            router_correct += 1

        # Stage 3: brain
        t0 = time.time()
        if intent is IntentKind.ACTION:
            raw = backend.chat_json("You are Spot.", [{"role": "user", "content": transcript}], timeout=15)
        else:
            raw = backend.chat_freeform("You are Spot.", [{"role": "user", "content": transcript}], timeout=30)
        elapsed = time.time() - t0
        (action_latencies if intent is IntentKind.ACTION else freeform_latencies).append(elapsed)

        # Stage 4: parse + diff
        action_match = params_match = None
        json_parses = False
        if intent is IntentKind.ACTION:
            try:
                parsed = json.loads(raw)
                json_parses = True
                actions = parsed.get("actions", [])
                if actions:
                    actual_action = actions[0].get("action")
                    actual_params = actions[0].get("params", {})
                    action_match = (actual_action == expected_action)
                    if expected_params:
                        params_match = all(
                            actual_params.get(k) == v for k, v in expected_params.items()
                        )
                    else:
                        params_match = True
                else:
                    action_match = (expected_action is None)
                    params_match = True
            except json.JSONDecodeError:
                json_parses = False
        else:
            json_parses = True  # not applicable
            action_match = True  # free-form is qualitative
            params_match = True

        results.append({
            "wav": entry["wav"],
            "transcript": transcript,
            "expected_intent": expected_intent,
            "actual_intent": intent_str,
            "router_ok": intent_str == expected_intent,
            "expected_action": expected_action,
            "actual_raw": raw[:120] + ("..." if len(raw) > 120 else ""),
            "json_parses": json_parses,
            "action_match": action_match,
            "params_match": params_match,
            "latency_s": round(elapsed, 2),
        })

    # Report
    n = len(corpus)
    action_results = [r for r in results if r["expected_intent"] == "action"]
    freeform_results = [r for r in results if r["expected_intent"] == "freeform"]

    print("=== per-utterance ===")
    for r in results:
        flag = "OK" if (r["router_ok"] and r["json_parses"] and r["action_match"] and r["params_match"]) else "FAIL"
        print(f"  [{flag}] {r['wav']:40s} route={r['actual_intent']:8s} "
              f"act={r.get('expected_action')!s:20s}  lat={r['latency_s']}s")
        if flag == "FAIL":
            print(f"          transcript={r['transcript']!r}")
            print(f"          raw={r['actual_raw']!r}")

    print("\n=== aggregate ===")
    print(f"  router accuracy: {router_correct}/{n} = {router_correct/n*100:.0f}%")
    action_ok = sum(1 for r in action_results if r["action_match"] and r["json_parses"])
    print(f"  action accuracy: {action_ok}/{len(action_results)}")
    if action_latencies:
        p50 = statistics.median(action_latencies)
        p95 = sorted(action_latencies)[int(len(action_latencies)*0.95)] if len(action_latencies) > 1 else action_latencies[0]
        print(f"  action latency p50={p50:.2f}s  p95={p95:.2f}s")
    if freeform_latencies:
        p50 = statistics.median(freeform_latencies)
        p95 = sorted(freeform_latencies)[int(len(freeform_latencies)*0.95)] if len(freeform_latencies) > 1 else freeform_latencies[0]
        print(f"  freeform latency p50={p50:.2f}s  p95={p95:.2f}s")

    # Pass gates (Stage 2A.3)
    gates = {
        "router_95pct": router_correct / n >= 0.95,
        "action_90pct": action_ok / max(1, len(action_results)) >= 0.90,
        "action_p50_lt_2s": (statistics.median(action_latencies) < 2.0) if action_latencies else False,
        "action_p95_lt_3_5s": (sorted(action_latencies)[int(len(action_latencies)*0.95)] < 3.5) if len(action_latencies) > 1 else False,
        "freeform_p95_lt_6s": (sorted(freeform_latencies)[int(len(freeform_latencies)*0.95)] < 6.0) if len(freeform_latencies) > 1 else False,
    }
    print("\n=== gates ===")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Pre-flight — both servers live**

```bash
# llama-server must be up (Task 4)
curl -fs http://127.0.0.1:11435/health || { echo "llama-server DOWN"; exit 1; }

# Parakeet ASR server must be up
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server &
ASR_PID=$!
sleep 8
nc -z localhost 50055 || { echo "ASR server DOWN"; kill $ASR_PID; exit 1; }
```

- [ ] **Step 4: Run mic-verify**

```bash
spot-env/bin/python scripts/mic_verify.py | tee /tmp/mic_verify_$(date +%Y%m%d_%H%M%S).log
echo "Exit: $?"
```

Expected: exit 0 + all gates PASS. Save the log.

If FAIL: follow the cascade in spec §2A.3:
1. Tighten/loosen GBNF
2. Tighten router regex in `brain/router.py`
3. Bump quant (re-pull Q6_K from `unsloth/gemma-4-e4b-it-GGUF`, update `setup_llamacpp_models.py`, re-export systemd unit)
4. Rollback to Ollama: `SPOT_BRAIN_BACKEND=ollama spot-env/bin/python scripts/mic_verify.py`
5. Rollback to qwen pair: edit `llm_brain.py` `DEFAULT_MODEL` + restart

```bash
kill $ASR_PID
```

- [ ] **Step 5: On pass — reclaim the qwen pair**

```bash
ollama list  # confirm gemma4 + qwen pair present
ollama rm qwen2.5:7b
ollama rm qwen2.5vl:7b
ollama list
df -h /mnt/ssd
```

Expected: ~10.7 GB freed on `/mnt/ssd/ollama-models/`.

- [ ] **Step 6: Update stage1-rollback.md to mark qwen layer dead**

Open `docs/project/stage1-rollback.md` and add at end of "Layer 2 — Brain model rollback" section:

```markdown
> **Stage 2A.3 update (mic-verify pass):** The qwen2.5:7b + qwen2.5vl:7b pair has been removed via `ollama rm`. This layer is no longer available without re-downloading the pair (~10.7 GB). The new rollback ladder is `SPOT_BRAIN_BACKEND=ollama` to the gemma4 Ollama path, and the new `stage2-rollback.md` for full Stage 2 rollback. See `docs/project/stage2-rollback.md`.
```

- [ ] **Step 7: Commit + push**

```bash
git add tests/utterances/action_corpus.yaml scripts/mic_verify.py docs/project/stage1-rollback.md
git commit -m "stage 2a.3: mic-verify script + action corpus + qwen pair removal"
git push origin tour_guide_upgrade_matteo
```

---

## Task 11 — Delete RivaBackend (Stage 2A.4 cleanup)

**Files:**
- Delete: `src/voice_control/asr/riva_backend.py`
- Modify: `src/voice_control/asr/__init__.py`
- Modify: `requirements.txt` (verify Riva client is gone)

- [ ] **Step 1: Confirm Parakeet has been stable for ≥1 full mic-verify run (Task 10 passed)**

If Task 10 was rolled back to Riva, do NOT delete riva_backend.py. Stop here.

- [ ] **Step 2: Delete riva_backend.py + drop from registry**

```bash
git rm src/voice_control/asr/riva_backend.py
```

Open `src/voice_control/asr/__init__.py`. Remove the `if name == "riva":` branch. Result:

```python
import os

_DEFAULT = "parakeet"


def get_backend():
    name = os.environ.get("SPOT_ASR_BACKEND", _DEFAULT).lower()
    if name == "parakeet":
        from .parakeet import ParakeetBackend
        return ParakeetBackend()
    raise ValueError(f"Unknown SPOT_ASR_BACKEND={name!r}; valid: parakeet (riva removed in stage 2a.4)")
```

- [ ] **Step 3: Verify nvidia-riva-client is gone from requirements.txt**

```bash
grep -nE 'riva' requirements.txt && echo "riva line(s) present; remove them" || echo "clean"
```

If present, delete the line(s).

- [ ] **Step 4: Verify imports still work**

```bash
spot-env/bin/python -c "from src.voice_control.asr import get_backend; print(get_backend().name)"
```

Expected: prints `parakeet`. (Should NOT print riva. Should NOT ImportError.)

- [ ] **Step 5: Commit + push**

```bash
git add src/voice_control/asr/__init__.py requirements.txt
git commit -m "stage 2a.4: delete riva_backend.py (parakeet stable per mic-verify)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 12 — LiveKit Wakeword swap

**Files:**
- Create: `src/voice_control/wake_word_livekit.py`
- Modify: `src/voice_control/wake_word.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Locate the LiveKit Wakeword loader**

Check current install/release status:

```bash
spot-env/bin/pip search livekit-wakeword 2>/dev/null || true
spot-env/bin/pip install --no-deps livekit-wakeword 2>&1 | tail -3
```

If PyPI install fails (package not yet on PyPI), clone + install from source:

```bash
git clone https://github.com/livekit/livekit-wakeword /mnt/ssd/livekit-wakeword
cd /mnt/ssd/livekit-wakeword
spot-env/bin/pip install --no-deps -e .
```

- [ ] **Step 2: Pull (or train) a "Hey Spot" model**

If LiveKit publishes a "hey_spot.onnx" on their model hub, download:

```bash
mkdir -p /mnt/ssd/wake-models
cd /mnt/ssd/wake-models
# Replace with actual URL from LiveKit's model registry once known
wget https://huggingface.co/livekit/wakeword-en-hey-spot/resolve/main/hey_spot.onnx
ls -lh hey_spot.onnx
```

If LiveKit does NOT publish "Hey Spot", train one using their Piper synthetic-positives pipeline per their README. Expect ~30-60 min on AGX Orin CPU.

- [ ] **Step 3: Write the LiveKit backend wrapper**

```python
# src/voice_control/wake_word_livekit.py
"""LiveKit Wakeword backend — ONNX, drop-in alternative to sherpa-onnx KWS.
Selected via SPOT_WAKE_BACKEND=livekit.
"""
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

LIVEKIT_MODEL = Path(os.environ.get("LIVEKIT_WAKE_MODEL", "/mnt/ssd/wake-models/hey_spot.onnx"))
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 1280   # 80 ms @ 16 kHz — typical for conv-attention wake models
WAKE_THRESHOLD = float(os.environ.get("LIVEKIT_WAKE_THRESHOLD", "0.5"))


class LiveKitWakeWord:
    name = "livekit"

    def __init__(self) -> None:
        if not LIVEKIT_MODEL.exists():
            raise FileNotFoundError(f"LiveKit wake model not found: {LIVEKIT_MODEL}")
        self.session = ort.InferenceSession(
            str(LIVEKIT_MODEL),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)

    def accept_chunk(self, pcm16_bytes: bytes) -> bool:
        """Feed a PCM16 chunk. Returns True if wake word detected this window."""
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, samples])
        detected = False
        while len(self.buffer) >= WINDOW_SAMPLES:
            window = self.buffer[:WINDOW_SAMPLES].reshape(1, -1)
            score = float(self.session.run(None, {self.input_name: window})[0].squeeze())
            if score >= WAKE_THRESHOLD:
                detected = True
            self.buffer = self.buffer[WINDOW_SAMPLES // 2 :]   # 50 % overlap
        return detected
```

- [ ] **Step 4: Add SPOT_WAKE_BACKEND env-var dispatch**

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
        # Existing sherpa-onnx KWS init code (preserve current implementation)
        print("[wake] backend=sherpa_onnx")
        # ... existing init returning a sherpa-onnx KeywordSpotter object
        ...
    raise ValueError(f"Unknown SPOT_WAKE_BACKEND={_WAKE_BACKEND_NAME!r}; valid: livekit|sherpa_onnx")
```

Refactor any direct sherpa-onnx KWS construction in this file into a `_make_wake_detector_sherpa()` helper, and call `_make_wake_detector()` from the public entry point. Existing callers (`client_mic.py`) need ZERO changes if the returned object exposes a compatible `accept_chunk(pcm_bytes) -> bool` interface.

If sherpa-onnx KWS returns a different API, write a small adapter so both backends expose `.accept_chunk(bytes) -> bool` and `.reset()`.

- [ ] **Step 5: Pin LiveKit in requirements.txt**

```
# Wake word — LiveKit Wakeword (Apache 2.0, conv-attention head, drop-in ONNX)
# Install with --no-deps:
#     pip install --no-deps livekit-wakeword
livekit-wakeword
```

- [ ] **Step 6: Smoke test both backends**

```bash
# LiveKit path
SPOT_WAKE_BACKEND=livekit spot-env/bin/python - <<'EOF'
from src.voice_control.wake_word import _make_wake_detector
d = _make_wake_detector()
import wave
with wave.open("tests/audio/idle_01.wav", "rb") as w:
    pcm = w.readframes(w.getnframes())
print("livekit FP test (idle):", d.accept_chunk(pcm))  # expect False
EOF

# Sherpa-onnx rollback path
SPOT_WAKE_BACKEND=sherpa_onnx spot-env/bin/python - <<'EOF'
from src.voice_control.wake_word import _make_wake_detector
d = _make_wake_detector()
print("sherpa-onnx instance:", d)
EOF
```

- [ ] **Step 7: Record a small Baker-Berry-style eval set (manual, optional but recommended)**

```bash
# Record 30 min of background noise (no wake word)
arecord -f S16_LE -r 16000 -c 1 -d 1800 tests/audio/background_baker_berry.wav

# Record 50 deliberate "Hey Spot" utterances spaced out (one per ~5 s)
arecord -f S16_LE -r 16000 -c 1 -d 250 tests/audio/wake_positives.wav
```

If you can't record both, skip — just confirm the loader works.

- [ ] **Step 8: Commit + push**

```bash
git add src/voice_control/wake_word_livekit.py src/voice_control/wake_word.py requirements.txt
git commit -m "stage 2a.4: LiveKit Wakeword backend behind SPOT_WAKE_BACKEND env (sherpa-onnx is rollback)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 13 — Write stage2-rollback.md

**Files:**
- Create: `docs/project/stage2-rollback.md`

- [ ] **Step 1: Write the rollback runbook**

```markdown
# Stage 2 Rollback Runbook

Stage 2A complete state (as of `stage2a-voice-complete` tag):
- Brain runtime: llama.cpp `llama-server` on port 11435 (systemd user unit)
- Brain model: gemma4-e4b-Q4_K_M.gguf + mmproj-BF16.gguf + E2B drafter at /mnt/ssd/llamacpp-models/
- ASR: Parakeet TDT 0.6B v3 at /mnt/ssd/parakeet-models/, served on port 50055
- VAD: Silero via sherpa-onnx
- Wake word: LiveKit "Hey Spot" model at /mnt/ssd/wake-models/hey_spot.onnx
- qwen pair removed (Stage 1 rollback layer dead — see stage1-rollback.md)

## Disk pressure guard

`df -h /mnt/ssd` — must show > 30 GB free. If below threshold, do NOT re-acquire qwen pair before reclaiming elsewhere.

## Layer A — Brain runtime rollback (cheap, < 1 min)

Use if llama-server crashes or gemma4 quality regresses on llama.cpp path.

```
SPOT_BRAIN_BACKEND=ollama python scripts/run_voice_control.py
```

Or set persistently in the systemd unit / shell rc. Ollama 0.24.0 + gemma4:e4b is still installed; latency reverts to ~5 s on JSON path due to bug #15260.

## Layer B — ASR backend rollback (cheap, < 1 min, requires Riva re-acquisition)

Parakeet is the default. Riva backend code was deleted in Task 11 / stage 2a.4. To re-enable:

1. Re-acquire Riva server per `stage1-rollback.md` Layer 4 (~30-60 min, 24 GB download).
2. Restore `src/voice_control/asr/riva_backend.py` from git history (`git show <stage2a-voice-complete>:src/voice_control/asr/riva_backend.py > src/voice_control/asr/riva_backend.py`).
3. Restore the `riva` branch in `src/voice_control/asr/__init__.py`.
4. Add `nvidia-riva-client>=2.17.0` back to `requirements.txt`, `pip install --no-deps nvidia-riva-client`.
5. Flip env: `SPOT_ASR_BACKEND=riva python -m src.voice_control.server`.

## Layer C — Wake-word rollback (instant)

```
SPOT_WAKE_BACKEND=sherpa_onnx python scripts/run_voice_control.py
```

Sherpa-onnx KWS Zipformer is still installed (didn't change in Stage 2A).

## Layer D — Full Stage 2A revert (destructive, requires user approval)

```
git checkout stage1.5-complete
# detached HEAD inspection; or to reset the branch:
git checkout tour_guide_upgrade_matteo
git reset --hard stage1.5-complete
git push origin tour_guide_upgrade_matteo --force-with-lease
```

This reverts llama-server systemd unit (still loaded — stop it: `systemctl --user disable --now llama-server`), Parakeet/Silero/LiveKit code, and re-installs `webrtcvad` + Riva imports (which will fail until Layer B steps complete).

## Layer E — Brain model rollback (Stage 1.5 layer — still applies)

The qwen pair is GONE. The remaining brain rollback is:
1. Bump to a higher gemma4 quant (Q6_K, Q8_0): re-pull via `setup_llamacpp_models.py` with updated filenames, update systemd unit `-m` flag, restart unit.
2. Pull a different brain model entirely (e.g., qwen3-vl:8b-instruct GGUF per Stage 2 design "Things deliberately NOT in Stage 2" §): manual `huggingface_hub.hf_hub_download`, update systemd unit, restart.

## Verification per layer

After any layer flip, run mic-verify:
```
spot-env/bin/python scripts/mic_verify.py
```
```

- [ ] **Step 2: Commit + push**

```bash
git add docs/project/stage2-rollback.md
git commit -m "stage 2a: stage2-rollback.md runbook (5 layers + verification)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 14 — Tag stage2a-voice-complete

- [ ] **Step 1: Confirm all preceding tasks committed + pushed**

```bash
git status
git log --oneline -15
```

Expected: clean working tree, recent commits from Tasks 1-13.

- [ ] **Step 2: Run mic-verify one more time as final smoke**

```bash
curl -fs http://127.0.0.1:11435/health
SPOT_ASR_BACKEND=parakeet spot-env/bin/python -m src.voice_control.server &
ASR_PID=$!
sleep 8
spot-env/bin/python scripts/mic_verify.py
echo "exit=$?"
kill $ASR_PID
```

Expected: exit 0, all gates PASS.

- [ ] **Step 3: Tag + push**

```bash
git tag -a stage2a-voice-complete -m "Stage 2A complete: llama.cpp brain (adaptive thinking) + Parakeet ASR + Silero VAD + LiveKit Wakeword. Mic-verify passes; qwen pair reclaimed."
git push origin stage2a-voice-complete
```

- [ ] **Step 4: Update MEMORY.md with current state**

In `/home/spotdog/.claude/projects/-home-spotdog-spot-dartmouth-spot-capstone/memory/MEMORY.md`, append a one-line pointer under "Linked memories":

```markdown
- [Stage 2A voice pipeline complete](../../...) — llama.cpp + adaptive thinking, Parakeet ASR, Silero VAD, LiveKit Wakeword. Tag: stage2a-voice-complete. qwen pair gone.
```

(The agent reading this plan should write a proper memory file with frontmatter per the auto-memory format, then add this pointer to MEMORY.md.)

---

## Self-Review

**Spec coverage:**
- §2A.0 brain runtime swap → Tasks 1, 2, 3, 4, 5 ✅
- §2A.1 audio corpus → Task 6 ✅
- §2A.2 Phase A (Parakeet backend) → Task 7 ✅
- §2A.2 Phase B (Silero VAD) → Task 8 ✅
- §2A.2 Phase C (Parakeet promotion) → Task 9 ✅
- §2A.3 mic-verify → Task 10 ✅
- §2A.4 RivaBackend deletion → Task 11 ✅
- §2A.4 LiveKit Wakeword swap → Task 12 ✅
- Stage 2 rollback runbook → Task 13 ✅
- Final tag → Task 14 ✅

**Placeholder scan:** Step 4 of Task 8 says "find every `vad.is_speech(frame_bytes, SAMPLE_RATE)` call" — this is a grep instruction, not a placeholder. Code template is complete; implementer fills based on actual call sites. Task 12 Step 4 has `...` in the sherpa-onnx branch — this represents the existing un-modified KWS init code in `wake_word.py`; the implementer must preserve current code, not write new. This is documented intent, not a gap.

**Type consistency:** `IntentKind.ACTION` / `IntentKind.FREEFORM` enum names match across `brain/router.py` (Task 5 Step 2), `brain/__init__.py` (Step 1), `llm_brain.py` (Step 5), `mic_verify.py` (Task 10 Step 2). `Backend.chat_json` / `chat_freeform` / `vlm_describe` method names match across `OllamaBackend` (Task 5 Step 3), `LlamaCppBackend` (Step 4), `llm_brain.py` (Step 5). `ParakeetBackend.transcribe(pcm_bytes)` signature matches `RivaBackend.transcribe()` and `ASRServicer.StreamingRecognize` consumer (Task 7 Steps 6, 7, 8). `accept_chunk(pcm16_bytes) -> bool` method on both `LiveKitWakeWord` and the existing sherpa-onnx adapter (Task 12 Step 4) — implementer must verify adapter exists or write one.

**Env var consistency:** `SPOT_BRAIN_BACKEND` (Tasks 5, 10, 13), `SPOT_ASR_BACKEND` (Tasks 7, 8, 9, 10, 11, 13), `SPOT_WAKE_BACKEND` (Tasks 12, 13). All defaults documented in respective `__init__.py` files.
