# Stage 2 Rollback Runbook

## Stage 2A complete state (tag `stage2a-voice-complete`)

- **Brain runtime:** llama.cpp via NVIDIA Jetson container, `gemma-4-E4B-Q8_0.gguf` + Q8 mmproj on `/mnt/ssd/llamacpp-models/`, served on `127.0.0.1:11435` by `llama-server.service`. Context = 32768 (system prompt is ~62 KB / ~15 K tokens after knowledge packs).
- **Grammar:** single unified GBNF at `src/voice_control/grammar/spot_action.gbnf`. Emits `{"actions":[...],"response":"..."}`; dispatcher bridges `"action"` → `"intent"` field.
- **ASR:** **Parakeet TDT 0.6B v3** (via `onnx-asr`, CUDA EP through our `onnxruntime-gpu 1.23.0` Jetson AI Lab wheel) at `/mnt/ssd/parakeet-models/`. Warm ~190 ms / 6.6 s wav (= 35× realtime). Nemotron Speech Streaming 0.6B int8 at `/mnt/ssd/nemotron-models/` is the **CPU-only** alternate (sherpa-onnx Python wheel has no aarch64 GPU build).
- **VAD:** Silero via sherpa-onnx (`/mnt/ssd/vad-models/silero_vad.onnx`). Replaced webrtcvad (2012-era false-positive prone).
- **Wake word:** sherpa-onnx KeywordSpotter (Zipformer int8 at `models/kws/`). LiveKit-Wakeword infra written but model not yet trained — flip default to `livekit` once `/mnt/ssd/wake-models/hey_spot.onnx` exists.
- **TTS:** Kokoro v1.0 (`af_sarah`, 24 kHz, CUDA EP). Streaming SSE → sentence chunker in `spot_tts.TTSChunker`.

## Disk pressure guard

`df -h /mnt/ssd` — must show > 25 GB free before re-acquiring qwen Stage 1 pair.

## Layer A — Brain runtime rollback (instant)

llama-server crash or Gemma 4 regression:

```bash
SPOT_BRAIN_BACKEND=ollama spot-env/bin/python scripts/run_voice_control.py
```

Ollama 0.27+ with `gemma4:e4b` still installed. Ollama bug #15260 fixed in PR
#15678 (Apr 2026), so format=json + thinking no longer pays the old penalty.

## Layer B — ASR backend rollback (instant)

Parakeet unstable (CUDA EP regression, model fetch issue, hallucinations
escaping the blocklist):

```bash
SPOT_ASR_BACKEND=nemotron spot-env/bin/python -m src.voice_control.server
```

Nemotron Streaming 0.6B int8 on /mnt/ssd/nemotron-models/. CPU only on
aarch64; ~700 ms for a 2 s command vs Parakeet's ~60-190 ms. Functional but
noticeably slower.

## Layer C — Wake word rollback (default in 2A)

sherpa-onnx KWS is the default. If you've trained the LiveKit model and it
false-rejects in some condition (loud room, unfamiliar accent), revert:

```bash
SPOT_WAKE_BACKEND=sherpa_onnx spot-env/bin/python scripts/run_voice_control.py
```

To swap forward to LiveKit once trained:

```bash
SPOT_WAKE_BACKEND=livekit spot-env/bin/python scripts/run_voice_control.py
```

## Layer D — Streaming TTS disable (instant)

If TTS sentence chunker mis-segments or skips text:

```bash
# In src/voice_control/client_mic.py, set `chunker = None` at the brain.process
# guard (around line 1135). Existing post-process tts.speak(response) then
# fires for every utterance (legacy non-streaming path).
```

No env-var toggle exists yet — flag-gate added in Stage 2B if needed.

## Layer E — Full Stage 2A revert (destructive)

```bash
sudo systemctl disable --now llama-server.service
git checkout tour_guide_upgrade_matteo
git reset --hard stage1.5-complete
git push origin tour_guide_upgrade_matteo --force-with-lease
```

Re-installs `webrtcvad` + Riva imports (Riva will fail without setup). Use
only after consultation. Recovers the qwen pair (Stage 1 rollback layer)
which was removed in T14.

## Layer F — Brain model quant bump

Q8_0 is the 2A baseline. To go heavier (e.g., bf16) for max quality:

1. Re-pull via `scripts/setup_llamacpp_models.py` with filename swap.
2. Edit `/etc/systemd/system/llama-server.service` `ExecStart` `-m` path.
3. `sudo systemctl daemon-reload && sudo systemctl restart llama-server`.
4. Re-run mic-verify (`scripts/mic_verify.py`) to confirm.

To go lighter (Q4_K_M or Q6_K) for VRAM headroom, same steps. Q4_K_M is
available in `ggml-org/gemma-4-E4B-it-GGUF`; Q6_K is on third-party repos
(unsloth, lmstudio-community) — see commit 9211fe8 rationale.

## Verification after any rollback

Re-run mic-verify (when implemented in T12):

```bash
spot-env/bin/python scripts/mic_verify.py
```

Save log to /tmp/ and compare gates against the T0 baseline snapshot at
`/tmp/stage2a_preflight_*.json`.

## After 2E.1 — TTS + persona

### Layer: revert TTS backend (cloud → local)

Symptom: ElevenLabs latency spike, API outage, free tier exhausted, network drop.

```bash
export SPOT_TTS_BACKEND=kokoro
# restart voice loop
```

Reverts to local kokoro-onnx v1.0. No code change required.

### Layer: collapse to single persona

Symptom: persona swap action causing dispatch errors or bad behavior.

```bash
export SPOT_PERSONA=tour_guide
```

Brain still loads registry but treats every utterance as default persona. To fully disable runtime swap, comment out the `set_persona` bullet in `SYSTEM_PROMPT` (llm_brain.py) so the LLM stops emitting the action.

### Layer: pin to Kokoro v1.0 directory (default)

Symptom: alternate Kokoro install attempted (e.g. future v1.1) corrupt OR voice quality regression.

```bash
export SPOT_KOKORO_MODEL_DIR=models/tts/kokoro-v1.0
```

`models/tts/kokoro-v1.0/` (54 voices, fp16-gpu ONNX) is the in-repo default; this env var just makes the pin explicit when an override was set.

### Hard rollback (full 2E.1 revert)

```bash
git revert <2e1-merge-commits>
# or, on stage2e1-tts-persona branch:
git reset --hard <pre-2e1-tag>
```

Pre-2E.1 commit hash: `d9468bf` — branch off this point.

## Stage 2E.2 — gripper puppet mouth (tag `stage2e2-complete`)

Mouth animation is **disabled by default** — `MouthDriver` starts `enabled=False` and is only created/enabled by the `enable_mouth` action (which also deploys the arm). So the safe state is the default; nothing to roll back unless the arm misbehaves.

### Layer: instant kill (voice)

Say *"stop moving your mouth"* → `disable_mouth` action: stops the gripper, closes it, stows the arm. Any safety command (`stop` / `freeze` / `estop`) also closes the gripper first via `_close_mouth()`.

### Layer: disable the actions (LLM can't trigger)

Remove `| "\"enable_mouth\"" | "\"disable_mouth\""` from the `action-name` rule in `src/voice_control/grammar/spot_action.gbnf`. The LLM then cannot emit them; TTS + playback are unchanged.

### Layer: full feature off (code)

The TTS→mouth hooks are no-ops whenever `tts.mouth is None`. Since the driver is only created inside `enable_mouth`, simply never enabling it leaves the feature dormant. To hard-disable, comment out the `enable_mouth`/`disable_mouth` branches in `dispatch_intent` (`spot_dispatch.py`).

### Hard rollback (full 2E.2 revert)

```bash
git revert ac9c227 26610db 40bf3a2 b849a44 a00d9e7
```

Touches only the 2E.2 surface (`animation/mouth.py`, `spot_tts.py` play-callback hooks, `spot_dispatch.py` actions, `session.py` arm helpers, `audio_player.py` on_play_end-on-error guard, `session_state.py`, `persona.py`, `personas.yaml`, grammar). Pre-2E.2 commit: `7f875ac`.
