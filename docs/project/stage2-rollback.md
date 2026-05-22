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
