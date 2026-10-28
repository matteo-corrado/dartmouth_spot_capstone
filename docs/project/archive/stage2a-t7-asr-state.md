# Stage 2A T7 ASR — research + bench state (2026-05-22)

Pre-compact snapshot. State of ASR investigation for Stage 2A T7.

## Verdict

Plan 2A model pick (Nemotron Speech Streaming 0.6B) **survives**. Runtime path needs change. Two candidates remain after wave-1+2 cross-validation and on-device bench:

1. **Parakeet TDT v3** via `onnx-asr` + `onnxruntime-gpu 1.23.0` (CUDA EP) — already installed, RTFx **48.82×** measured on Orin GPU. Offline.
2. **Nemotron Speech Streaming 0.6B** via sherpa-onnx **CPU EP** — RTFx **4.31×** measured on Orin CPU n=4. Streaming (true cache-aware Conformer-Transducer).

User has decided to build sherpa-onnx from source for CUDA EP to enable option-2 GPU. Build in progress at time of compact.

## Measured benchmark (same 2 wavs from Nemotron bundle test_wavs/, 23.34s total audio)

| ASR | Runtime | 6.62s wav | 16.71s wav | Total wall | **RTFx** | GPU? |
|---|---|---|---|---|---|---|
| Nemotron Streaming 560ms-int8 | sherpa-onnx CPU n=4 | 1.49 s | 3.92 s | 5.41 s | 4.31× | no |
| Parakeet TDT 0.6B v3 | onnx-asr CPU | 0.71 s | 1.67 s | 2.38 s | 9.81× | no |
| Parakeet TDT 0.6B v3 | onnx-asr **CUDA** | **0.19 s** | **0.29 s** | **0.48 s** | **48.82×** | yes |
| Nemotron Streaming 560ms-int8 | sherpa-onnx "cuda" (fell back to CPU n=2) | 2.17 s | 5.71 s | 7.88 s | 2.96× | no — see below |

Transcripts perfect on all 4 configs.

**Sherpa-onnx CUDA fall-back evidence**: pip-installed `sherpa_onnx 1.13.2` aarch64 wheel printed verbatim:
> `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available providers: CPUExecutionProvider. Fallback to cpu!`

Bundled `libonnxruntime_providers_cuda.so` in the wheel links against **CUDA 10.2 / cuDNN 8 / cuBLAS 10 / cuFFT 10** (JetPack 4 / Xavier era) — useless on JP6.2 (CUDA 12.6). Hence the build.

## Build state at compact

- **Background script**: `/tmp/build_now.sh` (kicked off via `nohup ... > /tmp/build.log 2>&1 &`)
- **Build root**: `/mnt/ssd/sherpa-onnx-build` (shallow clone v1.13.2, GPU=ON, ORT-GPU 1.23.2 aarch64, CUDA_ARCH 87)
- **Build phase at compact**: `cmake --build build -j8` running. cmake configure confirmed `SHERPA_ONNX_ENABLE_GPU ON`.
- **PID**: variable; check via `pgrep -af build_now.sh`
- **Log**: `tail -50 /tmp/build.log`
- **Expected wall time**: 30-60 min from kick-off.

CUDA dev packages installed during build prep (~211 MB on `/`):
- cuda-nvcc-12-6, cuda-cccl-12-6, cuda-crt-12-6, cuda-cudart-dev-12-6, cuda-driver-dev-12-6, cuda-nvvm-12-6
- cuda-libraries-dev-12-6, libcudnn9-dev-cuda-12

Reversible post-build via `sudo apt remove cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcudnn9-dev-cuda-12 cuda-libraries-dev-12-6` + autoremove.

## Bundle layout (on-disk at compact)

`/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/`:
- `encoder.int8.onnx` (623 MB)
- `decoder.int8.onnx` (7 MB)
- `joiner.int8.onnx` (1.7 MB)
- `tokens.txt`
- `test_wavs/{0,1}.wav` + `trans.txt`

Eight SKUs published on HF account `csukuangfj2` (NOT `csukuangfj`): 80/160/560/1120 ms × fp32/int8. All dated 2026-04-25.

## Plan 2A T7 corrections (regardless of build outcome)

In `src/voice_control/asr/nemotron.py` constructor:
- `feature_dim=80` → **`feature_dim=128`** (Nemotron uses 128-mel)
- `provider="cuda"` → **depends on build outcome**:
  - If build succeeds and CUDA EP works → keep `provider="cuda"`
  - If build fails or CPU sufficient → `provider="cpu"`, `num_threads=4`

In `scripts/setup_nemotron.sh`: archive name is correct, but HF account in any direct URL is `csukuangfj2` mirror.

Drop from plan (decided by research):
- `nvidia-riva-client` Python dep → Riva 2.24+ deprecated Jetson Orin entirely (Thor/JP7 only)
- Parakeet rollback in T7 step 5 → **flip default**: Parakeet GPU is faster offline; Nemotron streaming is for Stage 2B/C partial wiring

## Open questions post-build

1. Does the built sherpa-onnx CUDA EP actually engage (no silent fallback)?
2. Nemotron CUDA RTFx on AGX Orin sm_87 — first public number, no prior bench.
3. End-to-end mic-loop latency tail-after-EOS with GPU sherpa-onnx — gates Stage 2B/C partial-wiring decision.

## Research artifacts

Internal research scripts (not committed):
- `/tmp/bench_nemotron.py` — sherpa-onnx CPU vs CUDA bench (CPU n=4 baseline)
- `/tmp/bench_parakeet.py` — onnx-asr CPU vs CUDA bench
- `/tmp/build_now.sh` — sherpa-onnx GPU build script

External handoff for other agent conversation: `docs/project/archive/stage2a-t7-asr-handoff.md`
