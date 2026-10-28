# Stage 2A T7 ASR — handoff for parallel agent conversation

**Bottom line:** Plan 2A's Nemotron Speech Streaming model pick survives all verification. Sherpa-onnx is the right wrapper. **The GPU build was completed and benchmarked — CUDA EP is SLOWER than CPU for streaming Conformer on Orin.** Final pick: keep both Parakeet TDT v3 (offline, CUDA) and Nemotron Streaming (CPU n=4) installed; default = Parakeet for offline utterances, Nemotron for streaming use cases.

## What was investigated

5 agents (3 desk research + 1 cross-validator + 1 on-device empirical + 1 bundle-compatibility verifier) cross-validated 6 competing hypotheses about ASR options on Jetson AGX Orin (JetPack 6.2, aarch64, sm_87, CUDA 12.6, 64GB unified memory).

## Confirmed facts (HIGH confidence, multi-agent corroborated)

1. **Nemotron Speech Streaming 0.6B ONNX is true streaming** (cache-aware Conformer-Transducer, OnlineRecognizer API). PR k2-fsa/sherpa-onnx#3044 landed. **Eight published SKUs** dated **2026-04-25** at HF account **`csukuangfj2`** (mirror): {80, 160, 560, 1120} ms chunks × {fp32, int8}. 560ms-int8 archive ≈ 1.3 GB; full set on disk at `/mnt/ssd/nemotron-models/`.

2. **Sherpa-onnx Python aarch64 wheel from PyPI is CPU-only.** Bundled `libonnxruntime.so` has NO CUDA `DT_NEEDED` entries (verified by `readelf -d`). Passing `provider="cuda"` silently falls back to CPU on a stock pip install with the verbatim message `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available providers: CPUExecutionProvider. Fallback to cpu!`

3. **Sherpa-onnx CUDA aarch64 wheels exist on the HF mirror `csukuangfj2/sherpa-onnx-wheels` but only as legacy CUDA 10.2 / cuDNN 8 / cuBLAS 10 / cuFFT 10 builds** (verified by `readelf -d libonnxruntime_providers_cuda.so` on the cp310 aarch64 wheel — JetPack 4/Xavier-era). JetPack 6.2 ships CUDA 12.6, so the off-index aarch64 CUDA wheel is unusable. **From-source build is the only GPU path** for sherpa-onnx on Jetson Orin (in progress as of this handoff).

4. **`onnxruntime-gpu 1.23.0` is already installed in `spot-env` and fully CUDA-wired** — `libcudart.so.12`, `libcudnn.so.9`, providers = `[Tensorrt, CUDA, CPU]`, `get_device()='GPU'`. **Parakeet TDT v3 via `onnx-asr` benefits from this directly** with no sherpa-onnx involvement.

5. **`onnx-asr 0.11.0` (in spot-env) cannot load the sherpa-onnx Nemotron bundle.** Structural incompatibility, NOT a naming issue:
   - sherpa-onnx export: 3 files (encoder.int8.onnx + decoder.int8.onnx + joiner.int8.onnx), streaming encoder with `cache_last_channel` / `cache_last_time` / `cache_last_channel_len` inputs, 128-mel features, split predictor + joiner.
   - onnx-asr expects: 2 files (encoder-model + fused decoder_joint-model), stateless encoder, 80-mel, single fused predictor+joiner (the `istupakov/*-onnx` convention).
   - Verified by symlink-rename workaround — still fails because the I/O signatures differ. Confirmed via `onnx_asr/models/nemo.py:23,83-128` + `onnx_asr/resolver.py:16-32`.

6. **NVIDIA Riva 2.24+ deprecated Jetson Orin entirely** (verbatim release-note quote: "Supported Jetson platform is now NVIDIA Jetson Thor"). Last Orin-supporting Riva (2.19) never shipped Nemotron. Riva path is dead. Riva remnants on the device are trivial: just `nvidia-riva-client 2.24.0` (Python gRPC client lib) in `spot-env`. Cleanup = `pip uninstall nvidia-riva-client`.

## Falsified claims (do NOT take into the plan)

- **"sherpa-onnx publishes aarch64 CUDA wheels matched to JetPack 6 (ORT 1.18.1 + CUDA 12.6)"** — false. Only CUDA 10.2 aarch64 builds exist (legacy), and they're off-index on a separate HF account.
- **"Canary-Qwen 2.5B is WER king AND streaming-capable"** — false. Canary-Qwen is offline-only; the model card has no mention of streaming/Wait-K/AlignAtt. The streaming-capable Canary is Canary-1B FastConformer, which has an open stale-output bug (NeMo#15231).
- **"Sherpa-onnx Nemotron export is offline-only"** — false. PR #3044 + the `csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-*` archives are explicitly streaming (`OnlineRecognizer.from_transducer`).
- **"Moonshine sub-200ms on Orin GPU"** — unverified extrapolation from Apple M3 numbers. No Jetson benchmark exists.
- **"NeMo on Jetson is NVIDIA-blessed"** — community-validated by `raymondlo84` in `nvidia/parakeet-tdt-0.6b-v3` HF discussion #15, not staff-blessed. Recipe still works. URL is `pypi.jetson-ai-lab.io/jp6/cu126` (NOT `.dev`).

## Measured benchmark on AGX Orin

Same 2 test wavs (LibriSpeech material from bundle's `test_wavs/`, 23.34 s total audio, 16 kHz mono PCM16). Bench v2 used tegrastats `GR3D_FREQ` to verify GPU was actually engaged (no silent fallback).

| ASR | Runtime | Total wall | **RTFx** | GPU util (tegrastats) |
|---|---|---|---|---|
| Nemotron Streaming 560ms-int8 | sherpa-onnx **CPU n=4** | 5.60 s | **4.17×** | 0% (idle, proven) |
| Nemotron Streaming 560ms-int8 | sherpa-onnx **CUDA n=4** (from-source GPU build) | 6.74 s | **3.46×** | 28% mean / 48% peak |
| Parakeet TDT 0.6B v3 | onnx-asr CPU | 2.38 s | **9.81×** | 0% |
| Parakeet TDT 0.6B v3 | onnx-asr **CUDA** | **0.48 s** | **48.82×** | high (not measured, but obvious) |

Transcripts perfect on all configs. Voice command (3-5 s) on Parakeet CUDA = **wall 60-100 ms**; on Nemotron CPU = 700-1200 ms.

**Critical finding (post-build)**: Nemotron sherpa-onnx CUDA is **20% SLOWER than CPU n=4 on Orin**, despite genuine GPU engagement (GR3D_FREQ 48% peak). Reason: cache-aware streaming Conformer at 560ms chunks = many small per-chunk ops; CUDA kernel-launch overhead (~10-100 µs/op) dominates the actual compute. Mel features and decoder loop run on CPU regardless of provider, adding ping-pong. Orin's 12-core A78AE keeps data hot in L2/L3 and wins.

Parakeet (offline, single big batch) is the inverse workload and is where Orin GPU shines.

## Final T7 path — keep both, route by use case

Both backends installed. User-confirmed (2026-05-22). Default behavior:

| Use case | Backend | Reason |
|---|---|---|
| Offline utterance (VAD-segmented, push-to-talk) | **Parakeet TDT v3 CUDA** | RTFx 48.82×, ~60-100 ms wall on voice command. Lowest latency by 10×. |
| Streaming with mid-utterance partials | **Nemotron Streaming CPU n=4** | RTFx 4.17×, true chunk-level partials, `on_token` callback for barge-in. |

### Parakeet (default, offline)

```python
# src/voice_control/asr/parakeet.py
import os
os.environ["HF_HOME"] = "/mnt/ssd/parakeet-models"

import onnx_asr
self.model = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
# m.recognize(audio, sample_rate=16000) → str
# audio: float32 numpy in [-1, 1] (int16 / 32768.0)
```

### Nemotron (streaming, CPU only on Orin)

```python
# src/voice_control/asr/nemotron.py
self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens,
    sample_rate=SAMPLE_RATE,
    feature_dim=128,                 # Nemotron-0.6B uses 128-mel (NOT 80)
    provider="cpu",                  # CUDA proven slower than CPU on Orin — do not use "cuda"
    num_threads=4,                   # all of A78AE perf cluster
    decoding_method="greedy_search",
)
```

**Do not change `provider="cpu"` to `"cuda"`** even though the GPU sherpa-onnx build briefly existed. Measured slower; the build has been reverted to the pip CPU wheel (see "Build outcome" below).

## Bundle layout (already on disk)

`/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/`
- `encoder.int8.onnx` (623 MB), `decoder.int8.onnx` (7 MB), `joiner.int8.onnx` (1.7 MB), `tokens.txt`, `test_wavs/`, `README.md`

Eight SKUs available — switch chunk size by switching directory (80/160/560/1120 ms × fp32/int8).

## Benchmark gate — passed

Nemotron Streaming CPU n=4 RTFx 4.17× — passes the real-time-safe threshold by ~4×. GPU bench done and rejected (3.46× — slower than CPU). Bench scripts at `/tmp/bench_nemotron.py` and `/tmp/bench_nemotron_v2.py` (on tmpfs — gone on reboot).

## From-source GPU build — outcome (completed and reverted)

Build done and measured. Reverted because CUDA was slower than CPU. Key gotchas captured for any future attempt:

| Gotcha | Symptom | Fix |
|---|---|---|
| `SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.23.2` rejected | `CMake Error: Unsupported onnxruntime version 1.23.2` | Use **`1.18.1`** for CUDA 12.6 aarch64 (only valid value per `cmake/onnxruntime-linux-aarch64-gpu.cmake:50`). |
| `pip install -e .` ignored `SHERPA_ONNX_ENABLE_GPU=ON` from cmake build dir | Silent fallback to CPU build inside setup.py's own cmake re-run | Set `SHERPA_ONNX_CMAKE_ARGS` env var, OR do standalone cmake build + manually copy `.so` files into site-packages. |
| `pip install -e .` crashed on `/tmp/sherpa_onnx/bin/` copy | `[Errno 2] No such file or directory` at end of install | Skip the editable install. Use standalone cmake + manual `.so` swap. |
| cmake found wrong Python despite `-DPython3_EXECUTABLE` | `Manually-specified variables were not used by the project` | Project variable is **`-DPYTHON_EXECUTABLE`** (no "3"). 3.10.x patch versions are ABI-compat anyway. |
| `_sherpa_onnx.so` copied to wrong dir | `ImportError: VERS_1.24.4 not found` | `__init__.py` does `from sherpa_onnx.lib._sherpa_onnx import ...` — module must land in **`site-packages/sherpa_onnx/lib/`**, not the parent dir. |

Build artifacts and scripts (`/tmp/build_v2.sh`, `/tmp/build_v2.log`, `/mnt/ssd/sherpa-onnx-build/`) were deleted post-bench. CUDA dev apt packages (cuda-nvcc-12-6, cuda-cudart-dev-12-6, cuda-cccl-12-6, cuda-crt-12-6, cuda-driver-dev-12-6, cuda-nvvm-12-6, cuda-libraries-dev-12-6, libcudnn9-dev-cuda-12) **retained** (~201 MB eMMC) — useful for future ML work (NeMo native, TRT-LLM, etc).

sherpa-onnx reverted to pip wheel `sherpa-onnx==1.13.2` (CPU EP only). Nemotron CPU n=4 path verified post-revert at RTFx 4.37× (same as pre-build).

## Plan 2A T7 patches still needed

- **Drop `nvidia-riva-client` from `requirements.txt`** (and any code that imports it). Riva is dead on Orin.
- **Refresh `docs/project/archive/upgrade-plan.md` Phase 4 status** — disk pressure on `/` is gone (24 GB free post-cleanup; Riva 24 GB image purged; Docker root on SSD). Stale memory said 787 MB free.
- **In `src/voice_control/asr/nemotron.py` constructor**: set `feature_dim=128` (NOT 80 — Nemotron uses 128-mel) and `provider="cpu"` (CUDA is slower on Orin; do not change).
- **T7 step 5 (Parakeet rollback)**: keep as-is. Both backends ship. Backend selection via `SPOT_ASR_BACKEND` env (or current routing — user already has both wired up).

## Models considered and rejected with reasons

| Model | Why not |
|---|---|
| Parakeet TDT 0.6B v3 (as primary if option B) | Offline-only per k2-fsa/sherpa-onnx#2918; great offline perf but no streaming for 2B partials |
| Canary-Qwen 2.5B | Offline only (no streaming docs in model card despite a wave-1 agent's claim); 5+ GB VRAM cost |
| Canary-1B Flash | AlignAtt streaming exists but stale-output bug NeMo#15231 open |
| Whisper Large v3 Turbo | Mid-pack 2026 WER; faster-whisper CTranslate2 aarch64 CUDA build painful (15+ iters per jetson-containers#722) |
| Distil-Whisper | Same CTranslate2 build pain; English-only, ~1% WER hit |
| Moonshine | Jetson GPU latency unverified (sub-200ms claim = extrapolation from M3); WER ~6.6% Medium |
| SenseVoice Small | Non-autoregressive, fast, but English not class-leading |
| Voxtral Mini 3B / Phi-4 multimodal | 3-5B audio-LLMs — shares VRAM with Gemma 4 E4B; wrong architecture for pipelined STT→LLM |
| NeMo native (PyTorch checkpoint) | Works via community recipe but adds 5-8 GB PyTorch+CUDA stack, pynini build risk, NeMo "not formally supported on Jetson" (NVIDIA-NeMo#15014) |
| TensorRT-LLM custom Nemotron export | No public recipe; FastConformer encoder not upstream in TRT-LLM 0.12-jetson |

## Resolved questions (was: open at first handoff)

1. **Q: Does built sherpa-onnx CUDA EP actually engage (no silent fallback)?** A: **Yes** — tegrastats GR3D_FREQ measured 28% mean / 48% peak during CUDA bench, vs 0% during CPU bench. Build worked.
2. **Q: Nemotron Streaming 560ms-int8 CUDA RTFx on Orin sm_87 — expected ≥15×?** A: **No, 3.46×** — slower than CPU n=4 (4.17×). Streaming Conformer's small-chunk workload is CPU-favored. Per-chunk kernel-launch overhead + CPU-resident glue (mel features, decoder loop) negates GPU benefit.
3. **Q: End-to-end mic-loop latency tail-after-EOS with GPU sherpa-onnx — gates 2B/C wiring decision?** A: GPU path is dead; use CPU for streaming. Tail-after-EOS on Nemotron CPU = ~240 ms for 1-second utterance (1/4.17 × 1000 ms).

## Open questions still

1. If using Parakeet (option A path): does raw concatenated token output (no auto-punctuation) suffice for the LLM dispatcher, or does it need post-processing?
2. Whether Stage 2B/C will actually wire `on_token` partials. If no, Nemotron is dead weight and could be removed. If yes, current dual-install is correct.

## Source citations (verified during cross-validation)

- Nemotron streaming model card: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Sherpa-onnx Nemotron streaming bundles: https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25
- Sherpa-onnx PR #3044 (Nemotron support): k2-fsa/sherpa-onnx
- Sherpa-onnx aarch64 CUDA wheels: https://huggingface.co/csukuangfj2/sherpa-onnx-wheels (CUDA 10.2 only — unusable on JP6.2)
- ORT-GPU for JetPack 6.2 index: https://pypi.jetson-ai-lab.io/jp6/cu126
- Riva Orin deprecation: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/release-notes.html (2.24)
- NeMo on Jetson community recipe: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/15
- onnx-asr nemo loader source: `spot-env/lib/python3.10/site-packages/onnx_asr/models/nemo.py:23,83-128`
- MS CoreAI on-device streaming ASR benchmark: arXiv 2604.14493
- istupakov/parakeet-tdt-0.6b-v3-onnx (the export onnx-asr loads): https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
