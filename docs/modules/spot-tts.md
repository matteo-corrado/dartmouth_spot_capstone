# spot_tts.py

`src/voice_control/spot_tts.py` -- GPU-accelerated text-to-speech via
`kokoro-onnx` + `onnxruntime-gpu` (CUDA Execution Provider on the Jetson
AGX Orin).

Non-blocking TTS that runs Kokoro v1.0 on the GPU. Includes mic mute/unmute
callbacks to prevent TTS audio from feeding back into the ASR pipeline, and
an output gain knob for the USB speaker.

## Key Class

### `SpotTTS(voice, speed, output_device, on_mute, on_unmute, volume)`

```python
class SpotTTS:
    def __init__(
        self,
        voice: str = "af_sarah",       # Kokoro v1.0 voice name
        speed: float = 1.1,
        output_device=None,             # sounddevice device index
        on_mute=None,                   # Callback before playback
        on_unmute=None,                 # Callback after playback
        volume: float = 1.0,            # Output gain (0.0 - 1.5)
    )
```

#### `speak(text)`

Non-blocking. Spawns a daemon thread for synthesis and playback. Cancels any
in-progress speech before starting new synthesis.

#### `speak_sync(text)`

Blocking. Generates and plays audio in the calling thread. Returns after
playback completes.

#### `wait()`

Block until current background speech finishes. No-op if nothing is playing.

#### `is_speaking() -> bool`

Returns True if audio is currently playing.

#### `is_available() -> bool`

Returns True if the model loaded successfully and sounddevice is available.

#### `set_volume(volume) -> float`

Set output gain (clamped to 0.0..1.5). Snapshot is taken at the start of each
utterance, so a mid-utterance change won't tear the buffer.

### Module-Level Convenience

```python
def get_tts(voice="af_sarah", output_device=None, volume=1.0) -> SpotTTS  # Singleton
```

## Voice Configuration

Model: **Kokoro v1.0** (`kokoro-v1.0.fp16-gpu.onnx`), 24 kHz sample rate,
54 voices. Voices are referenced by string name, not integer ID.

Common voices (full list via `Kokoro.get_voices()` once loaded):

| Name | Description |
|------|-------------|
| `af_sarah` | American female (Sarah) -- **default** |
| `af_bella` | American female (Bella) |
| `af_nicole` | American female (Nicole) |
| `af_sky` | American female (Sky) |
| `am_adam` | American male (Adam) |
| `am_michael` | American male (Michael) |
| `bf_emma` | British female (Emma) |
| `bf_isabella` | British female (Isabella) |
| `bm_george` | British male (George) |
| `bm_lewis` | British male (Lewis) |

The naming convention is `{a|b}{f|m}_<name>` -- (a)merican / (b)ritish,
(f)emale / (m)ale.

## GPU Provider

The module forces the CUDA Execution Provider by setting
`ONNX_PROVIDER=CUDAExecutionProvider` **before** importing `kokoro_onnx`.
This is a workaround for a bug in `kokoro-onnx 0.4.9` at `__init__.py:41`,
which probes for a non-existent `onnxruntime-gpu` Python module to detect the
GPU build and falls back to CPU when the probe fails. The escape hatch at
`__init__.py:46` honors the env var.

After loading, the module logs which provider is actually active and prints a
warning if CUDA didn't engage:

```
[TTS] Ready -- voice=af_sarah, speed=1.1, provider=CUDAExecutionProvider, sample_rate=24000Hz
```

If you see `provider=CPUExecutionProvider`, check that
`onnxruntime-gpu==1.23.0` is installed from the Jetson AI Lab pip index and
that nothing imports `kokoro_onnx` before `spot_tts` does.

## Mute Callbacks

When `on_mute` and `on_unmute` are provided (typically `_mute_mic` and
`_unmute_mic` from `client_mic.py`), the mic is paused during TTS playback.
This prevents TTS audio captured by the mic from triggering VAD or corrupting
ASR transcripts.

Flow: `on_mute()` -> synthesize -> play -> wait -> `on_unmute()`.

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `DEFAULT_VOICE` | `"af_sarah"` | Warm/natural American female voice |
| `DEFAULT_SPEED` | `1.1` | Slightly faster than natural |
| `DEFAULT_LANG` | `"en-us"` | Phonemizer language |
| `DEFAULT_VOLUME` | `1.0` | Output gain (1.0 = full) |
| `MAX_VOLUME` | `1.5` | Clipping risk above this |
| `MODEL_DIR` | `models/tts/kokoro-v1.0` | Model directory |
| `MODEL_FILE` | `kokoro-v1.0.fp16-gpu.onnx` | FP16 GPU-optimized model |
| `VOICES_FILE` | `voices-v1.0.bin` | Packed voice embeddings |

## Usage Example

```python
from spot_tts import SpotTTS

tts = SpotTTS(voice="af_sarah", speed=1.1, output_device=24)
if tts.is_available():
    tts.speak("Hello! I am Spot.")    # non-blocking
    tts.wait()                         # wait for completion
    tts.speak_sync("Goodbye!")         # blocking
```

```bash
# CLI test
python src/voice_control/spot_tts.py "Hello from Spot"
python src/voice_control/spot_tts.py --voice am_adam --speed 0.9 "Slow Adam voice"
python src/voice_control/spot_tts.py --volume 1.3 "Louder"
python src/voice_control/spot_tts.py --wav output.wav "Save to file"
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `kokoro-onnx==0.4.9`, `onnxruntime-gpu==1.23.0`,
`sounddevice`, `numpy==1.26.4`, `joblib`

**Setup:**

```bash
# numpy must be 1.x for the onnxruntime-gpu wheel ABI
pip install --no-deps "numpy==1.26.4"
pip install --no-deps "opencv-python==4.11.0.86"
pip install --no-deps "kokoro-onnx==0.4.9"
pip install joblib
pip install --no-deps --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
            "onnxruntime-gpu==1.23.0"

# Download Kokoro v1.0 model files (~200 MB)
python scripts/setup_kokoro_v1.py
```

See `requirements.txt` for the canonical pin set and rationale.
