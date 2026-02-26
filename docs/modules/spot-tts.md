# spot_tts.py

`src/voice_control/spot_tts.py` -- Text-to-speech via sherpa-onnx Kokoro.

Non-blocking TTS that runs on CPU. Includes mic mute/unmute callbacks to
prevent TTS audio from feeding back into the ASR pipeline.

## Key Class

### `SpotTTS(speaker_id, speed, output_device, on_mute, on_unmute)`

```python
class SpotTTS:
    def __init__(
        self,
        speaker_id: int = 3,          # af_sarah
        speed: float = 1.1,
        output_device=None,            # sounddevice device index
        on_mute=None,                  # Callback before playback
        on_unmute=None                 # Callback after playback
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

### Module-Level Convenience

```python
def get_tts(speaker_id=3, output_device=None) -> SpotTTS  # Singleton
```

## Speaker Configuration

Model: **Kokoro v0.19** (kokoro-en-v0_19), 24 kHz sample rate.

| ID | Name | Description |
|----|------|-------------|
| 0 | af | American female |
| 1 | af_bella | American female (Bella) |
| 2 | af_nicole | American female (Nicole) |
| 3 | af_sarah | American female (Sarah) -- **default** |
| 4 | af_sky | American female (Sky) |
| 5 | am_adam | American male (Adam) |
| 6 | am_michael | American male (Michael) |
| 7 | bf_emma | British female (Emma) |
| 8 | bf_isabella | British female (Isabella) |
| 9 | bm_george | British male (George) |
| 10 | bm_lewis | British male (Lewis) |

## Mute Callbacks

When `on_mute` and `on_unmute` are provided (typically `_mute_mic` and
`_unmute_mic` from `client_mic.py`), the mic is paused during TTS playback.
This prevents TTS audio captured by the mic from triggering VAD or corrupting
ASR transcripts.

Flow: `on_mute()` -> synthesize -> play -> wait -> `on_unmute()`.

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `DEFAULT_SPEAKER_ID` | 3 | af_sarah speaker |
| `DEFAULT_SPEED` | 1.1 | Slightly faster than natural |
| `MODEL_DIR` | `models/tts/kokoro-en-v0_19` | Model directory path |

## Usage Example

```python
from spot_tts import SpotTTS

tts = SpotTTS(speaker_id=3, speed=1.1)
if tts.is_available():
    tts.speak("Hello! I am Spot.")   # non-blocking
    tts.wait()                        # wait for completion
    tts.speak_sync("Goodbye!")        # blocking
```

```bash
# CLI test
python src/voice_control/spot_tts.py "Hello from Spot"
python src/voice_control/spot_tts.py --sid 5 --speed 0.9 "Slow Adam voice"
python src/voice_control/spot_tts.py --wav output.wav "Save to file"
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `sherpa_onnx`, `sounddevice`, `numpy`

**Setup:** `pip install sherpa-onnx && python scripts/setup_kokoro.py`
