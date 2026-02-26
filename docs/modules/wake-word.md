# wake_word.py

`src/voice_control/wake_word.py` -- Wake word detection ("Hey Spot").

Streaming keyword spotter using sherpa-onnx's zipformer model. Runs on CPU
with ~0ms detection latency. Shares the sherpa-onnx dependency with TTS
(no additional packages needed).

## Key Class

### `WakeWordDetector(model_dir)`

```python
class WakeWordDetector:
    def __init__(self, model_dir=None)  # Default: models/kws/
```

#### `process_frame(pcm16_bytes) -> bool`

Feed a 16 kHz PCM16 audio frame (any size, typically 30ms / 960 bytes).
Returns True when "Hey Spot" is detected. Automatically resets the internal
stream after detection.

```python
def process_frame(self, pcm16_bytes: bytes) -> bool
```

#### `reset()`

Reset the detector stream for next detection. Called automatically after
a positive detection, but can also be called manually (e.g. when returning
to WAKE_WORD state after listening timeout).

#### `is_available() -> bool`

Returns True if the model loaded successfully.

## Model

**sherpa-onnx-kws-zipformer-gigaspeech-3.3M** (int8 quantized, ~5 MB total).

Files in `models/kws/`:
- `encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx`
- `decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx`
- `joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx`
- `tokens.txt`
- `keywords.txt` (contains "h ey s p aa t" phoneme sequence)

Download: `python scripts/setup_kws.py`

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `KEYWORDS_SCORE` | 1.5 | Boost keyword score (higher = easier to trigger) |
| `KEYWORDS_THRESHOLD` | 0.25 | Detection threshold (higher = harder to trigger) |
| `NUM_TRAILING_BLANKS` | 1 | Blank frames after keyword before firing |

## Integration with client_mic.py

The wake word detector runs on every audio frame in the `WAKE_WORD` state.
When detected:

1. `beep.wake_detected()` plays a rising chime.
2. State transitions to `LISTENING`.
3. Audio is NOT drained -- remaining speech in the buffer (e.g. "stand up"
   from "Hey Spot stand up") is preserved for VAD to pick up.
4. `client_mic.py` strips the wake phrase from the ASR transcript before
   sending to the LLM.

If the dedicated detector is unavailable, `client_mic.py` falls back to
ASR-based wake phrase detection (regex on the transcript).

## Usage Example

```python
from wake_word import WakeWordDetector

detector = WakeWordDetector()
if detector.is_available():
    # Feed audio frames from mic
    if detector.process_frame(pcm16_bytes):
        print("Hey Spot detected!")
        # detector auto-resets after detection
```

```bash
# Standalone mic test
python src/voice_control/wake_word.py --test-mic --device 24

# Quick load + silence test
python src/voice_control/wake_word.py
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `sherpa_onnx`, `numpy`

**Setup:** `python scripts/setup_kws.py`
