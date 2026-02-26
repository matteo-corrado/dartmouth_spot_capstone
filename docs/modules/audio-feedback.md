# audio_feedback.py

`src/voice_control/audio_feedback.py` -- Audio tone feedback.

Short beeps and chimes (60-250ms) that give the user instant audio feedback
for voice control events. Uses numpy-generated sine waves played through
sounddevice. Suppresses ALSA error messages via ctypes.

## Singleton

```python
from audio_feedback import beep
```

The module exports a pre-instantiated `AudioFeedback` singleton named `beep`.

## Methods

All methods are non-blocking -- each spawns a daemon thread for playback.

### `beep.wake_detected()`

Rising chime -- wake word heard. Three tones: C5 (523 Hz) -> E5 (659 Hz) ->
G5 (784 Hz), 70ms each, total ~210ms.

### `beep.listening()`

Soft blip -- ready for command. Single G5 (784 Hz) tone, 80ms.

### `beep.command_ok()`

Happy double-beep -- command accepted. C5 (523 Hz, 70ms) + C6 (1047 Hz, 60ms)
with 20ms gap, total ~150ms.

### `beep.error()`

Low buzz -- something failed. A3 (220 Hz, 120ms) -> F3 (175 Hz, 120ms)
descending, total ~250ms.

### `beep.chain_next()`

Short tick -- next action in a multi-command chain. E5 (659 Hz), 60ms.

## Class

### `AudioFeedback(output_device=None)`

```python
class AudioFeedback:
    def __init__(self, output_device=None)
```

`output_device` is a sounddevice device index. The default singleton uses
the system default output.

## Audio Generation

- Sample rate: 48 kHz (standard ALSA rate, avoids underrun with short clips).
- Sine wave tones with fade-in/fade-out ramps to avoid clicks.
- 80ms silence tail appended to each tone to let ALSA buffers drain.
- Volume range: 0.15 to 0.25 (quiet enough to not interfere with ASR).

## ALSA Error Suppression

At module load, ALSA's error handler is replaced with a null handler via
ctypes to suppress noisy warnings like "underrun occurred" that clutter
the terminal but are harmless for short tones.

```python
_libasound = ctypes.cdll.LoadLibrary("libasound.so.2")
_libasound.snd_lib_error_set_handler(_c_null_handler)
```

## Usage Example

```python
from audio_feedback import beep

beep.wake_detected()   # Rising chime
beep.command_ok()      # Double-beep
beep.error()           # Low buzz
```

```bash
# Play all tones sequentially
python src/voice_control/audio_feedback.py
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `sounddevice`, `numpy`, `ctypes` (stdlib)
