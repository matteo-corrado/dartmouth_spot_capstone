# audio_player

`src/voice_control/audio_player.py`

## Purpose

`AudioPlayer` owns the **single** persistent `sounddevice.OutputStream` for
the entire voice pipeline. Both `SpotTTS` and the beep generator in
`audio_feedback.py` submit playback through it instead of calling
`sd.play()` / `sd.wait()` directly.

## Why this exists

`sd.play()` / `sd.stop()` / `sd.wait()` operate on a *module-global*
`_last_callback` inside `sounddevice` and are not thread-safe. The previous
design spawned a fresh daemon thread per `tts.speak()` call AND played beeps
from yet another thread, which meant three concurrent producers fighting over
that single global. The race manifested as a permanent deadlock when
`capture_frame` failed: a TTS thread would never reach its `finally` block
to clear `_mic_muted`, and the input pipeline would block on an empty
`audio_queue` forever.

The fix is to centralize all audio playback through one stream and one
worker. The worker plays items in submission order; `is_busy()` is the
single source of truth for "should the mic be muted right now".

## Concurrency model

Three threads touch the object:

1. **Caller threads** — `enqueue_raw`, `enqueue_render`, `wait`,
   `is_busy`, `shutdown`. Called from `client_mic.main()` and from the
   audio callback (`is_busy()` only).
2. **Worker thread** (`_worker_loop`) — pulls items from the queue, runs
   the render function (e.g. Kokoro inference), and writes audio to the
   stream in ~100ms chunks. Chunked writes give the watchdog a place to
   interrupt a wedged write.
3. **Watchdog thread** (`_watchdog_loop`) — polls the current task age
   once a second. If a single task has been processing longer than
   `WATCHDOG_TIMEOUT_S` (30s), it calls `force_reset()` to recover from
   a wedged worker.

Two locks coordinate them:

| Lock | Protects | Held for |
|---|---|---|
| `_inflight_lock` | `_inflight` (the source of truth for `is_busy()`), `_stop` set/check, `_current_task_started_at` | nanoseconds — acquired ~33×/sec from the audio callback |
| `_reset_lock` | Serializes `force_reset` and `shutdown` so concurrent recovery attempts don't double-up streams or workers | up to ~2s (worker join timeout) |

## API

```python
from src.voice_control.audio_player import AudioPlayer

player = AudioPlayer(output_device=3)

# Submit raw int16/float samples (used by audio_feedback.beep):
player.enqueue_raw(samples_np, rate=24000)

# Submit a deferred render that runs INSIDE the worker (used by SpotTTS):
player.enqueue_render(lambda: kokoro.create("Hello"), label="tts:Hello")

# Mic-mute interlock (called from PortAudio callback):
if player.is_busy():
    return  # discard the input frame

# Wait for the queue to drain (used by tts.speak_sync):
player.wait()

# Tear down at process exit:
player.shutdown()
```

## Mic-mute interlock

`client_mic.audio_callback()` reads `_audio_player.is_busy()` on every
input frame. While the player is busy, frames are discarded so Spot's
own voice can never feed back into ASR. After the player goes idle, an
extra `SPEAKER_TAIL_COOLDOWN_S = 0.15` grace period gives the OS audio
buffer time to drain — see `client_mic.py:138`. **Do not reduce this
constant** without testing for TTS-tail bleed.

## Recovery and "leak the stream" safety

`force_reset()` and `shutdown()` deliberately do **not** call
`stream.close()` while the worker might still be inside `stream.write()`.
A previous version did, and we observed `malloc(): unaligned tcache chunk
detected` heap corruption inside PortAudio in dev. The branches that drop
references to the stream rather than closing it are intentional safety
gates, not a leak.

## Related files

- `src/voice_control/spot_tts.py` — renderer-only, submits via `enqueue_render`
- `src/voice_control/audio_feedback.py` — sample generator, submits via `enqueue_raw`
- `src/voice_control/client_mic.py` — owns the singleton, wires the mic-mute interlock
