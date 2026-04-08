"""Audio feedback tones for Spot voice control.

Short beeps/chimes (60-250ms) that give the user instant audio feedback
without interfering with ASR.

Architecture
------------
This module is a *sample generator*. It builds short numpy waveforms for
each event (wake_detected, command_ok, error, ...) and submits them to a
shared ``AudioPlayer`` instance owned by the voice pipeline. It does NOT
open a sounddevice stream of its own — that would race against ``SpotTTS``
over the global ``sd`` state and was the root cause of the mute interlock
deadlock we were debugging. See ``audio_player.py``.

Beeps queue *behind* TTS speech: if Spot is mid-sentence, a beep waits
until the sentence finishes. The beeps are 60-250ms each so the delay is
imperceptible in practice, and the serialization is what guarantees the
mic-mute invariant stays consistent.

Usage:
    from audio_feedback import beep
    beep.set_player(player)        # wire up before any beeps fire
    beep.wake_detected()            # rising chime — wake word heard
    beep.listening()                # soft blip — ready for command
    beep.command_ok()               # happy double-beep — command accepted
    beep.error()                    # low buzz — something failed
    beep.chain_next()               # short tick — next action in chain
"""

import numpy as np

from src.voice_control.audio_player import AudioPlayer

SAMPLE_RATE = 48000  # Standard ALSA rate — avoids underrun with short clips
TAIL_MS = 80         # Silence padding at end — gives ALSA buffer time to drain
DEFAULT_VOLUME = 1.0
MAX_VOLUME = 1.5     # per-tone amplitudes are 0.15-0.25, so 1.5x stays well below clipping


def _tone(freq: float, duration_ms: int, volume: float = 0.3) -> np.ndarray:
    """Generate a sine wave tone with silence tail."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
    samples = np.sin(2 * np.pi * freq * t) * volume
    # Fade in/out to avoid clicks
    fade = min(len(samples) // 8, 400)
    if fade > 0:
        samples[:fade] *= np.linspace(0, 1, fade)
        samples[-fade:] *= np.linspace(1, 0, fade)
    # Add silence tail so ALSA buffer doesn't underrun
    tail = np.zeros(int(SAMPLE_RATE * TAIL_MS / 1000), dtype=np.float64)
    return np.concatenate([samples, tail]).astype(np.float32)


class AudioFeedback:
    """Audio feedback tones for voice control events.

    Generates sine-wave tones and submits them to a shared AudioPlayer.
    Until ``set_player`` is called the singleton is a no-op (so importing
    the module never crashes if there is no audio hardware).
    """

    def __init__(self, player: AudioPlayer | None = None,
                 volume: float = DEFAULT_VOLUME):
        self._player = player
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        # ``device`` is kept as an attribute for backwards compatibility with
        # call sites that did ``beep.device = args.output_device`` before
        # the playback was migrated to AudioPlayer. It has no effect now —
        # the device is owned by the player. Setting it is harmless.
        self.device = None

    def set_player(self, player: AudioPlayer) -> None:
        """Wire up the shared AudioPlayer. Call this once at startup."""
        self._player = player

    def set_volume(self, volume: float) -> float:
        """Set master gain for all beeps. Clamped to 0.0..MAX_VOLUME."""
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        return self.volume

    def _play(self, samples: np.ndarray, label: str) -> None:
        """Apply master volume and submit to the player.

        No-op if no player has been wired up (e.g. when ``audio_feedback``
        is imported by a tool that doesn't run the full voice pipeline).
        """
        if self._player is None or not self._player.is_available():
            return
        gain = self.volume
        if gain != 1.0:
            samples = (samples * gain).astype(np.float32)
        self._player.enqueue_raw(samples, SAMPLE_RATE, label=f"beep:{label}")

    def wake_detected(self):
        """Rising chime — wake word heard (C5 -> E5 -> G5, 210ms)."""
        c5 = _tone(523, 70, 0.25)
        e5 = _tone(659, 70, 0.25)
        g5 = _tone(784, 70, 0.25)
        # Only need tail on the last tone
        samples = np.concatenate([c5[:-int(SAMPLE_RATE*TAIL_MS/1000)],
                                  e5[:-int(SAMPLE_RATE*TAIL_MS/1000)],
                                  g5])
        self._play(samples, "wake_detected")

    def listening(self):
        """Soft blip — ready for command (short G5, 80ms)."""
        samples = _tone(784, 80, 0.2)
        self._play(samples, "listening")

    def command_ok(self):
        """Happy double-beep — command accepted (C5, C6, 150ms)."""
        c5 = _tone(523, 70, 0.2)
        gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
        c6 = _tone(1047, 60, 0.2)
        samples = np.concatenate([c5[:-int(SAMPLE_RATE*TAIL_MS/1000)], gap, c6])
        self._play(samples, "command_ok")

    def error(self):
        """Low buzz — something failed (A3 descending, 250ms)."""
        a3 = _tone(220, 120, 0.25)
        gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
        f3 = _tone(175, 120, 0.25)
        samples = np.concatenate([a3[:-int(SAMPLE_RATE*TAIL_MS/1000)], gap, f3])
        self._play(samples, "error")

    def chain_next(self):
        """Short tick — next action in chain (E5, 60ms)."""
        samples = _tone(659, 60, 0.15)
        self._play(samples, "chain_next")


# Module-level singleton. Created without a player so importing the module
# is side-effect free; client_mic.py wires up the real player at startup
# via ``beep.set_player(...)``.
beep = AudioFeedback()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Audio feedback tones test")
    print("=" * 40)
    player = AudioPlayer()
    beep.set_player(player)

    for name in ["wake_detected", "listening", "command_ok", "error", "chain_next"]:
        print(f"  Playing: {name}")
        getattr(beep, name)()
        time.sleep(0.8)

    player.wait()
    print("Done!")
