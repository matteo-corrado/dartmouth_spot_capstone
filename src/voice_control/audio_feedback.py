"""Audio feedback tones for Spot voice control.

Short beeps/chimes (60-250ms) that give the user instant audio feedback
without interfering with ASR. Uses numpy sine waves + sounddevice (no new deps).

Usage:
    from audio_feedback import beep
    beep.wake_detected()   # rising chime — wake word heard
    beep.listening()       # soft blip — ready for command
    beep.command_ok()      # happy double-beep — command accepted
    beep.error()           # low buzz — something failed
    beep.chain_next()      # short tick — next action in chain
"""

import os
import threading
import numpy as np

# Suppress noisy ALSA warnings (must be set before importing sounddevice)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
_alsa_error_suppressed = False
try:
    import ctypes
    _libasound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _alsa_error_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                            ctypes.c_char_p, ctypes.c_int,
                                            ctypes.c_char_p)
    def _null_handler(filename, line, function, err, fmt):
        pass
    _c_null_handler = _alsa_error_handler(_null_handler)
    _libasound.snd_lib_error_set_handler(_c_null_handler)
    _alsa_error_suppressed = True
except Exception:
    pass

try:
    import sounddevice as sd
except ImportError:
    sd = None

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


def _play_blocking(samples: np.ndarray, device=None):
    """Play audio samples and wait."""
    if sd is None:
        return
    try:
        sd.play(samples, samplerate=SAMPLE_RATE, device=device, blocksize=2048)
        sd.wait()
    except Exception:
        pass


class AudioFeedback:
    """Audio feedback tones for voice control events."""

    def __init__(self, output_device=None, volume: float = DEFAULT_VOLUME):
        self.device = output_device
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))

    def set_volume(self, volume: float) -> float:
        """Set master gain for all beeps. Clamped to 0.0..MAX_VOLUME."""
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        return self.volume

    def _play(self, samples):
        """Apply master volume and launch playback in a daemon thread."""
        gain = self.volume
        if gain != 1.0:
            samples = (samples * gain).astype(np.float32)
        threading.Thread(target=_play_blocking, args=(samples, self.device), daemon=True).start()

    def wake_detected(self):
        """Rising chime — wake word heard (C5 -> E5 -> G5, 210ms)."""
        c5 = _tone(523, 70, 0.25)
        e5 = _tone(659, 70, 0.25)
        g5 = _tone(784, 70, 0.25)
        # Only need tail on the last tone
        samples = np.concatenate([c5[:-int(SAMPLE_RATE*TAIL_MS/1000)],
                                  e5[:-int(SAMPLE_RATE*TAIL_MS/1000)],
                                  g5])
        self._play(samples)

    def listening(self):
        """Soft blip — ready for command (short G5, 80ms)."""
        samples = _tone(784, 80, 0.2)
        self._play(samples)

    def command_ok(self):
        """Happy double-beep — command accepted (C5, C6, 150ms)."""
        c5 = _tone(523, 70, 0.2)
        gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
        c6 = _tone(1047, 60, 0.2)
        samples = np.concatenate([c5[:-int(SAMPLE_RATE*TAIL_MS/1000)], gap, c6])
        self._play(samples)

    def error(self):
        """Low buzz — something failed (A3 descending, 250ms)."""
        a3 = _tone(220, 120, 0.25)
        gap = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
        f3 = _tone(175, 120, 0.25)
        samples = np.concatenate([a3[:-int(SAMPLE_RATE*TAIL_MS/1000)], gap, f3])
        self._play(samples)

    def chain_next(self):
        """Short tick — next action in chain (E5, 60ms)."""
        samples = _tone(659, 60, 0.15)
        self._play(samples)


# Module-level singleton
beep = AudioFeedback()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Audio feedback tones test")
    print("=" * 40)

    for name in ["wake_detected", "listening", "command_ok", "error", "chain_next"]:
        print(f"  Playing: {name}")
        getattr(beep, name)()
        time.sleep(0.8)

    print("Done!")
