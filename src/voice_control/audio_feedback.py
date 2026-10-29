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


# --- Operator-approved "simple" set (2026-10-28) -----------------------------
# Plain beeps told apart by note count + pitch direction (earcon guidelines):
# up = "your turn", down = "done"; first note accented, last note longer.
def _beep(freq: float, dur_s: float, volume: float = 0.22) -> np.ndarray:
    """Beep: fundamental + soft 2nd/3rd harmonic, 5 ms fades, gentle decay. No tail."""
    t = np.arange(int(SAMPLE_RATE * dur_s)) / SAMPLE_RATE
    tone = (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
            + 0.12 * np.sin(6 * np.pi * freq * t)) / 1.37
    x = volume * np.exp(-t * 3.0) * tone
    n = int(SAMPLE_RATE * 0.005)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def _gap(s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * s))


def _seq(*parts: np.ndarray) -> np.ndarray:
    """Join parts and add the ALSA drain tail."""
    tail = np.zeros(int(SAMPLE_RATE * TAIL_MS / 1000))
    return np.concatenate([*parts, tail]).astype(np.float32)


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
        """Command accepted: two identical quick notes (C6 C6)."""
        self._play(_seq(_beep(1047, .07, .24), _gap(.06), _beep(1047, .1)), "command_ok")

    def error(self):
        """Something failed: two slow falling low notes (E4 -> C4)."""
        self._play(_seq(_beep(330, .16, .26), _gap(.08), _beep(262, .28)), "error")

    def chain_next(self):
        """Next action in chain: one very short high tick (G6)."""
        self._play(_seq(_beep(1568, .04, .16)), "chain_next")

    def thinking(self):
        """Heard you, working: one short low note (A4). Same as think_cue."""
        self._play(_seq(_beep(440, .12, .2)), "thinking")

    # --- Turn-taking cues (distinct from the power-on ``ready`` chime) ---
    def listen_cue(self):
        """Mic open, your turn: two quick rising notes (A5 -> E6)."""
        self._play(_seq(_beep(880, .09, .26), _gap(.04), _beep(1319, .14)), "listen_cue")

    def think_cue(self):
        """Heard you, working: one short low note (A4)."""
        self._play(_seq(_beep(440, .12, .2)), "think_cue")

    def speak_cue(self):
        """Soft C6 tick. Not mapped to RESPONDING any more (blue ring + voice are the cue)."""
        self._play(_seq(_beep(1047, .05, .12)), "speak_cue")

    def released_cue(self):
        """Conversation over: three quick high falling notes (G6 E6 C6)."""
        self._play(_seq(_beep(1568, .07, .22), _gap(.03), _beep(1319, .07, .2), _gap(.03), _beep(1047, .13, .18)), "released_cue")

    def ready(self):
        """Power-on: three rising notes, slow (C5 E5 G5)."""
        self._play(_seq(_beep(523, .12, .26), _gap(.06), _beep(659, .12), _gap(.06), _beep(784, .22)), "ready")


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
