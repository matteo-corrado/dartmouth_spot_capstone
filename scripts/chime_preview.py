#!/usr/bin/env python3
"""Play each voice-pipeline chime once, slowly, with what it means.

Uses the real tones from src/voice_control/audio_feedback.py, so what you hear
is what the robot plays. Also plays the louder/longer listen_cue option under
consideration so the two can be compared.

Usage (from repo root):
    spot-env/bin/python scripts/chime_preview.py
    spot-env/bin/python scripts/chime_preview.py --only listen_cue --repeat 3
    spot-env/bin/python scripts/chime_preview.py --output-device 3 --volume 1.0
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.voice_control.audio_feedback import (  # noqa: E402
    SAMPLE_RATE, TAIL_MS, AudioFeedback, _tone)
from src.voice_control.audio_player import AudioPlayer  # noqa: E402

SPEAKER_DEVICE_NAME = "UACDemoV1.0"  # Spot's USB speaker, same as client_mic

LEAD_S = 0.25  # silence first so the USB speaker is awake (it drops the start of a fresh stream)
GAP_S = 1.5

# (method name, when the robot plays it)
CHIMES = [
    ("ready", "Power-on: pipeline started and ready"),
    ("listen_cue", "Mic open, go ahead and talk (after wake phrase / after Spot finishes)"),
    ("think_cue", "Got it, working (utterance captured, thinking)"),
    ("speak_cue", "About to speak"),
    ("command_ok", "Command accepted"),
    ("chain_next", "Next action in a chained command"),
    ("error", "Something failed"),
    ("released_cue", "Speaker lock released (conversation over)"),
]


class _Capture:
    """Stands in for AudioPlayer: keeps the samples instead of playing them."""
    samples = None

    def is_available(self):
        return True

    def enqueue_raw(self, samples, rate, label=""):
        self.samples = samples


def listen_cue_louder():
    """Option under consideration: listen_cue at 200 ms, vol 0.35."""
    e5 = _tone(659, 100, 0.35)
    a5 = _tone(880, 100, 0.35)
    return np.concatenate([e5[:-int(SAMPLE_RATE * TAIL_MS / 1000)], a5])


def find_speaker():
    for i, d in enumerate(sd.query_devices()):
        if SPEAKER_DEVICE_NAME.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    return None


# --- Candidate redesign: distinct timbres, not just different sine notes ---
def _t(dur_s):
    return np.arange(int(dur_s * SAMPLE_RATE)) / SAMPLE_RATE


def _fade(x, ms=5):
    n = int(SAMPLE_RATE * ms / 1000)
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def bell(f, dur=0.45, vol=0.25):
    """FM bell: bright, ringing, long decay."""
    t = _t(dur)
    idx = 3.0 * np.exp(-t * 6)
    return _fade(vol * np.exp(-t * 7) * np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 1.4 * f * t)))


def pluck(f, dur=0.3, vol=0.3):
    """Karplus-Strong plucked string: warm, guitar-like."""
    n = int(dur * SAMPLE_RATE)
    period = int(SAMPLE_RATE / f)
    buf = np.random.default_rng(int(f)).uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = 0.996 * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return _fade(vol * out)


def wood(f=900, dur=0.07, vol=0.3):
    """Woodblock tick: short, dry, percussive."""
    t = _t(dur)
    return _fade(vol * np.exp(-t * 60) * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2.76 * f * t)), 2)


def whoosh(rising=True, dur=0.3, vol=0.25):
    """Filtered-noise swoosh with a sweeping centre."""
    t = _t(dur)
    noise = np.random.default_rng(7).standard_normal(len(t))
    f = np.linspace(400, 3000, len(t)) if rising else np.linspace(3000, 400, len(t))
    carrier = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    env = np.sin(np.pi * t / dur)
    sm = np.convolve(noise, np.ones(24) / 24, mode="same")
    return _fade(vol * env * (0.6 * carrier + 0.4 * sm / (np.abs(sm).max() + 1e-9)))


def buzz(f, dur=0.18, vol=0.18):
    """Square-ish buzz: harsh, unmistakably 'wrong'."""
    t = _t(dur)
    return _fade(vol * np.tanh(4 * np.sin(2 * np.pi * f * t)) * np.exp(-t * 4))


def gap(s):
    return np.zeros(int(s * SAMPLE_RATE))


def seq(*parts):
    return np.concatenate(parts).astype(np.float32)


NEW = [
    ("new_ready", "Power-on: rising bell arpeggio", lambda: seq(bell(523, .25), bell(659, .25), bell(784, .25), bell(1047, .5))),
    ("new_listen_A", "Mic open: two bright rising bells", lambda: seq(bell(1319, .18), bell(1760, .4))),
    ("new_listen_B", "Mic open: rising whoosh", lambda: whoosh(True)),
    ("new_think_A", "Thinking: three soft woodblock ticks", lambda: seq(wood(700), gap(.08), wood(700), gap(.08), wood(700))),
    ("new_think_B", "Thinking: low pluck", lambda: pluck(196, .4)),
    ("new_speak", "About to speak: single soft pluck (or drop it: the speech itself is the cue)", lambda: pluck(392, .25, .2)),
    ("new_ok", "Command accepted: quick bright pluck pair", lambda: seq(pluck(523, .12), pluck(784, .3))),
    ("new_chain", "Next chained action: one woodblock tick", lambda: wood(1100)),
    ("new_error", "Something failed: two low buzzes", lambda: seq(buzz(147), gap(.06), buzz(110, .28))),
    ("new_released", "Conversation over: falling whoosh + low bell", lambda: seq(whoosh(False, .25), bell(392, .5))),
]


def play(player, samples):
    """Play through the pipeline's own AudioPlayer (same stream, resampling and device)."""
    lead = np.zeros(int(LEAD_S * SAMPLE_RATE), dtype=np.float32)
    player.enqueue_raw(np.concatenate([lead, samples.astype(np.float32)]), SAMPLE_RATE, label="preview")
    player.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="play one chime by name (e.g. listen_cue, new_listen_A)")
    ap.add_argument("--set", choices=["current", "new", "both"], default="current",
                    help="current pipeline chimes, the candidate redesign, or both")
    ap.add_argument("--repeat", type=int, default=1, help="times to play each chime")
    ap.add_argument("--volume", type=float, default=1.0, help="master gain, same as the pipeline --volume")
    ap.add_argument("--output-device", type=int, default=None, help="sounddevice output index")
    args = ap.parse_args()

    device = args.output_device if args.output_device is not None else find_speaker()
    print(f"output device: {device if device is not None else 'system default'}")
    player = AudioPlayer(output_device=device)
    if not player.is_available():
        sys.exit("audio player failed to open the output device")

    cap = _Capture()
    fb = AudioFeedback(player=cap, volume=args.volume)
    items = [(n, d, None) for n, d in CHIMES]
    items.append(("listen_cue_louder", "OPTION: listen_cue at 200 ms, vol 0.35 (compare with listen_cue)",
                  listen_cue_louder() * fb.volume))
    new_items = [(n, d, f() * fb.volume) for n, d, f in NEW]
    if args.only:
        items = items + new_items
    elif args.set == "new":
        items = new_items
    elif args.set == "both":
        items = items + new_items
    if args.only:
        items = [i for i in items if i[0] == args.only]
        if not items:
            sys.exit(f"unknown chime {args.only!r}; choose from: "
                     + ", ".join([n for n, _ in CHIMES] + ["listen_cue_louder"] + [n for n, _, _ in NEW]))

    for name, desc, samples in items:
        if samples is None:
            getattr(fb, name)()
            samples = cap.samples
        print(f"{name:18s} {len(samples) / SAMPLE_RATE * 1000:4.0f} ms  {desc}")
        for _ in range(args.repeat):
            play(player, samples)
            time.sleep(GAP_S)
    player.shutdown()


if __name__ == "__main__":
    main()
