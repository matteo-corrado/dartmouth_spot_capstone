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


def play(samples, device):
    lead = np.zeros(int(LEAD_S * SAMPLE_RATE), dtype=np.float32)
    sd.play(np.concatenate([lead, samples.astype(np.float32)]), SAMPLE_RATE, device=device)
    sd.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="play one chime by name (e.g. listen_cue, listen_cue_louder)")
    ap.add_argument("--repeat", type=int, default=1, help="times to play each chime")
    ap.add_argument("--volume", type=float, default=1.0, help="master gain, same as the pipeline --volume")
    ap.add_argument("--output-device", type=int, default=None, help="sounddevice output index")
    args = ap.parse_args()

    cap = _Capture()
    fb = AudioFeedback(player=cap, volume=args.volume)
    items = [(n, d, None) for n, d in CHIMES]
    items.append(("listen_cue_louder", "OPTION: listen_cue at 200 ms, vol 0.35 (compare with listen_cue)",
                  listen_cue_louder() * fb.volume))
    if args.only:
        items = [i for i in items if i[0] == args.only]
        if not items:
            sys.exit(f"unknown chime {args.only!r}; choose from: "
                     + ", ".join([n for n, _ in CHIMES] + ["listen_cue_louder"]))

    for name, desc, samples in items:
        if samples is None:
            getattr(fb, name)()
            samples = cap.samples
        print(f"{name:18s} {len(samples) / SAMPLE_RATE * 1000:4.0f} ms  {desc}")
        for _ in range(args.repeat):
            play(samples, args.output_device)
            time.sleep(GAP_S)


if __name__ == "__main__":
    main()
