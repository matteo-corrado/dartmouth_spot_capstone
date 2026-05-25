"""Audition every persona × both TTS backends. Plays one in-character line per
pair through PulseAudio default sink, prompts user y/n/s, writes verdict
table to docs/project/persona-audition-<date>.md.

Use --dry-run to print the matrix without playing audio or hitting ElevenLabs.
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# One in-character line per persona. Same persona → same text across backends
# so the user is comparing voice timbre + delivery, not content.
SAMPLE_TEXTS = {
    "tour_guide":     "Welcome to Dartmouth. Follow me and I'll show you the green.",
    "pirate":         "Arr matey, hoist the colors and set course for the library!",
    "snarky":         "Oh great, another tour. I'll try to contain my excitement.",
    "butler":         "Good afternoon. Shall I escort you to the dining hall?",
    "shakespeare":    "Hark! What fair stranger doth grace our hallowed grounds?",
    "gen_z":          "Yo, lowkey welcome to Dartmouth, this place hits different.",
    "surfer":         "Whoa dude, welcome to the campus — gnarly day for a tour.",
    "narrator":       "Welcome to Dartmouth College, where centuries of tradition meet modern inquiry.",
    "news_anchor":    "Good evening. Coming up: a tour of Dartmouth's historic campus.",
}

BACKENDS = [("kokoro", "kokoro_v1"), ("elevenlabs", "elevenlabs")]
SAMPLE_RATE = 24000  # both backends emit 24kHz int16 PCM


def load_personas(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print persona × backend matrix, skip audio + ElevenLabs calls")
    ap.add_argument("--only", default=None,
                    help="Comma-separated persona names to audition (default: all)")
    args = ap.parse_args()

    personas = load_personas(ROOT / "config" / "personas.yaml")
    names = list(personas.keys())
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        names = [n for n in names if n in wanted]
        missing = wanted - set(names)
        if missing:
            print(f"WARN: requested personas not in yaml: {sorted(missing)}")

    print(f"Auditioning {len(names)} personas × {len(BACKENDS)} backends "
          f"= {len(names) * len(BACKENDS)} clips")
    for name in names:
        text = SAMPLE_TEXTS.get(name, f"Hello, I am Spot in {name} mode.")
        for backend_name, voice_key in BACKENDS:
            voice = personas[name].get("voices", {}).get(voice_key)
            status = "READY" if voice else "NO VOICE_ID"
            print(f"  [{name:14s}] {backend_name:11s} voice={voice or '-':25s} "
                  f"text={text!r}  [{status}]")
    if args.dry_run:
        print("\n--dry-run: matrix printed, no audio played, no API hit.")
        return 0
    print("\nTODO Task 2: wire backend instantiation + playback + verdict loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
