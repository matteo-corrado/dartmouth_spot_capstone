#!/usr/bin/env python3
"""Interactive audio-corpus recorder for Stage 2A T6.

Captures from XVF3800 channel 0 (beamformed), 16 kHz mono PCM16, into
tests/audio/<name>.wav. Walks the user through every utterance the plan
demands, with distance variations for wake phrases.
"""
from __future__ import annotations
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "audio"
SAMPLE_RATE = 16000
DEVICE_NAME_HINT = "reSpeaker"
DEFAULT_DURATION = 3.0


def find_device() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and DEVICE_NAME_HINT in d["name"]:
            return i
    raise SystemExit(f"No input device matching {DEVICE_NAME_HINT!r}; aborting.")


def record_one(device: int, out_path: Path, duration: float) -> None:
    if out_path.exists():
        ans = input(f"  {out_path.name} exists — overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("  skipped")
            return
    print(f"  Recording {duration:.1f}s -> {out_path.name}")
    for n in (3, 2, 1):
        print(f"    {n}...", end="", flush=True)
        time.sleep(0.5)
    print(" GO")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=2, dtype="int16", device=device)
    sd.wait()
    mono = audio[:, 0]
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(mono.tobytes())
    peak = float(np.abs(mono).max()) / 32768.0
    print(f"  done — peak {peak:.2f} {'(too quiet!)' if peak < 0.05 else ''}")


# (utterance_name, prompt_phrase, distance_label, duration_s)
WAKE_PROMPTS = [
    ("hey_spot_close", "Hey Spot", "close ~30 cm", 2.5),
    ("hey_spot_mid", "Hey Spot", "mid ~1.5 m", 2.5),
    ("hey_spot_far", "Hey Spot", "far ~3 m", 2.5),
    ("hey_spot_stand_up", "Hey Spot, stand up", "close ~30 cm", 3.0),
    ("hey_spot_walk_forward", "Hey Spot, walk forward", "close ~30 cm", 3.0),
]

COMMAND_PROMPTS = [
    ("stand_up", "stand up", "close", 2.5),
    ("sit_down", "sit down", "close", 2.5),
    ("walk_forward_1m", "walk forward one meter", "close", 3.0),
    ("walk_forward_two_meters", "walk forward two meters", "close", 3.0),
    ("turn_left_90", "turn left 90 degrees", "close", 3.0),
    ("turn_around", "turn around", "close", 2.5),
    ("go_to_the_kitchen", "go to the kitchen", "close", 3.0),
    ("come_here", "come here", "close", 2.5),
    ("follow_me", "follow me", "close", 2.5),
    ("stop", "stop", "close", 2.0),
    ("battery_status", "what's your battery status", "close", 3.0),
    ("what_do_you_see", "what do you see", "close", 3.0),
    ("describe_surroundings", "describe your surroundings", "close", 3.5),
    ("tell_me_about_dartmouth_hall", "tell me about Dartmouth Hall", "close", 3.5),
    ("find_a_chair", "find a chair", "close", 3.0),
]

ADVERSARIAL_PROMPTS = [
    ("adv_spot_the_difference", "spot the difference", "mid ~1.5 m", 3.0),
    ("adv_on_the_spot", "on the spot", "mid ~1.5 m", 2.5),
    ("adv_x_marks_the_spot", "X marks the spot", "mid ~1.5 m", 3.0),
    ("adv_hey_dog", "hey dog", "mid ~1.5 m", 2.0),
    ("adv_hey_stop", "hey stop", "mid ~1.5 m", 2.0),
    ("adv_hot_spot", "hot spot", "mid ~1.5 m", 2.0),
]

# Idle clips: no speaker, ambient room noise.
IDLE_PROMPTS = [
    ("idle_01", "[silent — just room noise]", "mic position", 5.0),
    ("idle_02", "[silent — just room noise]", "mic position", 5.0),
    ("idle_03", "[silent — just room noise]", "mic position", 5.0),
    ("idle_04", "[silent — just room noise]", "mic position", 5.0),
    ("idle_05", "[silent — just room noise]", "mic position", 5.0),
]


def section(title: str, prompts: list) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print("Press Enter to start each, type 's' to skip, 'q' to abort.")


def run_section(device: int, title: str, prompts: list) -> None:
    section(title, prompts)
    for name, phrase, distance, dur in prompts:
        out = OUT_DIR / f"{name}.wav"
        print(f"\n[{name}] distance={distance}")
        print(f'  Say: "{phrase}"')
        ans = input("  Ready? [Enter=record, s=skip, q=quit] ").strip().lower()
        if ans == "q":
            print("aborted")
            sys.exit(0)
        if ans == "s":
            print("  skipped")
            continue
        record_one(device, out, dur)


def _suffix(room: str | None) -> str:
    return f"_{room}" if room else ""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 2A T6 audio corpus recorder")
    ap.add_argument("--room", type=str, default=None,
                    help="Room label appended to filenames (e.g. 'atrium', 'lab'). "
                         "For tour-guide cross-room eval; re-run per environment.")
    ap.add_argument("--only", choices=["wake", "commands", "adversarial", "idle"],
                    default=None, help="Skip everything except this section.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = find_device()
    sfx = _suffix(args.room)
    print(f"Using input device {device}: {sd.query_devices()[device]['name']}")
    print(f"Output dir: {OUT_DIR}")
    if args.room:
        print(f"Room label: {args.room!r} (files get '{sfx}' suffix)")
    print()
    print("Spot's beamformed channel 0 is captured (matches client_mic production path).")
    print("Sit Spot powered-on at its normal position. Speak naturally.")
    input("Press Enter to begin.")
    sections = [
        ("wake", "Wake phrases — vary distances", WAKE_PROMPTS),
        ("commands", "Commands — close (~30 cm)", COMMAND_PROMPTS),
        ("adversarial", "Adversarial confusables — mid distance", ADVERSARIAL_PROMPTS),
        ("idle", "Idle / room noise — silent", IDLE_PROMPTS),
    ]
    for key, title, prompts in sections:
        if args.only and args.only != key:
            continue
        labeled = [(name + sfx, phrase, dist, dur) for name, phrase, dist, dur in prompts]
        run_section(device, title, labeled)
    print(f"\nDone. Files in {OUT_DIR}")
    print("Next: T12 mic-verify will replay these against the live pipeline.")
    print()
    print("To add another room later:")
    print("  spot-env/bin/python scripts/record_corpus.py --room <label> --only wake")
    print("  spot-env/bin/python scripts/record_corpus.py --room <label> --only idle")


if __name__ == "__main__":
    main()
