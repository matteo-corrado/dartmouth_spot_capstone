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

    # --- live audition ---
    from src.voice_control.tts.kokoro import KokoroBackend
    from src.voice_control.tts.elevenlabs import ElevenLabsBackend

    print("\nWarming backends (Kokoro loads ~85MB ONNX, ElevenLabs needs API key)...")
    kokoro = KokoroBackend()
    eleven = ElevenLabsBackend()  # raises if ELEVENLABS_API_KEY unset
    backends = {"kokoro": kokoro, "elevenlabs": eleven}

    # verdicts[name][backend_name] = "y" | "n" | "s" | "ERR:<msg>"
    verdicts: dict[str, dict[str, str]] = {n: {} for n in names}

    print("\nStarting audition. After each clip, type: y (fit), n (miss), "
          "s (skip), r (replay), q (quit early).\n")
    for name in names:
        text = SAMPLE_TEXTS.get(name, f"Hello, I am Spot in {name} mode.")
        desc = personas[name].get("prompt_prefix", "")[:80]
        print(f"\n=== {name} ===  {desc!r}")
        print(f"  text: {text!r}")
        for backend_name, voice_key in BACKENDS:
            voice = personas[name].get("voices", {}).get(voice_key)
            if not voice:
                verdicts[name][backend_name] = "s"
                print(f"  [{backend_name}] NO VOICE_ID — auto-skip")
                continue
            while True:
                print(f"  [{backend_name}] voice={voice} — rendering...", flush=True)
                try:
                    pcm = backends[backend_name].synthesize(text, voice)
                    audio = np.frombuffer(pcm, dtype="<i2")
                    sd.play(audio, SAMPLE_RATE)
                    sd.wait()
                except Exception as e:
                    print(f"    ERROR: {e}")
                    verdicts[name][backend_name] = f"ERR:{type(e).__name__}"
                    break
                v = input(f"    [{name}/{backend_name}] fit? (y/n/s/r/q): ").strip().lower()
                if v == "r":
                    continue
                if v == "q":
                    print("Early quit — writing partial results.")
                    write_report(verdicts, names)
                    return 0
                if v not in ("y", "n", "s"):
                    v = "s"
                verdicts[name][backend_name] = v
                break

    write_report(verdicts, names)
    return 0


def write_report(verdicts: dict, names: list) -> None:
    date = dt.date.today().isoformat()
    out = ROOT / "docs" / "project" / f"persona-audition-{date}.md"
    lines = [
        f"# Persona × Backend Audition — {date}",
        "",
        "Verdicts: `y` = fits persona, `n` = misses, `s` = skipped / no voice, "
        "`ERR:<type>` = synth failure.",
        "",
        "| persona | kokoro | elevenlabs | sample text |",
        "|---|---|---|---|",
    ]
    for name in names:
        k = verdicts.get(name, {}).get("kokoro", "-")
        e = verdicts.get(name, {}).get("elevenlabs", "-")
        text = SAMPLE_TEXTS.get(name, "").replace("|", "\\|")
        lines.append(f"| {name} | {k} | {e} | {text} |")

    n_k_fit = sum(1 for n in names if verdicts.get(n, {}).get("kokoro") == "y")
    n_e_fit = sum(1 for n in names if verdicts.get(n, {}).get("elevenlabs") == "y")
    lines += [
        "",
        f"**Kokoro fit:** {n_k_fit}/{len(names)}",
        f"**ElevenLabs fit:** {n_e_fit}/{len(names)}",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
