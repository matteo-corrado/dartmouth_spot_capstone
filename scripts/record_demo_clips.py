#!/usr/bin/env python3
"""Guided demo-clip recorder (quiet / loud / overlap rooms).

Records stereo 16 kHz int16 clips from the XVF3800 plus a DoA log per clip into
tests/audio/demo/<condition>/. Optional MicArray supplies the DoA log.

  spot-env/bin/python scripts/record_demo_clips.py --condition quiet
  spot-env/bin/python scripts/record_demo_clips.py --condition loud --phrases "Hey Spot|Listen up Spot"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:  # optional: absent without Task 2 / without pyusb
    from src.voice_control.mic_array import MicArray
except Exception:  # pragma: no cover
    MicArray = None

OUT_ROOT = REPO / "tests" / "audio" / "demo"
DEFAULT_PHRASES_FILE = REPO / "config" / "wake_phrases.txt"
SAMPLE_RATE = 16000
DEVICE_NAME_HINT = "reSpeaker"
ANGLES = (("front", 0), ("left", 90), ("back", 180), ("right", 270))
ANGLE_NAME = {a: n for n, a in ANGLES}
QUESTION = "what is this poster about?"
PAUSE_TEXT = "can you tell me about the project"


@dataclass(frozen=True)
class Prompt:
    clip: str
    kind: str          # wake | wake_pause | quick_reply | overlap | background
    say: str
    angle_deg: int
    duration_s: float
    phrase: str = ""
    expected_text: str = ""


def slug(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def readable(phrase: str) -> str:
    """'LISTEN UP SPOT' -> 'Listen up Spot' (first word capitalised, 'spot' -> 'Spot')."""
    words = ["Spot" if w == "spot" else w for w in phrase.strip().lower().split()]
    if words and words[0] != "Spot":
        words[0] = words[0].capitalize()
    return " ".join(words)


def build_prompts(phrases: list[str]) -> list[Prompt]:
    out: list[Prompt] = []
    for raw in phrases:
        p, s = readable(raw), slug(raw)
        for name, ang in ANGLES:
            out.append(Prompt(f"wake_{s}_{name}", "wake", f"{p}, {QUESTION}", ang, 4.0,
                              p, f"{raw.lower()} {QUESTION}"))
        out.append(Prompt(f"pause_{s}", "wake_pause",
                          f"{p}, can you tell me... [pause 1 s] ...about the project", 0, 6.0,
                          p, f"{raw.lower()} {PAUSE_TEXT}"))
    for i in (1, 2, 3):
        out.append(Prompt(f"quick_reply_{i}", "quick_reply",
                          'Wait for the ready chime, then immediately say: "what\'s your name"',
                          0, 4.0, "", "what's your name"))
    first = phrases[0] if phrases else "HEY SPOT"
    for i in (1, 2, 3):
        out.append(Prompt(f"overlap_{i}", "overlap",
                          f'Visitor stands FRONT of Spot; a SECOND person stands LEFT or BACK '
                          f'and talks continuously. Visitor says: "{readable(first)}, {QUESTION}"',
                          0, 6.0, readable(first), f"{first.lower()} {QUESTION}"))
    out.append(Prompt("background", "background",
                      "Stay silent (no wake phrase). Normal room noise / chatter.", 0, 120.0))
    return out


def load_phrases(phrases_arg: str | None, phrases_file: Path) -> list[str]:
    if phrases_arg:
        found = [p.strip() for p in phrases_arg.split("|") if p.strip()]
    elif phrases_file.exists():
        found = [ln.strip() for ln in phrases_file.read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    else:
        found = []
    return found or ["HEY SPOT"]


def find_device() -> int:
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 2 and DEVICE_NAME_HINT in d["name"]:
            return i
    raise SystemExit(f"No 2-channel input device matching {DEVICE_NAME_HINT!r}; aborting.")


def write_wav(path: Path, audio: np.ndarray) -> None:
    """audio: (N, 2) int16."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.ascontiguousarray(audio, dtype=np.int16).tobytes())


def write_doa_csv(path: Path, samples: list, t0: float) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s", "deg", "speech"])
        for t, deg, speech in samples:
            w.writerow([f"{t - t0:.3f}", int(deg), int(bool(speech))])


def collect_doa(mic, t0: float, duration: float, clock=time.monotonic, sleep=time.sleep) -> list:
    """Drain mic history during a clip (its deque is short, so poll in chunks)."""
    got: list = []
    last = t0
    while True:
        now = clock()
        for s in mic.samples_between(last, now):
            if s[0] > last:
                got.append(s)
                last = s[0]
        if now - t0 >= duration:
            return got
        sleep(0.5)


CHIME_LEAD_S = 0.5


def chime_samples() -> tuple[np.ndarray, int, str]:
    """The production 'ready' cue (AudioFeedback.ready), captured via a fake player."""
    try:
        from src.voice_control.audio_feedback import AudioFeedback, SAMPLE_RATE as FB_RATE

        class _Cap:
            got = None

            def is_available(self):
                return True

            def enqueue_raw(self, samples, rate, label=""):
                self.got = (np.asarray(samples, dtype=np.float32), int(rate))

        cap = _Cap()
        fb = AudioFeedback(player=cap)
        getattr(fb, "listen_cue", fb.ready)()
        if cap.got is not None:
            return cap.got[0], cap.got[1], "audio_feedback.ready"
    except Exception as e:
        print(f"  (AudioFeedback unavailable: {e})")
    print("  WARNING: using GENERATED fallback tone, not the production chime")
    t = np.arange(int(0.24 * 48000)) / 48000
    return (0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), 48000, "generated_tone"


def play_samples(samples: np.ndarray, rate: int, output_device) -> None:
    """Blocking playback on its own OutputStream (safe alongside sd.rec)."""
    import sounddevice as sd
    with sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                         device=output_device) as out:
        out.write(samples.reshape(-1, 1))


def record_clip(device: int, prompt: Prompt, mic, output_device=None):
    """Returns (audio (N,2) int16, t0, doa_samples, chime_info | None)."""
    import threading
    import sounddevice as sd
    for n in (3, 2, 1):
        print(f"    {n}...", end="", flush=True)
        time.sleep(0.5)
    print(" GO")
    t0 = time.monotonic()
    audio = sd.rec(int(prompt.duration_s * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=2, dtype="int16", device=device)
    info: dict = {}
    th = None
    if prompt.kind == "quick_reply":
        samples, rate, source = chime_samples()

        def _chime():
            time.sleep(CHIME_LEAD_S)
            info["chime_start_s"] = round(time.monotonic() - t0, 3)
            try:
                play_samples(samples, rate, output_device)
            except Exception as e:
                print(f"  chime playback failed: {e}")
                info["error"] = str(e)
            info["chime_end_s"] = round(time.monotonic() - t0, 3)
            info["source"] = source

        th = threading.Thread(target=_chime, daemon=True)
        th.start()
    doa = collect_doa(mic, t0, prompt.duration_s) if mic is not None else []
    sd.wait()
    if th is not None:
        th.join(timeout=2.0)
    return audio, t0, doa, (info or None)


def save_manifest_entry(path: Path, prompt: Prompt) -> None:
    try:
        entries = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        os.replace(path, path.with_name(f"manifest.bad.{int(time.time())}.json"))
        entries = []
    entries = [e for e in entries if e.get("clip") != prompt.clip]
    entries.append({"clip": prompt.clip, "kind": prompt.kind, "phrase": prompt.phrase,
                    "expected_text": prompt.expected_text, "angle_deg": prompt.angle_deg})
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, path)


def run_session(prompts: list[Prompt], out_dir: Path, record_fn, input_fn=input) -> int:
    """record_fn(prompt) -> (audio, t0, doa_samples, chime_info|None). Returns clips written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    written = 0
    try:
        for i, p in enumerate(prompts, 1):
            wav = out_dir / f"{p.clip}.wav"
            redo = False
            while True:
                exists = wav.exists() and not redo
                print(f"\n[{i}/{len(prompts)}] {p.clip}  {p.duration_s:.0f}s"
                      f"{'  (exists)' if exists else ''}")
                if p.kind in ("wake", "wake_pause"):
                    print(f"  Speak from the {ANGLE_NAME.get(p.angle_deg, 'front').upper()} of Spot")
                print(f"  Say: {p.say}")
                if exists:
                    ans = input_fn("  [Enter=skip (keep existing), r=redo, q=quit] ").strip().lower()
                    if ans == "":
                        ans = "s"
                else:
                    ans = input_fn("  [Enter=record, s=skip, q=quit] ").strip().lower()
                if ans == "q":
                    print("aborted")
                    return written
                if ans == "s":
                    print("  skipped")
                    break
                try:
                    audio, t0, doa, chime = record_fn(p)
                    for stale in (f"{p.clip}.doa.csv", f"{p.clip}.chime.json"):
                        (out_dir / stale).unlink(missing_ok=True)
                    write_wav(wav, audio)
                    if doa:
                        write_doa_csv(out_dir / f"{p.clip}.doa.csv", doa, t0)
                    if chime:
                        (out_dir / f"{p.clip}.chime.json").write_text(json.dumps(chime))
                    save_manifest_entry(manifest, p)
                except Exception as e:
                    print(f"  RECORD FAILED: {e!r}  (try again, or s to skip)")
                    continue
                written += 1
                peak = float(np.abs(audio).max()) / 32768.0
                print(f"  done, peak {peak:.2f}{' (too quiet)' if peak < 0.02 else ''}")
                if input_fn("  [Enter=keep, r=redo] ").strip().lower() != "r":
                    break
                redo = True
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted; clips so far are saved")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided demo-clip recorder")
    ap.add_argument("--condition", required=True, help="quiet | loud | overlap | any label")
    ap.add_argument("--phrases", default=None, help="'|'-separated wake phrases (overrides file)")
    ap.add_argument("--output-device", type=int, default=None,
                    help="sounddevice output index for the quick_reply chime (Spot speaker)")
    ap.add_argument("--phrases-file", type=Path, default=DEFAULT_PHRASES_FILE)
    args = ap.parse_args()

    phrases = load_phrases(args.phrases, args.phrases_file)
    prompts = build_prompts(phrases)
    out_dir = OUT_ROOT / args.condition
    device = find_device()
    mic = None
    if MicArray is not None:
        mic = MicArray.open()
        if mic is not None:
            mic.start_poller()
    print(f"Condition: {args.condition}   phrases: {phrases}")
    print(f"DoA log: {'on' if mic is not None else 'off (no MicArray)'}   output: {out_dir}")
    input("Spot powered on, mic in its Thursday position. Press Enter to begin.")
    try:
        n = run_session(prompts, out_dir, lambda p: record_clip(device, p, mic, args.output_device))
    finally:
        if mic is not None:
            mic.close()
    print(f"\n{n} clips written to {out_dir}")


if __name__ == "__main__":
    main()
