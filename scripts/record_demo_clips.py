#!/usr/bin/env python3
"""Guided demo-clip recorder (quiet / loud / overlap rooms).

Records stereo 16 kHz int16 clips from the XVF3800 plus a DoA log per clip into
tests/audio/demo/<condition>/. Optional MicArray supplies the DoA log.

  spot-env/bin/python scripts/record_demo_clips.py --condition quiet
  spot-env/bin/python scripts/record_demo_clips.py --condition loud --phrases "Hey Spot|Listen up Spot"

Beam-steering A/B (same room, back to back; analyse with tests/audio/beam_ab.py):
  spot-env/bin/python scripts/record_demo_clips.py --condition ab_off --set overlap --interferer-deg 90
  spot-env/bin/python scripts/record_demo_clips.py --condition ab_on  --set overlap --interferer-deg 90 --beam
--beam points both XVF3800 fixed beams at each clip's speaker angle (oracle
direction, the best case for live steering) and turns them off after the clip.
Every clip with a MicArray also gets <clip>.beams.csv (all 4 beam azimuths +
speech energies, 10 Hz).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    out.extend(SAFETY_PROMPTS)
    return out


# Safety words for the always-on stop listener (--set safety). No wake phrase.
SAFETY_PROMPTS = [
    Prompt(f"safety_{word}_{style}", "safety", say, 0, 3.0, "", expected)
    for word, expected, variants in (
        ("stop", "stop", (("normal", 'Say "stop" (normal voice)'),
                          ("quiet", 'Say "stop" QUIETLY'),
                          ("sentence", 'Say "okay Spot, stop"'))),
        ("freeze", "freeze", (("normal", 'Say "freeze" (normal voice)'),
                              ("quiet", 'Say "freeze" QUIETLY'),
                              ("sentence", 'Say "okay Spot, freeze"'))),
    )
    for style, say in variants
] + [Prompt(f"safety_{w}_far", "safety", f'Step back ~3 m, then say "{w}" (normal voice)', 0, 3.0, "", w)
     for w in ("stop", "freeze")
] + [Prompt(f"safety_{w}_quiet{i}", "safety", f'Say "{w}" QUIETLY (take {i})', 0, 3.0, "", w)
     for w in ("stop", "freeze") for i in (2, 3)
] + [Prompt(f"safety_{w}_far_quiet", "safety", f'Step back ~3 m, then say "{w}" QUIETLY', 0, 3.0, "", w)
     for w in ("stop", "freeze")]


# Short sets: only the clips a noise condition changes (--set). "full" = everything.
SETS = {
    "babble": lambda p: (p.kind == "wake" and p.clip.endswith("_front"))
    or p.clip in ("quick_reply_1", "quick_reply_2", "background"),
    "overlap": lambda p: p.kind == "overlap" or (p.kind == "wake" and p.clip.endswith("_front")),
    "safety": lambda p: p.kind == "safety",
}


def select_prompts(prompts: list[Prompt], set_name: str) -> list[Prompt]:
    if set_name == "full":
        return prompts
    return [p for p in prompts if SETS[set_name](p)]


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


def body_to_mic_deg(body_deg, offset_deg, sign) -> float:
    """Inverse of speaker_lock.mic_to_body_deg (sign is +-1)."""
    return (sign * body_deg + offset_deg) % 360.0


BEAM_COLS = ("az1", "az2", "az_free", "az_auto", "en1", "en2", "en_free", "en_auto")


def write_beams_csv(path: Path, samples: list, t0: float) -> None:
    """samples: (t, az_rad[4], energy[4]); azimuths written in mic-frame degrees."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s", *BEAM_COLS])
        for t, az, en in samples:
            w.writerow([f"{t - t0:.3f}", *(f"{math.degrees(a) % 360.0:.1f}" for a in az),
                        *(f"{e:.4g}" for e in en)])


def collect_beams(mic, t0: float, duration: float, hz=10.0, clock=time.monotonic,
                  sleep=time.sleep) -> list:
    """Poll all 4 beams' azimuth + speech energy for `duration` s (read errors skipped)."""
    got: list = []
    while clock() - t0 < duration:
        t = clock()
        try:
            az, en = mic.read_beams()
            got.append((t, az, en))
        except Exception:  # noqa: BLE001 - a missed sample is fine
            pass
        sleep(1.0 / hz)
    return got


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
CHIME_DEVICE_LEAD_S = 0.25  # silence before the cue so the USB speaker is awake


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
        name = "listen_cue" if hasattr(fb, "listen_cue") else "ready"
        getattr(fb, name)()
        if cap.got is not None:
            return cap.got[0], cap.got[1], f"audio_feedback.{name}"
    except Exception as e:
        print(f"  (AudioFeedback unavailable: {e})")
    print("  WARNING: using GENERATED fallback tone, not the production chime")
    t = np.arange(int(0.24 * 48000)) / 48000
    return (0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), 48000, "generated_tone"


def play_samples(samples: np.ndarray, rate: int, output_device) -> None:
    """Blocking playback on its own OutputStream (safe alongside sd.rec)."""
    import sounddevice as sd
    # USB speakers drop the first ~100 ms of a freshly opened stream; lead with
    # silence so a short cue (listen_cue is 140 ms) isn't swallowed. The live
    # AudioPlayer keeps its stream open, so it doesn't need this.
    lead = np.zeros(int(CHIME_DEVICE_LEAD_S * rate), dtype=np.float32)
    with sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                         device=output_device) as out:
        out.write(np.concatenate([lead, samples]).reshape(-1, 1))


def record_clip(device: int, prompt: Prompt, mic, output_device=None, beam_mic_deg=None):
    """Returns (audio (N,2) int16, t0, doa_samples, chime_info | None, beam_samples).

    beam_mic_deg: point both fixed beams there (mic frame) for this clip, off after."""
    import threading
    import sounddevice as sd
    if beam_mic_deg is not None:
        mic.set_fixed_beams(math.radians(beam_mic_deg))
        time.sleep(0.3)  # let the beamformer settle before GO
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
            info["chime_start_s"] = round(time.monotonic() - t0 + CHIME_DEVICE_LEAD_S, 3)
            try:
                play_samples(samples, rate, output_device)
            except Exception as e:
                print(f"  chime playback failed: {e}")
                info["error"] = str(e)
            info["chime_end_s"] = round(max(time.monotonic() - t0,
                                            info["chime_start_s"] + len(samples) / rate), 3)
            info["source"] = source

        th = threading.Thread(target=_chime, daemon=True)
        th.start()
    beams: list = []
    bt = None
    if mic is not None:
        bt = threading.Thread(target=lambda: beams.extend(collect_beams(mic, t0, prompt.duration_s)),
                              daemon=True)
        bt.start()
    try:
        doa = collect_doa(mic, t0, prompt.duration_s) if mic is not None else []
        sd.wait()
        if th is not None:
            th.join(timeout=2.0)
        if bt is not None:
            bt.join(timeout=2.0)
    finally:
        if beam_mic_deg is not None:
            mic.set_fixed_beams(None)
    return audio, t0, doa, (info or None), beams


def save_manifest_entry(path: Path, prompt: Prompt, extra: dict | None = None) -> None:
    try:
        entries = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        os.replace(path, path.with_name(f"manifest.bad.{int(time.time())}.json"))
        entries = []
    entries = [e for e in entries if e.get("clip") != prompt.clip]
    entries.append({"clip": prompt.clip, "kind": prompt.kind, "phrase": prompt.phrase,
                    "expected_text": prompt.expected_text, "angle_deg": prompt.angle_deg,
                    **(extra or {})})
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, path)


def run_session(prompts: list[Prompt], out_dir: Path, record_fn, input_fn=input,
                manifest_extra: dict | None = None) -> int:
    """record_fn(prompt) -> (audio, t0, doa_samples, chime_info|None[, beam_samples]).
    Returns clips written. manifest_extra is merged into every manifest entry."""
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
                    audio, t0, doa, chime, *rest = record_fn(p)
                    beams = rest[0] if rest else None
                    for stale in (f"{p.clip}.doa.csv", f"{p.clip}.chime.json", f"{p.clip}.beams.csv"):
                        (out_dir / stale).unlink(missing_ok=True)
                    write_wav(wav, audio)
                    if doa:
                        write_doa_csv(out_dir / f"{p.clip}.doa.csv", doa, t0)
                    if beams:
                        write_beams_csv(out_dir / f"{p.clip}.beams.csv", beams, t0)
                    if chime:
                        (out_dir / f"{p.clip}.chime.json").write_text(json.dumps(chime))
                    save_manifest_entry(manifest, p, manifest_extra)
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
    ap.add_argument("--set", choices=("full", *SETS), default="full",
                    help="babble: front wakes + 2 quick replies + background; "
                         "overlap: front wakes + overlap clips; full: everything")
    ap.add_argument("--beam", action="store_true",
                    help="aim both fixed beams at each clip's speaker angle (beam-steering A/B)")
    ap.add_argument("--interferer-deg", type=float, default=None,
                    help="body-frame angle of the second talker / podcast (0 front, 90 left), "
                         "saved in the manifest for tests/audio/beam_ab.py")
    args = ap.parse_args()

    phrases = load_phrases(args.phrases, args.phrases_file)
    prompts = select_prompts(build_prompts(phrases), args.set)
    print(f"{len(prompts)} clips in set '{args.set}' (~{sum(p.duration_s for p in prompts) / 60:.1f} min of recording)")
    out_dir = OUT_ROOT / args.condition
    device = find_device()
    mic = None
    if MicArray is not None:
        mic = MicArray.open()
        if mic is not None:
            mic.start_poller()
    print(f"Condition: {args.condition}   phrases: {phrases}")
    print(f"DoA log: {'on' if mic is not None else 'off (no MicArray)'}   output: {out_dir}")
    beam_of = lambda p: None  # noqa: E731
    if args.beam:
        if mic is None:
            raise SystemExit("--beam needs the XVF3800 control link (MicArray); aborting.")
        from src.voice_control.tuning import load_tuning
        t = load_tuning()
        beam_of = lambda p: body_to_mic_deg(p.angle_deg, t.mic_yaw_offset_deg, t.mic_yaw_sign)  # noqa: E731
        print(f"Beam steering ON: fixed beams aimed at each clip's speaker angle "
              f"(front = {body_to_mic_deg(0, t.mic_yaw_offset_deg, t.mic_yaw_sign):.0f} deg mic). "
              "Keep the second talker in the SAME spot for both A/B runs.")
    input("Spot powered on, mic in its Thursday position. Press Enter to begin.")
    try:
        n = run_session(prompts, out_dir,
                        lambda p: record_clip(device, p, mic, args.output_device, beam_of(p)),
                        manifest_extra={"beam_steered": bool(args.beam),
                                        "interferer_body_deg": args.interferer_deg})
    finally:
        if mic is not None:
            mic.close()
    print(f"\n{n} clips written to {out_dir}")


if __name__ == "__main__":
    main()
