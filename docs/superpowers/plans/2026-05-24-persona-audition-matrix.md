# Persona × Backend Audition Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audition every persona in `config/personas.yaml` through both TTS backends (Kokoro v1 + ElevenLabs), capture human fit/miss verdicts, surface weak persona-voice pairings (suspected: Kokoro voices generic).

**Architecture:** One interactive Python script. For each persona, render the same in-character line twice — first Kokoro, then ElevenLabs — play through current PulseAudio default sink (Jieli USB DAC), prompt user `y/n/s` per clip, write results to a markdown table at `docs/project/persona-audition-<date>.md`. Backends instantiated directly (bypassing `get_backend()` env singleton) so both coexist in process.

**Tech Stack:** existing `tts.kokoro.KokoroBackend`, `tts.elevenlabs.ElevenLabsBackend`, `sounddevice` for playback (24 kHz PCM int16), `yaml` for personas, stdlib `input()` for verdicts.

**Out of scope:** No code change to backends, persona schema, or dispatcher. No new tests (interactive audio tool — TDD doesn't fit; sanity dry-run mode covers structural correctness).

---

## File Structure

- **Create:** `scripts/audition_personas.py` — interactive audition driver
- **Create:** `docs/project/persona-audition-2026-05-24.md` — verdict table (written by script)
- **Modify:** none

---

### Task 1: Sample-text catalog + dry-run script skeleton

**Files:**
- Create: `scripts/audition_personas.py`

- [ ] **Step 1: Create script with sample-text dict + persona-loading + `--dry-run` mode**

```python
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
```

- [ ] **Step 2: Run dry-run to verify matrix**

```bash
source spot-env/bin/activate && python scripts/audition_personas.py --dry-run
```

Expected: prints 9 personas × 2 backends = 18 rows. Each row shows voice_id (or `-`) + `READY` or `NO VOICE_ID`. Exits 0. No audio. No network.

Pause and read output: confirm 9 persona names match `config/personas.yaml`. Confirm Kokoro voice_ids are short slugs (`af_sarah`, `am_fenrir`, …) and ElevenLabs voice_ids are 20-char alphanumeric strings.

- [ ] **Step 3: Commit dry-run scaffold**

```bash
git add scripts/audition_personas.py
git commit -m "stage 2e1 audition: persona × backend matrix dry-run scaffold"
```

---

### Task 2: Backend instantiation + playback + verdict loop

**Files:**
- Modify: `scripts/audition_personas.py`

- [ ] **Step 1: Replace `main()` body — instantiate both backends, render+play per pair, collect verdicts**

Replace the loop in `main()` from Task 1 (everything after the persona-name filtering block, BEFORE `if args.dry_run`) with this. Keep the dry-run block intact above; the new code runs only when `not args.dry_run`.

```python
    # (kept from Task 1)
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
```

- [ ] **Step 2: Verify dry-run still works**

```bash
source spot-env/bin/activate && python scripts/audition_personas.py --dry-run
```

Expected: identical output to Task 1 Step 2 (matrix printed, exit 0, no audio).

- [ ] **Step 3: Smoke-test live mode with one persona**

```bash
set -a && source .env && set +a && source spot-env/bin/activate && \
  python scripts/audition_personas.py --only tour_guide
```

Expected: Kokoro renders + plays through Jieli speaker; user types verdict; ElevenLabs renders + plays; user types verdict; report written to `docs/project/persona-audition-<today>.md` with one row.

Verify report file exists and has 1 data row + tally lines.

- [ ] **Step 4: Commit playback loop**

```bash
git add scripts/audition_personas.py
git commit -m "stage 2e1 audition: live playback + verdict capture + markdown report"
```

---

### Task 3: Full audition run + verdict capture

**Files:**
- Create: `docs/project/persona-audition-2026-05-24.md` (written by script)

- [ ] **Step 1: Confirm audio path before launching the full run**

```bash
pactl info | grep "Default Sink"
```

Expected: `Default Sink: alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.analog-stereo`. If not, set it:

```bash
pactl set-default-sink alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.analog-stereo
```

Confirm the Jieli USB DAC is plugged into the Spot speaker and volume is up.

- [ ] **Step 2: Run the full audition**

```bash
set -a && source .env && set +a && source spot-env/bin/activate && \
  python scripts/audition_personas.py
```

Listen to all 18 clips (9 personas × 2 backends). Type `y` if the voice fits the persona, `n` if it misses, `s` to skip, `r` to replay a clip, `q` to quit early (writes partial report).

Take note of any pair where Kokoro feels generic vs ElevenLabs in-character — those are the rows we need to either swap Kokoro slug for or accept ElevenLabs-only.

- [ ] **Step 3: Inspect the generated report**

```bash
cat docs/project/persona-audition-$(date +%F).md
```

Expected: 9 data rows + Kokoro/ElevenLabs fit tallies. Eyeball: does the tally match the verbal verdicts you gave?

- [ ] **Step 4: Commit the report**

```bash
git add docs/project/persona-audition-*.md
git commit -m "stage 2e1 audition: persona × backend verdict matrix (run 1)"
```

---

### Task 4 (conditional): Triage outcomes

Only do this task if the report shows ≥2 Kokoro `n` verdicts. Otherwise skip — Kokoro is fine and the suspicion was wrong.

**Files:**
- Modify: `config/personas.yaml` (one line per remediated persona)
- Reference: `models/tts/kokoro-en-v0_19/voices.json` for available Kokoro slugs

- [ ] **Step 1: List candidate Kokoro slug swaps for each Kokoro miss**

For each persona where Kokoro got `n`, look up alternative slugs in the Kokoro voice manifest:

```bash
python -c "
import onnx, json
# voices.json sits next to the model — adjust path if different
from pathlib import Path
p = Path('models/tts/kokoro-en-v0_19/voices.json')
if p.exists():
    print(json.dumps(list(json.loads(p.read_text()).keys()), indent=2))
else:
    print('voices.json not at expected path — check models/tts/ tree')
"
```

Expected: list of Kokoro speaker slugs (`af_sarah`, `am_fenrir`, `am_onyx`, `bm_george`, `bm_lewis`, `af_nova`, …).

For each miss, propose a swap rationale (e.g. `pirate kokoro=am_fenrir → am_onyx` because Onyx is gravellier).

- [ ] **Step 2: STOP — present swap proposals to the user before editing yaml**

Do not silently rewrite `config/personas.yaml`. Surface the proposed swaps as a short markdown table and wait for the user to approve, reject, or amend each row. Persona-voice fit is the user's call, not the agent's.

- [ ] **Step 3: After approval, apply swaps and re-run audition for changed personas only**

```bash
python scripts/audition_personas.py --only persona1,persona2
```

Verify the new Kokoro slugs now get `y` verdicts. If still `n`, loop back to Step 1 with the next candidate.

- [ ] **Step 4: Commit swaps + delta report**

```bash
git add config/personas.yaml docs/project/persona-audition-*.md
git commit -m "stage 2e1 audition: swap Kokoro slugs for personas X, Y (run 2)"
```

---

## Self-Review

**Spec coverage:** Audition every persona × both backends ✓ (Task 2 + 3). Capture user verdicts ✓ (verdict loop). Surface Kokoro weakness ✓ (Kokoro fit tally in report + Task 4 triage gate). Suspected-generic hypothesis testable ✓ (per-pair verdicts make it falsifiable).

**Placeholder scan:** No TBDs. Every step has runnable code or a runnable command. Task 4 is explicitly conditional with a clear gate.

**Type consistency:** `verdicts: dict[str, dict[str, str]]` — keys `kokoro` / `elevenlabs` match `BACKENDS` tuple names. `SAMPLE_TEXTS` keys match `names` from personas.yaml. `write_report` reads the same dict shape that the loop writes.

**Open risks:**
- 5 of the 8 prewarmed stock personas didn't get an ElevenLabs voice_id (cowboy, robot, wizard, drill_sergeant, scientist — Library NO MATCH after prewarm rerun). They are NOT in personas.yaml so won't appear in the audition. The 9 present ones (tour_guide, pirate, snarky, butler, shakespeare, gen_z, surfer, narrator, news_anchor) all have both voice_ids.
- ElevenLabs synth makes a real API call per clip. 9 calls. Cost is negligible (Creator tier, short lines) but it does hit the network — `--dry-run` skips this.
- `sounddevice.play` routes through ALSA default → PulseAudio default sink. Task 3 Step 1 verifies the sink before the run.
