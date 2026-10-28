# Stage 2B — Spot End-to-End Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **HUMAN-IN-LOOP REQUIRED.** This plan validates the voice loop on the real Boston Dynamics Spot robot. Tasks 3-7 require Spot powered, E-stop in another terminal, clear lab space (≥3m × 3m), and a human operator within line of sight at all times. Robot motion is NOT delegable to a subagent.

**Goal:** Validate Stage 2A voice pipeline (llama.cpp brain + Nemotron Speech Streaming ASR + Silero VAD + LiveKit Wakeword) end-to-end on the real Spot robot. On pass, merge `tour_guide_upgrade_matteo` → `main` and tag `stage2b-e2e-complete`.

**Architecture:** No new code paths. This is a regression matrix + safety audit + merge ceremony. The only new artifact is `scripts/run_regression_matrix.py` (operator-driven checklist runner with timing capture) and `docs/project/stage2-rollback.md` updates.

**Tech Stack:** Existing voice loop entry `python3 -m src.voice_control.client_mic`, bosdyn-client 5.0.1.1, llama.cpp HTTP @ port 11435, Nemotron Speech Streaming ASR (Parakeet kept as rollback layer), Silero VAD, LiveKit Wakeword. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-15-stage2-design.md` section 2B.1.

---

## Pre-conditions (verify before Task 1)

- Plan 2A complete: `git tag` shows `stage2a-voice-complete` exists on `tour_guide_upgrade_matteo`.
- `llama-server` daemon running (verify: `curl -s http://localhost:11435/health` returns `{"status":"ok"}`).
- Nemotron Speech Streaming model downloaded to `/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/` (Parakeet rollback model in `/mnt/ssd/parakeet-models/` optional).
- `SPOT_BRAIN_BACKEND=llamacpp` and `SPOT_ASR_BACKEND=nemotron` set in environment or `.env`.
- Spot powered, on the floor, ≥3m clear in all directions, E-stop tested.
- A second terminal is open with `python -m src.voice_control.estop_run` ready (DO NOT skip).
- Human operator has line of sight to Spot at all times.
- `tegrastats` available on PATH (verify: `which tegrastats`).
- `set_persona` is present in the Stage 2A GBNF grammar but its dispatcher handler ships in 2E.1 T16. If the LLM emits `set_persona` during 2B testing, the dispatcher logs an unknown-action warning. This is acceptable; do not block the matrix on it.

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `scripts/run_regression_matrix.py` | Create (~250 LOC) | Operator-driven regression checklist runner; captures latency + tegrastats; writes `logs/stage2b/regression_<timestamp>.json` |
| `tests/regression/stage2b_matrix.yaml` | Create (~80 LOC) | Declarative test matrix: 7 items, each with `name`, `command`, `pass_criteria`, `safety_notes`, `exploratory` flag |
| `docs/project/stage2-rollback.md` | Modify | Append "After 2B" layer documenting merge-revert procedure |
| `logs/stage2b/` | Create dir | Symlinked to `/mnt/ssd/spot-logs/stage2b/` per Stage 1.5; regression JSON outputs |
| `docs/superpowers/specs/2026-05-15-stage2-design.md` | (no change) | Reference only |
| (no other files) | — | Plan 2B intentionally creates no production code changes |

---

## Task 1: Pre-flight environment audit

**Files:**
- Run only: shell commands
- Read only: `scripts/setup_nemotron.py`, `scripts/setup_parakeet.py`, `scripts/setup_llamacpp_models.py`, `.env`, `~/.config/systemd/user/llama-server.service`

- [ ] **Step 1: Verify Plan 2A tag exists**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone tag --list 'stage2a-voice-complete'
```
Expected output: `stage2a-voice-complete` (one line).
If empty: STOP. Plan 2A is not complete. Do not proceed.

- [ ] **Step 2: Verify llama-server daemon health**

```bash
systemctl --user status llama-server.service --no-pager
curl -sf http://localhost:11435/health
curl -sf http://localhost:11435/v1/models | python3 -m json.tool
```
Expected: systemd shows `active (running)`; `/health` returns `{"status":"ok"}`; `/v1/models` lists `gemma-4-E4B-Q6_K.gguf`.

- [ ] **Step 3: Verify Nemotron + Silero (+ optional Parakeet) models present**

```bash
ls -lh /mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/*.onnx
ls -lh /mnt/ssd/parakeet-models/*.onnx 2>/dev/null || echo "(no Parakeet — rollback layer unavailable)"
ls -lh /mnt/ssd/silero-vad/*.onnx 2>/dev/null || ls -lh $(python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)" | xargs dirname)/../../share/sherpa-onnx/ 2>/dev/null
```
Expected: at least one `.onnx` in the Nemotron dir. Parakeet absence is non-fatal (rollback only). Silero may live inside sherpa-onnx package — locate it.

- [ ] **Step 4: Verify env backend flags**

```bash
grep -E '^SPOT_(BRAIN|ASR|WAKE)_BACKEND' /home/spotdog/spot/dartmouth_spot_capstone/.env || \
  echo "MISSING — set in .env: SPOT_BRAIN_BACKEND=llamacpp SPOT_ASR_BACKEND=nemotron SPOT_WAKE_BACKEND=livekit"
```
Expected: all three vars set. If missing, append to `.env`. The pass criteria of 2B assume llama.cpp + Nemotron + LiveKit are the live paths.

- [ ] **Step 5: Verify Spot connection without lease**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "from src.session import quick_robot; r = quick_robot(); print('robot id:', r.serial_number); print('battery:', r.get_battery_percentage(), '%')"
```
Expected: prints serial number + battery %. Battery must be ≥30 % for the matrix run (the walk + follow segments draw ~5 % each).
If connection fails: check `.env` for `BOSDYN_ROBOT_IP / USERNAME / PASSWORD`. Do not proceed until clean.

- [ ] **Step 6: Verify E-stop terminal ready**

In a separate terminal (NOT this one):
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && python3 -m src.voice_control.estop_run
```
Expected: stays running, prompts for keypress. Leave it running for the duration of Task 3–6.
**Do not proceed until the E-stop terminal is open and within reach of the operator.**

- [ ] **Step 7: Commit pre-flight log**

```bash
mkdir -p /mnt/ssd/spot-logs/stage2b
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  ./scripts/preflight_log.sh > /mnt/ssd/spot-logs/stage2b/preflight_$(date +%Y%m%d_%H%M%S).log 2>&1 || \
  echo "(preflight_log.sh does not exist yet; capture manually: env | grep SPOT_; ollama list; pip freeze | grep -E 'llama|nemotron|parakeet|sherpa|kokoro|bosdyn')" >> /mnt/ssd/spot-logs/stage2b/preflight_manual.log
```
(The script may not exist yet; the manual fallback captures the same data.)

---

## Task 2: Build the regression matrix runner

**Files:**
- Create: `tests/regression/stage2b_matrix.yaml`
- Create: `scripts/run_regression_matrix.py`

- [ ] **Step 1: Write the regression matrix YAML**

Create `tests/regression/stage2b_matrix.yaml` with the following content (the runner reads this; do not modify item order without updating Task 3 step references):

```yaml
# Stage 2B end-to-end regression matrix.
# Items are executed in order. Operator confirms pass/fail interactively.
# exploratory=true items may fail without blocking the merge.

items:
  - name: stand
    intent: '{"action": "stand", "params": {}}'
    pass_criteria: 'Spot stands within 5 s; no error in [Brain-timing] log; no e-stop.'
    safety_notes: 'Operator confirms clear 1m radius before triggering.'
    timeout_s: 15
    exploratory: false

  - name: walk_forward_1m
    intent: '{"action": "walk", "params": {"distance_m": 1.0, "speed_mps": 0.4}}'
    pass_criteria: 'Spot walks forward ≈1m within 8 s; ends standing; no e-stop.'
    safety_notes: '≥1.5m clear in front; floor flat and dry.'
    timeout_s: 20
    exploratory: false

  - name: sit
    intent: '{"action": "sit", "params": {}}'
    pass_criteria: 'Spot sits within 5 s; no e-stop.'
    safety_notes: 'Verify legs not pinched by cables.'
    timeout_s: 15
    exploratory: false

  - name: describe_vlm
    utterance: 'what do you see?'
    pass_criteria: 'Brain returns coherent natural-language description of front camera within 8 s; mentions at least one real object in scene; latency captured.'
    safety_notes: 'Stationary; no motion.'
    timeout_s: 15
    exploratory: false

  - name: graphnav_known_location
    utterance: 'go to {LOCATION}'   # operator substitutes a known saved location
    pass_criteria: 'Spot navigates to within 0.5m of saved waypoint; reports arrival; no e-stop.'
    safety_notes: 'Verify route clear of obstacles end-to-end before triggering. Operator follows along beside Spot.'
    timeout_s: 60
    exploratory: false

  - name: follow_then_stop
    utterance_1: 'follow me'
    utterance_2: 'stop'
    pass_criteria: 'Spot follows operator for ≥5s; operator says "stop"; Spot stops within 2s and idles.'
    safety_notes: 'Operator backs up slowly. If Spot accelerates unexpectedly, e-stop immediately. Vision swap (2C) will replace this path; flaky perception is expected.'
    timeout_s: 30
    exploratory: false

  - name: smoke_door
    utterance: 'open the door'
    pass_criteria: 'Either: (a) Spot identifies a push-bar door + reports attempting AutoPushCommand, OR (b) Spot gracefully declines with "I don''t see a door I can open". Either outcome is a PASS; a crash/freeze is a FAIL.'
    safety_notes: 'Operator stands beside Spot with hand on e-stop. Use ONLY a designated test door — never a fire door.'
    timeout_s: 30
    exploratory: true   # door opening is on ice per MEMORY.md — pass not required for merge

  - name: smoke_follow_long
    utterance_1: 'follow me'
    utterance_2: 'stop'
    pass_criteria: 'Follow for 30 s through a slow walk + 2 turns; stop on command. Drops or misdetects are expected per current YOLOv8n+fisheye limitations.'
    safety_notes: 'Same as follow_then_stop; longer duration → higher risk; cancel any time.'
    timeout_s: 60
    exploratory: true   # Long-follow is exploratory; 2C fixes the perception path
```

- [ ] **Step 2: Write the runner skeleton**

Create `scripts/run_regression_matrix.py`:

```python
#!/usr/bin/env python3
"""Stage 2B end-to-end regression matrix runner.

Reads tests/regression/stage2b_matrix.yaml and walks the operator through
each item interactively. Records timing + outcome + tegrastats snapshot to
logs/stage2b/regression_<timestamp>.json.

USAGE:
    python3 scripts/run_regression_matrix.py [--dry-run]

This script does NOT autonomously drive the robot. Each item prompts the
operator to confirm safety pre-conditions, then triggers the voice loop
or dispatches an intent, then asks the operator to mark pass/fail.

E-STOP MUST BE OPEN IN A SEPARATE TERMINAL.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
MATRIX = REPO / "tests/regression/stage2b_matrix.yaml"
LOG_DIR = REPO / "logs/stage2b"


def load_matrix() -> dict:
    with open(MATRIX) as f:
        return yaml.safe_load(f)


def capture_tegrastats(duration_s: float = 2.0) -> dict:
    """Capture a short tegrastats sample as a sanity snapshot."""
    proc = subprocess.Popen(
        ["tegrastats", "--interval", "1000"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    time.sleep(duration_s)
    proc.terminate()
    out, _ = proc.communicate(timeout=2)
    return {"raw": out.strip().split("\n")[-1] if out else "(empty)"}


def prompt_yes_no(question: str) -> bool:
    while True:
        ans = input(f"{question} [y/n/abort]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        if ans in ("a", "abort"):
            print("Aborting matrix run.")
            sys.exit(2)


def run_item(item: dict, dry_run: bool) -> dict:
    print("\n" + "=" * 70)
    print(f"ITEM: {item['name']}  (exploratory={item.get('exploratory', False)})")
    print("=" * 70)
    print(f"SAFETY: {item['safety_notes']}")
    print(f"PASS CRITERIA: {item['pass_criteria']}")
    if not prompt_yes_no("Safety pre-conditions confirmed?"):
        return {"name": item["name"], "outcome": "SKIPPED", "reason": "safety not confirmed"}

    if "utterance" in item:
        print(f"\n>>> SPEAK NOW: '{item['utterance']}'")
    elif "utterance_1" in item:
        print(f"\n>>> SPEAK NOW: '{item['utterance_1']}'")
        print(f"    THEN AFTER {item.get('timeout_s', 30)//2}s: '{item['utterance_2']}'")
    elif "intent" in item:
        print(f"\n>>> DISPATCHING INTENT: {item['intent']}")
        if not dry_run:
            # Operator dispatches via separate terminal; this script does not bypass voice loop.
            print("    (Open the voice loop terminal and dispatch this intent manually,")
            print("     or speak an equivalent utterance.)")

    start = time.time()
    teg_before = capture_tegrastats(1.0)
    input("Press ENTER when item is complete (or e-stopped)...")
    elapsed = time.time() - start
    teg_after = capture_tegrastats(1.0)

    passed = prompt_yes_no(f"Did this item PASS the criteria above?")
    notes = input("Operator notes (one line, optional): ").strip()

    return {
        "name": item["name"],
        "outcome": "PASS" if passed else "FAIL",
        "elapsed_s": round(elapsed, 2),
        "tegrastats_before": teg_before,
        "tegrastats_after": teg_after,
        "operator_notes": notes,
        "exploratory": item.get("exploratory", False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk the matrix without prompting for actual robot action")
    args = parser.parse_args()

    matrix = load_matrix()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"regression_{ts}.json"

    print("Stage 2B Regression Matrix Runner")
    print(f"Items: {len(matrix['items'])}")
    print(f"Log:   {out_path}")
    print("E-STOP MUST BE OPEN IN A SEPARATE TERMINAL.")
    if not prompt_yes_no("E-stop confirmed open?"):
        return 2

    results = []
    for item in matrix["items"]:
        result = run_item(item, dry_run=args.dry_run)
        results.append(result)
        # Persist after each item so a crash mid-matrix doesn't lose data.
        out_path.write_text(json.dumps({"items": results, "matrix": str(MATRIX)}, indent=2))

    # Summary.
    non_exp = [r for r in results if not r["exploratory"]]
    failed_non_exp = [r for r in non_exp if r["outcome"] == "FAIL"]
    exp = [r for r in results if r["exploratory"]]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Non-exploratory: {len(non_exp) - len(failed_non_exp)}/{len(non_exp)} PASS")
    print(f"Exploratory:     {sum(1 for r in exp if r['outcome'] == 'PASS')}/{len(exp)} PASS")
    print(f"Merge gate:      {'CLEAR' if not failed_non_exp else 'BLOCKED'}")
    print(f"Log:             {out_path}")
    return 0 if not failed_non_exp else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke-test the runner with --dry-run (no robot needed)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/run_regression_matrix.py --dry-run
```
Walk through the 8 items, marking all PASS. Expected: writes `logs/stage2b/regression_<ts>.json`; summary shows 6/6 non-exploratory + 2/2 exploratory PASS; merge gate CLEAR.
**If the script crashes or the YAML fails to parse, fix before proceeding to Task 3.**

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add \
  scripts/run_regression_matrix.py tests/regression/stage2b_matrix.yaml
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b runner: regression matrix yaml + interactive runner script"
```

---

## Task 3: Caveman diff review of runner + matrix

**Files:**
- Read only: the diff of Task 2

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer on the Task 2 diff**

Dispatch via the Agent tool:

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "Review Task 2 diff",
  prompt: "Review the most recent commit on branch tour_guide_upgrade_matteo at /home/spotdog/spot/dartmouth_spot_capstone — it adds scripts/run_regression_matrix.py and tests/regression/stage2b_matrix.yaml. This is a Stage 2B regression matrix runner. Operator-driven (not autonomous). Verify:\n1. The runner does NOT bypass the voice loop or directly call BD SDK motion commands without operator confirmation.\n2. The YAML schema is self-consistent (every item has name + pass_criteria + safety_notes + timeout_s).\n3. The exploratory flag is honored in the summary logic (exploratory failures do not block the merge gate).\n4. Logs are written incrementally (a crash mid-matrix should not lose prior items).\nReport severity-tagged findings only; skip formatting nits."
})
```

- [ ] **Step 2: Address every finding tagged BLOCKER or HIGH**

If the reviewer flags any BLOCKER or HIGH item, fix it and amend the commit (or add a new commit). MEDIUM/LOW may be deferred to Task 7 final review.

- [ ] **Step 3: Commit fixes if any**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b runner: address review findings"
```
Skip this step if no fixes were needed.

---

## Task 4: Execute the regression matrix on the real Spot

**Files:**
- Run only: voice loop + regression runner
- Write: `logs/stage2b/regression_<timestamp>.json`

> **Operator presence required for every item. Do NOT delegate to a subagent. Do NOT batch-execute.**

- [ ] **Step 1: Start the voice loop in terminal A**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && python3 -m src.voice_control.client_mic
```
Wait for `[VoiceLoop] ready` (or equivalent) message. Verify it picked up the llama.cpp backend (look for `SPOT_BRAIN_BACKEND=llamacpp` line) and Nemotron ASR (`SPOT_ASR_BACKEND=nemotron`).

- [ ] **Step 2: Confirm E-stop in terminal B**

E-stop terminal from Task 1 Step 6 must still be running and responsive.

- [ ] **Step 3: Start the matrix runner in terminal C**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && python3 scripts/run_regression_matrix.py
```
Walk through items 1–6 (non-exploratory). For each:
- Confirm safety pre-conditions.
- Speak the utterance into the voice loop (terminal A).
- Watch the robot execute.
- Mark pass/fail in the runner (terminal C).

- [ ] **Step 4: After item 6, decide whether to attempt exploratory items**

Items 7 (smoke_door) and 8 (smoke_follow_long) are exploratory. Skip if:
- Lab space is unsuitable (no test door for item 7; no clear 30s walk path for item 8).
- Operator fatigue is high (the matrix to this point is non-trivial cognitive load).
- Any item 1–6 was a near-miss safety incident.

Otherwise, run them and accept any outcome (PASS or FAIL — both are valid for exploratory items).

- [ ] **Step 5: Review the summary**

The runner prints PASS/FAIL counts + merge gate status. The log at `logs/stage2b/regression_<ts>.json` is the authoritative record.

- [ ] **Step 6: Power Spot down and stow safely**

```
# in voice loop
"sit"  →  wait until sit  →  power off via Bd app or via voice "power off"
```
Tether to charger if battery < 50 %.

- [ ] **Step 7: Commit the regression log**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  ls -lt logs/stage2b/regression_*.json | head -1
# Note the path; commit it.
git -C /home/spotdog/spot/dartmouth_spot_capstone add logs/stage2b/regression_*.json
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b: regression matrix run $(date +%Y-%m-%d) — see log for outcomes"
```
**NOTE:** `logs/` is a symlink to SSD per Stage 1.5 but is committed (`.gitignore` line 28 ignores `logs` directory itself; the symlink target's files are followed). If `git add` fails with "ignored", explicitly add with `-f`:
```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -f logs/stage2b/regression_*.json
```

---

## Task 5: Safety-reviewer audit of any code touched since stage2a-voice-complete

**Files:**
- Read only: `git diff stage2a-voice-complete..HEAD -- src/voice_control/spot_dispatch.py src/voice_control/intent.py src/voice_control/llm_brain.py src/voice_control/estop_run.py`

- [ ] **Step 1: List code files touched since 2A tag**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone diff --name-only stage2a-voice-complete..HEAD -- 'src/**/*.py' 'scripts/**/*.py' | sort
```
Expected: nothing from Plan 2B should touch motion/dispatch/llm_brain code (this plan only adds the runner + matrix YAML). If anything in this list looks unexpected, investigate before proceeding.

- [ ] **Step 2: Dispatch safety-reviewer on the full diff scope**

Dispatch via the Agent tool:

```
Agent({
  subagent_type: "safety-reviewer",
  description: "Stage 2B safety audit",
  prompt: "Audit all changes since git tag stage2a-voice-complete on branch tour_guide_upgrade_matteo at /home/spotdog/spot/dartmouth_spot_capstone before merging to main. Focus on stop/freeze/e-stop bypass coverage, shutdown sequencing, motion paths. Pay particular attention to src/voice_control/spot_dispatch.py, src/voice_control/intent.py, src/voice_control/llm_brain.py action handling, src/voice_control/estop_run.py, and the new scripts/run_regression_matrix.py runner. Report any safety regressions or new motion paths that lack stop coverage. This is the merge-to-main gate — be strict."
})
```

- [ ] **Step 3: Address every BLOCKER or HIGH finding**

Stage 2B is the merge gate. ANY safety-reviewer BLOCKER/HIGH must be fixed before the merge. MEDIUM is operator judgment. LOW may be deferred.

If fixes are needed:
```bash
# Make the fixes, then:
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b: safety review fixes — <one-line summary>"
```

- [ ] **Step 4: Re-run the relevant matrix items if safety fixes changed motion code**

If Step 3 modified `spot_dispatch.py`, `intent.py`, `llm_brain.py`, or `estop_run.py`, re-run Task 4 items 1, 2, 3, 6 (stand, walk, sit, follow+stop) at minimum.
Skip this step if no safety code was changed.

---

## Task 6: Update stage2-rollback.md with merge-revert procedure

**Files:**
- Modify: `docs/project/stage2-rollback.md`

- [ ] **Step 1: Append the merge-revert section**

Append (2A T13 creates `docs/project/stage2-rollback.md`; do not duplicate the env-var rollback layers already in that doc — add only the merge-revert section):

```markdown
## After 2B — merge-to-main

Once `stage2b-e2e-complete` is tagged on `main`, `tour_guide_upgrade_matteo` has been merged. To revert the merge:

### Soft revert (recommended)
```bash
# Find the merge commit on main
git -C /home/spotdog/spot/dartmouth_spot_capstone log --merges --oneline -5 main
# Revert it
git -C /home/spotdog/spot/dartmouth_spot_capstone checkout main
git -C /home/spotdog/spot/dartmouth_spot_capstone revert -m 1 <merge-commit-sha>
git -C /home/spotdog/spot/dartmouth_spot_capstone push origin main
```
Preserves history; both PR commits and the revert are visible.

### Hard reset (NOT recommended on shared main; ASK user before doing this)
```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone checkout main
git -C /home/spotdog/spot/dartmouth_spot_capstone reset --hard stage1.5-complete
git -C /home/spotdog/spot/dartmouth_spot_capstone push --force-with-lease origin main
```
Only do this if main has not been pulled by anyone else. **Confirm with user before force-pushing main.**

### Sub-system runtime fallbacks
See the env-var flip ladder already documented in this file (added by Stage 2A T13). **Caveat:** the `SPOT_BRAIN_BACKEND=ollama` layer requires re-pulling the qwen2.5:7b + qwen2.5vl:7b pair (~10.7 GB) — 2A T12 removed it via `ollama rm`. The other layers (ASR=parakeet, Wake=sherpa_onnx) work as documented. Restart the voice loop (`python3 -m src.voice_control.client_mic`) after any env var flip.
```

- [ ] **Step 2: Commit the rollback doc update**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add docs/project/stage2-rollback.md
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b: rollback doc — merge-revert + runtime-only fallbacks"
```

---

## Task 7: Final review pass (caveman + code-reviewer)

**Files:**
- Read only: `git diff stage2a-voice-complete..HEAD`

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer on the full 2B diff**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "Full 2B diff review",
  prompt: "Review every change between git tag stage2a-voice-complete and HEAD on branch tour_guide_upgrade_matteo at /home/spotdog/spot/dartmouth_spot_capstone. This is the Stage 2B plan: regression matrix runner + matrix YAML + rollback doc update + regression log. Verify the runner correctly enforces operator confirmation, the matrix is internally consistent, the rollback doc covers all paths, no unexpected files were committed. Report severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch feature-dev:code-reviewer for deeper quality audit**

```
Agent({
  subagent_type: "feature-dev:code-reviewer",
  description: "Quality audit of Stage 2B",
  prompt: "Code-review the entire Stage 2B diff (between tags stage2a-voice-complete and HEAD on tour_guide_upgrade_matteo at /home/spotdog/spot/dartmouth_spot_capstone). This is the merge-to-main gate plan. Look for: bugs, logic errors, security issues, adherence to project conventions (see CLAUDE.md / MEMORY.md), and high-confidence improvements only. Files: scripts/run_regression_matrix.py, tests/regression/stage2b_matrix.yaml, docs/project/stage2-rollback.md, logs/stage2b/regression_*.json. Be strict on the runner since it gates the merge; lenient on the YAML."
})
```

- [ ] **Step 3: Address BLOCKER/HIGH findings; document deferrals**

For each BLOCKER/HIGH: fix + amend commit OR new commit. For MEDIUM/LOW that you defer: add a one-line entry to `docs/project/stage2-followups.md` (create if missing) so the deferred items are tracked.

- [ ] **Step 4: Commit any fixes**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2b: address review findings"
```
Skip if no fixes.

---

## Task 8: Verify all merge gate criteria are met

**Files:**
- Read only: `logs/stage2b/regression_*.json`

- [ ] **Step 1: Confirm regression log shows merge gate CLEAR**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
import json, glob, sys
logs = sorted(glob.glob('logs/stage2b/regression_*.json'))
if not logs:
    print('NO REGRESSION LOG'); sys.exit(2)
data = json.loads(open(logs[-1]).read())
items = data['items']
non_exp = [i for i in items if not i.get('exploratory', False)]
failed_non_exp = [i for i in non_exp if i['outcome'] == 'FAIL']
print(f'Latest log: {logs[-1]}')
print(f'Non-exp PASS: {len(non_exp) - len(failed_non_exp)}/{len(non_exp)}')
print(f'Merge gate: {\"CLEAR\" if not failed_non_exp else \"BLOCKED\"}')
sys.exit(0 if not failed_non_exp else 1)
"
```
Expected: prints `Merge gate: CLEAR` and exits 0.
If BLOCKED: do not proceed. Fix the regressing item (re-run Plan 2A's relevant sub-task, then re-run the matrix item).

- [ ] **Step 2: Confirm no safety-reviewer BLOCKER outstanding**

Manually re-read the Task 5 + Task 7 reviewer outputs. If anything BLOCKER/HIGH is unresolved, STOP and address.

- [ ] **Step 3: Confirm clean git state**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone status
```
Expected: `working tree clean` (or only untracked files unrelated to this plan).

---

## Task 9: Open PR and merge to main

> **Confirm with user before pushing to remote or creating PR.** This is the irreversible step.

- [ ] **Step 1: Push the branch**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone push origin tour_guide_upgrade_matteo
```
Cached HTTPS creds work per MEMORY.md.

- [ ] **Step 2: Create the PR**

```bash
gh pr create --title "Stage 2: voice pipeline + Spot e2e (merge-to-main)" --body "$(cat <<'EOF'
## Summary

Stage 2A (voice pipeline, llama.cpp brain + Nemotron Speech Streaming ASR + Silero VAD + LiveKit Wakeword) + Stage 2B (Spot e2e regression matrix) complete. All non-exploratory regression items pass on the real robot. Safety-reviewer audit clean. Ready to merge to main.

TTS + multi-persona (2E.1) ships as a follow-up PR same week (Tuesday hard deadline). Vision overhaul (2C) and smart chatbot (2D) will follow as separate PRs on top of main.

## What's in this PR
- llama.cpp + GBNF-grammar adaptive-thinking brain (replaces Ollama as default)
- Nemotron Speech Streaming 0.6B INT8 ASR (replaces broken Riva bridge; Parakeet TDT 0.6B v3 kept as rollback layer)
- Silero VAD via sherpa-onnx (replaces webrtcvad)
- LiveKit Wakeword (replaces openWakeWord plan)
- Stage 2B regression matrix runner + matrix YAML
- Rollback docs

## Test plan
- [x] Stage 2A mic-verify passed (see scripts/mic_verify.py output)
- [x] Stage 2B regression matrix non-exploratory items 6/6 PASS (see logs/stage2b/regression_*.json)
- [x] Safety-reviewer audit clean (no outstanding BLOCKER/HIGH)
- [x] caveman:cavecrew-reviewer final review clean
- [x] feature-dev:code-reviewer quality audit addressed

## Rollback
See docs/project/stage2-rollback.md — env-var-driven runtime rollback for brain, ASR, wake word, vision. Git revert procedure documented for merge revert.
EOF
)"
```

- [ ] **Step 3: Self-merge (or wait for review per repo conventions)**

If the repo has no PR-review gate (single-contributor capstone), self-merge:
```bash
gh pr merge --merge   # NOT --squash; preserve sub-task commit history
```
If review is required, wait for approval.

- [ ] **Step 4: Tag the merge commit on main**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone checkout main
git -C /home/spotdog/spot/dartmouth_spot_capstone pull origin main
git -C /home/spotdog/spot/dartmouth_spot_capstone tag -a stage2b-e2e-complete -m "stage 2b: voice loop + Spot e2e validated on real robot; merge gate clear"
git -C /home/spotdog/spot/dartmouth_spot_capstone push origin stage2b-e2e-complete
```

- [ ] **Step 5: Verify tag visible on remote**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone ls-remote --tags origin | grep stage2b-e2e-complete
```
Expected: prints the tag SHA. If missing, the push failed; re-push.

- [ ] **Step 6: Update TodoWrite / mark Plan 2B done**

The dispatcher marks this task complete in TodoWrite. Stage 2B done.

---

## Self-review

**Spec coverage:** Section 2B.1 in `docs/superpowers/specs/2026-05-15-stage2-design.md` requires:
- ✅ stand → walk forward 1 m → sit (matrix items 1-3)
- ✅ "what do you see?" VLM (matrix item 4)
- ✅ "go to <known location>" GraphNav (matrix item 5)
- ✅ "follow me" then "stop" (matrix item 6)
- ✅ Smoke door + smoke follow LAST, exploratory (matrix items 7-8)
- ✅ Safety review on motion/dispatch/llm_brain/estop_run code (Task 5)
- ✅ Pass criteria: all 5 non-exploratory succeed without manual intervention (Task 8 gate check)
- ✅ Merge `tour_guide_upgrade_matteo` → `main`, tag `stage2b-e2e-complete` on merge commit on main (Task 9)

**Placeholder scan:** No TBD / TODO / "implement later" in any task. The `{LOCATION}` placeholder in the GraphNav item YAML is intentional — the operator substitutes a real saved location at runtime (matrix items are tested generically, not against a fixed waypoint).

**Type consistency:** YAML schema (`name`, `pass_criteria`, `safety_notes`, `timeout_s`, `exploratory`, plus optional `utterance` / `utterance_1`+`utterance_2` / `intent`) is the same across all 8 items and across both Task 2 (write) and the Python runner (read). The `outcome` strings in the runner JSON output (`PASS` / `FAIL` / `SKIPPED`) match the strings the gate check in Task 8 looks for.

**Ambiguity check:** 
- "merge gate CLEAR" = zero non-exploratory FAIL items in the latest regression log (Task 8 Step 1 enforces this).
- "exploratory" = `exploratory: true` in YAML; failures excluded from merge gate but logged.
- "safety-reviewer BLOCKER" = any finding the agent tags severity BLOCKER. Operator may not waive without user discussion.
- "force-push to main" requires explicit user OK per Task 6 rollback section.

**Reviewer dispatch points (per "build in periodic review passes" mandate):**
- Task 3: caveman:cavecrew-reviewer on Task 2 diff (mid-plan checkpoint #1)
- Task 5: safety-reviewer on all code since 2A tag (motion-touching audit)
- Task 7: caveman:cavecrew-reviewer + feature-dev:code-reviewer on full plan diff (final checkpoint)
- pin-guardian: not invoked — Plan 2B touches no production code or requirements.txt.

**Out of scope:** This plan does not modify production code. All matrix items use the voice loop as-is from Plan 2A. Vision improvements (2C) and chatbot improvements (2D) are explicitly separate plans, executed after merge.
