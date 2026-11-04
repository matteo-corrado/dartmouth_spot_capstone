# Vision actions: validation and next steps

Context: `c2edcab` on `vision/perception-v2` (built in worktree `../spot_vision`, from demo `81fedfe`). 1034 tests pass, 3 skip (fixtures with people, gitignored). Not merged, not on the demo or restructure branch, **never run on the robot**. Spec: `docs/superpowers/specs/2026-10-29-vision-upgrade-design.md` (on the vision branch); ledger with rulings R1-R30 and every deferred minor: `../spot_vision/.superpowers/sdd/2026-10-29-vision-upgrade/progress.md` (gitignored). `vision/friendly-social` (see [friendly-spot-social-mode.md](friendly-spot-social-mode.md)) is built on top of this branch.

What it is: package `src/voice_control/vision/` + thin hooks in `client_mic.py`, `spot_dispatch.py`, `llm_brain.py`, `grammar/spot_action.gbnf` (`find`, `look_around`), `intent.py`.
- Models: `yolo26l-seg` (main), `yolo26l-pose` (raised hand, follow), `yoloe-26l-seg` (150 extra nouns), exported `end2end=True`, run on our own onnxruntime TensorRT runtime (FP16: 12 / 10 / 12.5 ms). Models + engine cache in `models/vision/` (gitignored), made by `scripts/setup_vision_models.py`.
- Gemma 4 E4B on llama-server for descriptions, thinking off.
- Directions always from camera transforms: bearing in degrees, + = left. Front cameras are cross-mounted (frontleft looks ~34 deg right).

| Action | Says / does |
|---|---|
| Describe ("what do you see", "what's on your left", "look around") | Stitched front pair + gripper in one VLM call, then "I can also spot ... behind me" from YOLO on the other cameras |
| Find ("where is the bottle") | Direction + distance, no movement |
| Look at speaker | Mic direction, refined +/-30 deg by the nearest detected person |
| Go to ("come to me", "go to the person in the red shirt", "go to the chair") | Walks in odom; stops 1.5 m from a person, 1.0 m from an object; asks for a raised hand only when there are several people |
| Follow ("follow me") | Pose tracker, holds ~2 m, vx <= 0.5 m/s, never backs up; "lost you" after 3 s; ends at 15 m or 120 s |

---

## 0. Before any robot run

1. **Bring the safety wake-word fixes in.** The demo branch has 9 commits the vision branch lacks, including `bd384b4`..`44125a7` (quiet "stop"/"freeze" was being missed). Merge `demo-poster-2026-10-29` into `vision/perception-v2` (or transplant first, see section 3) and re-run the full suite. **Do not test go-to or follow without these.**
2. **Wire the worktree to shared assets.** The worktree has only `models/vision/`. Symlink `models/kws`, `models/speaker`, `models/tts` from the main checkout, and load the main checkout's env file explicitly (the worktree has none).
3. **Warm the engines.** Run `scripts/setup_vision_models.py`; it exits non-zero if any model is not on TensorRT. A cold first build blocks a dispatch for minutes.
4. **Demo stopped.** Vision and the demo can't share the lease or the mic, and YOLO on the GPU slows Gemma (22.5 -> ~15 tok/s at 30 img/s), which also skews the latency numbers.

## 1. Validate on the robot (in this order)

Steps 1.1-1.4 do not walk. 1.5-1.9 move the robot: operator present, e-stop in hand, clear 4 m area.

### 1.1 Snapshot latency (read-only, gates follow)
Script: `scripts/vision_snapshot_latency.py` on `vision/perception-v2` (`d85b2e1`). Times `cameras.capture` + `world.detect_frames` for 2, 3 and 5 cameras, 8 runs each, no lease. Run it from a vision worktree; `--env-dir` points at the main checkout's credentials file.
- **Pass:** median < 0.5 s, max < 0.8 s for the 2-camera case.
- **If slow:** follow stays off near people (safety review condition). Try fewer cameras, skip depth on cameras not in use, or batch frames into one inference.
- Also note: whether all images come back upright (dump one set), and Gemma tok/s while it runs.

### 1.2 Describe
- "What do you see" with objects in front and one behind. **Pass:** answer starts with the front, mentions what is in the gripper view, ends with "I can also spot ... behind me", no front object repeated in that line. Reply < ~4 s.
- "What's on your left / right / behind you", "look around". **Pass:** each side names what is really there.
- **Directions:** place one known object at front, left, right, back. **Pass:** each is reported on the right side. This is the main check that the rotation table and transforms are right on the real robot.

### 1.3 Find
- One COCO object (chair, bottle) and one YOLOE-only object (backpack, plant, whiteboard). **Pass:** right direction, distance within ~0.5 m of a tape measure, no motion. Unknown object -> clear "I can't see a ..." reply.

### 1.4 Look at the speaker
Two people ~60 deg apart, take turns talking. **Pass:** arm gaze lands on the talker within ~1 s; with only the mic (person out of view) it still turns roughly the right way; it never swings to a silent person.

### 1.5 Come to me
- One person, 4 m away. **Pass:** walks over, stops 1.5 m +/- 0.3 m (tape on floor), says it arrived.
- Two people. **Pass:** asks for a raised hand, picks the right one, walks to them.
- Person walks out of view mid-approach. **Pass:** stops within ~3 s and says it lost them; never walks on without a fresh distance.

### 1.6 Person by clothing
Two people, one in red. "Go to the person in the red shirt". **Pass:** picks the red shirt, no gaze swinging during the check, gives up cleanly (no motion) if neither matches.

### 1.7 Go to an object
"Go to the chair" from ~4 m. **Pass:** stops 1.0 m +/- 0.3 m. Object behind an obstacle: Spot's own avoidance routes around or it reports it's in the way after ~4 s.

### 1.8 Follow (only after 1.1 passes)
Walk a slow loop. **Pass:** holds ~2 m, never closer than ~1.2 m, no backing up; hand-off between cameras keeps the same person when a second person crosses; walk out of view -> "lost you" after ~3 s and stops; ends by itself at 15 m / 120 s.

### 1.9 Stops
Say "stop" and "freeze" during go-to (person and object) and during follow, plus once during the raised-hand selection. **Pass:** motion stops within ~0.5 s every time, no motion afterwards, no spoken answer contains the word "stop" (the speech filter rewrites it).

Record results in `docs/project/vision-robot-checklist.md`, then update `status.md` / `bugs.md`.

## 2. Unvalidated assumptions to watch while testing
- Raised-hand rule (COCO keypoints 5/6/9/10, wrist above shoulder, held 2 frames) only tested on stills.
- Ground-plane distance fallback (`GROUND_Z_DEFAULT = -0.5`) when depth misses: too far if feet are cut off; goto treats ground range < 4 m as no distance.
- First TRT build on a cold cache blocks the dispatch thread (stop still works).
- `follow` never reports ok, so the arm isn't restored afterwards (M8, left as is).

## 3. Before merging
- Transplant onto `restructure-modular` (tag `restructure-verified-2026-11-03`): copy `src/voice_control/vision/` + tests, re-apply the thin hooks in the restructured `client_mic` / `spot_dispatch` / `llm_brain` / gbnf / `intent`, then run the full suite there. Update the gbnf allowlist with `find` and `look_around` (else the LLM silently misfires).
- Re-run the safety reviewer on the transplanted dispatcher hooks.
- Fixtures captured with `scripts/capture_vision_fixture.py` that contain people stay gitignored.

---

## 4. Improvement directions (ranked)

### A. Faster snapshot -> better follow
Batch all frames through one TensorRT call, skip depth for cameras without a target, and keep the tracker on the 1-2 cameras that see the person. Lets follow run faster than its current snapshot rate and tightens the 2 m hold.

### B. Colour and attributes for objects
"Red chair" today picks the nearest chair. Reuse the clothing yes/no VLM check (or a cheap mask-colour histogram from the seg mask) for any adjective + noun.

### C. Groups and plurals
"Come to us" / "the kids" behave like "me". Irregular plurals (knives, shelves) and trailing punctuation don't resolve. Small parser fixes in `vision/phrases.py` and `vocab.resolve_label`.

### D. Remember what it saw
Each command takes a fresh snapshot. A short-lived world memory (last seen bearing/odom position per label, ~30 s) would answer "where did you see the bottle" and let go-to start without a fresh look.

### E. Two-turn follow-ups
Follow-ups are single-turn through Gemma (operator choice 2a). A short context of the last describe would let "what colour is it" or "go to that" work.

### F. Better YOLOE vocabulary
The 150-noun list was picked by hand. Tune from what the robot actually misses in its rooms; `bag -> backpack` synonym currently shadows YOLOE's `bag`.

### G. Robustness minors from the ledger
Worth a single cleanup pass: `capture` only catches `ValueError`; dedupe is global not per camera (two same-label objects 10 deg apart merge); yes/no parse reads "**Yes**" as no (safe direction); `io.snapshot` has no timeout; target can switch to another same-label detection within 25 deg during go-to. Full list in the ledger ("minor (deferred)").

### H. Stairs
Follow has no stair handling (Spot climbs forward, descends backwards). Until it does, follow must end at the top of stairs; the 15 m radius is the only guard today.
