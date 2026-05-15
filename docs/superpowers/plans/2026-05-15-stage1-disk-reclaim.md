# Stage 1 Disk-Reclaim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim ~27 GB on the Jetson eMMC and land all peripheral-free code work + LLM/VLM brain swap from the master plan, while keeping voice control rollback intact via git tags + qwen pair retention.

**Architecture:** Stage 1 runs entirely without mic / speaker / Spot. Sequence is (1) tag pre-Stage-1 stable + write rollback runbook → (2) off-tree disk reclaim → (3) code commits in dependency order (Phase 0.2 fixes → Phase 0.3 latency wiring → Phase 4 cosmetic Riva removal) → (4) Ollama upgrade + gemma4:e4b pull + brain code swap + text-only verification → (5) tag stage1-complete. The qwen pair stays on disk as the rollback path until Stage 2 mic-verifies gemma4 in production.

**Tech Stack:** bash (Docker, Ollama, git), Python 3 (in-place edits to existing modules; no new packages installed in Stage 1).

**Spec:** `docs/superpowers/specs/2026-05-15-disk-reclaim-staging-design.md`

---

## Scope note

This plan covers Stage 1 + Stage 1.5 (SSD population, gated on the thesis plan's SSD reformat + mount). **Stage 2 is its own future plan** — written when peripherals return — because two of its picks (LiveKit Wakeword vs openWakeWord; YOLO26/YOLOE26 vs YOLO11n/YOLOE-11s) need fresh measurement before they can be reduced to bite-sized tasks.

**Cross-plan dependency (Stage 1.5 only):** The destructive USB SSD reformat (exFAT → ext4), `/mnt/ssd` mount, `/etc/fstab` entry, and the three Spot placeholder subdirs `/mnt/ssd/{ollama-models,clip-cache,spot-logs}/` are owned by the thesis plan at `~/spot/mc_thesis/social-cfm-mppi-thesis-docs/docs/superpowers/plans/2026-05-15-jetson-orin-port.md` (Tasks 1-2). Spot Tasks 1-20 (Stage 1 proper) are independent. Spot T21 must not run until thesis Tasks 0-2 are complete; thesis Task 0 in turn gates on the Spot Stage 1 exit state (eMMC ≥ 25 GB free + gemma4 pulled + Riva gone), so the natural order is: Spot T1-T20 → thesis T0-T2 → Spot T21.

## File structure (Stage 1)

| File | Action | Responsibility |
|---|---|---|
| `docs/project/stage1-rollback.md` | Create | The rollback runbook (committed FIRST) |
| `src/voice_control/spot_dispatch.py` | Modify | Phase 0.2 nav-feedback fix + name normalization (3 sites); Phase 0.3 state-dict additions |
| `src/voice_control/spot_tts.py` | Modify | Phase 0.3 latency callbacks at `enqueue_render` |
| `src/voice_control/client_mic.py` | Modify | Phase 0.3 argparse flags, `init_recorder()`, greet block, per-utterance `begin_utterance / complete` brackets, span wraps |
| `src/voice_control/llm_brain.py` | Modify | Phase 2 model defaults; `warm_up_vlm` no-op; clean up `_vlm_warmed` |
| `scripts/run_voice_control.py` | Modify | Phase 4 cosmetic: delete `RIVA_CONTAINER`, `RIVA_PORT`, `ensure_riva()`, Riva startup branch |
| `requirements.txt` | Modify | Phase 4 cosmetic: drop `nvidia-riva-client>=2.17.0` |
| `scripts/setup_riva.sh` | Delete | Phase 4 cosmetic |
| `README.md` | Modify | Phase 0.3 latency section + Phase 4 pipeline diagram update |
| `docs/architecture/overview.md` | Modify | Phase 4 pipeline diagram |
| `docs/project/status.md` | Modify | Phase 4 model lineup |
| `~/ollama-0.16.1.backup` | Create (off-tree) | Binary backup before Ollama upgrade |
| `git tag pre-stage1-stable` | Create + push | Restore point at HEAD `625184b` |
| `git tag stage1-complete` | Create + push | Stage 1 end |

---

## Task 1: Tag pre-Stage-1 stable + push to origin

**Files:**
- (no file changes — git tag + push only)

- [ ] **Step 1: Verify HEAD is the expected commit and tree is clean**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone rev-parse HEAD
git -C /home/spotdog/spot/dartmouth_spot_capstone status --short
```

Expected: HEAD is `625184b...` (the upgrade-plan consolidation commit OR the design doc commit `39083d2` if you ran the brainstorming skill earlier today — both are acceptable Stage-1-stable points). `git status --short` shows nothing tracked (untracked `.claude/` is fine).

- [ ] **Step 2: Create the annotated tag**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git tag -a pre-stage1-stable -m "Pre-Stage 1: Riva live, Ollama 0.16.1 + qwen pair, no gemma4, no Stage 1 code edits"
```

- [ ] **Step 3: Push the tag and the branch to origin**

```bash
git push origin pre-stage1-stable
git push origin tour_guide_upgrade_matteo
```

If the user's terminal is required for HTTPS auth (per CLAUDE.md, no credential helper), the operator runs these manually.

- [ ] **Step 4: Verify the tag is on origin**

```bash
git ls-remote --tags origin pre-stage1-stable
```

Expected: a single line showing the tag's SHA.

---

## Task 2: Write the rollback runbook (FIRST commit)

The runbook is committed and pushed BEFORE any code or disk operation, so it's findable from GitHub if the local checkout dies.

**Files:**
- Create: `/home/spotdog/spot/dartmouth_spot_capstone/docs/project/stage1-rollback.md`

- [ ] **Step 1: Capture the live NGC URL for Riva quickstart re-acquisition**

Browse to https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64 and copy the exact 2.17.0 download URL into a scratchpad. (URL paths sometimes change between NGC catalog updates; capturing it now ensures the runbook has a working URL even if the catalog reorganizes later.)

- [ ] **Step 2: Write `docs/project/stage1-rollback.md` with the content below**

```markdown
# Stage 1 Rollback Runbook

This runbook restores the Jetson + repo to its pre-Stage-1 state.

## What "stable" means (pre-Stage-1 inventory, 2026-05-15)

- **Branch / tag:** `tour_guide_upgrade_matteo` at tag `pre-stage1-stable`.
- **Disk:** 53 GB used / ~787 MB free / 99% on `/dev/mmcblk0p1`.
- **Riva Docker image:** `nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64` resident at `/var/lib/docker` (24.04 GB).
- **Riva quickstart dir:** `~/riva_quickstart_arm64_v2.17.0/` (2.1 GB).
- **Ollama:** version 0.16.1 with `qwen2.5:7b` (4.7 GB) + `qwen2.5vl:7b` (6.0 GB) resident.
- **VS Code CLI versions:** 4 directories under `~/.vscode-server/cli/servers/`. Most-recent per `lru.json`: `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f`.

## Disk-pressure guard (run before any rollback that re-acquires artifacts)

```
df -h /
```

If free space is < 30 GB AND the rollback step pulls Riva back, abort and `ollama rm gemma4:e4b` first to free 9.6 GB.

## Layer 1 — Code rollback (cheap, ~1 minute)

Use this if a Stage 1 code commit broke something or you want to reset the branch.

Inspection-only (non-destructive):
```
git checkout pre-stage1-stable
```
This goes into detached-HEAD mode. To return: `git checkout tour_guide_upgrade_matteo`.

Destructive (only with explicit approval):
```
git checkout tour_guide_upgrade_matteo
git reset --hard pre-stage1-stable
git push origin tour_guide_upgrade_matteo --force-with-lease
```

Verification:
```
git rev-parse HEAD
```
Expected: matches `git rev-parse pre-stage1-stable`.

## Layer 2 — Brain model rollback (cheap, ~1 minute)

Use this if gemma4:e4b regresses on action selection or VLM quality.

Edit `src/voice_control/llm_brain.py:28-29`:
```python
DEFAULT_MODEL = "qwen2.5:7b"
VLM_MODEL = "qwen2.5vl:7b"
```

Restart `run_voice_control.py`. The qwen pair takes over instantly (still on disk per Stage 1 design).

Verification:
```
grep -n 'DEFAULT_MODEL\|VLM_MODEL' src/voice_control/llm_brain.py
ollama list   # should still show qwen2.5:7b and qwen2.5vl:7b
```

To free the 9.6 GB the dead gemma4 occupies:
```
ollama rm gemma4:e4b
```

## Layer 3 — Ollama binary downgrade (cheap, ~2 minutes)

Use this if the Ollama 0.20.x upgrade breaks the daemon.

```
sudo systemctl stop ollama
sudo cp ~/ollama-0.16.1.backup /usr/local/bin/ollama
sudo systemctl start ollama
ollama --version
```

Expected: `ollama version is 0.16.1`.

Models on disk (under `/usr/share/ollama/.ollama/models/`) survive the binary swap.

## Layer 4 — Riva re-acquisition (slow, 30-60 minutes, requires NGC auth)

Use only if you genuinely need Riva back (Parakeet broken AND no other STT works).

```
# 1. Pull the Docker image (~24 GB download)
docker pull nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64

# 2. Re-download the quickstart tarball (URL captured at runbook-write time)
cd ~ && wget "<NGC_URL_GOES_HERE>" -O riva_quickstart_arm64_v2.17.0.tar.gz
tar -xzf riva_quickstart_arm64_v2.17.0.tar.gz

# 3. Re-init the model_repository (this is bind-mounted into the container)
cd ~/riva_quickstart_arm64_v2.17.0
bash riva_init.sh

# 4. Start the container
bash riva_start.sh
```

Verification:
```
docker images | grep riva-speech   # 24 GB
docker ps | grep riva-speech       # running
nc -zv localhost 50051             # accepting connections
```

NGC authentication: if `docker pull` fails with auth error, run `ngc config set` to configure your NGC API key (https://ngc.nvidia.com/setup).

## Layer 5 — VS Code CLI restoration (no manual action)

Old VS Code CLI versions under `~/.vscode-server/cli/servers/Stable-*/` re-download automatically the next time you connect from a new VS Code remote session that requests an older version. No manual step required.

## Decision tree

| Symptom | Layer to use |
|---|---|
| gemma4 says wrong things / picks wrong action | Layer 2 (brain rollback) |
| Stage 1 code commit broke imports or behavior | Layer 1 (code rollback) |
| Ollama daemon crashes / upgrade misbehaves | Layer 3 (binary downgrade) |
| Need Riva back | Layer 4 (slow, last resort) |
| Wrong VS Code version on next connect | Layer 5 (do nothing) |

## Stage 2 rollback (placeholder — to be written as part of Stage 2)

Stage 2 will add Parakeet model files, openWakeWord/LiveKit Wakeword model, ONNX YOLO exports, and possibly a Tavily API key. A `docs/project/stage2-rollback.md` will be written when those land.
```

Replace `<NGC_URL_GOES_HERE>` with the URL captured in Step 1.

- [ ] **Step 3: Verify the runbook exists and reads cleanly**

```bash
ls -l /home/spotdog/spot/dartmouth_spot_capstone/docs/project/stage1-rollback.md
wc -l /home/spotdog/spot/dartmouth_spot_capstone/docs/project/stage1-rollback.md
```

Expected: file exists, ~110 lines.

- [ ] **Step 4: Commit and push**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git add docs/project/stage1-rollback.md
git commit -m "stage1: add rollback runbook"
git push origin tour_guide_upgrade_matteo
```

- [ ] **Step 5: Verify the runbook is on origin**

```bash
git log --oneline origin/tour_guide_upgrade_matteo -3
```

Expected: top entry is `<sha> stage1: add rollback runbook`.

---

## Task 3: Off-tree disk reclaim — Riva quickstart dir + tarball

**Files:**
- Delete (off-tree): `/home/spotdog/riva_quickstart_arm64_v2.17.0/`
- Delete (off-tree): `/home/spotdog/riva_quickstart_arm64_v2.17.0.tar.gz` (already 0 bytes)

- [ ] **Step 1: Confirm pre-state**

```bash
df -h /
du -sh ~/riva_quickstart_arm64_v2.17.0
ls -lh ~/riva_quickstart_arm64_v2.17.0.tar.gz
```

Expected: ~787 MB free on `/`; quickstart dir ~2.1 GB; tarball 0 bytes.

- [ ] **Step 2: Delete the quickstart dir**

```bash
rm -rf ~/riva_quickstart_arm64_v2.17.0
```

- [ ] **Step 3: Delete the empty tarball**

```bash
rm -f ~/riva_quickstart_arm64_v2.17.0.tar.gz
```

- [ ] **Step 4: Verify reclaim**

```bash
df -h /
ls -ld ~/riva_quickstart_arm64_v2.17.0 ~/riva_quickstart_arm64_v2.17.0.tar.gz 2>&1
```

Expected: free space increased by ~2.1 GB; both paths report `No such file or directory`.

(No git commit — these are off-tree operations.)

---

## Task 4: Off-tree disk reclaim — Riva Docker container + image

**Files:**
- Off-tree: `riva-speech` Docker container
- Off-tree: `nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64` Docker image

- [ ] **Step 1: Confirm pre-state**

```bash
docker images | grep riva-speech
docker ps -a | grep riva-speech
docker system df
```

Expected: image listed at 24.04 GB; container exists (likely stopped); `docker system df` shows 100% reclaimable.

- [ ] **Step 2: Stop and remove the container**

```bash
docker stop riva-speech 2>/dev/null || true
docker rm riva-speech 2>/dev/null || true
```

- [ ] **Step 3: Remove the image**

```bash
docker rmi nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64
```

This frees ~24 GB. The command may take a few seconds.

- [ ] **Step 4: Verify reclaim**

```bash
docker images
docker ps -a
df -h /
```

Expected: no riva-speech entries; `df -h /` shows ~26 GB more free than the Task 3 baseline.

(No git commit — off-tree.)

---

## Task 5: Off-tree disk reclaim — VS Code CLI old versions

**Files:**
- Off-tree: 3 of 4 dirs under `~/.vscode-server/cli/servers/`

- [ ] **Step 1: Confirm pre-state and identify the active version**

```bash
cat ~/.vscode-server/cli/servers/lru.json
ls -d ~/.vscode-server/cli/servers/Stable-*
du -sh ~/.vscode-server/cli/
```

Expected: `lru.json` shows 4 SHA entries with `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f` first (most-recent); 4 directories listed; total ~1.3 GB.

- [ ] **Step 2: Delete the 3 older versions (keep `Stable-0958...` only)**

```bash
rm -rf ~/.vscode-server/cli/servers/Stable-8b640eef5a6c6089c029249d48efa5c99adf7d51
rm -rf ~/.vscode-server/cli/servers/Stable-41dd792b5e652393e7787322889ed5fdc58bd75b
rm -rf ~/.vscode-server/cli/servers/Stable-e7fb5e96c0730b9deb70b33781f98e2f35975036
```

- [ ] **Step 3: Verify reclaim**

```bash
ls -d ~/.vscode-server/cli/servers/Stable-*
du -sh ~/.vscode-server/cli/
df -h /
```

Expected: only `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f` remains; `cli/` total ~325 MB; `df` shows ~975 MB more free than Task 4 baseline.

(No git commit — off-tree.)

---

## Task 6: Phase 0.2 — log + break in nav-feedback swallow

**Files:**
- Modify: `src/voice_control/spot_dispatch.py:437-438`

- [ ] **Step 1: Read the current swallow site to confirm exact text**

```bash
sed -n '428,442p' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```

Expected: lines around 432-440 contain a `STATUS_LOST` handler with `try: ... except Exception: pass` swallowing the re-localization attempt.

- [ ] **Step 2: Apply the edit**

In `src/voice_control/spot_dispatch.py`, replace:

```python
                            try:
                                if _spot_session and _ensure_localized(graph_nav_client, _spot_session):
                                    print(f"[Spot] Re-localized, retrying navigation to '{name}'")
                                    nav_to_cmd_id = None
                                    continue
                            except Exception:
                                pass
                            break
```

with:

```python
                            try:
                                if _spot_session and _ensure_localized(graph_nav_client, _spot_session):
                                    print(f"[Spot] Re-localized, retrying navigation to '{name}'")
                                    nav_to_cmd_id = None
                                    continue
                            except Exception as e:
                                print(f"[Spot] Re-localize failed during nav-feedback recovery: "
                                      f"{type(e).__name__}: {e}")
                            break
```

- [ ] **Step 3: Verify no `except Exception: pass` remains in `_navigate_waypoint_sequence`**

```bash
awk '/^def _navigate_waypoint_sequence/,/^def [a-z_]/{print NR": "$0}' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py | grep -E 'except Exception:\s*pass'
```

Expected: no output.

- [ ] **Step 4: Verify the file imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "from src.voice_control.spot_dispatch import _navigate_waypoint_sequence; print('ok')"
```

Expected: `ok`. (No `ImportError`; the helper exists at module scope.)

- [ ] **Step 5: Hold the commit until Task 7 is also done — both belong in the same `phase 0.2` commit**

---

## Task 7: Phase 0.2 — name normalization at 3 dispatch sites

**Files:**
- Modify: `src/voice_control/spot_dispatch.py:1138, 1222, 1291`

The `normalize_location_name` helper (`src/location_manager.py:46`) is already imported as `_normalize_location_name` at `src/voice_control/spot_dispatch.py:23` and used in `save_location` at `:1107`. Three more callers still do the transformation inline.

- [ ] **Step 1: Confirm the 3 sites still use raw `.lower()` / `.strip().lower().replace(" ", "_")`**

```bash
sed -n '1136,1140p;1219,1224p;1288,1293p' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```

Expected: line 1138 has `params.get("location", "").lower()`; lines 1222 and 1291 have `loc_name.strip().lower().replace(" ", "_")`.

- [ ] **Step 2: Edit `:1138` (the `go_to` handler)**

Replace:
```python
            location_name = params.get("location", "").lower()
```
with:
```python
            location_name = _normalize_location_name(params.get("location", ""))
```

- [ ] **Step 3: Edit `:1222` (tour handler inner loop)**

Replace:
```python
                        loc_name = loc_name.strip().lower().replace(" ", "_")
```
with:
```python
                        loc_name = _normalize_location_name(loc_name)
```

- [ ] **Step 4: Edit `:1291` (patrol handler inner loop)**

Same replacement as Step 3 — replace:
```python
                        loc_name = loc_name.strip().lower().replace(" ", "_")
```
with:
```python
                        loc_name = _normalize_location_name(loc_name)
```

- [ ] **Step 5: Verify only one remaining inline `.strip().lower().replace(" ", "_")` (line 879 — `desc_normalized` for object descriptions, NOT a location name)**

```bash
grep -n '\.strip().lower().replace(" ", "_")\|"location".*\.lower()' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```

Expected: exactly one hit at line 879 (`desc_normalized = description.strip().lower().replace(" ", "_")` — different concept, leave alone). No other matches.

- [ ] **Step 6: Verify all 4 callers (the 3 we just edited + the existing `save_location`) use the helper**

```bash
grep -n '_normalize_location_name' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```

Expected: 5 lines — 1 import (line 23) + 4 callers.

- [ ] **Step 7: Audit `locations.json` for pre-existing space-named entries that need migration**

```bash
python -c "import json; d=json.load(open('/home/spotdog/spot/dartmouth_spot_capstone/locations.json')); maps=d.get('maps', {}); spaces=[(m, k) for m, slc in maps.items() for k in slc if ' ' in k]; print('Space-named entries:', spaces if spaces else 'NONE')"
```

If any entries print, manually re-key them to underscore form via:
```bash
python -c "import json; p='/home/spotdog/spot/dartmouth_spot_capstone/locations.json'; d=json.load(open(p)); maps=d.get('maps', {}); changed=False
for m, slc in maps.items():
  for k in list(slc.keys()):
    if ' ' in k:
      slc[k.strip().lower().replace(' ', '_')] = slc.pop(k); changed=True
if changed: json.dump(d, open(p, 'w'), indent=2); print('migrated')
else: print('nothing to migrate')"
```

Expected output on a clean file: `Space-named entries: NONE` (skip the migration step).

- [ ] **Step 8: Verify `spot_dispatch.py` imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "import src.voice_control.spot_dispatch; print('ok')"
```

Expected: `ok`.

- [ ] **Step 9: Commit Phase 0.2 (both Task 6 and Task 7 changes together)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git add src/voice_control/spot_dispatch.py locations.json 2>/dev/null
git commit -m "phase 0.2: log on nav-feedback recovery failure, normalize location names in go_to/tour/patrol"
git push origin tour_guide_upgrade_matteo
```

(Add `locations.json` only if Step 7 actually rewrote it. The bare `git add locations.json` will succeed even if unchanged.)

---

## Task 8: Phase 0.3 Wave 2 — spot_dispatch.py state-dict additions

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (around `_PROCESS_START_TS` at module top, and `get_robot_state_dict()` body)

The state dict already includes `battery_percent` and `current_map`. We add 4 missing fields plus a module-level process start timestamp.

- [ ] **Step 1: Locate the existing module-level constants region**

```bash
grep -n '^_ESTOP_STATE_NAMES\|^CAMERA_SOURCES\|^_spot_session\|^_nav_thread' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py | head -10
```

Expected: 4-5 lines showing the module-level globals around lines 30-50.

- [ ] **Step 2: Add `_PROCESS_START_TS` constant near the other module globals**

In `src/voice_control/spot_dispatch.py`, find the line `_ESTOP_STATE_NAMES = {0: "unknown", 1: "cut", 2: "not_cut", 3: "soft_stop"}` (near the top of the file, around line 30) and add immediately after it:

```python

# Set once at module import for uptime computation in get_robot_state_dict().
_PROCESS_START_TS = time.time()
```

- [ ] **Step 3: Locate the `get_robot_state_dict` body**

```bash
grep -n 'def get_robot_state_dict\|def _list_saved_locations' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py | head -5
```

- [ ] **Step 4: Add 4 new fields to the state dict initial-values block**

Find the dict literal that contains `"battery_percent": "unknown",` ... `"current_map": "unknown",` (around lines 88-99). After the `"current_map": "unknown",` line, before the closing `}`, add:

```python
        "uptime_s": int(time.time() - _PROCESS_START_TS),
        "estop_holder": "unknown",
        "num_saved_locations": len(saved_locs),
        "process_pid": os.getpid(),
```

- [ ] **Step 5: Add `import os` at the module top if not already present**

```bash
grep -n '^import os' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py | head -1
```

If empty, add `import os` near the existing `import sys` line at the top.

- [ ] **Step 6: Wire `estop_holder` from the live state read (after the `_spot_session is None` early return)**

Find the block that reads `robot_state = state_client.get_robot_state()` inside `get_robot_state_dict`. After it pulls the existing `estop_status`, add:

```python
        # E-Stop holder: who currently holds the E-Stop endpoint
        try:
            holders = []
            for entry in robot_state.estop_states:
                if entry.state == entry.STATE_NOT_ESTOPPED:
                    holders.append(entry.name)
            state["estop_holder"] = ", ".join(holders) if holders else "none"
        except Exception:
            state["estop_holder"] = "unknown"
```

(Place this near the existing estop_status read, inside the same `try:` block — do not introduce a second `try` if one already wraps the whole `get_robot_state` call.)

- [ ] **Step 7: Verify the file imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "from src.voice_control import spot_dispatch; d = spot_dispatch.get_robot_state_dict(); print(sorted(d.keys()))"
```

Expected: a sorted key list including `'battery_percent'`, `'current_map'`, `'estop_holder'`, `'num_saved_locations'`, `'process_pid'`, `'uptime_s'`.

- [ ] **Step 8: Hold the commit — Phase 0.3 lands as one commit after Tasks 8/9/10/11 are all done**

---

## Task 9: Phase 0.3 Wave 2 — spot_tts.py callback wiring at `enqueue_render`

**Files:**
- Modify: `src/voice_control/spot_tts.py` around line 190 (the `self._player.enqueue_render(...)` call inside `speak()`)

The `_Task` dataclass (`audio_player.py:91-109`) already accepts `on_render_start / on_render_end / on_play_start / on_play_end`. The TTS speak path currently passes only the render closure and label.

- [ ] **Step 1: Read the current speak/enqueue region**

```bash
sed -n '155,205p' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_tts.py
```

Expected: visible `def speak(self, text, voice=None, speed=None, gain=1.0):` and a closing `self._player.enqueue_render(_render, label=f"tts:{text[:40]}")`.

- [ ] **Step 2: Add `from .latency import get_recorder` at the top of `spot_tts.py`**

```bash
grep -n '^from \.\|^from src\.' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_tts.py
```

If `from .latency import get_recorder` isn't present, add it next to the other relative imports near the top of the file.

- [ ] **Step 3: Build closures and pass them to enqueue_render**

In `src/voice_control/spot_tts.py`, replace the line:

```python
        self._player.enqueue_render(_render, label=f"tts:{text[:40]}")
```

with:

```python
        # Latency hooks: stamp render/play boundaries on the current trace if
        # the latency recorder is initialized. No-op when disabled.
        rec = get_recorder()
        trace = rec.current() if rec else None
        if trace is not None:
            cb_render_start = lambda: trace.mark("tts_render_start")
            cb_render_end = lambda: trace.mark("tts_render_end")
            cb_play_start = lambda: trace.mark("tts_play_start")
            cb_play_end = lambda: trace.mark("tts_play_end")
            trace.mark("tts_enqueue")
        else:
            cb_render_start = cb_render_end = cb_play_start = cb_play_end = None

        self._player.enqueue_render(
            _render,
            label=f"tts:{text[:40]}",
            on_render_start=cb_render_start,
            on_render_end=cb_render_end,
            on_play_start=cb_play_start,
            on_play_end=cb_play_end,
        )
```

- [ ] **Step 4: Verify `enqueue_render` accepts the on_* kwargs**

```bash
grep -nA2 'def enqueue_render' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/audio_player.py | head -15
```

If `enqueue_render` does NOT have `on_*` keyword args (only `enqueue_raw` does in some builds), use `_render`'s captured closure to dispatch hooks at the right wrapper points instead. Check `audio_player.py` and adjust accordingly. The simplest fallback: if `enqueue_render` only takes `(render_fn, label)`, wrap `_render` to fire `cb_render_start` at start, `cb_render_end` after generation; the play_*/play_end hooks then fire from the player's existing worker code if it consults `_Task.on_play_*` or are skipped entirely if not wired through.

- [ ] **Step 5: Verify the file imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "from src.voice_control.spot_tts import SpotTTS; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Hold the commit — wait for Tasks 10/11 before committing Phase 0.3**

---

## Task 10: Phase 0.3 Wave 2 — client_mic.py argparse flags + greet block

**Files:**
- Modify: `src/voice_control/client_mic.py` (argparse around L517-529, greet/warmup region around L580-605)

- [ ] **Step 1: Add `from .latency import init_recorder, get_recorder` to client_mic.py imports**

```bash
grep -n '^from \.\|^from src\.\|^import ' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/client_mic.py | head -20
```

Add (in the same import block as the other module imports, sorted/grouped to fit the existing style):

```python
from src.voice_control.latency import init_recorder, get_recorder
from src.voice_control import startup_status
```

- [ ] **Step 2: Add `--latency` and `--latency-out` argparse flags after the existing `--volume` flag (around line 526)**

After the existing `parser.add_argument("--map", ...)` line (around line 527-528), add:

```python
    parser.add_argument(
        "--latency",
        choices=["off", "ring", "file", "all"],
        default="off",
        help="Latency telemetry mode: off (default), ring (in-memory only), file (write JSONL), all (both)",
    )
    parser.add_argument(
        "--latency-out",
        type=str,
        default=None,
        help="Latency JSONL file path (default: logs/latency-{timestamp}.jsonl). Only used when --latency=file or all.",
    )
```

- [ ] **Step 3: Initialize the recorder right after `args = parser.parse_args()` (around line 529)**

After `args = parser.parse_args()`, add:

```python

    # Latency telemetry — no-op when --latency=off
    init_recorder(mode=args.latency, file_path=args.latency_out)
```

- [ ] **Step 4: Wrap the brain warmup with a startup_phase context**

Find the existing block at `client_mic.py:589-594` that reads:

```python
            brain_warm = threading.Thread(target=brain.warm_up, daemon=True)
            vlm_warm = threading.Thread(target=brain.warm_up_vlm, daemon=True)
            brain_warm.start()
            vlm_warm.start()
            brain_warm.join()
            vlm_warm.join()
```

After Phase 2 lands (Task 16), the `vlm_warm` thread will be removed because `warm_up_vlm()` becomes a no-op. **For now, keep both threads.** Wrap them as:

```python
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain_warm = threading.Thread(target=brain.warm_up, daemon=True)
                vlm_warm = threading.Thread(target=brain.warm_up_vlm, daemon=True)
                brain_warm.start()
                vlm_warm.start()
                brain_warm.join()
                vlm_warm.join()
```

Add `from contextlib import nullcontext` to the imports at the top of the file if not already present.

- [ ] **Step 5: Verify the file imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "import src.voice_control.client_mic; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Hold the commit — Tasks 11 also touches client_mic.py for per-utterance wiring**

---

## Task 11: Phase 0.3 Wave 2 — client_mic.py per-utterance brackets

**Files:**
- Modify: `src/voice_control/client_mic.py` `process_utterance` body (around L958+)

- [ ] **Step 1: Locate process_utterance**

```bash
grep -n '^def process_utterance' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/client_mic.py
```

Expected: one hit, around line 958.

- [ ] **Step 2: Wrap the body in begin_utterance / complete brackets**

Inside `process_utterance(...)`, immediately after the function signature/docstring (before any work begins), add:

```python
    rec = get_recorder()
    trace = rec.begin_utterance() if rec else None
    try:
```

Then INDENT THE ENTIRE existing body of `process_utterance` one level. Find where the function returns and add a matching `finally`:

```python
    finally:
        if rec is not None and trace is not None:
            rec.complete(trace)
```

(If `process_utterance` has multiple return points, the `finally` clause fires regardless.)

- [ ] **Step 3: Wrap the VLM call with `trace.span("vlm")`**

Find the line `vlm_response = brain.query_vlm(image_bytes, clean, yolo_hint=yolo_hint)` (around L1183). Wrap:

```python
                            with (trace.span("vlm") if trace else nullcontext()):
                                vlm_response = brain.query_vlm(image_bytes, clean, yolo_hint=yolo_hint)
```

- [ ] **Step 4: Mark dispatch boundaries around the action loop**

Find `result = brain.process(clean, state)` (around L1095). Immediately AFTER the LLM result is parsed (when actions are about to be dispatched), add `if trace: trace.mark("intent_dispatch")`. After the dispatch loop completes, add `if trace: trace.mark("dispatch_complete")`.

For example, if the dispatch happens shortly after `result = brain.process(...)`:

```python
        result = brain.process(clean, state)
        ...
        if trace:
            trace.mark("intent_dispatch")
        for action_dict in result.get("actions", []):
            dispatch_intent(...)
        if trace:
            trace.mark("dispatch_complete")
```

(Place these `mark` calls precisely at the code points that bound the dispatch phase. Read `client_mic.py` around L1095-1180 to find the natural enclosure.)

- [ ] **Step 5: Verify the file imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "import src.voice_control.client_mic; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Verify `--latency` flag parses without crashing**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python src/voice_control/client_mic.py --help 2>&1 | grep -A1 latency
```

Expected: shows `--latency {off,ring,file,all}` and `--latency-out` in help output.

- [ ] **Step 7: Hold the commit — Task 12 (README) finishes Phase 0.3**

---

## Task 12: Phase 0.3 Wave 2 — README "Latency Metrics" section + commit

**Files:**
- Modify: `README.md` (append a section)

- [ ] **Step 1: Confirm the README does not already have a Latency section**

```bash
grep -n -i 'latency\|## ' /home/spotdog/spot/dartmouth_spot_capstone/README.md | head -20
```

- [ ] **Step 2: Append a `## Latency Metrics` section to the end of `README.md`**

```markdown

## Latency Metrics

Per-utterance timing data is captured by `src/voice_control/latency.py` and surfaces via the `--latency` flag on `client_mic.py` (or `run_voice_control.py`):

- `--latency off` (default): zero overhead, no recording
- `--latency ring`: in-memory ring buffer (last N traces), accessible via SIGUSR1 dump
- `--latency file`: write JSONL to `logs/latency-{timestamp}.jsonl`
- `--latency all`: ring + file

Spans captured per utterance: VAD onset/offset, ASR request/response, LLM request/response, optional VLM request/response, intent dispatch, TTS render/play start/end. Schema is documented in the `Trace` dataclass at `src/voice_control/latency.py`.

To dump the in-memory ring buffer to stderr: `kill -USR1 <pid>`.

Startup-phase metrics are written once per boot to `logs/startup-{timestamp}.json` when `--latency=file` or `all`.
```

- [ ] **Step 3: Commit Phase 0.3 (Tasks 8 + 9 + 10 + 11 + 12 together)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git add src/voice_control/spot_dispatch.py src/voice_control/spot_tts.py src/voice_control/client_mic.py README.md
git status --short
```

Verify only the expected files are staged. Then:

```bash
git commit -m "phase 0.3 wave 2: wire latency producers into client_mic, spot_tts, spot_dispatch"
git push origin tour_guide_upgrade_matteo
```

- [ ] **Step 4: Verify the commit landed on origin**

```bash
git log --oneline origin/tour_guide_upgrade_matteo -2
```

Expected: top entry is `<sha> phase 0.3 wave 2: wire latency producers...`.

---

## Task 13: Phase 4 cosmetic — run_voice_control.py + requirements.txt + setup_riva.sh

**Files:**
- Modify: `scripts/run_voice_control.py` (delete L31-32 RIVA constants; delete L103-114 `ensure_riva()`; delete L292-296 startup branch; update docstring at top)
- Modify: `requirements.txt` (delete L15 `nvidia-riva-client>=2.17.0`)
- Delete: `scripts/setup_riva.sh`

- [ ] **Step 1: Delete `RIVA_CONTAINER` and `RIVA_PORT` constants in run_voice_control.py**

In `scripts/run_voice_control.py`, find and DELETE the two lines:

```python
RIVA_CONTAINER = "riva-speech"
RIVA_PORT = 50051
```

Keep `OLLAMA_PORT = 11434` and `ASR_PORT = 50055`.

- [ ] **Step 2: Delete the entire `ensure_riva()` function**

Find `def ensure_riva():` (around line 103) and delete the function and its body (down to the next `def`). The function ends just before `def ensure_ollama():` or similar.

```bash
grep -n '^def ensure_riva\|^def ensure_' /home/spotdog/spot/dartmouth_spot_capstone/scripts/run_voice_control.py
```

Use the line range to identify the start/end and delete.

- [ ] **Step 3: Delete the Riva startup branch in `main()`**

Find the block in `main()` (around line 292):

```python
    if not args.skip_services:
        if not ensure_riva():
            print("\n  Riva is required for ASR. Exiting.")
            return 1

        if not args.no_brain:
            if not ensure_ollama():
                print("\n  WARNING: Ollama not available. LLM brain will be disabled.")
```

Delete the `if not ensure_riva(): ... return 1` block (the Riva check). Keep the Ollama check. Result:

```python
    if not args.skip_services:
        if not args.no_brain:
            if not ensure_ollama():
                print("\n  WARNING: Ollama not available. LLM brain will be disabled.")
```

- [ ] **Step 4: Update the docstring at the top of run_voice_control.py**

Find the line:
```
    Mic -> VAD -> Riva ASR (Canary-Qwen-2.5B) -> LLM Brain -> Action + TTS Response
```
and replace `Riva ASR (Canary-Qwen-2.5B)` with `ASR (server.py backend)`. Also remove "Riva Docker" from the auto-start description if present.

```bash
grep -n -i 'riva' /home/spotdog/spot/dartmouth_spot_capstone/scripts/run_voice_control.py
```

Expected after this step: zero matches.

- [ ] **Step 5: Drop nvidia-riva-client from requirements.txt**

```bash
sed -n '13,17p' /home/spotdog/spot/dartmouth_spot_capstone/requirements.txt
```

Confirm line 15 is `nvidia-riva-client>=2.17.0`. Then delete that single line:

```bash
sed -i '/^nvidia-riva-client>=2\.17\.0$/d' /home/spotdog/spot/dartmouth_spot_capstone/requirements.txt
grep -n riva /home/spotdog/spot/dartmouth_spot_capstone/requirements.txt
```

Expected: no output.

- [ ] **Step 6: Delete scripts/setup_riva.sh**

```bash
rm /home/spotdog/spot/dartmouth_spot_capstone/scripts/setup_riva.sh
ls /home/spotdog/spot/dartmouth_spot_capstone/scripts/setup_riva.sh 2>&1
```

Expected: `No such file or directory`.

- [ ] **Step 7: Verify run_voice_control.py imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('rvc', pathlib.Path('scripts/run_voice_control.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
```

Expected: `ok`. (If the script reads `argparse` at module scope, this may print help text — that's also OK as long as no exception fires.)

- [ ] **Step 8: Hold the commit — Task 14 (docs) lands in the same Phase 4 cosmetic commit**

---

## Task 14: Phase 4 cosmetic — docs (README + overview + status)

**Files:**
- Modify: `README.md` (drop Riva quickstart references in setup section)
- Modify: `docs/architecture/overview.md` (drop Riva from pipeline diagram)
- Modify: `docs/project/status.md` (drop Riva from feature/model lineup)

- [ ] **Step 1: Audit existing Riva references in docs**

```bash
grep -rn -i 'riva' /home/spotdog/spot/dartmouth_spot_capstone/README.md /home/spotdog/spot/dartmouth_spot_capstone/docs/architecture/overview.md /home/spotdog/spot/dartmouth_spot_capstone/docs/project/status.md
```

- [ ] **Step 2: Update README.md**

For each Riva mention found in Step 1, replace it with "ASR backend (currently the existing server.py wrapper; Parakeet swap pending in Stage 2)" or remove the surrounding setup-step entirely if it was Riva-specific. Do NOT delete sections about ASR generally — only the Riva-specific bits.

- [ ] **Step 3: Update docs/architecture/overview.md**

Find the pipeline diagram and replace the `Riva ASR` node label with `ASR (server.py)`. Adjust any prose paragraphs that describe Riva's role.

- [ ] **Step 4: Update docs/project/status.md**

Find the model/feature lineup section. Drop any "Riva" entry from the live-models table and add a note in the Status section that Riva removal landed in Stage 1 (link to `docs/superpowers/specs/2026-05-15-disk-reclaim-staging-design.md`).

- [ ] **Step 5: Verify zero unintended Riva references remain**

```bash
grep -rn -i 'riva' /home/spotdog/spot/dartmouth_spot_capstone/README.md /home/spotdog/spot/dartmouth_spot_capstone/docs/ --include='*.md' | grep -v 'docs/superpowers/specs/' | grep -v 'docs/project/stage1-rollback.md' | grep -v 'docs/project/upgrade-plan.md'
```

Expected: no matches. (The spec, runbook, and master plan are allowed to mention Riva — those are historical/rollback references.)

- [ ] **Step 6: Commit Phase 4 cosmetic (Tasks 13 + 14 together)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git add scripts/run_voice_control.py requirements.txt README.md docs/architecture/overview.md docs/project/status.md
git rm scripts/setup_riva.sh
git status --short
```

Verify staged set. Then:

```bash
git commit -m "phase 4 (partial): remove Riva orchestration from non-server code + docs"
git push origin tour_guide_upgrade_matteo
```

---

## Task 15: Backup Ollama 0.16.1 binary

**Files:**
- Create (off-tree): `~/ollama-0.16.1.backup`

- [ ] **Step 1: Confirm current Ollama version**

```bash
ollama --version
```

Expected: `ollama version is 0.16.1`.

- [ ] **Step 2: Stop Ollama service**

```bash
sudo systemctl stop ollama
```

- [ ] **Step 3: Copy the binary**

```bash
sudo cp /usr/local/bin/ollama ~/ollama-0.16.1.backup
ls -lh ~/ollama-0.16.1.backup
```

Expected: ~36 MB file.

- [ ] **Step 4: Confirm the backup is owned by spotdog**

```bash
sudo chown spotdog:spotdog ~/ollama-0.16.1.backup
ls -l ~/ollama-0.16.1.backup
```

Expected: `-rwxr-xr-x 1 spotdog spotdog ...`.

- [ ] **Step 5: Leave Ollama stopped — Task 16 starts the upgrade**

---

## Task 16: Ollama in-place upgrade to ≥ 0.20.0

**Files:**
- Off-tree: `/usr/local/bin/ollama`

- [ ] **Step 1: Run the official Ollama install script**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The script downloads ~36 MB and replaces `/usr/local/bin/ollama`. The systemd unit file is also touched but the user-owned override at `/etc/systemd/system/ollama.service.d/override.conf` survives.

- [ ] **Step 2: Start Ollama service**

```bash
sudo systemctl start ollama
sleep 3
sudo systemctl status ollama --no-pager | head -15
```

Expected: status `active (running)`.

- [ ] **Step 3: Verify the upgraded version**

```bash
ollama --version
```

Expected: `ollama version is 0.20.x` or higher.

- [ ] **Step 4: Verify the systemd override survived**

```bash
cat /etc/systemd/system/ollama.service.d/override.conf
```

Expected: still contains `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`.

- [ ] **Step 5: Verify existing models are still resident and accessible**

```bash
ollama list
ollama run qwen2.5:7b "say hi" --verbose 2>&1 | head -5
```

Expected: list shows `qwen2.5:7b` and `qwen2.5vl:7b`; the run command returns a token stream within ~10s.

- [ ] **Step 6: If any of Steps 2-5 fail, immediately roll back per `docs/project/stage1-rollback.md` Layer 3**

(No git commit — off-tree.)

---

## Task 17: Pull gemma4:e4b

**Files:**
- Off-tree: `/usr/share/ollama/.ollama/models/blobs/` (new manifest + blob files)

- [ ] **Step 1: Disk pre-flight**

```bash
df -h /
```

Expected: ≥ 12 GB free. (After Tasks 3-5 we should have ~28 GB free.) If not, abort and investigate before pulling.

- [ ] **Step 2: Pull the model**

```bash
ollama pull gemma4:e4b
```

Pulls ~9.6 GB. Takes 10-30 minutes depending on network.

- [ ] **Step 3: Verify**

```bash
ollama list
df -h /
```

Expected: list now includes `gemma4:e4b` ~9.6 GB; `df` shows ~9.6 GB consumed.

(No git commit — off-tree.)

---

## Task 18: Phase 2 brain code edits

**Files:**
- Modify: `src/voice_control/llm_brain.py:28-29` (model defaults)
- Modify: `src/voice_control/llm_brain.py:265-314` (warm_up_vlm becomes no-op; remove `_vlm_warmed`)
- Modify: `src/voice_control/llm_brain.py:539-540` (remove `_vlm_warmed` lazy-warm call in `query_vlm`)
- Modify: `src/voice_control/client_mic.py:590` (drop the `vlm_warm` thread; warm_up_vlm is now a no-op anyway, so this is cleanup not behavior change)

- [ ] **Step 1: Change model defaults**

In `src/voice_control/llm_brain.py`, replace:
```python
DEFAULT_MODEL = "qwen2.5:7b"
VLM_MODEL = "qwen2.5vl:7b"
```
with:
```python
DEFAULT_MODEL = "gemma4:e4b"
VLM_MODEL = "gemma4:e4b"
```

- [ ] **Step 2: Convert warm_up_vlm to a no-op + remove `_vlm_warmed` state**

Find `def warm_up_vlm(self):` (line 265). Replace its entire body (everything indented under the def, down to the next top-level def) with:

```python
    def warm_up_vlm(self):
        """No-op since LLM and VLM are now the same model (gemma4:e4b).
        Kept for API compatibility with callers that still invoke it.
        """
        return
```

Also delete the `self._vlm_warmed = False` line in `__init__` (around line 205). Search for any other `_vlm_warmed` references and remove them:

```bash
grep -n '_vlm_warmed' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/llm_brain.py
```

Expected after edits: no matches except possibly the no-op stub (which should NOT reference it either after this step).

- [ ] **Step 3: Clean up the lazy-warm call inside query_vlm**

Find around line 539:
```python
        if not self._vlm_warmed:
            self.warm_up_vlm()
```
and DELETE both lines (the LLM and VLM are the same model and pre-warmed via `warm_up()`).

- [ ] **Step 4: Remove the `vlm_warm` thread in client_mic.py**

In `src/voice_control/client_mic.py`, find the brain warmup block (around line 589-594):

```python
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain_warm = threading.Thread(target=brain.warm_up, daemon=True)
                vlm_warm = threading.Thread(target=brain.warm_up_vlm, daemon=True)
                brain_warm.start()
                vlm_warm.start()
                brain_warm.join()
                vlm_warm.join()
```

Simplify to:

```python
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain.warm_up()
```

(Single model now, no need for parallel warm.)

- [ ] **Step 5: Verify everything imports cleanly**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "from src.voice_control.llm_brain import SpotBrain; b = SpotBrain(); print('model:', b.model)"
```

Expected: `model: gemma4:e4b`.

```bash
python -c "import src.voice_control.client_mic; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Hold the commit — Task 19 verifies before committing**

---

## Task 19: Phase 2 text-only verification

**Files:**
- (no file changes — verification only)

- [ ] **Step 1: Run the existing CLI test against representative typed transcripts**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python src/voice_control/llm_brain.py gemma4:e4b
```

The CLI test at `:608-657` walks through a fixed set of test utterances. For each one, confirm in the printed output:
- The JSON parses cleanly (no `JSONDecodeError`).
- The chosen `action` field matches one of the 28-action names in the catalog.
- `response` is non-empty and reads naturally.
- The first run cold-starts the model (~15-20 s); subsequent calls are warm (~2-6 s).

If any utterance produces malformed JSON or wrong action selection, **abort and roll back per `docs/project/stage1-rollback.md` Layer 2** before committing the brain edits.

- [ ] **Step 2: Smoke-test the VLM path with a sample image**

Find a sample JPEG anywhere in the repo (e.g., `door_calibration/door_view.jpg` or any frame in `models/`):

```bash
find /home/spotdog/spot/dartmouth_spot_capstone -name '*.jpg' -size +10k -not -path '*/.git/*' | head -3
```

Pick one, then:

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
python -c "
from src.voice_control.llm_brain import SpotBrain
import sys
img = open(sys.argv[1], 'rb').read()
b = SpotBrain()
print(b.query_vlm(img, 'describe this image briefly'))
" door_calibration/door_view.jpg
```

(Adjust path to a JPEG that exists.) Expected: a coherent 1-2 sentence description.

- [ ] **Step 3: Resource check**

In a second terminal, run:
```bash
tegrastats --interval 1000 --logfile /tmp/tegra.gemma4.log
```

Then re-run Step 1 or Step 2 in the first terminal. After ~30 seconds, stop tegrastats with Ctrl-C and inspect the log:

```bash
head -30 /tmp/tegra.gemma4.log | grep -o 'RAM [0-9]*' | head -10
```

Expected: RAM usage rises by ~6-8 GB (gemma4:e4b VRAM forecast) during inference.

- [ ] **Step 4: All 3 verifications must pass before committing. If any fails, roll back per `docs/project/stage1-rollback.md` Layer 2 and investigate.**

- [ ] **Step 5: Commit Phase 2**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git add src/voice_control/llm_brain.py src/voice_control/client_mic.py
git status --short
git commit -m "phase 2: collapse LLM + VLM to gemma4:e4b (text-verified, mic-verify pending)"
git push origin tour_guide_upgrade_matteo
```

---

## Task 20: Tag stage1-complete + push

**Files:**
- (no file changes — git tag + push only)

- [ ] **Step 1: Verify HEAD is the Phase 2 commit**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
git log --oneline -5
```

Expected: top entry is `<sha> phase 2: collapse LLM + VLM to gemma4:e4b ...`.

- [ ] **Step 2: Create the annotated tag**

```bash
git tag -a stage1-complete -m "Stage 1 complete: ~27 GB reclaimed, Phase 0.2/0.3/4 + Phase 2 landed, gemma4:e4b text-verified, qwen pair retained as rollback"
```

- [ ] **Step 3: Push the tag**

```bash
git push origin stage1-complete
```

- [ ] **Step 4: Verify the tag is on origin**

```bash
git ls-remote --tags origin stage1-complete
```

Expected: a single line.

- [ ] **Step 5: Final state check**

```bash
df -h /
ollama list
git log --oneline -8
```

Expected:
- `df`: ~18-19 GB free (started at 787 MB, freed ~27 GB, consumed 9.6 GB for gemma4 = ~18 GB net).
- `ollama list`: shows gemma4:e4b + qwen2.5:7b + qwen2.5vl:7b.
- Recent commits: stage1-rollback runbook, phase 0.2, phase 0.3 wave 2, phase 4 partial, phase 2.

Stage 1 is complete. Stage 2 fires when peripherals return.

---

## Task 21 (gated): Stage 1.5 — SSD migration

> **Cross-plan coordination:** The destructive SSD reformat (exFAT → ext4), `/mnt/ssd` mount, fstab entry, and the placeholder subdirs `/mnt/ssd/{ollama-models,clip-cache,spot-logs}/` are owned by the thesis plan at `~/spot/mc_thesis/social-cfm-mppi-thesis-docs/docs/superpowers/plans/2026-05-15-jetson-orin-port.md` (Tasks 1-2). **Do not run Spot T21 until that plan's Tasks 0-2 are complete.** This task only populates the pre-created Spot subdirs.

**Precondition (verify ALL — if any fails, stop and finish the thesis plan first):**

```bash
# 1. Mount exists and is ext4 with the thesis-plan label
mount | grep -E '/mnt/ssd .*ext4' || { echo "FAIL: /mnt/ssd not mounted ext4"; exit 1; }
lsblk -f | grep jetson_ssd || { echo "FAIL: SSD label != jetson_ssd"; exit 1; }

# 2. fstab entry present (so reboots survive)
grep -q '/mnt/ssd ext4' /etc/fstab || { echo "FAIL: /mnt/ssd absent from /etc/fstab"; exit 1; }

# 3. Thesis Task 2 placeholder subdirs exist
test -d /mnt/ssd/ollama-models -a -d /mnt/ssd/clip-cache -a -d /mnt/ssd/spot-logs \
  || { echo "FAIL: thesis Task 2 subdirs missing"; exit 1; }

# 4. Mount writable by spotdog
touch /mnt/ssd/.write-test && rm /mnt/ssd/.write-test || { echo "FAIL: /mnt/ssd not writable"; exit 1; }

echo "OK — thesis Tasks 0-2 satisfied; safe to populate."
```

If any check fails, **stop here**. This task fires after the thesis plan reformats + mounts the SSD.

**Files:**
- Off-tree: `/usr/share/ollama/.ollama/models/` (move to SSD)
- Off-tree: `~/.cache/clip` (symlink to SSD)
- Off-tree: `~/spot/dartmouth_spot_capstone/logs` (symlink to SSD)
- Modify: `/etc/systemd/system/ollama.service.d/override.conf` (add `OLLAMA_MODELS`)

- [ ] **Step 1: Pin SSD root (mount path owned by thesis plan)**

```bash
SSD_ROOT=/mnt/ssd   # fixed by thesis plan Task 1; do not change
# subdirs already exist (thesis plan Task 2); verify rather than mkdir
ls -la $SSD_ROOT/ollama-models $SSD_ROOT/clip-cache $SSD_ROOT/spot-logs
```

Expected: all three dirs listed, owned by spotdog.

- [ ] **Step 2: Stop Ollama and move the model store**

```bash
sudo systemctl stop ollama
sudo mv /usr/share/ollama/.ollama/models $SSD_ROOT/ollama-models/
ls $SSD_ROOT/ollama-models/
```

Expected: directory exists, contains `blobs/`, `manifests/`.

- [ ] **Step 3: Add the OLLAMA_MODELS env var to the systemd override**

Create or edit `/etc/systemd/system/ollama.service.d/override.conf`. Add (or append) inside the `[Service]` block:

```
[Service]
Environment="OLLAMA_MODELS=/mnt/ssd/ollama-models"
```

`/mnt/ssd` is pinned by the thesis plan; no need to adjust.

- [ ] **Step 4: Reload systemd and start Ollama**

```bash
sudo systemctl daemon-reload
sudo systemctl start ollama
sleep 3
ollama list
```

Expected: list still shows gemma4:e4b + qwen pair (running off SSD now).

- [ ] **Step 5: Symlink the CLIP cache**

`$SSD_ROOT/clip-cache/` already exists (thesis Task 2). Move contents in, don't nest.

```bash
if [ -d ~/.cache/clip ] && [ ! -L ~/.cache/clip ]; then
  mv ~/.cache/clip/* $SSD_ROOT/clip-cache/ 2>/dev/null || true
  rmdir ~/.cache/clip
fi
ln -sfn $SSD_ROOT/clip-cache ~/.cache/clip
ls -la ~/.cache/clip
du -sh -L ~/.cache/clip/
```

Expected: symlink resolves to `/mnt/ssd/clip-cache`; size ~343 MB on SSD. Verify YOLO-World still loads encoder:

```bash
uv run python -c "from ultralytics import YOLOWorld; m = YOLOWorld('yolov8s-world.pt'); m.set_classes(['person']); print('clip ok')"
```

Expected: prints `clip ok` (no re-download — uses cache from SSD).

- [ ] **Step 6: Symlink the logs directory**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone
if [ -d logs ] && [ ! -L logs ]; then
  mv logs/* $SSD_ROOT/spot-logs/ 2>/dev/null || true
  rmdir logs 2>/dev/null || true
fi
ln -sfn $SSD_ROOT/spot-logs logs
ls -la logs
```

Expected: symlink resolves to `/mnt/ssd/spot-logs`.

- [ ] **Step 7: Verify reclaim on eMMC**

```bash
df -h /
df -h $SSD_ROOT
```

Expected: `/` shows ~10-15 GB MORE free than the Stage 1 end state; SSD shows the model store + clip cache.

- [ ] **Step 8: Commit a note about SSD migration to docs**

Update `docs/project/stage1-rollback.md` with a note: "Ollama models live on SSD at `/mnt/ssd/ollama-models` as of Stage 1.5 (SSD owned by thesis plan, see `~/spot/mc_thesis/social-cfm-mppi-thesis-docs/docs/superpowers/plans/2026-05-15-jetson-orin-port.md`). Layer 3 (Ollama binary downgrade) still works; the binary on `/usr/local/bin/ollama` is unchanged. Restoring models from a backup requires symlinking `/usr/share/ollama/.ollama/models` back, OR setting `OLLAMA_MODELS` via the systemd override. If the SSD ever fails or is unmounted, `nofail` in `/etc/fstab` lets the system boot but Ollama will fail to start until the override is removed and models are restored to eMMC."

```bash
git add docs/project/stage1-rollback.md
git commit -m "stage 1.5: ollama models, clip cache, logs migrated to SSD"
git push origin tour_guide_upgrade_matteo
git tag stage1.5-complete
git push origin stage1.5-complete
```

---

## Self-review notes

**Spec coverage:** Every requirement in `docs/superpowers/specs/2026-05-15-disk-reclaim-staging-design.md` Stage 1 sections (A, B, C) and Stage 1.5 maps to one or more tasks above. Stage 2 sections are intentionally out of scope (separate plan).

**Placeholder scan:** No `TBD`, `TODO` (in plan tasks; runbook content includes intentional `<NGC_URL_GOES_HERE>` that is captured in Task 2 Step 1), or unimplemented references. The "fallback" in Task 9 Step 4 about `enqueue_render` kwargs is a defensive branch in case the audio_player API differs from the spec's assumption — verified by reading `audio_player.py:230-260` that `enqueue_raw` accepts the kwargs but `enqueue_render` may need adjusting at execution time.

**Type consistency:** `_normalize_location_name` is the import alias; `normalize_location_name` is the source name in `location_manager.py`. Used consistently. `LatencyRecorder.startup_phase` (context manager for boot phases) and `Trace.span` (context manager for per-utterance phases) are the real APIs used throughout, NOT the spec's pseudo-names `time_phase` / `trace_span`. `_PROCESS_START_TS` is the module-level constant added in Task 8 and read in `get_robot_state_dict`.

**Frequent commits:** 5 commits in Stage 1 (runbook, phase 0.2, phase 0.3, phase 4 partial, phase 2) + 1 in Stage 1.5. Each commit is independently revertable. Push after every commit.

**Rollback hooks:** `pre-stage1-stable` tag locks the entry point; `stage1-complete` tag locks the exit point; runbook layers cover code, brain, Ollama binary, Riva re-acquisition, VS Code restoration; qwen pair stays on disk through Stage 1 → Stage 2 boundary.
