# Stage 2C — Vision Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Spot perception to the Phase 9.B revised baseline: per-camera rotation table, distortion re-check + warp assessment (undistortion DEFERRED — see Task 5), gripper color camera plumbing, batched 5-camera scan, `look_around` action, and **GPU YOLO via an isolated `yolo-server` sidecar** — `yolo26m` person-follow (+ByteTrack `track()`) and `yoloe-26x-seg` open-vocab scan (dynamic `set_classes`). Exit tag: `stage2c-vision-complete`.

**Architecture:**
1. Introduce a `Frame` namedtuple as the single perception data type carrying `pil`, `jpeg_bytes`, `source`. All capture helpers return `Frame`. **Frame does NOT carry the `ImageResponse` proto** — paths that need `transforms_snapshot`/`frame_name_image_sensor` (WalkToObjectInImage, world-object publish) stay on `visual_nav._capture_with_response` (returns `(rotated_PIL, image_response)`), never route through bare `Frame`.
2. Bake rotation per camera (not a global 90° CW). **Undistortion is DEFERRED** (grounded 2026-06-01: all cameras report the `pinhole` oneof with NO distortion coefficients; BD ships the KannalaBrandt/BrownConrady proto *schemas* but populates neither, and provides no undistort helper — the only path is checkerboard self-calibration). Task 5 re-checks whether coeffs appear (firmware may change) and captures real frames to grade warp severity; 2C ships raw pinhole frames.
3. Replace the serial 5-camera object scan with a single batched `_capture_multi` + batched detector call.
4. Add `look_around` action and wire to the `src/voice_control/grammar/spot_action.gbnf` grammar from Plan 2A.
5. Run YOLO on the GPU in a **separate venv** (`/mnt/ssd/yolo-env`, CUDA-torch + ultralytics) behind a localhost HTTP service (`yolo-server`, port 11436, mirrors `llama-server`). `visual_nav` becomes a thin `requests` client. spot-env's hard pins are never touched (the only CUDA-torch wheel drags numpy 2.x → isolation avoids it). Detection→walk stays YOLO-pixel → `WalkToObjectInImage`, but fed a **mask-derived** ground-contact pixel (yoloe-seg masks) instead of bbox center. Found objects + the followed person are published to Spot's world-object service via `mutate_world_objects`.

**Tech Stack:** spot-env (client): cv2 (opencv-python==4.11.0.86), bosdyn-client 5.1.4, `requests` 2.26.0, PIL — **no ultralytics/torch for inference**. yolo-env (service, `/mnt/ssd/yolo-env`, isolated): CUDA-torch 2.10.0 + torchvision 0.25.0 (jetson-ai-lab jp6/cu126) + nvidia-cudss-cu12 + ultralytics 8.4.58 + flask; GPU via native ultralytics `device=0` (CUDA 12.6, Orin). **Rejected paths:** ultralytics `.engine`/ONNX-ORT inside spot-env (CPU-only torch + no `tensorrt` python; CUDA-torch would break the numpy pin), NetworkComputeBridge (tours use GraphNav direct, not Autowalk missions → no payoff), `yolo26-seg` (redundant vs yoloe-seg).

**Spec reference:** `docs/superpowers/specs/2026-05-15-stage2-design.md` sections 2C.0 / 2C.0.5 / 2C.1 / 2C.2 / 2C.3 / 2C.4. Fisheye coordinate transforms reference: `fisheye-coords` skill.

**Hard ordering:** Branch off `main` (carries everything through Stage 2F P2). The `stage2b-e2e-complete` tag does **not** exist (2B never ran) — the 2B gate is **skipped** by decision; follow/scan are re-validated directly in Tasks 18 + 21.

**Task execution order (dependency exception):** Tasks 9-11 (batched scan) are written against `yolo_client` (created in Task 17). Execute **1-8, 12-17, then 9-11, then 18-22**. Task numbering is kept stable to preserve cross-references.

**Progression (this plan ships the baseline; cross-camera fusion is deferred — decision B):**
- **2C (this plan):** GPU sidecar; `yolo26m` follow + ByteTrack `track()` for *within-camera* ID lock; `yoloe-26x-seg` dynamic scan + masks→WalkTo pixel; publish detections as world objects; rotation/warp-assessment/gripper/batched scan/look_around. Cross-camera handoff **keeps the existing per-camera locking** in `follow_person`.
- **Follow-on 1 — cross-camera fusion:** overhaul `follow_person` to project each per-camera `track()` to a single **odom-frame target** (reuse `_bearing_in_body_frame`/`pixel_to_camera_space`/`transforms_snapshot`), Kalman **motion-coast across non-overlapping blind gaps**, geometric association. The fused target *is* the published world object. Needs the robot to tune gating/coast.
- **Follow-on 2 (optional) — appearance re-ID:** only for long blind-gap reacquire + crowd disambiguation (geometry+coast can't). Heavy on greyscale fisheye; defer until testing shows real blind-gap loss.

---

## Pre-conditions

- Branch off main: `git checkout -b stage2c-vision-overhaul main` (or a worktree per superpowers:using-git-worktrees). No 2B gate (see Hard ordering).
- `yolo-env` GPU venv provisioned (Task 14; already built 2026-06-01 at `/mnt/ssd/yolo-env` — but **flask is MISSING from it**, verified 2026-07-23; Task 14 Step 2b installs it) and `yolo-server` reachable for Tasks 17-18 and 9-11.
- llama-server daemon running and responsive (vision touches no brain code but smoke tests use the voice loop).
- cv2 available: `python3 -c "import cv2; print(cv2.__version__)"` → `4.11.0.86`.
- Spot powered + on the floor for Tasks 3, 5, 8, 10, 18, 21 (any task touching live capture). E-stop terminal open.

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `src/voice_control/perception/frame.py` | Create (~40 LOC) | `Frame(NamedTuple)`: `pil`, `jpeg_bytes`, `source` |
| `src/voice_control/spot_dispatch.py` | Modify (~+90 LOC, ~-30 LOC) | `CAMERA_ROTATION_DEG` table, `_rotate_for_source`, rewritten `capture_frame`, `gripper_camera_available`, `hand_color_image` in `CAMERA_SOURCES` |
| `src/voice_control/client_mic.py` | Modify (~+60 LOC) | `capture_frame` caller updated to `Frame`; `look_around` VLM branch (mirrors the existing `describe` branch) |
| `scripts/preflight_2c_capture.py` | Commit (already written 2026-06-02) | One frame per camera → warp assessment evidence (Task 5) |
| `scripts/probe_distortion.py` | Create (~40 LOC) | Print each image source's distortion oneof + coefficient values (Task 5 re-check; re-run after firmware updates) |
| `src/voice_control/visual_nav.py` | Modify (~+80 LOC, ~-50 LOC) | Replace serial scan loop with `_scan_all_cameras` + `_find_object_batch`; replace in-process `_get_world_model`/`_get_person_model` (drop ultralytics import) with `yolo_client` HTTP calls; follow loop uses `detect_person(track=True)`; WalkTo pixel from the returned mask point; bbox readers → plain numpy |
| `src/voice_control/intent.py` | Modify (~+8 LOC) | Add `"look_around"` to intent enum if a static enum exists; otherwise N/A |
| `scripts/setup_yolo_env.sh` | Create (~30 LOC) | Reproducible build of the isolated `yolo-env` GPU venv (validated recipe: jetson torch 2.10.0 + torchvision 0.25.0 + nvidia-cudss-cu12 + ultralytics + flask) |
| `requirements-yolo.txt` | Create | Pinned deps for `yolo-env` (separate from spot-env's `requirements.txt`) |
| `scripts/yolo_server.py` | Create (~150 LOC) | Flask GPU service (yolo-env, `127.0.0.1:11436`): loads `yolo26m` + `yoloe-26x-seg` on `device=0`; `/health`, `/detect_person` (optional `track=1` ByteTrack), `/find_object` (dynamic `set_classes`, cached text-PE, returns boxes + mask-derived walk-to pixel) |
| `systemd/yolo-server.service` | Create | systemd unit (mirrors `llama-server.service`); `Environment=LD_LIBRARY_PATH=…/nvidia/cu12/lib:/usr/local/cuda-12.6/lib64`, `Restart=on-failure` |
| `src/voice_control/perception/yolo_client.py` | Create (~90 LOC) | spot-env `requests` client → yolo-server; PIL→jpeg; returns ultralytics-`Results`-compatible `Detections` (`.boxes.xyxy/.conf/.cls` + `walk_xy`); timeout + fail-safe-empty |
| `src/voice_control/world_objects.py` | Modify (~+40 LOC) | Add `publish_detection(...)` → `mutate_world_objects(ADD, IMAGE_COORDINATES)` so found objects + the followed person land in Spot's frame graph |
| `requirements.txt` | Modify | No new inference deps (`requests` already present); note ultralytics/torch now optional in spot-env |
| `docs/project/stage2-rollback.md` | Modify | Append vision-rollback layer (yolo-server down → fail-safe, rebuild yolo-env, scan/rotation revert) |
| `/mnt/ssd/yolo-models/` | Create dir | ultralytics weight cache for yolo-env (gitignored) |
| `tests/captures/` | Create dir | 20-30 sample Spot frames for the follow/scan smoke + client tests |
| `.gitignore` | Modify | Broaden `yolov8*.pt` → `yolo*.pt` + `yoloe*.pt`; add `tests/captures/` if large |

---

## Task 1: Define the `Frame` namedtuple

**Files:**
- Create: `src/voice_control/perception/frame.py`

- [ ] **Step 1: Write the Frame module**

Create `src/voice_control/perception/frame.py`:

```python
"""Frame: the single perception data type used across capture and inference.

Replaces the previous mix of raw PIL.Image / JPEG bytes / source string
that callers had to assemble themselves. capture_frame() returns Frame;
all downstream consumers (visual_nav, vlm_describe, llm_brain) accept Frame.
"""

from typing import NamedTuple
from PIL import Image


class Frame(NamedTuple):
    """A single captured + rotated camera frame.

    Attributes:
        pil: PIL.Image.Image, RGB, in the upright view.
        jpeg_bytes: bytes — JPEG-encoded view of `pil` (cached so VLM
            uploads do not re-encode).
        source: str — the BD SDK source name (e.g., ``"frontleft_fisheye_image"``).
    """
    pil: Image.Image
    jpeg_bytes: bytes
    source: str
```

- [ ] **Step 2: Verify imports work**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "from src.voice_control.perception.frame import Frame; f = Frame(pil=None, jpeg_bytes=b'', source='test'); print(f)"
```
Expected: prints `Frame(pil=None, jpeg_bytes=b'', source='test')`.

- [ ] **Step 3: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/perception/frame.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: Frame namedtuple — single perception data type"
```

---

## Task 2: Add CAMERA_ROTATION_DEG table + _rotate_for_source helper

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (near line 35-46, alongside CAMERA_SOURCES)

- [ ] **Step 1: Add the rotation table**

In `src/voice_control/spot_dispatch.py`, immediately after the `CAMERA_SOURCES` dict (current location lines 38-46), insert:

```python
# Per-camera rotation in degrees CCW required to bring the image upright.
# Negative = clockwise. Values measured empirically per camera mount.
# Front cameras are sideways-mounted (-90° CCW = 90° CW to upright).
# Side and back cameras have other mountings; right is upside-down (180°).
#
# Reference: see fisheye-coords skill for the inverse-mapping math when
# converting bbox pixels in the rotated frame back to native sensor frame
# (still used by visual_nav._unrotate_pixel after this change — the only
# thing that changes is which angle is applied per source).
CAMERA_ROTATION_DEG: dict[str, int] = {
    "frontleft_fisheye_image":  -90,
    "frontright_fisheye_image": -90,
    "left_fisheye_image":         0,
    "right_fisheye_image":      180,
    "back_fisheye_image":         0,
    "hand_color_image":           0,  # gripper color camera (added in Task 8)
}
```

- [ ] **Step 2: Add the rotation helper**

In the same file, near the existing `capture_frame` (around line 182, before that function), insert:

```python
from PIL import Image


def _rotate_for_source(pil_image: Image.Image, source_name: str) -> Image.Image:
    """Rotate a PIL image to upright orientation for the given source.

    Returns a new PIL.Image. Uses ``CAMERA_ROTATION_DEG`` for the per-source
    angle; unknown sources return the image unchanged with no warning (so
    callers can pass arbitrary sources during prototyping).
    """
    angle = CAMERA_ROTATION_DEG.get(source_name, 0)
    if angle == 0:
        return pil_image
    # PIL.rotate is CCW for positive angles, expand=True to avoid clipping.
    return pil_image.rotate(angle, expand=True)
```

- [ ] **Step 3: Smoke-test the helper offline**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from PIL import Image
from src.voice_control.spot_dispatch import _rotate_for_source, CAMERA_ROTATION_DEG
img = Image.new('RGB', (640, 480), 'red')
for src, deg in CAMERA_ROTATION_DEG.items():
    out = _rotate_for_source(img, src)
    print(f'{src}: {deg}deg -> {out.size}')
"
```
Expected: front-left/right become (480, 640); left/back stay (640, 480); right stays (640, 480); hand_color_image stays (640, 480).

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/spot_dispatch.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: CAMERA_ROTATION_DEG table + _rotate_for_source helper"
```

---

## Task 3: Rewrite capture_frame to return Frame

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (existing `capture_frame` around lines 182-207)

- [ ] **Step 1: Read the current capture_frame implementation**

```bash
grep -n -A 30 'def capture_frame' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```
Confirm it returns JPEG bytes today and takes a `camera="front"` argument.

- [ ] **Step 2: Rewrite capture_frame**

Replace the existing `capture_frame` body with this version (preserve the function name + the `camera="front"` parameter for backward compat):

```python
from io import BytesIO
from src.voice_control.perception.frame import Frame
from bosdyn.api import image_pb2


def capture_frame(camera: str = "front") -> Frame | None:
    """Capture a single frame, rotated upright per CAMERA_ROTATION_DEG.

    Returns Frame(pil, jpeg_bytes, source) or None on failure.

    Pipeline: capture → decode → rotate → encode. (Undistortion deferred —
    see Task 5. If a future calibration spike lands, its remap step slots
    between decode and rotate.)
    """
    source = CAMERA_SOURCES.get(camera, camera)
    from src.session import quick_robot
    from bosdyn.client.image import ImageClient

    try:
        robot = quick_robot()
        image_client = robot.ensure_client(ImageClient.default_service_name)
        responses = image_client.get_image_from_sources([source])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("capture_frame %s: %s", source, e)
        return None
    if not responses:
        return None
    img_resp = responses[0]

    # Decode raw bytes → PIL.
    pixel_fmt = img_resp.shot.image.pixel_format
    raw = img_resp.shot.image.data
    if pixel_fmt == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
        pil = Image.frombytes(
            "RGB",
            (img_resp.shot.image.cols, img_resp.shot.image.rows),
            raw,
        )
    else:
        # Default: JPEG-encoded (fisheye monochrome).
        pil = Image.open(BytesIO(raw)).convert("RGB")

    # Rotate per source.
    pil = _rotate_for_source(pil, source)

    # Encode JPEG once for downstream VLM upload.
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()

    return Frame(pil=pil, jpeg_bytes=jpeg, source=source)
```

- [ ] **Step 3: Update callers that expected raw bytes**

Find every caller of `capture_frame`:
```bash
grep -rn 'capture_frame' /home/spotdog/spot/dartmouth_spot_capstone/src/ /home/spotdog/spot/dartmouth_spot_capstone/scripts/
```
For each call site, replace `bytes_result = capture_frame(...)` with `frame = capture_frame(...); if frame: bytes_result = frame.jpeg_bytes`. The PIL access path is `frame.pil`.

Verified call sites (grep 2026-07-23 — NOT `llm_brain.py`):
- `src/voice_control/client_mic.py` ~line 1550 (the `describe` VLM branch: `image_bytes = capture_frame(camera)` → `frame.jpeg_bytes`)
- `tests/audio/vlm_probe/probe.py` ~line 97 (untracked test harness — update if committing it, else skip)

Update each call site to access `.jpeg_bytes` or `.pil` explicitly. Do NOT change call site count semantics.

- [ ] **Step 4: Smoke-test live capture**

Requires Spot powered. With Spot on the floor:
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from src.voice_control.spot_dispatch import capture_frame, CAMERA_SOURCES
for nick in ['front', 'left', 'right', 'back']:
    f = capture_frame(nick)
    if f is None:
        print(f'{nick}: NONE')
        continue
    print(f'{nick}: source={f.source} pil={f.pil.size} jpeg={len(f.jpeg_bytes)}b')
    f.pil.save(f'/tmp/cap_{nick}.jpg')
"
ls -la /tmp/cap_*.jpg
```
Expected: 4 JPEGs saved. **Open each in an image viewer and confirm they are upright** (front cameras no longer sideways). This is the visual smoke test that Task 2's rotation table is correct per-camera.

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/spot_dispatch.py src/voice_control/client_mic.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: capture_frame returns Frame; per-source rotation applied"
```

---

## Task 4: Mid-plan checkpoint #1 — caveman:cavecrew-reviewer on Tasks 1-3

**Files:**
- Read only: diff of Tasks 1-3

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 1 review",
  prompt: "Review the last 3 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone (commits adding src/voice_control/perception/frame.py, CAMERA_ROTATION_DEG table, _rotate_for_source helper, and rewriting capture_frame to return Frame). This is Stage 2C of a Boston Dynamics Spot robot perception overhaul. Critical concerns:\n1. Every existing caller of capture_frame is updated to access .jpeg_bytes or .pil (no broken callers).\n2. The rotation table covers every source in CAMERA_SOURCES (no KeyError at runtime).\n3. PIL.Image.rotate convention (CCW positive) matches the documented intent.\n4. RGB_U8 decode branch handles only the gripper color path (raw bytes, NOT JPEG) — fisheye monochrome must still hit the JPEG-decode path.\n5. No memory leak from PIL JPEG re-encode per frame (Frame.jpeg_bytes is cached, not re-computed).\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Fix BLOCKER/HIGH findings**

If anything is BLOCKER/HIGH: fix in a new commit. Re-dispatch reviewer if substantial changes.

- [ ] **Step 3: Commit fixes (if any)**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address checkpoint 1 review findings"
```

---

## Task 5: Distortion re-check + warp assessment (undistortion DEFERRED)

> **Decision (2026-07-23, user):** DEFER undistortion. Grounded verdict 2026-06-01: every camera reports the `pinhole` oneof with NO distortion coefficients; bosdyn ships the `KannalaBrandtIntrinsics`/Brown-Conrady proto *schemas* but populates neither and has no undistort/rectify helper anywhere in the SDK. Intrinsics (K) alone cannot flatten a fisheye — the warp lives in the k1..k4 coefficients BD doesn't provide. The only path is checkerboard self-calibration (`cv2.fisheye.calibrate` → k1..k4 → `initUndistortRectifyMap` + remap). 2C ships raw pinhole frames; this task (a) keeps re-checking whether BD starts populating coeffs, and (b) grades real-world warp severity to decide if a calibration spike is ever warranted.

**Files:**
- Create: `scripts/probe_distortion.py` (~40 LOC)
- Commit: `scripts/preflight_2c_capture.py` (already written 2026-06-02)
- Create: `tests/captures/` (warp-evidence frames)

- [ ] **Step 1: Write the distortion probe**

Create `scripts/probe_distortion.py` — for each image source, print which distortion oneof is set (`WhichOneof("camera_models")` — verify the actual oneof name via `image_pb2.ImageSource.DESCRIPTOR.oneofs`) and any coefficient values. Read-only, no lease. Re-run after every robot firmware update; if coeffs ever appear, reopen the undistort decision.

- [ ] **Step 2: Run probe (robot powered, read-only)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && python3 scripts/probe_distortion.py
```
Expected (as of 2026-06-01 firmware): every source prints `pinhole`, no coefficients. Record the output + firmware version in the commit message.

- [ ] **Step 3: Capture warp-assessment frames**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && python3 scripts/preflight_2c_capture.py
cp /tmp/preflight_2c/*.jpg tests/captures/
```
Point at least one camera at straight vertical edges (door frame, wall corner) before capturing.

- [ ] **Step 4: Grade warp severity**

Open each frame. Check: are straight lines near the image EDGES visibly curved? Center is expected-good (why `pixel_to_camera_space`/walk-to already work). Verdict options:
- **Mild/center-usable** → done; 2C ships raw frames. Note verdict below.
- **Severe at edges AND edge detections matter** → schedule a self-calibration spike (checkerboard session, ~half day with robot) as a follow-on; do NOT block 2C.

Record the verdict here when run: `WARP VERDICT (date): ______`

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add scripts/probe_distortion.py scripts/preflight_2c_capture.py tests/captures/
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: distortion probe + warp-assessment captures (undistort deferred — BD provides no coeffs)"
```

---

## Task 6: (REMOVED — undistortion deferred)

Undistort module + capture_frame integration removed from 2C by decision 2026-07-23 (see Task 5 rationale: BD provides no distortion coefficients; self-calibration is the only path and is not warranted until Task 5's warp verdict says so). If a future calibration spike produces per-camera k1..k4, revive this as its own plan: `cv2.fisheye.initUndistortRectifyMap` (D as `(4,1)`) + cached `cv2.remap` per source, slotted between decode and rotate in `capture_frame`. No code lands in 2C.

---

## Task 7: Mid-plan checkpoint #2 — safety-reviewer + caveman on capture pipeline

**Files:**
- Read only: diff of Tasks 3, 5, 8

- [ ] **Step 1: Dispatch safety-reviewer**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C capture pipeline safety",
  prompt: "Audit changes to src/voice_control/spot_dispatch.py, src/voice_control/client_mic.py, and scripts/probe_distortion.py in the recent commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. This is Stage 2C of a Boston Dynamics Spot robot perception overhaul. Focus on:\n1. Whether capture_frame failures (None return) propagate safely through callers — does any downstream code crash on None frame and bring down the brain or block estop? The describe branch in client_mic.py now reads frame.jpeg_bytes.\n2. Whether any change to spot_dispatch.py inadvertently changes motion path behavior (the file mixes capture and motion — the rotation additions should be capture-only).\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch caveman:cavecrew-reviewer in parallel**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 2 diff review",
  prompt: "Review the recent commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone — Stage 2C tasks 3+5 (capture_frame Frame rewrite, caller updates, distortion probe). Verify:\n1. Every capture_frame caller reads .jpeg_bytes/.pil (client_mic.py describe branch especially).\n2. capture_frame pipeline is decode → rotate → encode with no undistort remnants.\n3. probe_distortion.py is read-only (no lease, no power, no motion).\nReport severity-tagged findings only."
})
```

- [ ] **Step 3: Address BLOCKER/HIGH findings; commit fixes**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address checkpoint 2 review findings"
```
Skip if no fixes needed.

---

## Task 8: Gripper color camera plumbing

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (CAMERA_SOURCES, capture_frame RGB_U8 branch already added in Task 3; verify; add `gripper_camera_available` helper)

- [ ] **Step 1: Add hand_color_image to CAMERA_SOURCES**

In `src/voice_control/spot_dispatch.py` near line 38 (the dict), add:

```python
CAMERA_SOURCES = {
    # ... existing entries ...
    "gripper":          "hand_color_image",
    "hand":             "hand_color_image",
    "hand_color":       "hand_color_image",
}
```

The Task 2 `CAMERA_ROTATION_DEG` table already includes `hand_color_image: 0`. Verify it's still there.

- [ ] **Step 2: Add gripper_camera_available pre-flight**

Add to `spot_dispatch.py` after the helpers:

```python
from src.session import quick_robot
from bosdyn.client.image import ImageClient


def gripper_camera_available(robot=None) -> bool:
    """True iff hand_color_image is in the robot's image source list.

    Spot's arm must be installed + powered for the gripper camera to appear.
    """
    if robot is None:
        robot = quick_robot()
    try:
        image_client = robot.ensure_client(ImageClient.default_service_name)
        sources = image_client.list_image_sources()
        return any(s.name == "hand_color_image" for s in sources)
    except Exception:
        return False
```

- [ ] **Step 3: Verify capture path via Task 3's RGB_U8 branch**

The RGB_U8 decode branch added in Task 3 step 2 already handles gripper color frames. Confirm:

```bash
grep -n 'PIXEL_FORMAT_RGB_U8' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/spot_dispatch.py
```
Expected: at least one match in `capture_frame`. If missing, the rewrite from Task 3 step 2 was incomplete — go back and fix.

- [ ] **Step 4: Smoke-test live (requires arm-equipped Spot, powered)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from src.voice_control.spot_dispatch import gripper_camera_available, capture_frame
if not gripper_camera_available():
    print('Gripper camera NOT available (arm not powered or no arm) — skipping')
else:
    f = capture_frame('gripper')
    if f:
        f.pil.save('/tmp/gripper.jpg')
        print(f'gripper.jpg saved: {f.pil.size}')
    else:
        print('capture_frame returned None')
"
```
Expected when arm is powered: saves a color JPEG. Open and verify it is a color image (not grayscale).

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/spot_dispatch.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: gripper color camera (hand_color_image) plumbing + pre-flight gate"
```

---

## Task 9: Add _find_object_batch + _scan_all_cameras helpers

> **Ordering:** requires `yolo_client` (Task 17) + a running yolo-server. Execute AFTER Task 17 (see "Task execution order", top of plan).

**Files:**
- Modify: `src/voice_control/visual_nav.py` (around the existing `_find_object` at line 175)

- [ ] **Step 1: Add the batched detector wrapper**

Insert in `visual_nav.py` near `_find_object`. The GPU sidecar's `/find_object` is single-image; "batched" here means one client call per frame against an already-warm GPU (~55 ms each, ~280 ms for a full 5-camera one-shot scan — well within budget). Each hit carries `walk_xy` (mask-derived ground-contact pixel) which MUST be threaded through to WalkToObjectInImage (Task 17d).

```python
def _find_object_batch(pil_images: list, description: str) -> list:
    """Detect `description` in each image via yolo-server.

    Returns a list (same order as input) of None or dict:
        {"bbox": [x1,y1,x2,y2], "conf": float, "walk_xy": [x,y]}
    Coordinates are in the ROTATED image frame (input frame).
    """
    out = []
    for img in pil_images:
        det = yolo_client.find_object(img, [description])
        if len(det.boxes) == 0:
            out.append(None)
            continue
        best = int(det.boxes.conf.argmax())
        out.append({
            "bbox": det.boxes.xyxy[best].tolist(),
            "conf": float(det.boxes.conf[best]),
            "walk_xy": det.boxes.walk[best],
        })
    return out
```

- [ ] **Step 2: Add _scan_all_cameras helper**

Insert just after `_find_object_batch`. **Capture goes through `_capture_with_response` (visual_nav.py:354)** — NOT `_capture_multi` (line 200), which returns raw `{source: ImageResponse}` protos, and NOT bare `Frame` — because the downstream WalkToObjectInImage + `publish_detection` need each hit's `ImageResponse` (`transforms_snapshot`, `frame_name_image_sensor`).

```python
def _scan_all_cameras(session, query: str) -> dict:
    """One-shot 5-camera scan for an object via yolo-server.

    Returns dict: source → {"bbox", "conf", "walk_xy", "rotated_size",
    "image_response"} for sources with a hit. Sources with no hit or a
    failed capture are omitted.
    """
    hits = {}
    captures = []  # (source, rotated_PIL, image_response)
    for source, _ in ALL_CAMERAS:  # ALL_CAMERAS at visual_nav.py:38-46
        cap = _capture_with_response(session, source)
        if cap is None:
            continue
        pil_img, image_response = cap
        captures.append((source, pil_img, image_response))
    if not captures:
        return {}
    dets = _find_object_batch([c[1] for c in captures], query)
    for (source, pil_img, image_response), det in zip(captures, dets):
        if det is None:
            continue
        det["rotated_size"] = pil_img.size
        det["image_response"] = image_response
        hits[source] = det
    return hits
```

(Adapt the `_capture_with_response` return-shape access to the actual signature — read it first; docstring says `(rotated_PIL, image_response)`.)

- [ ] **Step 3: Smoke-test offline (yolo-server running)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from PIL import Image
from src.voice_control.visual_nav import _find_object_batch
imgs = [Image.new('RGB', (320, 240), c) for c in ('red', 'green', 'blue')]
out = _find_object_batch(imgs, 'cat')
print('batch result:', out)  # [None, None, None] on solid-color inputs
"
```
Expected: no crash, `[None, None, None]`. Also verify fail-safe: stop yolo-server, rerun — still `[None, None, None]`, no traceback.

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/visual_nav.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: _find_object_batch + _scan_all_cameras (yolo-server, walk_xy + ImageResponse threaded)"
```

---

## Task 10: Replace serial scan loop in visual_nav.py with batched scan

> **Ordering:** execute AFTER Tasks 17 + 9 (see "Task execution order", top of plan).

**Files:**
- Modify: `src/voice_control/visual_nav.py` (existing serial scan loop — per spec at lines 595-609)

- [ ] **Step 1: Locate the existing serial scan loop**

```bash
sed -n '590,615p' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/visual_nav.py
```
This is the loop in (most likely) `_seek_object_in_all_cameras` or similar that iterates camera-by-camera and runs `_find_object` per camera. **Read the actual code; line numbers may have shifted by ±5 due to prior tasks' edits.**

- [ ] **Step 2: Replace the loop body with a single _scan_all_cameras call**

Replace the per-camera loop with:

```python
# Stage 2C: one-shot GPU scan (was serial CPU loop pre-2026-05-15).
hits = _scan_all_cameras(session, query)
if not hits:
    return None
# Pick the largest bbox across cameras (closest object). Existing
# selection logic ran per-camera; now we pick across the batched dict.
def _area(h):
    x1, y1, x2, y2 = h["bbox"]
    return (x2 - x1) * (y2 - y1)
best_source, best = max(hits.items(), key=lambda kv: _area(kv[1]))
# best carries: bbox, conf, walk_xy (rotated coords), rotated_size,
# image_response. Downstream (turn-to-bearing, WalkToObjectInImage per
# Task 17d) consumes walk_xy — unrotated via _unrotate_pixel — and
# image_response for transforms_snapshot/frame_name_image_sensor.
```

**Important:** preserve the function's outer signature. The behavioral changes: 5 sequential detect calls → one warm-GPU pass, and the WalkTo pixel comes from `best["walk_xy"]` (mask ground-contact) instead of bbox center. Pixel un-rotation + bearing math + WalkToObjectInImage submission machinery are otherwise unchanged.

- [ ] **Step 3: Smoke-test the replaced scan on Spot**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/test_visual_nav.py --target "chair" --no-walk
```
(`--no-walk` is hypothetical; if the script doesn't support it, comment out the WalkToObjectInImage submission temporarily for a dry run.)
Expected: prints which camera saw a chair and the bbox. Time the run; should be < 1 s for the scan portion (vs > 2 s for the prior serial loop).

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/visual_nav.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: replace serial 5-camera scan with single batched detect call"
```

---

## Task 11: Mid-plan checkpoint #3 — caveman:cavecrew-reviewer on Tasks 8-10

**Files:**
- Read only: diff of Tasks 8-10

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 3 review",
  prompt: "Review the last 3 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone — Stage 2C tasks 8-10 (gripper plumbing + batched scan helpers + serial-loop replacement). Verify:\n1. gripper_camera_available does NOT call image_client without try/except (arm not powered = exception, NOT crash).\n2. _find_object_batch is fail-safe per frame: yolo_client returning empty Detections yields None for that frame, never a raise; empty input list returns [].\n3. _scan_all_cameras zips captures and detections in the same order, and every hit dict carries walk_xy + rotated_size + image_response.\n4. The serial-loop replacement preserves the original function's selection semantics (largest-bbox across cameras, not first-hit) and the WalkTo pixel now comes from walk_xy, unrotated via _unrotate_pixel.\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Fix BLOCKER/HIGH findings; commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address checkpoint 3 review findings"
```

---

## Task 12: Add look_around action (client_mic VLM branch, mirrors describe)

> **Grounded 2026-07-23:** there is no `describe_image()` anywhere. The real VLM entry point is `LLMBrain.query_vlm(image_bytes, question, yolo_hint="")` (instance method, `llm_brain.py:571`), and the `describe` action is NOT handled in `dispatch_intent` — it is special-cased in `client_mic.py` (~line 1545, `cmd == "describe"` branch) where `brain` and `tts` are in scope. `look_around` follows the exact same pattern. `dispatch_intent` (`spot_dispatch.py:755`, returns bool, uses local `name`) is untouched. No `intent.py` regex fast-path either — look_around is grammar/LLM-only, same as describe (decision: LOW review item).

**Files:**
- Modify: `src/voice_control/client_mic.py` (new `look_around` branch beside the `describe` branch)

- [ ] **Step 1: Read the describe branch**

```bash
sed -n '1493,1600p' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/client_mic.py
```
Note: how it suppresses the LLM's hallucinated `response`, speaks a stock ack, calls `brain.query_vlm`, speaks the VLM answer via `tts`, and skips `dispatch_intent` for that action. Mirror all of it.

- [ ] **Step 2: Add the look_around branch**

In the action loop, next to `if cmd == "describe":`, add `elif cmd == "look_around":`:

```python
elif cmd == "look_around":
    vlm_used = True
    from src.voice_control.spot_dispatch import (
        capture_frame, gripper_camera_available)
    nicks = ["front", "left", "right", "back"]
    if gripper_camera_available():
        nicks.append("gripper")
    captions = []
    for nick in nicks:
        frame = capture_frame(nick)
        if frame is None:
            continue
        try:
            cap = brain.query_vlm(
                frame.jpeg_bytes,
                "In one short sentence, what is visible in this direction?")
            captions.append(f"{nick}: {cap}")
        except Exception as e:
            print(f"[look_around] VLM failed on {nick}: {e}")
    summary = " ".join(captions) if captions else "I couldn't see anything."
    print(f"\nSPOT (VLM): \"{summary}\"")
    if tts:
        tts.wait()
        tts.speak(summary, voice=voice_id)
```

Also extend the pre-dispatch ack suppression: the `describe_action = next(...)` stock-ack logic (~line 1500) should treat `look_around` the same way (LLM `response` is a hallucination before the VLM has run) — e.g. match `a.get("intent") in ("describe", "look_around")` and use a "Let me look around..." ack.

**Latency note:** 4-5 sequential VLM calls ≈ 15-30 s; the branch runs on the dispatch path, same blocking profile as `describe`. The safety reviewer in Task 18 checks stop/E-stop coverage during it.

- [ ] **Step 3: Smoke-test (robot + llama-server up)**

Run the voice loop, say "look around". Expected: stock ack immediately, then per-direction one-liners spoken as one summary. Confirm a follow-up "stop" utterance is still handled after the summary.

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/client_mic.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: look_around — per-camera VLM sweep in client_mic (mirrors describe branch)"
```

---

## Task 13: Wire look_around into the spot_action.gbnf grammar

**Files:**
- Modify: `src/voice_control/grammar/spot_action.gbnf` (created in Plan 2A Task 3)

- [ ] **Step 1: Add look_around to the action enum production**

> **Grounded 2026-07-23:** the real rule is **`action-name`** (hyphen, NOT `action_name`), it lives on **ONE line** (`spot_action.gbnf:14`), and the grammar emits `{"actions":[{"action":...}]}`. Keep it one line — llama.cpp's GBNF parser silently fails on multi-line `|` continuations. Do NOT reformat the rule.

Append ` | "\"look_around\""` to the END of the existing single-line rule at `src/voice_control/grammar/spot_action.gbnf:14`:

```gbnf
action-name ::= "\"stop\"" | "\"freeze\"" | ... | "\"enable_mouth\"" | "\"disable_mouth\"" | "\"look_around\""
```

(One-token edit at end of line 14; every existing alternative preserved verbatim.)

- [ ] **Step 2: Re-run mic-verify smoke against look_around**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  echo "look around" | python3 scripts/mic_verify.py --text-mode --intent look_around
```
(Assumes mic_verify supports text mode for grammar-only smoke. If not, the smoke is deferred to Stage 2B-style live mic test.)
Expected: brain returns `{"action": "look_around", ...}` valid JSON.

- [ ] **Step 3: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/grammar/spot_action.gbnf
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: spot_action.gbnf — add look_around to action enum"
```

---

## Task 14: Provision the isolated `yolo-env` GPU venv

> **Why a separate venv:** ultralytics' GPU path needs a CUDA-enabled torch. The only CUDA-torch wheel for jp6/cu126 (jetson-ai-lab) drags `numpy 2.x`, which breaks spot-env's hard `numpy==1.26.4` pin (segfault vs onnxruntime-gpu/opencv/kokoro). Isolating YOLO in its own venv keeps spot-env's pins untouched; spot-env talks to it over HTTP (Tasks 15-17). **Already built + validated 2026-06-01** — this task makes it reproducible.

**Files:**
- Create: `scripts/setup_yolo_env.sh`
- Create: `requirements-yolo.txt`
- Provision: `/mnt/ssd/yolo-env`

- [ ] **Step 1: Write the setup script**

Create `scripts/setup_yolo_env.sh`:

```bash
#!/usr/bin/env bash
# Build the isolated GPU venv for the yolo-server sidecar (Stage 2C).
# Validated 2026-06-01 on JetPack 6.2 / L4T R36.5 / CUDA 12.6 / Orin.
set -euo pipefail
VENV=/mnt/ssd/yolo-env
IDX=https://pypi.jetson-ai-lab.io/jp6/cu126   # .io host (.dev is dead DNS)

python3.10 -m venv "$VENV"                     # explicit 3.10; plain venv (NOT --system-site-packages)
"$VENV/bin/pip" install --upgrade pip
# CUDA torch + MATCHING torchvision. 0.25.0 pairs with torch 2.10.0; the index
# default 0.26.0 is the torch-2.11 partner and silently breaks torchvision::nms.
"$VENV/bin/pip" install "torch==2.10.0" "torchvision==0.25.0" --index-url "$IDX"
# The jetson torch wheel links libcudss.so.0 but does NOT declare it as a dep.
"$VENV/bin/pip" install "nvidia-cudss-cu12==0.7.1.6" --index-url "$IDX"
"$VENV/bin/pip" install "ultralytics==8.4.58" flask   # pins match the validated 2026-06-01 build
echo "Done. yolo-server needs LD_LIBRARY_PATH (baked into the systemd unit, Task 16)."
```

- [ ] **Step 2: Run it (skip if `/mnt/ssd/yolo-env` already exists from the 2026-06-01 build)**

```bash
bash /home/spotdog/spot/dartmouth_spot_capstone/scripts/setup_yolo_env.sh
```

- [ ] **Step 2b: Close the flask gap in the EXISTING env**

The 2026-06-01 build predates the flask requirement — **flask is NOT installed in `/mnt/ssd/yolo-env`** (verified `pip list` 2026-07-23). If skipping Step 2:

```bash
/mnt/ssd/yolo-env/bin/pip install flask
/mnt/ssd/yolo-env/bin/python -c "import flask; print(flask.__version__)"
```

- [ ] **Step 3: Verify GPU + the two gotchas**

```bash
export LD_LIBRARY_PATH=/mnt/ssd/yolo-env/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.6/lib64
/mnt/ssd/yolo-env/bin/python -c "
import torch, torchvision
from torchvision.ops import nms
print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA unavailable — check LD_LIBRARY_PATH / cudss'
"
```
Expected: `torch 2.10.0 tv 0.25.0 cuda True` (no `nms` ImportError). If `cuda False` → LD_LIBRARY_PATH missing the `nvidia/cu12/lib` dir (no RPATH to bundled `libcudss.so.0`). If `torchvision::nms does not exist` → torchvision drifted off 0.25.0.

- [ ] **Step 4: Pin yolo-env deps + commit**

Create `requirements-yolo.txt` (documents the env; install via `setup_yolo_env.sh`, NOT this file directly — torch needs the jetson index):
```
# yolo-env (isolated GPU venv for yolo-server). Install via scripts/setup_yolo_env.sh.
# jetson-ai-lab jp6/cu126 index for torch/torchvision/cudss; PyPI for the rest.
torch==2.10.0
torchvision==0.25.0      # MUST match torch 2.10.0 (NOT the index-default 0.26.0)
nvidia-cudss-cu12==0.7.1.6   # libcudss.so.0 — undeclared dep of the jetson torch wheel
ultralytics==8.4.58
flask
```

Also broaden `.gitignore` (File-Map item with no other home): change `yolov8*.pt` → `yolo*.pt` and add `yoloe*.pt` (weights live in `/mnt/ssd/yolo-models`, never in-repo).

```bash
chmod +x scripts/setup_yolo_env.sh
git -C /home/spotdog/spot/dartmouth_spot_capstone add scripts/setup_yolo_env.sh requirements-yolo.txt .gitignore
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: provision isolated yolo-env GPU venv (recipe + pins) + gitignore yolo weights"
```

---

## Task 15: `yolo_server.py` — GPU inference HTTP service

**Files:**
- Create: `scripts/yolo_server.py` (runs in yolo-env; standalone — does NOT import the spot `src` package)
- Output: a Flask service on `127.0.0.1:11436` (11434/11435/50055 already taken)

- [ ] **Step 1: Write the server**

Create `scripts/yolo_server.py`:

```python
#!/usr/bin/env python3
"""GPU YOLO inference sidecar for Spot (Stage 2C). Runs in /mnt/ssd/yolo-env.

Loads the follow + open-vocab scan models once on the GPU and serves them over
localhost HTTP so spot-env (CPU-only torch, hard-pinned) never imports torch.

Endpoints:
  GET  /health                         -> {"status","models"}
  POST /detect_person  [?track=1]       body=jpeg -> {"boxes":[[x1,y1,x2,y2,conf,id],...]}
  POST /find_object    ?classes=a,b,c   body=jpeg -> {"objects":[{xyxy,conf,name,walk_xy},...]}
"""
import io
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO, YOLOE

DEVICE = 0
PERSON_WEIGHTS = "yolo26m.pt"        # follow (class 0); mAP 53.1, ~32 ms GPU
SCAN_WEIGHTS = "yoloe-26x-seg.pt"    # open-vocab seg; dynamic set_classes, ~55 ms GPU
CONF = 0.25

app = Flask(__name__)
_person = YOLO(PERSON_WEIGHTS)
_scan = YOLOE(SCAN_WEIGHTS)
_pe_cache = {}                       # frozenset(classes) -> CLIP text embeddings

# Flask runs threaded; both models are shared mutable state. _person_lock
# serializes track() (ByteTrack keeps internal per-stream state — concurrent
# calls corrupt IDs); _scan_lock serializes set_classes()+predict() (a race
# runs predict with the wrong vocab).
_person_lock = threading.Lock()
_scan_lock = threading.Lock()

_warm = Image.fromarray(np.full((480, 640, 3), 127, "uint8"))
_person.predict(_warm, device=DEVICE, classes=[0], conf=CONF, verbose=False)
with _scan_lock:                     # warm the scan path too (CLIP text encoder + first CUDA graph)
    _pe_cache[frozenset(["person"])] = _scan.get_text_pe(["person"])
    _scan.set_classes(["person"], _pe_cache[frozenset(["person"])])
    _scan.predict(_warm, device=DEVICE, conf=CONF, verbose=False)


def _pil():
    return Image.open(io.BytesIO(request.get_data())).convert("RGB")


def _walk_xy(box_xyxy, mask_xy):
    """Ground-contact pixel for WalkToObjectInImage: lowest interior mask point,
    else bbox bottom-center. mask_xy = (N,2) polygon in ORIGINAL-image coords."""
    if mask_xy is not None and len(mask_xy) > 0:
        pts = np.asarray(mask_xy)
        ymax = pts[:, 1].max()
        xs = pts[pts[:, 1] >= ymax - 2][:, 0]     # cols at the lowest rows
        return [float(np.median(xs)), float(ymax)]
    x1, y1, x2, y2 = box_xyxy
    return [float((x1 + x2) / 2), float(y2)]      # bbox bottom-center fallback


@app.get("/health")
def health():
    return jsonify(status="ok", models={"person": PERSON_WEIGHTS, "scan": SCAN_WEIGHTS})


@app.post("/detect_person")
def detect_person():
    """track=1 CONTRACT: ByteTrack state is a SINGLE stream. Only ONE caller
    (the follow loop's primary front camera) may send track=1; every other
    camera/caller must use plain detection, or IDs get corrupted by
    interleaved frames from different viewpoints."""
    track = request.args.get("track") == "1"
    img = _pil()
    with _person_lock:
        if track:                                 # ByteTrack IDs persist across calls (one stream)
            res = _person.track(img, device=DEVICE, classes=[0], conf=CONF,
                                persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        else:
            res = _person.predict(img, device=DEVICE, classes=[0], conf=CONF, verbose=False)[0]
    b = res.boxes
    out = []
    if b is not None and len(b):
        xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy() if (track and b.id is not None) else [-1] * len(conf)
        for i in range(len(conf)):
            out.append([*[float(v) for v in xyxy[i]], float(conf[i]), int(ids[i])])
    return jsonify(boxes=out)


@app.post("/find_object")
def find_object():
    classes = [c.strip() for c in request.args.get("classes", "").split(",") if c.strip()]
    if not classes:
        return jsonify(error="no classes"), 400
    key = frozenset(classes)
    img = _pil()
    with _scan_lock:                              # set_classes + predict must be atomic
        if key not in _pe_cache:                  # cache text embeddings per vocab
            _pe_cache[key] = _scan.get_text_pe(classes)
        _scan.set_classes(classes, _pe_cache[key])
        res = _scan.predict(img, device=DEVICE, conf=CONF, verbose=False)[0]
    b = res.boxes
    objs = []
    if b is not None and len(b):
        xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        masks = res.masks.xy if res.masks is not None else [None] * len(conf)
        for i in range(len(conf)):
            mask = masks[i] if i < len(masks) else None   # guard: masks can be shorter than boxes
            objs.append({"xyxy": [float(v) for v in xyxy[i]], "conf": float(conf[i]),
                         "name": res.names[int(cls[i])], "walk_xy": _walk_xy(xyxy[i], mask)})
    return jsonify(objects=objs)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=11436, threaded=True)
```

- [ ] **Step 2: Manual launch (foreground, for testing)**

```bash
cd /mnt/ssd/yolo-models    # ultralytics caches weights in cwd
export LD_LIBRARY_PATH=/mnt/ssd/yolo-env/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.6/lib64
/mnt/ssd/yolo-env/bin/python /home/spotdog/spot/dartmouth_spot_capstone/scripts/yolo_server.py
```
Expected: Flask on 127.0.0.1:11436; both models load (`yolo26m` + `yoloe-26x-seg` download on first run; `yoloe` also fetches ultralytics/CLIP.git for the text encoder).

- [ ] **Step 3: Smoke the endpoints (second terminal)**

```bash
python3 -c "from PIL import Image; Image.new('RGB',(640,480),'gray').save('/tmp/t.jpg')"
curl -s localhost:11436/health
curl -s --data-binary @/tmp/t.jpg localhost:11436/detect_person
curl -s --data-binary @/tmp/t.jpg "localhost:11436/detect_person?track=1"
curl -s --data-binary @/tmp/t.jpg "localhost:11436/find_object?classes=chair,backpack"
```
Expected: health JSON; `{"boxes":[]}` / `{"objects":[]}` on a gray image (empty arrays + HTTP 200 proves the path works; `track=1` boxes carry a 6th element = track id).

- [ ] **Step 4: Confirm GPU engaged (third terminal, while looping curl)**

```bash
tegrastats --interval 500   # GR3D_FREQ must spike > 0 % during inference
```

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add scripts/yolo_server.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: yolo_server — GPU sidecar (yolo26m follow + track, yoloe-26x-seg dynamic scan + mask walk-pixel)"
```

---

## Task 16: `yolo-server.service` systemd unit

**Files:**
- Create: `systemd/yolo-server.service`

- [ ] **Step 1: Write the unit** (mirrors `systemd/llama-server.service`)

Create `systemd/yolo-server.service`:

```ini
# systemd/yolo-server.service
# Install:
#   sudo cp systemd/yolo-server.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now yolo-server.service
[Unit]
Description=GPU YOLO inference sidecar for Spot (Stage 2C)
After=network.target

[Service]
Type=simple
User=spotdog
WorkingDirectory=/mnt/ssd/yolo-models
# Jetson torch wheel has no RPATH to its bundled libcudss.so.0 — MUST set this,
# or torch.cuda silently fails and inference falls back to CPU.
Environment=LD_LIBRARY_PATH=/mnt/ssd/yolo-env/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.6/lib64
ExecStart=/mnt/ssd/yolo-env/bin/python /home/spotdog/spot/dartmouth_spot_capstone/scripts/yolo_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Install + enable**

```bash
sudo cp /home/spotdog/spot/dartmouth_spot_capstone/systemd/yolo-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-server.service
```

- [ ] **Step 3: Verify up + healthy + on GPU**

```bash
systemctl status yolo-server.service --no-pager | head -10
sleep 10 && curl -s localhost:11436/health     # allow model load + CLIP fetch on first start
```
Expected: `active (running)`; health JSON. If it fails: `journalctl -u yolo-server -n 40` — usual culprits: missing `LD_LIBRARY_PATH` (→ `libcudss.so.0` error), weights/CLIP still downloading, or port already bound. Confirm GPU (not CPU fallback) with `tegrastats` while curling an endpoint — `GR3D_FREQ` must be > 0 %.

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add systemd/yolo-server.service
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: yolo-server systemd unit (LD_LIBRARY_PATH baked in)"
```

---

## Task 17: `yolo_client.py` + swap visual_nav to the service

**Files:**
- Create: `src/voice_control/perception/yolo_client.py`
- Test: `tests/perception/test_yolo_client.py`
- Modify: `src/voice_control/visual_nav.py` (loaders 103-126; `_find_object`/`_find_person` 175-197; `detect_in_image` 325-333; the WalkToObjectInImage pixel at ~653; follow-loop `model.predict` sites)

- [ ] **Step 1: Write the failing client test**

Create `tests/perception/test_yolo_client.py`:

```python
import numpy as np
from PIL import Image
from src.voice_control.perception import yolo_client


def test_find_object_parses_server_json(monkeypatch):
    class _Resp:
        def json(self):
            return {"objects": [{"xyxy": [10, 20, 30, 40], "conf": 0.9,
                                 "name": "chair", "walk_xy": [20, 40]}]}
        def raise_for_status(self): pass
    monkeypatch.setattr(yolo_client.requests, "post", lambda *a, **k: _Resp())
    det = yolo_client.find_object(Image.new("RGB", (64, 64)), ["chair"])
    assert len(det.boxes) == 1
    assert list(det.boxes.xyxy[0]) == [10, 20, 30, 40]
    assert det.boxes.names[0] == "chair"
    assert det.boxes.walk[0] == [20, 40]


def test_service_down_is_failsafe(monkeypatch):
    def _boom(*a, **k): raise yolo_client.requests.ConnectionError()
    monkeypatch.setattr(yolo_client.requests, "post", _boom)
    assert len(yolo_client.detect_person(Image.new("RGB", (64, 64))).boxes) == 0  # empty, not crash
```

- [ ] **Step 2: Run it (fails — module missing)**

Run: `cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -m pytest tests/perception/test_yolo_client.py -v`
Expected: FAIL (`ModuleNotFoundError: ...yolo_client`).

- [ ] **Step 3: Write the client**

Create `src/voice_control/perception/yolo_client.py`:

```python
"""Thin HTTP client to the yolo-server GPU sidecar (Stage 2C).

Keeps all torch/ultralytics out of spot-env. Returns a minimal
ultralytics-`Results`-compatible object so existing visual_nav consumers
(`.boxes.xyxy` / `.conf`) work with small edits. Service down/timeout ->
empty detections (fail-safe: the follow loop must degrade, never crash).
"""
import io
import logging
import numpy as np
import requests

_log = logging.getLogger(__name__)
_URL = "http://127.0.0.1:11436"
_TIMEOUT = 2.0


class _Boxes:
    def __init__(self, xyxy, conf, ids=None, names=None, walk=None):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.id = np.asarray(ids) if ids is not None else None
        self.names = names or []          # class-name strings (find_object only)
        self.walk = walk or []            # [x,y] ground-contact pixel per det (find_object only)
    def __len__(self):
        return len(self.conf)


class Detections:
    def __init__(self, boxes):
        self.boxes = boxes


def _to_jpeg(pil):
    buf = io.BytesIO(); pil.convert("RGB").save(buf, format="JPEG"); return buf.getvalue()


def detect_person(pil, track=False):
    """track=True ONLY from the follow loop's single primary camera —
    server-side ByteTrack state is one stream (see yolo_server contract)."""
    try:
        r = requests.post(f"{_URL}/detect_person", data=_to_jpeg(pil),
                          params={"track": "1"} if track else {}, timeout=_TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("boxes", [])
    except (requests.RequestException, ValueError) as e:
        _log.warning("detect_person failed (%s); no detections", e); rows = []
    ids = [int(row[5]) for row in rows] if rows and len(rows[0]) > 5 else None
    return Detections(_Boxes([row[:4] for row in rows], [row[4] for row in rows], ids=ids))


def find_object(pil, classes):
    try:
        r = requests.post(f"{_URL}/find_object", data=_to_jpeg(pil),
                          params={"classes": ",".join(classes)}, timeout=_TIMEOUT)
        r.raise_for_status()
        objs = r.json().get("objects", [])
    except (requests.RequestException, ValueError) as e:
        _log.warning("find_object failed (%s); no detections", e); objs = []
    return Detections(_Boxes([o["xyxy"] for o in objs], [o["conf"] for o in objs],
                             names=[o["name"] for o in objs], walk=[o["walk_xy"] for o in objs]))
```

- [ ] **Step 4: Run the test (passes)**

Run: `cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -m pytest tests/perception/test_yolo_client.py -v`
Expected: PASS (both).

- [ ] **Step 5: Swap visual_nav loaders + detection helpers**

In `src/voice_control/visual_nav.py`:

(a) Delete `_get_world_model`/`_get_person_model`/`_world_model`/`_person_model` (103-126) and gut `preload_models` (perception now lives in the service — make it a no-op or drop it). Add at top: `from src.voice_control.perception import yolo_client`.

(b) Replace the detection helpers:
```python
def _find_object(img, description):
    """Detect described object via yolo-server. Returns [x1,y1,x2,y2] or None."""
    det = yolo_client.find_object(img, [description])
    if len(det.boxes) == 0:
        return None
    return det.boxes.xyxy[det.boxes.conf.argmax()]

def _find_person(img):
    """Detect nearest person via yolo-server. Returns [x1,y1,x2,y2] or None."""
    det = yolo_client.detect_person(img)
    if len(det.boxes) == 0:
        return None
    xy = det.boxes.xyxy
    areas = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    return xy[areas.argmax()]
```

(c) In `detect_in_image` (~325-333) replace the in-process model block with:
```python
        det = yolo_client.find_object(img, [description])
        if len(det.boxes) > 0:
            best = det.boxes.conf.argmax()
            bbox = det.boxes.xyxy[best]
            conf = float(det.boxes.conf[best])
            # ... keep the existing w/h + position logic below unchanged ...
```

(d) **WalkToObjectInImage pixel (~653):** instead of bbox-center, use the server's mask-derived `walk_xy`. Capture it from the scan that found the target (return it alongside the bbox from `_find_object_batch`/`navigate_to_object`), then:
```python
    # walk_xy is the mask ground-contact point in ROTATED image coords (same frame as bbox).
    cx_rot, cy_rot = walk_xy if walk_xy is not None else ((bbox[0]+bbox[2])/2.0, bbox[3])
    px_native, py_native = _unrotate_pixel(cx_rot, cy_rot, rot_w, rot_h)
```

(e) Follow loop: replace each `model = _get_person_model()` + `model.predict(img, device=DEVICE, classes=[0], ...)` + `results[0].boxes…` site with `_find_person(img)` (returns best person bbox or None). To enable the ByteTrack lock, the follow loop's primary front-camera call should use `yolo_client.detect_person(img, track=True)` and key on `det.boxes.id`. Remove the now-dead `DEVICE`/`.cpu()` reads.

- [ ] **Step 6: Smoke import + a live call (yolo-server must be running)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -c "
from PIL import Image
from src.voice_control.visual_nav import _find_person, _find_object
img = Image.new('RGB', (640, 480), 'gray')
print('person:', _find_person(img))            # None on gray (client+service wired)
print('chair :', _find_object(img, 'chair'))
"
```
Expected: `person: None` / `chair: None`, no traceback (service reachable, returns empties on gray).

- [ ] **Step 7: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add \
  src/voice_control/perception/yolo_client.py tests/perception/test_yolo_client.py src/voice_control/visual_nav.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: visual_nav uses yolo-server HTTP client (drops in-process ultralytics; track + mask walk-pixel)"
```

---

## Task 18: Publish detections as world objects + integration verification + checkpoint #4

**Files:**
- Modify: `src/voice_control/world_objects.py` (add `publish_detection`)
- Read only: live system; diff of Tasks 12-17

- [ ] **Step 1: Add `publish_detection` to world_objects.py**

Lands YOLO detections in Spot's native frame graph (queryable via `list_world_objects`, auto-expiring) — aligns the scan path with the SDK world model. API verified: `WorldObjectClient.mutate_world_objects(req)`, `ImageProperties.coordinates` is a `bosdyn.api.Polygon`.

```python
import time
from bosdyn.api import world_object_pb2, geometry_pb2
from bosdyn.client.world_object import WorldObjectClient

def publish_detection(robot, name, image_response, bbox_native, lifetime_s=10):
    """Add a YOLO detection to Spot's world-object service as IMAGE_COORDINATES.
    bbox_native = (x1,y1,x2,y2) in the NATIVE (unrotated) sensor frame.

    Best-effort telemetry — NEVER block perception on it. Verify on-robot that
    the firmware accepts external ACTION_ADD of this type; if rejected, this
    logs and returns None (perception unaffected)."""
    x1, y1, x2, y2 = bbox_native
    # NB: the proto field is `vertexes` (sic) — `vertices=` raises ValueError.
    poly = geometry_pb2.Polygon(vertexes=[
        geometry_pb2.Vec2(x=x1, y=y1), geometry_pb2.Vec2(x=x2, y=y1),
        geometry_pb2.Vec2(x=x2, y=y2), geometry_pb2.Vec2(x=x1, y=y2)])
    img_props = world_object_pb2.ImageProperties(
        camera_source=image_response.source.name,
        coordinates=poly,
        frame_name_image_coordinates=image_response.shot.frame_name_image_sensor)
    wo = world_object_pb2.WorldObject(
        id=0,                          # 0 → server assigns an id
        name=f"yolo_{name}",
        acquisition_time=robot.time_sync.robot_timestamp_from_local_secs(time.time()),
        image_properties=img_props)
    wo.object_lifetime.seconds = lifetime_s
    req = world_object_pb2.MutateWorldObjectRequest(
        mutation=world_object_pb2.MutateWorldObjectRequest.Mutation(
            action=world_object_pb2.MutateWorldObjectRequest.ACTION_ADD, object=wo))
    try:
        client = robot.ensure_client(WorldObjectClient.default_service_name)
        return client.mutate_world_objects(req).mutated_object_id
    except Exception as e:
        print(f"[world_objects] publish_detection skipped ({e})")
        return None
```

- [ ] **Step 2: Wire it into the scan path**

`bbox_native` does NOT exist in current code — `visual_nav.py` unrotates only a single pixel via `_unrotate_pixel` (~line 656). Compute it here: unrotate all four corners of the hit's rotated-frame `bbox` with `_unrotate_pixel(x, y, rot_w, rot_h)` (rot_w/rot_h from the hit's `rotated_size`), then take min/max per axis (rotation can swap corner order). Then in `visual_nav.navigate_to_object`, once a target is found, call `publish_detection(robot, description, image_response, bbox_native)` using the hit's `image_response` from `_scan_all_cameras`. Best-effort — never gate navigation on its return.

- [ ] **Step 3: Verify publishing on-robot**

After a `go to <obj>` run, list world objects and confirm ours appears:
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -c "
# (use the repo's normal robot handle / session bootstrap)
from bosdyn.client.world_object import WorldObjectClient
c = robot.ensure_client(WorldObjectClient.default_service_name)
print([o.name for o in c.list_world_objects().world_objects])
"
```
Expected: a `yolo_<obj>` entry. **If `mutate_world_objects` raises permission/unsupported** → external ADD of IMAGE_COORDINATES is blocked on this firmware; leave `publish_detection` as the logged no-op (perception unaffected) and note it in the rollback doc. *(This is the on-robot check flagged during planning.)*

- [ ] **Step 4: End-to-end latency through the service**

With `yolo-server` running + Spot powered (or a saved frame):
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -c "
import time, pathlib
from PIL import Image
from src.voice_control.perception import yolo_client
p = pathlib.Path('tests/captures/pose0_front_left.jpg')
img = Image.open(p) if p.exists() else Image.new('RGB',(640,480),'gray')
for _ in range(3): yolo_client.detect_person(img)
t=time.time()
for _ in range(20): yolo_client.detect_person(img)
print(f'detect_person end-to-end (HTTP+GPU): {(time.time()-t)/20*1000:.1f} ms')
"
```
Expected: well under 100 ms (10 Hz budget); ~35-60 ms typical (~32 ms GPU + jpeg + localhost roundtrip).

- [ ] **Step 5: Confirm GPU + fail-safe**

```bash
# GPU: run `tegrastats --interval 500` in another terminal during Step 4 → GR3D_FREQ > 0 %.
# Fail-safe (service down must NOT crash the loop):
sudo systemctl stop yolo-server.service
cd /home/spotdog/spot/dartmouth_spot_capstone && ./spot-env/bin/python -c "
from PIL import Image
from src.voice_control.perception import yolo_client
print('down ->', len(yolo_client.detect_person(Image.new('RGB',(640,480))).boxes), 'boxes')"
sudo systemctl start yolo-server.service
```
Expected: `down -> 0 boxes`, no traceback (follow must stop cleanly on empty, never drive blind).

- [ ] **Step 6: Dispatch safety-reviewer on look_around + service swap**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C look_around + yolo-server swap safety",
  prompt: "Audit src/voice_control/spot_dispatch.py (look_around) and src/voice_control/visual_nav.py (yolo-server HTTP client swap + ByteTrack follow) in the recent commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. Focus on:\n1. look_around: does it block the dispatch thread for 15-30s (5 sequential VLM calls)? Can the user still issue stop during it? E-stop coverage?\n2. Follow loop now depends on a network service. If yolo-server HANGS (slow, not down), does the 2.0s client timeout stall the follow loop past safe limits? Does the loop still cap velocity at 10 Hz and stop cleanly when detect_person returns empty (person 'lost')?\n3. Fail-safe: when the service is down/empty, does follow decline gracefully (stop) rather than drive blind?\nReport severity-tagged findings only."
})
```

- [ ] **Step 7: Dispatch caveman:cavecrew-reviewer on the diff**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 4 diff",
  prompt: "Review the recent Stage 2C commits (tasks 12-17 + world-object publish) on the current branch at /home/spotdog/spot/dartmouth_spot_capstone: look_around, spot_action.gbnf, yolo-env, yolo_server, systemd unit, yolo_client + visual_nav swap, publish_detection. Verify:\n1. visual_nav no longer imports ultralytics/torch anywhere.\n2. yolo_client returns empty (not None/crash) on BOTH timeout and connection error.\n3. Every former results[0].boxes consumer now reads the numpy Detections shape (no leftover .cpu()).\n4. publish_detection is best-effort (try/except, never blocks navigation).\n5. The systemd unit LD_LIBRARY_PATH matches the validated path.\nReport severity-tagged findings only."
})
```

- [ ] **Step 8: Fix BLOCKER/HIGH; commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: publish detections as world objects + checkpoint 4 fixes"
```

---

## Task 19: Update stage2-rollback.md with vision-rollback layer

**Files:**
- Modify: `docs/project/stage2-rollback.md`

- [ ] **Step 1: Append the vision-rollback section**

Append:

```markdown
## After 2C — vision

### Layer: yolo-server (GPU sidecar) down or misbehaving

`visual_nav` is fail-safe: if `yolo-server` (127.0.0.1:11436) is unreachable or times out, `yolo_client` returns empty detections and the follow loop stops cleanly (never drives blind). Recover the service:
```bash
sudo systemctl restart yolo-server.service
journalctl -u yolo-server -n 40         # won't start: usually LD_LIBRARY_PATH or weights still downloading
```
If the yolo-env itself is broken (e.g. a system CUDA/JetPack bump invalidates the torch wheel), rebuild it:
```bash
rm -rf /mnt/ssd/yolo-env && bash scripts/setup_yolo_env.sh
```
To run perception-less on purpose, just stop the service — the loop degrades to "no detections", not a crash.

### Layer: revert to single-camera serial scan (batched scan regresses something)

```bash
# In src/voice_control/visual_nav.py, revert _scan_all_cameras to call _find_object
# in a loop. Single commit reverted via:
git revert <stage-2c-task-10-sha>
```

### Layer: revert per-camera rotation (rotation table is wrong)

```bash
# Comment out CAMERA_ROTATION_DEG (defaults all to 0) — frames stay sideways
# but reach downstream code without crashing.
```

### Hard rollback (full 2C revert)

```bash
git checkout main
git reset --hard <pre-2C main sha>   # the stage2b-e2e-complete tag never existed; use the commit main pointed to before 2C merged
git push --force-with-lease origin main   # ASK USER FIRST
```
```

- [ ] **Step 2: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add docs/project/stage2-rollback.md
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: rollback doc — vision layers (yolo-server, scan, rotation)"
```

---

## Task 20: Final review — caveman + feature-dev + safety on full 2C diff

**Files:**
- Read only: `git diff main..HEAD` (2C branched off main; the `stage2b-e2e-complete` tag does not exist)

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer on full 2C diff**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "Full 2C final review",
  prompt: "Review every change in `git diff main..HEAD` on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. This is the complete Stage 2C vision overhaul: per-camera rotation, distortion probe (undistort deferred), gripper plumbing, batched 5-camera scan, look_around action, GPU yolo-server sidecar (yolo26m follow + ByteTrack, yoloe-26x-seg dynamic scan), yolo_client swap, world-object publishing. Report severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch feature-dev:code-reviewer for quality audit**

```
Agent({
  subagent_type: "feature-dev:code-reviewer",
  description: "2C quality audit",
  prompt: "Code-review the full Stage 2C diff (`git diff main..HEAD` on the current branch at /home/spotdog/spot/dartmouth_spot_capstone). Focus on high-confidence issues: bugs, logic errors, security holes, code-quality issues, and adherence to project conventions (see CLAUDE.md / MEMORY.md / docs/superpowers/specs/2026-05-15-stage2-design.md). Report only issues you are 80%+ confident on. Files touched: src/voice_control/perception/*.py (incl. yolo_client.py), src/voice_control/spot_dispatch.py, src/voice_control/visual_nav.py, src/voice_control/world_objects.py, src/voice_control/grammar/spot_action.gbnf, scripts/setup_yolo_env.sh, scripts/yolo_server.py, systemd/yolo-server.service, requirements-yolo.txt, .gitignore, docs/project/stage2-rollback.md."
})
```

- [ ] **Step 3: Dispatch safety-reviewer for final motion-path audit**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C final safety pass",
  prompt: "Final safety audit of the complete Stage 2C diff (`git diff main..HEAD` on the current branch at /home/spotdog/spot/dartmouth_spot_capstone). Confirm:\n1. E-stop coverage unchanged across all new code paths.\n2. look_around action cannot wedge the dispatcher in a multi-VLM call that prevents the next utterance's stop command from running.\n3. The follow loop now depends on the yolo-server network service: a slow/hung service (within the 2.0s client timeout) must not stall the loop past safe limits, and empty/failed detections must stop the robot (not drive blind). GPU speedup must not bypass the 10 Hz velocity rate limiter.\nReport BLOCKER/HIGH severity findings; this is the merge gate."
})
```

- [ ] **Step 4: Address BLOCKER/HIGH from all 3 agents; commit fixes**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address final review findings"
```

---

## Task 21: Live regression — manual smoke checklist on Spot

> **Decision (2026-07-23, user):** `scripts/run_regression_matrix.py` + `tests/regression/stage2b_matrix.yaml` were never built (2B never ran). Task downgraded to a manual on-robot smoke checklist; building a real harness stays a separate future project.

**Files:**
- Create: `docs/project/stage2c-smoke-results.md` (filled-in checklist)

- [ ] **Step 1: Run the smoke checklist on Spot**

Operator at the robot, E-stop in another terminal, voice loop running. Record PASS/FAIL + notes per item in `docs/project/stage2c-smoke-results.md`:

1. **Wake + basic motion** — wake word, "stand", "sit". Voice loop unaffected by vision changes.
2. **Locomotion** — "walk forward", "turn left". Motion path untouched by 2C.
3. **GraphNav** — "go to <saved location>". Fiducial init + nav unchanged.
4. **describe** — "what do you see". capture_frame→Frame refactor + VLM path works, frames upright.
5. **go_to_object** — "go to the chair". Full new path: batched GPU scan → walk_xy → unrotate → WalkToObjectInImage; world object published (or logged no-op).
6. **follow_me** — follow across a room. yolo26m + ByteTrack via sidecar; clean stop when person lost AND when yolo-server is stopped mid-follow (fail-safe check).
7. **look_around** — "look around". Per-camera VLM sweep speaks one summary; "stop" still handled afterward.

- [ ] **Step 2: Gate on results**

All 7 PASS → proceed to Task 22. Any FAIL: do NOT merge — investigate, fix, re-test the failed item.

- [ ] **Step 3: Commit the results**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add docs/project/stage2c-smoke-results.md
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: on-robot smoke checklist results — vision overhaul verified live"
```

---

## Task 22: Push, PR, merge to main, tag stage2c-vision-complete

> **Confirm with user before pushing to remote or creating PR.**

- [ ] **Step 1: Push branch**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone push origin stage2c-vision-overhaul
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "Stage 2C: vision overhaul (rotation, gripper, batched scan, look_around, GPU yolo-server)" --body "$(cat <<'EOF'
## Summary
- Per-camera rotation table (`CAMERA_ROTATION_DEG`) — front cameras no longer hardcoded to 90° CW
- Undistortion DEFERRED with evidence: distortion probe confirms BD populates no coefficients; warp-assessment frames captured (calibration spike only if ever warranted)
- Frame namedtuple — single perception data type across capture and inference
- Gripper color camera (`hand_color_image`) plumbing — RGB_U8 decode + pre-flight gate
- Batched 5-camera scan — one detector call instead of 5 sequential
- `look_around` action — batched VLM caption across all cameras
- GPU YOLO via isolated `yolo-server` sidecar — `yolo26m` follow (+ByteTrack track), `yoloe-26x-seg` dynamic open-vocab scan; `visual_nav` is a thin HTTP client
- Mask-derived WalkToObjectInImage pixel; detections published to Spot's world-object service
- spot-env untouched (no ultralytics/torch for inference; hard pins intact)
- Vision rollback layer documented

## Test plan
- [x] Per-camera rotation visually verified on all 5 fisheye sources
- [x] Distortion probe run + warp verdict recorded (undistort deferred)
- [x] Gripper camera live test passed (color frame saved)
- [x] Batched scan smoke-test (chair detection) — single batched call
- [x] look_around action returns per-camera VLM captions
- [x] yolo-server end-to-end latency < 100 ms (10 Hz follow budget); GPU confirmed via tegrastats
- [x] Fail-safe verified: yolo-server down → empty detections, follow stops cleanly (no crash)
- [x] World-object publish verified on-robot (or logged no-op if firmware blocks external ADD)
- [x] Manual smoke checklist (7 items, docs/project/stage2c-smoke-results.md) PASS on Spot

## Rollback
See docs/project/stage2-rollback.md — layered rollback (yolo-server down/rebuild, scan revert, rotation revert, hard reset).
EOF
)"
```

- [ ] **Step 3: Self-merge (if single-contributor convention) or wait for review**

```bash
gh pr merge --merge
```

- [ ] **Step 4: Tag on main**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone checkout main
git -C /home/spotdog/spot/dartmouth_spot_capstone pull origin main
git -C /home/spotdog/spot/dartmouth_spot_capstone tag -a stage2c-vision-complete -m "stage 2c: vision overhaul merged; GPU yolo-server (yolo26m + yoloe-26x-seg) live"
git -C /home/spotdog/spot/dartmouth_spot_capstone push origin stage2c-vision-complete
```

- [ ] **Step 5: Verify tag on remote**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone ls-remote --tags origin | grep stage2c-vision-complete
```

---

## Self-review

**Spec coverage:** Every sub-section from spec section 2C:
- ✅ 2C.0 per-camera rotation + Frame namedtuple → Tasks 1, 2, 3
- ✅ 2C.0.5 fisheye undistortion → DEFERRED by decision 2026-07-23 (Task 5 = coeff re-check + warp evidence; Task 6 removed — BD provides no distortion coefficients, self-calibration not warranted yet)
- ✅ 2C.1 gripper color camera → Task 8
- ✅ 2C.2 batched 5-camera scan → Tasks 9, 10
- ✅ 2C.3 look_around action → Tasks 12, 13
- ✅ 2C.4 GPU YOLO sidecar (yolo26m follow + yoloe-26x-seg dynamic scan) → Tasks 14, 15, 16, 17, 18

**Placeholder scan:** No TBD / TODO / "implement later" anywhere. The `{LOCATION}` placeholder in Stage 2B YAML is intentional (operator substitution at runtime). The `<pre-2C main sha>` in Task 19's hard-rollback is an intentional runtime value (implementer fills in main's pre-2C commit).

**Type consistency:** `Frame(pil, jpeg_bytes, source)` defined in Task 1 is used as `frame.pil` / `frame.jpeg_bytes` / `frame.source` throughout Tasks 3, 8, 12. `_scan_all_cameras` hit dicts (`bbox`/`conf`/`walk_xy`/`rotated_size`/`image_response`, Task 9) are consumed by Task 10's selection + Task 17d's WalkTo pixel + Task 18's publish with matching keys. `CAMERA_ROTATION_DEG` keyed by full SDK source name (e.g. `"frontleft_fisheye_image"`); `CAMERA_SOURCES` keyed by nickname → SDK name; `_rotate_for_source(image, source_name)` takes the SDK name — internally consistent across Tasks 2, 3, 8.

**Reviewer dispatch points (per "build in periodic review passes" mandate):**
- Task 4: caveman:cavecrew-reviewer (post Tasks 1-3) — Frame + rotation + capture rewrite
- Task 7: safety-reviewer + caveman:cavecrew-reviewer (post Tasks 3+5) — capture pipeline + distortion probe
- Task 11: caveman:cavecrew-reviewer (post Tasks 8-10) — gripper + batched scan
- Task 18: safety-reviewer + caveman:cavecrew-reviewer (post Tasks 12-17) — look_around + yolo-server swap + world-object publish
- Task 20: caveman + feature-dev:code-reviewer + safety-reviewer (final pass, full diff)

**Ambiguity check:**
- "median IoU >= 0.9 AND class match >= 99 %" is explicit numeric gate (Task 17).
- "GPU utilization spike" = `tegrastats` `GR3D_FREQ > 0 %` during inference (Task 17 Step 4).
- "spot-env pins intact" = nothing is installed into spot-env for YOLO; the entire GPU stack lives in the separate `/mnt/ssd/yolo-env` (Task 14), so numpy/onnxruntime-gpu/opencv-python/kokoro-onnx cannot shift.

**Risks not yet mitigated in this plan (deferred):**
- yolo-env install can break if the jetson-ai-lab jp6/cu126 index drifts (it has thinned before); `setup_yolo_env.sh` pins exact versions, and a system CUDA/JetPack bump may force `rm -rf /mnt/ssd/yolo-env && bash scripts/setup_yolo_env.sh`.
- Fisheye edge warp on raw pinhole frames may hurt edge-of-frame detections/walk-to accuracy — Task 5 grades severity; checkerboard self-calibration spike is the escape hatch if it ever matters.
- AprilTag follow as exploratory backup is OUT OF SCOPE for Stage 2C (per spec out-of-scope list); Stage 3 spike if yolo26m proves insufficient on Spot.
- Cross-camera follow fusion + appearance re-ID are deferred (see Progression, top of plan) — 2C ships within-camera `track()` only.
