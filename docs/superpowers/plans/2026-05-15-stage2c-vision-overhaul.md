# Stage 2C — Vision Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Spot perception to the Phase 9.B revised baseline: per-camera rotation table, fisheye undistortion via cv2, gripper color camera plumbing, batched 5-camera scan, `look_around` action, TensorRT FP16 export of YOLO26-N (follow) + YOLOE26-S (object scan). Exit tag: `stage2c-vision-complete`.

**Architecture:**
1. Introduce a `Frame` namedtuple as the single perception data type carrying `pil`, `jpeg_bytes`, `source`. All capture helpers return `Frame`.
2. Bake rotation per camera (not a global 90° CW). Bake undistortion using BD's distortion params via `cv2.fisheye.undistortImage` (Kannala-Brandt) or `cv2.undistort` (Brown-Conrady) with `cv2.initUndistortRectifyMap` pre-computed at startup.
3. Replace the serial 5-camera object scan with a single batched `_capture_multi` + batched detector call.
4. Add `look_around` action and wire to the action.gbnf grammar from Plan 2A.
5. Export YOLO26-N + YOLOE26-S to `.engine` FP16 once; cache to SSD; swap ultralytics loaders to engine paths. CPU pinholeline of action stays (LLM VRAM contention) but inference is GPU.

**Tech Stack:** cv2 (opencv-python==4.11.0.86), bosdyn-client 5.0.1.1, ultralytics (bump for YOLO26+YOLOE26+engine), TensorRT 10.3.0.30 (already on JetPack 6.2.1), onnxruntime-gpu 1.23.0 (TRT EP exposed but used only as fallback), PIL.

**Spec reference:** `docs/superpowers/specs/2026-05-15-stage2-design.md` sections 2C.0 / 2C.0.5 / 2C.1 / 2C.2 / 2C.3 / 2C.4. Fisheye coordinate transforms reference: `fisheye-coords` skill.

**Hard ordering:** Plan 2C depends on Plan 2B merge to main (per spec section "Hard ordering constraints #7"). Branch off `main` at `stage2b-e2e-complete` tag.

---

## Pre-conditions

- `stage2b-e2e-complete` tag exists on `main`.
- Branch off main: `git checkout -b stage2c-vision-overhaul main` (or use a worktree per superpowers:using-git-worktrees).
- llama-server daemon running and responsive (vision touches no brain code but smoke tests use the voice loop).
- cv2 available: `python3 -c "import cv2; print(cv2.__version__)"` → `4.11.0.86`. Verify `cv2.fisheye.undistortImage` exists.
- Spot powered + on the floor for Tasks 4, 6, 10, 17, 18, 21 (any task touching live capture). E-stop terminal open.

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `src/voice_control/perception/frame.py` | Create (~40 LOC) | `Frame(NamedTuple)`: `pil`, `jpeg_bytes`, `source` |
| `src/voice_control/spot_dispatch.py` | Modify (~+120 LOC, ~-30 LOC) | `CAMERA_ROTATION_DEG` table, `_rotate_for_source`, `_undistort_for_source`, rewritten `capture_frame`, `look_around` action handler, `hand_color_image` in `CAMERA_SOURCES` |
| `src/voice_control/perception/camera_intrinsics.py` | Modify (~+80 LOC) | Extend `Intrinsics` dataclass with `D` (distortion coeffs), `distortion_model` ("pinhole" / "brown_conrady" / "kannala_brandt"); add `get_undistort_maps(robot, source_name)` precompute helper |
| `src/voice_control/perception/undistort.py` | Create (~120 LOC) | `_undistort_for_source(pil_image, source_name, robot)` + module-level cache of `(map1, map2)` per (robot, source) |
| `src/voice_control/visual_nav.py` | Modify (~+90 LOC, ~-25 LOC) | Replace serial scan loop with `_scan_all_cameras` + `_find_object_batch`; switch `_get_world_model` / `_get_person_model` to `.engine` paths |
| `src/voice_control/intent.py` | Modify (~+8 LOC) | Add `"look_around"` to intent enum if a static enum exists; otherwise N/A |
| `scripts/export_yolo_engines.py` | Create (~70 LOC) | One-shot ultralytics `.engine` export for YOLO26-N + YOLOE26-S; output to `/mnt/ssd/cfm_mppi_caches/yolo_engines/` |
| `scripts/verify_trt_numerics.py` | Create (~120 LOC) | Compares `.pt` vs `.engine` outputs on a held-out 20-30 frame set; reports mAP delta; FAIL if > 1 % |
| `requirements.txt` | Modify | Bump `ultralytics` to a YOLO26-supporting version (verify during Task 14) |
| `docs/project/stage2-rollback.md` | Modify | Append vision-rollback layer (engine swap, .pt fallback, undistort disable) |
| `tests/captures/` | Create dir | 20-30 sample Spot frames for numerics verification; one per camera × diverse scenes |
| `/mnt/ssd/cfm_mppi_caches/yolo_engines/` | Create dir | `.engine` cache (gitignored — engines are 30-100 MB and JetPack-bound) |
| `.gitignore` | Modify | Add `tests/captures/` if large; YOLO26 `.pt` weights covered by existing `yolov8*.pt` (need broader pattern) |

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
    """A single captured + rotated + undistorted camera frame.

    Attributes:
        pil: PIL.Image.Image, RGB, in the upright + undistorted view.
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

    NOTE: Undistortion is added in Task 6 (this task's pipeline is capture
    → decode → rotate → encode). The undistort step slots in between
    decode and rotate once Task 6 lands.
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

Likely call sites:
- `src/voice_control/llm_brain.py` (VLM path — uploads jpeg_bytes to llama.cpp `--mmproj` endpoint)
- `src/voice_control/spot_dispatch.py` (the `describe_*` action handlers)
- Possibly `scripts/test_visual_nav.py` (if it imports capture_frame)

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
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/spot_dispatch.py src/voice_control/llm_brain.py
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

## Task 5: Extend camera_intrinsics.py with distortion params caching

**Files:**
- Modify: `src/voice_control/perception/camera_intrinsics.py`

- [ ] **Step 1: Extend Intrinsics dataclass with distortion fields**

Modify the `Intrinsics` dataclass (current lines 35-57) to add distortion fields:

```python
@dataclass
class Intrinsics:
    """Pinhole + distortion parameters for one image source.

    Distortion model is one of:
        "pinhole"        — no distortion (D is empty array)
        "brown_conrady"  — cv2.undistort with (k1, k2, p1, p2, k3) coefficients
        "kannala_brandt" — cv2.fisheye.undistortImage with (k1, k2, k3, k4) coefficients
    """

    source_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    cols: int
    rows: int
    image_source_proto: image_pb2.ImageSource
    # NEW: distortion model + coefficients (numpy arrays for cv2 compat)
    distortion_model: str           # one of "pinhole" / "brown_conrady" / "kannala_brandt"
    D: "np.ndarray"                 # (5,) for brown_conrady, (4,) for kannala_brandt, (0,) for pinhole
```

Add `import numpy as np` at the top of the file if not present.

- [ ] **Step 2: Extend _ensure_loaded to populate distortion fields**

Replace the existing pinhole-only loop in `_ensure_loaded` (current lines ~72-124) to also extract distortion params. Pseudocode:

```python
import numpy as np

def _ensure_loaded(robot) -> None:
    key = _robot_key(robot)
    if key in _sources_cache:
        return
    sources = robot.ensure_client('image').list_image_sources()
    _sources_cache[key] = sources
    per_source: dict[str, Intrinsics] = {}
    for s in sources:
        # Default: empty distortion (pinhole)
        d_model = "pinhole"
        d_arr = np.zeros((0,), dtype=np.float64)
        # Branch on which distortion oneof the proto has.
        if s.HasField("pinhole_brown_conrady"):
            d_model = "brown_conrady"
            bc = s.pinhole_brown_conrady
            # BD's proto: k1, k2, p1, p2, k3 (Brown-Conrady standard order)
            d_arr = np.array([bc.k1, bc.k2, bc.p1, bc.p2, bc.k3], dtype=np.float64)
            fx = bc.intrinsics.focal_length.x
            fy = bc.intrinsics.focal_length.y
            cx = bc.intrinsics.principal_point.x
            cy = bc.intrinsics.principal_point.y
        elif s.HasField("kannala_brandt"):
            d_model = "kannala_brandt"
            kb = s.kannala_brandt
            d_arr = np.array([kb.k1, kb.k2, kb.k3, kb.k4], dtype=np.float64)
            fx = kb.intrinsics.focal_length.x
            fy = kb.intrinsics.focal_length.y
            cx = kb.intrinsics.principal_point.x
            cy = kb.intrinsics.principal_point.y
        elif s.HasField("pinhole"):
            ph = s.pinhole
            fx = ph.focal_length.x
            fy = ph.focal_length.y
            cx = ph.principal_point.x
            cy = ph.principal_point.y
        else:
            logger.debug("source %s has no recognized distortion model; skipping", s.name)
            continue

        per_source[s.name] = Intrinsics(
            source_name=s.name,
            fx=fx, fy=fy, cx=cx, cy=cy,
            cols=s.cols, rows=s.rows,
            image_source_proto=s,
            distortion_model=d_model,
            D=d_arr,
        )
    _intrinsics_cache[key] = per_source
```

**Caveat:** BD SDK proto field names for distortion oneofs (`pinhole_brown_conrady`, `kannala_brandt`) and their sub-fields (`k1`, `k2`, `p1`, `p2`, `k3`, intrinsics) MUST be verified against the actual SDK. Run `scripts/probe_image_sources.py` (existing diagnostic per recon) to print the actual proto schema. Adjust field accesses if names differ.

- [ ] **Step 3: Verify via probe script**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/probe_image_sources.py
```
Expected: prints each source name + which distortion oneof is set + the values. Use this output to confirm Step 2's field accesses are correct.

- [ ] **Step 4: Unit-test the extended intrinsics on real robot**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from src.session import quick_robot
from src.voice_control.perception.camera_intrinsics import get_intrinsics, _ensure_loaded
robot = quick_robot()
_ensure_loaded(robot)
for src in ['frontleft_fisheye_image', 'left_fisheye_image', 'back_fisheye_image']:
    i = get_intrinsics(robot, src)
    if i is None:
        print(f'{src}: NONE'); continue
    print(f'{src}: model={i.distortion_model} D={i.D.tolist()} fx={i.fx:.1f} fy={i.fy:.1f}')
"
```
Expected: every source returns a non-None Intrinsics with a `distortion_model` and a non-empty `D` for non-pinhole sources. Fisheye cameras typically show `kannala_brandt`; gripper color is usually `brown_conrady`. Adjust Step 2 if any source returns None unexpectedly.

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/perception/camera_intrinsics.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: camera_intrinsics caches distortion model + coeffs per source"
```

---

## Task 6: Add _undistort_for_source helper + precompute remap tables

**Files:**
- Create: `src/voice_control/perception/undistort.py`
- Modify: `src/voice_control/spot_dispatch.py` (integrate into capture_frame)

- [ ] **Step 1: Write the undistort module**

Create `src/voice_control/perception/undistort.py`:

```python
"""Undistortion of Spot fisheye + Brown-Conrady frames via cv2 remap.

Precomputes per-source remap tables at first use; per-frame cost is a
single cv2.remap (~3-5ms CPU on AGX Orin per 640x480 frame).

Distortion models supported:
  - "pinhole"         : no-op, returns input
  - "brown_conrady"   : cv2.initUndistortRectifyMap + cv2.remap
  - "kannala_brandt"  : cv2.fisheye.initUndistortRectifyMap + cv2.remap
"""

import cv2
import numpy as np
from PIL import Image

from .camera_intrinsics import get_intrinsics, _robot_key


# Cache: (robot_key, source_name) -> (map1, map2)
_remap_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _build_maps(intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Build (map1, map2) for cv2.remap. CV_16SC2 fixed-point for speed."""
    K = np.array(
        [[intrinsics.fx, 0, intrinsics.cx],
         [0, intrinsics.fy, intrinsics.cy],
         [0, 0, 1]],
        dtype=np.float64,
    )
    D = intrinsics.D
    size = (intrinsics.cols, intrinsics.rows)  # cv2 wants (width, height)
    if intrinsics.distortion_model == "kannala_brandt":
        # Fisheye: use cv2.fisheye namespace; identity rotation.
        # alpha behavior: preserve full FOV by passing K as the new camera matrix
        # (some pixels go out of the image; alternative cv2.getOptimalNewCameraMatrix
        # with alpha=1 would also work but adds complexity).
        D_kb = D.reshape(4, 1)  # cv2.fisheye expects (4,1) or (1,4)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D_kb, np.eye(3), K, size, cv2.CV_16SC2
        )
    elif intrinsics.distortion_model == "brown_conrady":
        # Standard pinhole undistort.
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, np.eye(3), K, size, cv2.CV_16SC2
        )
    else:
        # Pinhole — identity maps. Caller is expected to skip the remap,
        # but build trivial maps for uniformity if forced.
        xs, ys = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        map1 = np.stack([xs, ys], axis=-1).astype(np.int16)
        map2 = np.zeros(size[::-1], dtype=np.uint16)
    return map1, map2


def _get_or_build_maps(robot, source_name: str):
    """Return cached (map1, map2) or build + cache."""
    key = (_robot_key(robot), source_name)
    if key in _remap_cache:
        return _remap_cache[key]
    intr = get_intrinsics(robot, source_name)
    if intr is None:
        return None
    maps = _build_maps(intr)
    _remap_cache[key] = maps
    return maps


def undistort_for_source(pil_image: Image.Image, source_name: str, robot) -> Image.Image:
    """Undistort a PIL image for the given source.

    Returns a new PIL.Image (same size). For pinhole sources or unknown
    sources, returns the input unchanged (no-op).
    """
    intr = get_intrinsics(robot, source_name)
    if intr is None or intr.distortion_model == "pinhole":
        return pil_image
    maps = _get_or_build_maps(robot, source_name)
    if maps is None:
        return pil_image
    map1, map2 = maps
    np_img = np.array(pil_image)  # H, W, 3 for RGB
    if np_img.ndim == 2:
        np_img = np_img[..., None]  # 1-channel fisheye safety
    remapped = cv2.remap(np_img, map1, map2, cv2.INTER_LINEAR)
    if remapped.ndim == 3 and remapped.shape[2] == 1:
        remapped = remapped[..., 0]
    return Image.fromarray(remapped)
```

- [ ] **Step 2: Integrate into capture_frame**

In `src/voice_control/spot_dispatch.py`, in the `capture_frame` rewritten in Task 3, insert the undistort step between decode and rotate. **Order matters: undistort operates on the NATIVE sensor frame, before rotation.**

```python
# Before rotation: undistort using the BD intrinsics.
from src.voice_control.perception.undistort import undistort_for_source
pil = undistort_for_source(pil, source, robot)
# Then rotate as before.
pil = _rotate_for_source(pil, source)
```

The capture pipeline becomes: capture → decode → **undistort** → rotate → encode.

- [ ] **Step 3: Smoke-test undistort offline (no Spot needed)**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from src.voice_control.perception.undistort import undistort_for_source
# Mock intrinsics: pinhole returns input unchanged
img = Image.new('RGB', (640, 480), 'red')
# Test pinhole branch with mock — straight no-op
class _MockIntr:
    distortion_model = 'pinhole'
import src.voice_control.perception.undistort as u
import src.voice_control.perception.camera_intrinsics as ci
ci.get_intrinsics = lambda r, s: _MockIntr()
out = undistort_for_source(img, 'fake_source', MagicMock())
print('pinhole no-op:', out.size, '==', img.size, out is img)
"
```
Expected: `pinhole no-op: (640, 480) == (640, 480) True`.

- [ ] **Step 4: Smoke-test undistort live on Spot**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from src.voice_control.spot_dispatch import capture_frame
for nick in ['front_left', 'front_right', 'left', 'right', 'back']:
    f = capture_frame(nick)
    if f: f.pil.save(f'/tmp/undist_{nick}.jpg')
    print(f'{nick}:', 'ok' if f else 'NONE')
"
ls -la /tmp/undist_*.jpg
```
**Manually open each /tmp/undist_*.jpg and verify:**
- The frames no longer show fisheye barrel distortion (straight lines should be straight).
- Frames are upright (rotation from Task 2 still applies post-undistort).
- Frames are not cropped excessively at corners (if so, switch to `cv2.getOptimalNewCameraMatrix(K, D, ..., alpha=1)` per spec risks table).

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add \
  src/voice_control/perception/undistort.py src/voice_control/spot_dispatch.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: fisheye + brown-conrady undistortion via cv2 remap; cached per source"
```

---

## Task 7: Mid-plan checkpoint #2 — safety-reviewer + caveman on capture pipeline

**Files:**
- Read only: diff of Tasks 4-6

- [ ] **Step 1: Dispatch safety-reviewer**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C capture pipeline safety",
  prompt: "Audit changes to src/voice_control/spot_dispatch.py and src/voice_control/perception/ in the last 4 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. This is Stage 2C of a Boston Dynamics Spot robot perception overhaul. Focus on:\n1. Whether capture_frame failures (None return) propagate safely through callers — does any downstream code crash on None frame and bring down the brain or block estop?\n2. Whether the undistort pipeline could hang on a single bad frame (cv2.remap blocking on a corrupted map).\n3. Whether any change to spot_dispatch.py inadvertently changes motion path behavior (the file mixes capture and motion — the rotation + undistort additions should be capture-only).\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch caveman:cavecrew-reviewer in parallel**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 2 diff review",
  prompt: "Review the last 4 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone — Stage 2C tasks 4-6 (camera_intrinsics extension + undistort module + capture_frame integration). Verify:\n1. BD SDK proto field accesses (pinhole_brown_conrady, kannala_brandt, k1/k2/p1/p2/k3) are spelled correctly.\n2. cv2.fisheye.initUndistortRectifyMap takes D as (4,1) NOT (4,) — common mistake.\n3. The remap cache key is per-(robot, source) not per-call.\n4. capture_frame order is decode → undistort → rotate (not rotate → undistort, which would be wrong).\nReport severity-tagged findings only."
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

**Files:**
- Modify: `src/voice_control/visual_nav.py` (around the existing `_find_object` at line 175)

- [ ] **Step 1: Add the batched detector wrapper**

Insert in `visual_nav.py` near the existing `_find_object` (line 175):

```python
def _find_object_batch(pil_images: list, description: str) -> list:
    """Run YOLO-World detection on a batch of PIL images in one model call.

    Returns a list of bbox-or-None, one per input image, same order.
    Sets `model.set_classes([description])` once for the whole batch.
    """
    model = _get_world_model()
    model.set_classes([description])
    # ultralytics accepts list[PIL] → batched inference in a single forward pass
    results = model.predict(pil_images, device="cpu", conf=0.25, verbose=False)
    out = []
    for res in results:
        if res.boxes is None or len(res.boxes) == 0:
            out.append(None)
            continue
        # Take highest-confidence bbox per image.
        box = res.boxes[res.boxes.conf.argmax()]
        bbox = box.xyxy.cpu().numpy().flatten().tolist()  # [x1,y1,x2,y2]
        out.append(bbox)
    return out
```

- [ ] **Step 2: Add _scan_all_cameras helper**

Insert just after `_find_object_batch`:

```python
def _scan_all_cameras(session, query: str) -> dict:
    """Batched 5-camera scan for an object.

    Returns dict mapping rotated source name → bbox-or-None.
    Uses _capture_multi (which exists at visual_nav.py:200-225 per recon)
    to fetch all camera images in parallel, then runs YOLO-World once
    on the batch.
    """
    sources = [s for s, _ in ALL_CAMERAS]  # ALL_CAMERAS list at visual_nav.py:38-46
    frames = _capture_multi(session, sources)
    # frames is dict[source -> PIL.Image]; order both lists consistently
    ordered_sources = [s for s in sources if s in frames]
    ordered_images = [frames[s] for s in ordered_sources]
    if not ordered_images:
        return {}
    bboxes = _find_object_batch(ordered_images, query)
    return {src: bbox for src, bbox in zip(ordered_sources, bboxes)}
```

- [ ] **Step 3: Smoke-test the batched detector offline**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from PIL import Image
from src.voice_control.visual_nav import _find_object_batch
imgs = [Image.new('RGB', (320, 240), c) for c in ('red', 'green', 'blue')]
out = _find_object_batch(imgs, 'cat')
print('batch result:', out)  # likely [None, None, None] on solid-color inputs
"
```
Expected: no crash. Output is a list of length 3, each None (no cat in solid red/green/blue).

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/visual_nav.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: _find_object_batch + _scan_all_cameras for batched 5-camera scan"
```

---

## Task 10: Replace serial scan loop in visual_nav.py with batched scan

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
# Stage 2C: batched scan (was serial loop pre-2026-05-15).
results = _scan_all_cameras(session, query)
hits = {src: bbox for src, bbox in results.items() if bbox is not None}
if not hits:
    return None
# Pick the largest bbox across cameras (closest object). Existing
# selection logic ran per-camera; now we pick across the batched dict.
best_source, best_bbox = max(
    hits.items(),
    key=lambda kv: (kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1]),
)
# ... rest of the original function (turn-to-bearing, walk-to-object) follows.
```

**Important:** preserve the function's outer signature + downstream selection of the `best_source` + `best_bbox` semantics. The behavioral change is: 5 sequential detect calls → 1 batched detect call. Pixel un-rotation + bearing math + WalkToObjectInImage submission are unchanged.

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
  prompt: "Review the last 3 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone — Stage 2C tasks 8-10 (gripper plumbing + batched scan helpers + serial-loop replacement). Verify:\n1. gripper_camera_available does NOT call image_client without try/except (arm not powered = exception, NOT crash).\n2. _find_object_batch handles empty input list (currently it would call model.predict([]) which may raise).\n3. _scan_all_cameras's dict iteration is order-preserving (Python 3.7+ guarantees, but the zip pairing relies on it).\n4. The serial-loop replacement preserves the original function's selection semantics (largest-bbox across cameras, not first-hit).\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Fix BLOCKER/HIGH findings; commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address checkpoint 3 review findings"
```

---

## Task 12: Add look_around action to spot_dispatch

**Files:**
- Modify: `src/voice_control/spot_dispatch.py` (insert new action handler, register in dispatch table)

- [ ] **Step 1: Add the look_around handler**

In `spot_dispatch.py`, before `dispatch_intent` (around line 552), add:

```python
def _action_look_around(intent: dict) -> dict:
    """Capture frames from all body cameras + gripper (if available),
    run VLM caption on each, return per-camera descriptions.

    Returns dict for the brain's response generation:
        {"action": "look_around", "results": {<source>: <caption>}}
    """
    from src.voice_control.llm_brain import describe_image  # VLM call
    sources = list(CAMERA_SOURCES.keys())
    # De-dup (multiple keys can alias to the same source)
    seen = set()
    unique_sources = []
    for nick in sources:
        actual = CAMERA_SOURCES[nick]
        if actual not in seen:
            seen.add(actual)
            unique_sources.append(nick)

    if "gripper" in unique_sources and not gripper_camera_available():
        unique_sources.remove("gripper")

    captions: dict[str, str] = {}
    for nick in unique_sources:
        frame = capture_frame(nick)
        if frame is None:
            continue
        try:
            cap = describe_image(frame.jpeg_bytes, prompt="Describe what is visible in one short sentence.")
        except Exception as e:
            cap = f"(VLM failed: {e})"
        captions[frame.source] = cap
    return {"action": "look_around", "results": captions}
```

- [ ] **Step 2: Register in the dispatch table**

In `dispatch_intent` (line ~552), add the elif branch:

```python
elif intent_name == "look_around":
    return _action_look_around(intent)
```

Place it near other perception-style actions (`describe`, `check_obstacles`) for grep-ability.

- [ ] **Step 3: Smoke-test the action without the brain**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from src.voice_control.spot_dispatch import dispatch_intent
r = dispatch_intent({'intent': 'look_around', 'params': {}})
import json; print(json.dumps(r, indent=2))
"
```
Expected: prints a JSON dict with `action: look_around` + `results: {<source>: <caption>}` for 4-5 cameras. VLM latency × 5 = ~15-30 s on first run (cold cache).

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/spot_dispatch.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: look_around action — batched VLM caption across all cameras"
```

---

## Task 13: Wire look_around into action.gbnf grammar

**Files:**
- Modify: `src/voice_control/action.gbnf` (created in Plan 2A Task 3)

- [ ] **Step 1: Add look_around to the action enum production**

Read the existing grammar:
```bash
cat /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/action.gbnf
```
The `action_name` rule should be of the form `action_name ::= "\"stand\"" | "\"sit\"" | "\"walk\"" | ...`. Add `| "\"look_around\""` to the alternatives:

```gbnf
action_name ::= "\"stand\""
              | "\"sit\""
              | "\"walk\""
              | "\"go_to\""
              | "\"follow\""
              | "\"stop\""
              | "\"describe\""
              | "\"check_obstacles\""
              | "\"look_around\""
              | "\"battery_status\""
              | "\"power_off\""
              | "\"estop\""
              # ... (preserve all existing alternatives)
```

The exact set of existing alternatives depends on Plan 2A Task 3 output; preserve every alternative present.

- [ ] **Step 2: Re-run mic-verify smoke against look_around**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  echo "look around" | python3 scripts/mic_verify.py --text-mode --intent look_around
```
(Assumes mic_verify supports text mode for grammar-only smoke. If not, the smoke is deferred to Stage 2B-style live mic test.)
Expected: brain returns `{"action": "look_around", ...}` valid JSON.

- [ ] **Step 3: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/action.gbnf
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: action.gbnf — add look_around to action enum"
```

---

## Task 14: pin-guardian audit before ultralytics bump

**Files:**
- Read only: `requirements.txt` (pre-bump state)

- [ ] **Step 1: Capture pre-bump baseline**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  pip freeze | grep -E '(ultralytics|numpy|opencv|onnxruntime|torch)' > /tmp/pre_yolo26_pins.txt
cat /tmp/pre_yolo26_pins.txt
```

- [ ] **Step 2: Determine minimum ultralytics version supporting YOLO26 + engine export**

```bash
pip index versions ultralytics | head -5
# Check release notes for the first version that ships YOLO26 (>= 8.4 per spec target)
```
The spec calls for YOLO26 (released 2026-01-14). Verify the version with:
```bash
pip download ultralytics==X.Y.Z --no-deps -d /tmp/ul_test
python3 -c "from ultralytics import YOLO; help(YOLO('yolo26n.pt').export)" 2>&1 | head -20
```
Pin to the lowest version that supports YOLO26 and `.engine` export.

- [ ] **Step 3: Dispatch pin-guardian on the proposed change**

```
Agent({
  subagent_type: "pin-guardian",
  description: "Audit ultralytics bump",
  prompt: "I'm about to bump ultralytics from >=8.1.0 to the lowest version supporting YOLO26 + .engine export (likely >=8.4.x, exact version TBD by Step 2 of Task 14). The Jetson AGX Orin / JetPack 6.2.1 stack has these hard pins (segfault risk on upgrade): numpy==1.26.4, onnxruntime-gpu==1.23.0, kokoro-onnx==0.4.9, opencv-python==4.11.0.86, bosdyn-*==5.0.1.1. Ultralytics has a habit of bumping torch / numpy / opencv transitively. Verify whether bumping ultralytics requires --no-deps or breaks any of the hard pins. Report:\n1. The transitive-dep delta the bump would cause.\n2. Whether --no-deps + manual extra-dep install is needed.\n3. Whether any hard pin would be silently overridden.\n4. Concrete pip command sequence that respects the pins."
})
```

- [ ] **Step 4: Apply the pip command from pin-guardian's recommendation**

Run the exact command(s) the agent returns. Likely shape:
```bash
pip install --no-deps "ultralytics>=8.4.0"   # version per Step 2
pip check
```
If `pip check` reports unmet deps, install only the missing ones (NOT a bulk `pip install ultralytics` which would rewrite numpy/torch/opencv pins).

- [ ] **Step 5: Verify hard pins still intact**

```bash
pip freeze | grep -E '^(numpy|onnxruntime-gpu|opencv-python|kokoro-onnx|bosdyn-)' 
diff /tmp/pre_yolo26_pins.txt <(pip freeze | grep -E '(ultralytics|numpy|opencv|onnxruntime|torch)')
```
Expected: only `ultralytics` changes. If `numpy`, `opencv-python`, or `onnxruntime-gpu` shifted, revert immediately:
```bash
pip install --force-reinstall "numpy==1.26.4" "opencv-python==4.11.0.86"
pip install --no-deps --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 "onnxruntime-gpu==1.23.0"
```

- [ ] **Step 6: Update requirements.txt**

Edit `requirements.txt` line 44:
```
ultralytics>=8.4.0   # Bumped 2026-05-XX for YOLO26 + engine export support
```

- [ ] **Step 7: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add requirements.txt
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: bump ultralytics for YOLO26 + .engine export (pins preserved)"
```

---

## Task 15: Download + export YOLO26-N and YOLOE26-S to TensorRT engines

**Files:**
- Create: `scripts/export_yolo_engines.py`
- Output: `/mnt/ssd/cfm_mppi_caches/yolo_engines/yolo26n.engine` and `yoloe26s.engine`

- [ ] **Step 1: Write the export script**

Create `scripts/export_yolo_engines.py`:

```python
#!/usr/bin/env python3
"""Export YOLO26-N (follow loop) + YOLOE26-S (object scan) to TensorRT FP16.

Output to /mnt/ssd/cfm_mppi_caches/yolo_engines/. Engines are JetPack-bound
(invalidated on any JetPack/TRT bump); re-run this script after any system
upgrade.

NOTE: Ultralytics auto-downloads weights from their model hub on first
YOLO('yolo26n.pt') call. If yolov8-style cached weights already exist at
the project root, ultralytics will reuse them — but YOLO26 vs YOLOE26 are
new models with no cached weights yet on this machine.
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

ENGINE_DIR = Path("/mnt/ssd/cfm_mppi_caches/yolo_engines")
ENGINE_DIR.mkdir(parents=True, exist_ok=True)


def export(weights: str, out_name: str) -> Path:
    """Download weights (if needed), export to TRT FP16 engine, move to ENGINE_DIR."""
    model = YOLO(weights)
    # Export creates <weights_stem>.engine next to the .pt file
    exported = model.export(format="engine", half=True, device=0, dynamic=True)
    src = Path(exported)
    dst = ENGINE_DIR / out_name
    shutil.move(str(src), str(dst))
    print(f"Exported {weights} -> {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    print("Exporting YOLO26-N (follow loop)...")
    export("yolo26n.pt", "yolo26n.engine")
    print()
    print("Exporting YOLOE26-S (object scan)...")
    export("yoloe26s.pt", "yoloe26s.engine")
    print()
    print(f"All engines cached at: {ENGINE_DIR}")
```

- [ ] **Step 2: Pre-check existing .pt cache (avoid duplicate download per feedback-check-existing-artifacts)**

```bash
ls -lh /home/spotdog/spot/dartmouth_spot_capstone/yolo*.pt 2>/dev/null
ls -lh /home/spotdog/spot/dartmouth_spot_capstone/*.pt 2>/dev/null
```
If `yolo26n.pt` / `yoloe26s.pt` already exist at the project root, pass those paths into the export script. Avoid letting ultralytics re-download.

- [ ] **Step 3: Run the export**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/export_yolo_engines.py 2>&1 | tee /mnt/ssd/spot-logs/yolo_engine_export_$(date +%Y%m%d).log
```
Expected: ~3-5 min per model. Final output: 2 `.engine` files in `/mnt/ssd/cfm_mppi_caches/yolo_engines/`. If export fails:
- `CUDA OOM`: kill llama-server temporarily (`systemctl --user stop llama-server`), re-run export, then restart llama-server.
- `Onnx export error`: ultralytics version may not support YOLO26; check Task 14 Step 2 again.
- `TRT not found`: confirm `dpkg -l | grep tensorrt` shows 10.3.0.30; check `/usr/src/tensorrt/`.

- [ ] **Step 4: Verify engines load via ultralytics**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from ultralytics import YOLO
m1 = YOLO('/mnt/ssd/cfm_mppi_caches/yolo_engines/yolo26n.engine')
print('yolo26n engine loaded:', type(m1.model).__name__)
m2 = YOLO('/mnt/ssd/cfm_mppi_caches/yolo_engines/yoloe26s.engine')
print('yoloe26s engine loaded:', type(m2.model).__name__)
"
```
Expected: both load without error. ultralytics auto-detects `.engine` extension.

- [ ] **Step 5: Update .gitignore for new .pt files**

```bash
grep 'yolov8' /home/spotdog/spot/dartmouth_spot_capstone/.gitignore
```
Current pattern `yolov8*.pt` does NOT match `yolo26n.pt` or `yoloe26s.pt`. Update to broader:
```bash
# In .gitignore, replace `yolov8*.pt` with:
yolo*.pt
yoloe*.pt
```

- [ ] **Step 6: Commit script + gitignore + export log**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add \
  scripts/export_yolo_engines.py .gitignore
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: YOLO26-N + YOLOE26-S TRT FP16 export script + gitignore broadening"
```

---

## Task 16: Swap visual_nav model loaders to .engine paths

**Files:**
- Modify: `src/voice_control/visual_nav.py` (lines 107-126 — `_get_world_model` and `_get_person_model`)

- [ ] **Step 1: Replace the loader functions**

Replace lines 107-126 with:

```python
from pathlib import Path

ENGINE_DIR = Path("/mnt/ssd/cfm_mppi_caches/yolo_engines")
WORLD_ENGINE = ENGINE_DIR / "yoloe26s.engine"   # object scan
PERSON_ENGINE = ENGINE_DIR / "yolo26n.engine"   # follow loop

_world_model = None
_person_model = None


def _get_world_model():
    """YOLOE26-S for object scan (prompt-driven via set_classes)."""
    global _world_model
    if _world_model is None:
        from ultralytics import YOLO
        if not WORLD_ENGINE.exists():
            # Fallback to .pt if engine missing (re-export needed after JetPack bump)
            pt_path = _PROJECT_ROOT / "yoloe26s.pt"
            if pt_path.exists():
                _world_model = YOLO(str(pt_path))
            else:
                # Last fallback: legacy YOLO-Worldv2
                _world_model = YOLO(str(_PROJECT_ROOT / "yolov8s-worldv2.pt"))
        else:
            _world_model = YOLO(str(WORLD_ENGINE))
    return _world_model


def _get_person_model():
    """YOLO26-N for follow loop (single-class person at high FPS)."""
    global _person_model
    if _person_model is None:
        from ultralytics import YOLO
        if not PERSON_ENGINE.exists():
            pt_path = _PROJECT_ROOT / "yolo26n.pt"
            if pt_path.exists():
                _person_model = YOLO(str(pt_path))
            else:
                # Last fallback: legacy YOLOv8n
                _person_model = YOLO(str(_PROJECT_ROOT / "yolov8n.pt"))
        else:
            _person_model = YOLO(str(PERSON_ENGINE))
    return _person_model
```

The fallback chain (engine → new .pt → legacy .pt) provides graceful degradation if the engine cache is stale or missing.

- [ ] **Step 2: Update device kwarg for engine path**

`.engine` files run on GPU by definition. The existing `device="cpu"` in `.predict()` calls (per recon at visual_nav.py:175-197) MUST be removed for engine-loaded models — ultralytics will raise if you ask a GPU engine to run on CPU.

Find every `predict(...)` call and either remove `device="cpu"` or branch on engine vs .pt:

```bash
grep -n 'device="cpu"' /home/spotdog/spot/dartmouth_spot_capstone/src/voice_control/visual_nav.py
```

For each line, replace `device="cpu"` with `device=0` (CUDA device 0). VRAM contention with llama.cpp is acceptable; YOLO engines are tiny (~7-25 MB) and brief.

- [ ] **Step 3: Smoke-test engine inference**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from PIL import Image
from src.voice_control.visual_nav import _get_person_model
import time
m = _get_person_model()
img = Image.new('RGB', (640, 640), 'red')
# Warmup
m.predict(img, device=0, conf=0.25, verbose=False)
# Time
t0 = time.time()
for _ in range(20):
    m.predict(img, device=0, classes=[0], conf=0.25, verbose=False)
print(f'avg per-frame: {(time.time()-t0)/20*1000:.1f} ms')
"
```
Expected: < 5 ms per frame on AGX Orin (target: 2.62 ms per Ultralytics' own benchmark).
If > 20 ms, the engine likely fell back to CPU silently — investigate via `tegrastats` (next task).

- [ ] **Step 4: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add src/voice_control/visual_nav.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: visual_nav swaps to .engine YOLO26-N + YOLOE26-S; .pt fallback chain"
```

---

## Task 17: TRT numerics verification (mAP delta < 1 %)

**Files:**
- Create: `scripts/verify_trt_numerics.py`
- Create dir: `tests/captures/` (populate with 20-30 Spot frames before running)

- [ ] **Step 1: Capture verification frames**

With Spot in a typical lab/tour environment, capture 20-30 frames covering diverse scenes (objects: chair, person, door, computer, plant, sign; cameras: all 5 fisheye + gripper).

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && mkdir -p tests/captures
python3 -c "
from src.voice_control.spot_dispatch import capture_frame
import time
for i in range(5):
    print(f'Pose {i+1}/5: reposition Spot, press ENTER')
    input()
    for nick in ['front_left', 'front_right', 'left', 'right', 'back']:
        f = capture_frame(nick)
        if f: f.pil.save(f'tests/captures/pose{i}_{nick}.jpg')
"
ls tests/captures/ | wc -l
```
Expected: ~25 JPEGs total.

- [ ] **Step 2: Write the numerics verifier**

Create `scripts/verify_trt_numerics.py`:

```python
#!/usr/bin/env python3
"""Compare .pt baseline vs .engine FP16 outputs on tests/captures/.

Reports per-frame IoU + class-match deltas. PASS if median IoU >= 0.9
and class-match rate >= 0.99 across the corpus.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
CAPS = REPO / "tests/captures"
ENGINE = Path("/mnt/ssd/cfm_mppi_caches/yolo_engines/yolo26n.engine")
PT = REPO / "yolo26n.pt"


def iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def first_box(res):
    if res.boxes is None or len(res.boxes) == 0:
        return None
    box = res.boxes[res.boxes.conf.argmax()]
    return tuple(box.xyxy.cpu().numpy().flatten().tolist())


def main():
    if not ENGINE.exists() or not PT.exists():
        print(f"Missing: ENGINE={ENGINE.exists()} PT={PT.exists()}"); sys.exit(2)
    m_pt = YOLO(str(PT))
    m_en = YOLO(str(ENGINE))
    images = sorted(CAPS.glob("*.jpg"))
    if not images:
        print(f"No images in {CAPS}; capture first"); sys.exit(2)
    ious, cls_matches = [], []
    for p in images:
        img = Image.open(p)
        r_pt = m_pt.predict(img, device=0, conf=0.25, verbose=False)[0]
        r_en = m_en.predict(img, device=0, conf=0.25, verbose=False)[0]
        b_pt, b_en = first_box(r_pt), first_box(r_en)
        i = iou(b_pt, b_en)
        cls_pt = None if r_pt.boxes is None or len(r_pt.boxes) == 0 else int(r_pt.boxes.cls[r_pt.boxes.conf.argmax()].item())
        cls_en = None if r_en.boxes is None or len(r_en.boxes) == 0 else int(r_en.boxes.cls[r_en.boxes.conf.argmax()].item())
        match = cls_pt == cls_en
        ious.append(i)
        cls_matches.append(match)
        print(f"{p.name}: iou={i:.3f} cls_pt={cls_pt} cls_en={cls_en} match={match}")
    med_iou = float(np.median(ious))
    cls_rate = sum(cls_matches) / len(cls_matches)
    print(f"\nMedian IoU: {med_iou:.3f}")
    print(f"Class match rate: {cls_rate:.3f}")
    ok = med_iou >= 0.9 and cls_rate >= 0.99
    print(f"VERDICT: {'PASS' if ok else 'FAIL — investigate FP16 numerics or fall back to FP32'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run verification**

```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/verify_trt_numerics.py 2>&1 | tee /mnt/ssd/spot-logs/trt_verify_$(date +%Y%m%d).log
```
Expected: VERDICT: PASS.
If FAIL with low IoU but high class-match: bbox precision regressed slightly under FP16 — acceptable; document.
If FAIL with low class-match: re-export FP32 engines (set `half=False` in export script) and accept the latency penalty.

- [ ] **Step 4: Verify GPU is actually being used**

In one terminal:
```bash
tegrastats --interval 500
```
In another, while inference runs:
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 -c "
from PIL import Image
from src.voice_control.visual_nav import _get_person_model
m = _get_person_model()
img = Image.new('RGB', (640, 640), 'red')
for _ in range(100):
    m.predict(img, device=0, classes=[0], conf=0.25, verbose=False)
"
```
Expected in tegrastats output: `GR3D_FREQ` shows > 0 % during the 100 iterations. **If GR3D_FREQ stays at 0 %, the engine silently fell back to CPU — the swap is broken. Debug before merging.**

- [ ] **Step 5: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add scripts/verify_trt_numerics.py
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: TRT numerics verifier (mAP/cls IoU delta gate) + lab capture corpus"
```

If you commit `tests/captures/*.jpg` (~25 × 50-200 KB = ~1-5 MB total), add them; otherwise add `tests/captures/` to .gitignore.

---

## Task 18: Mid-plan checkpoint #4 — safety + caveman + numerics

**Files:**
- Read only: diff of Tasks 12-17

- [ ] **Step 1: Dispatch safety-reviewer on look_around + visual_nav engine swap**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C look_around + engine swap safety",
  prompt: "Audit changes to src/voice_control/spot_dispatch.py (look_around action) and src/voice_control/visual_nav.py (engine swap) in the last 6 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. Focus on:\n1. look_around action: does it block the dispatch thread for 15-30s (5 sequential VLM calls)? If so, can the user still issue a stop command during that time? E-stop bypass coverage?\n2. visual_nav engine swap: the follow loop now runs YOLO26-N on GPU at ~400 FPS theoretical. Does the throttle still cap downstream velocity commands at 10 Hz, or could the perception speedup cascade into faster motion than safe?\n3. Engine fallback chain (.engine -> new .pt -> legacy .pt): if all 3 fail, does the follow loop crash or gracefully decline?\nReport severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch caveman:cavecrew-reviewer on the same diff**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "2C checkpoint 4 diff",
  prompt: "Review the last 6 commits on the current branch at /home/spotdog/spot/dartmouth_spot_capstone — Stage 2C tasks 12-17 (look_around action, action.gbnf update, ultralytics bump, engine export + swap, numerics verifier). Verify:\n1. Every device= kwarg in visual_nav.py is consistent with the engine-vs-pt mode (engine MUST be device=0).\n2. The engine fallback chain order is correct (engine first, new .pt second, legacy .pt third).\n3. .gitignore now matches yolo26n.pt + yoloe26s.pt (the broader 'yolo*.pt' pattern).\n4. Numerics verifier handles None bboxes (when no object detected by either model — both None should NOT be a class-match miss).\nReport severity-tagged findings only."
})
```

- [ ] **Step 3: Fix BLOCKER/HIGH; commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address checkpoint 4 review findings"
```

---

## Task 19: Update stage2-rollback.md with vision-rollback layer

**Files:**
- Modify: `docs/project/stage2-rollback.md`

- [ ] **Step 1: Append the vision-rollback section**

Append:

```markdown
## After 2C — vision

### Layer: engine cache invalidation (after JetPack bump)

```bash
# Re-export engines after any sudo apt upgrade that touches TensorRT.
python3 scripts/export_yolo_engines.py
```

### Layer: fall back from .engine to .pt (engine broken / mAP regression)

The visual_nav loaders fall back automatically if the .engine file is missing:
1. `.engine` (preferred, GPU FP16)
2. `<weight>.pt` (CPU/GPU, .pt cache)
3. Legacy YOLOv8 `.pt` (last resort)

To force .pt mode without deleting engines, rename the engine dir:
```bash
mv /mnt/ssd/cfm_mppi_caches/yolo_engines{,.disabled}
```

### Layer: disable fisheye undistort

If undistort produces bad frames (e.g., cv2 build mismatch, wrong BD intrinsics):

```python
# In src/voice_control/perception/undistort.py, top-level:
DISABLED = True

def undistort_for_source(pil_image, source_name, robot):
    if DISABLED:
        return pil_image
    # ... existing logic ...
```

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
git reset --hard stage2b-e2e-complete
git push --force-with-lease origin main   # ASK USER FIRST
```
```

- [ ] **Step 2: Commit**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add docs/project/stage2-rollback.md
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: rollback doc — vision layers (engine, undistort, scan, rotation)"
```

---

## Task 20: Final review — caveman + feature-dev + safety on full 2C diff

**Files:**
- Read only: `git diff stage2b-e2e-complete..HEAD`

- [ ] **Step 1: Dispatch caveman:cavecrew-reviewer on full 2C diff**

```
Agent({
  subagent_type: "caveman:cavecrew-reviewer",
  description: "Full 2C final review",
  prompt: "Review every change between git tag stage2b-e2e-complete and HEAD on the current branch at /home/spotdog/spot/dartmouth_spot_capstone. This is the complete Stage 2C vision overhaul: per-camera rotation, fisheye undistortion, gripper plumbing, batched 5-camera scan, look_around action, TRT YOLO26+YOLOE26 engine swap. Report severity-tagged findings only."
})
```

- [ ] **Step 2: Dispatch feature-dev:code-reviewer for quality audit**

```
Agent({
  subagent_type: "feature-dev:code-reviewer",
  description: "2C quality audit",
  prompt: "Code-review the full Stage 2C diff (between tags stage2b-e2e-complete and HEAD on the current branch at /home/spotdog/spot/dartmouth_spot_capstone). Focus on high-confidence issues: bugs, logic errors, security holes, code-quality issues, and adherence to project conventions (see CLAUDE.md / MEMORY.md / docs/superpowers/specs/2026-05-15-stage2-design.md). Report only issues you are 80%+ confident on. Files touched: src/voice_control/perception/*.py, src/voice_control/spot_dispatch.py, src/voice_control/visual_nav.py, src/voice_control/action.gbnf, scripts/export_yolo_engines.py, scripts/verify_trt_numerics.py, requirements.txt, .gitignore, docs/project/stage2-rollback.md."
})
```

- [ ] **Step 3: Dispatch safety-reviewer for final motion-path audit**

```
Agent({
  subagent_type: "safety-reviewer",
  description: "2C final safety pass",
  prompt: "Final safety audit of the complete Stage 2C diff (between stage2b-e2e-complete and HEAD on the current branch at /home/spotdog/spot/dartmouth_spot_capstone). Confirm:\n1. E-stop coverage unchanged across all new code paths.\n2. look_around action cannot wedge the dispatcher in a multi-VLM call that prevents the next utterance's stop command from running.\n3. YOLO26-N follow speedup (CPU -> GPU TRT) has not bypassed any rate limiter on velocity commands.\n4. Undistortion failures degrade gracefully (no crash propagation to motion path).\nReport BLOCKER/HIGH severity findings; this is the merge gate."
})
```

- [ ] **Step 4: Address BLOCKER/HIGH from all 3 agents; commit fixes**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -u
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: address final review findings"
```

---

## Task 21: Live regression — re-run Stage 2B matrix items 1-6 with vision changes

**Files:**
- Read only: `tests/regression/stage2b_matrix.yaml`

- [ ] **Step 1: Run the regression runner on Spot**

Operator at the robot, E-stop in another terminal:
```bash
cd /home/spotdog/spot/dartmouth_spot_capstone && \
  python3 scripts/run_regression_matrix.py
```
Walk through items 1-6 (skip 7, 8 — exploratory unless time permits).

- [ ] **Step 2: Verify all non-exploratory items still PASS**

Expected: 6/6 non-exploratory PASS. The vision changes should improve, not regress, items 4 (describe), 5 (GraphNav), 6 (follow).

If anything regressed: do NOT proceed to merge. Investigate, fix, re-test.

- [ ] **Step 3: Commit regression log**

```bash
git -C /home/spotdog/spot/dartmouth_spot_capstone add -f logs/stage2b/regression_*.json
git -C /home/spotdog/spot/dartmouth_spot_capstone commit -m "stage 2c: re-run 2B matrix on Spot — vision changes pass regression"
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
gh pr create --title "Stage 2C: vision overhaul (rotation, undistort, gripper, batched scan, look_around, TRT engines)" --body "$(cat <<'EOF'
## Summary
- Per-camera rotation table (`CAMERA_ROTATION_DEG`) — front cameras no longer hardcoded to 90° CW
- Fisheye undistortion via cv2 (Kannala-Brandt + Brown-Conrady) — precomputed remap, ~3-5ms per frame
- Frame namedtuple — single perception data type across capture and inference
- Gripper color camera (`hand_color_image`) plumbing — RGB_U8 decode + pre-flight gate
- Batched 5-camera scan — one detector call instead of 5 sequential
- `look_around` action — batched VLM caption across all cameras
- TRT FP16 YOLO26-N (follow) + YOLOE26-S (object scan) — engine cache on SSD
- Ultralytics bumped to YOLO26-supporting version (hard pins preserved)
- Vision rollback layer documented

## Test plan
- [x] Per-camera rotation visually verified on all 5 fisheye sources
- [x] Fisheye undistort visually verified (straight lines straight)
- [x] Gripper camera live test passed (color frame saved)
- [x] Batched scan smoke-test (chair detection) — single batched call
- [x] look_around action returns per-camera VLM captions
- [x] TRT numerics PASS (median IoU >= 0.9, class match >= 99 %)
- [x] tegrastats confirms GPU utilization during engine inference
- [x] Stage 2B regression matrix items 1-6 PASS on Spot with vision changes

## Rollback
See docs/project/stage2-rollback.md — layered rollback (engine cache, undistort flag, scan revert, rotation revert, hard reset).
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
git -C /home/spotdog/spot/dartmouth_spot_capstone tag -a stage2c-vision-complete -m "stage 2c: vision overhaul merged; TRT YOLO26+YOLOE26 live"
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
- ✅ 2C.0.5 fisheye undistortion → Tasks 5, 6
- ✅ 2C.1 gripper color camera → Task 8
- ✅ 2C.2 batched 5-camera scan → Tasks 9, 10
- ✅ 2C.3 look_around action → Tasks 12, 13
- ✅ 2C.4 TRT YOLO26 + YOLOE26 → Tasks 14, 15, 16, 17

**Placeholder scan:** No TBD / TODO / "implement later" anywhere. The `{LOCATION}` placeholder in Stage 2B YAML is intentional (operator substitution at runtime). The "exact ultralytics version" in Task 14 Step 2 is a runtime discovery, not a placeholder — Step 3 dispatches pin-guardian to resolve it before pinning.

**Type consistency:** `Frame(pil, jpeg_bytes, source)` defined in Task 1 is used as `frame.pil` / `frame.jpeg_bytes` / `frame.source` throughout Tasks 3, 8, 12. `Intrinsics` dataclass extended in Task 5 with `distortion_model` (str) + `D` (np.ndarray) is consumed by `_build_maps` in Task 6 with matching field accesses. `CAMERA_ROTATION_DEG` keyed by full SDK source name (e.g. `"frontleft_fisheye_image"`); `CAMERA_SOURCES` keyed by nickname → SDK name; `_rotate_for_source(image, source_name)` takes the SDK name — internally consistent across Tasks 2, 3, 8.

**Reviewer dispatch points (per "build in periodic review passes" mandate):**
- Task 4: caveman:cavecrew-reviewer (post Tasks 1-3) — Frame + rotation + capture rewrite
- Task 7: safety-reviewer + caveman:cavecrew-reviewer (post Tasks 5-6) — intrinsics + undistort pipeline
- Task 11: caveman:cavecrew-reviewer (post Tasks 8-10) — gripper + batched scan
- Task 14: pin-guardian (pre ultralytics bump)
- Task 18: safety-reviewer + caveman:cavecrew-reviewer (post Tasks 12-17) — look_around + engine swap
- Task 20: caveman + feature-dev:code-reviewer + safety-reviewer (final pass, full diff)

**Ambiguity check:**
- "median IoU >= 0.9 AND class match >= 99 %" is explicit numeric gate (Task 17).
- "GPU utilization spike" = `tegrastats` `GR3D_FREQ > 0 %` during inference (Task 17 Step 4).
- "pin preserved" = `pip freeze` for numpy/onnxruntime-gpu/opencv-python/kokoro-onnx exactly matches pre-bump baseline (Task 14 Step 5).
- "TRT FP16 numerics regress" = Task 17 verifier exits non-zero — re-export FP32 per spec risks table.

**Risks not yet mitigated in this plan (deferred):**
- YOLO26 ultralytics support landing might require git-main install, not pypi — pin-guardian in Task 14 will catch this.
- cv2 corner cropping might require `cv2.getOptimalNewCameraMatrix(K, D, ..., alpha=1)` — Task 6 Step 4 visual check + spec risks table mitigation.
- AprilTag follow as exploratory backup is OUT OF SCOPE for Stage 2C (per spec out-of-scope list); Stage 3 spike if YOLO26-N + undistort proves insufficient on Spot.
