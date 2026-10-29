# Vision Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five well-defined vision actions (describe, find, look at speaker, go to, follow) running YOLO26-l / YOLOE-26-l on the Jetson GPU via TensorRT, wired into the voice pipeline through thin hooks.

**Architecture:** A self-contained package `src/voice_control/vision/` holds camera geometry, the front stitch, a GPU detector runtime (onnxruntime + TensorRT, no Ultralytics at runtime), a world snapshot, a thinking-off VLM client and the five actions. Motion actions talk to the robot through a small injected interface (`NavIO`) so they are testable without a robot; the dispatcher supplies the real one. Hooks in `client_mic.py`, `spot_dispatch.py`, `llm_brain.py` and the GBNF grammar are a few lines each, so the package can be transplanted after the operator's restructure.

**Tech Stack:** Python 3.10, numpy, OpenCV 4.11 (`cv2`), onnxruntime-gpu 1.23 (TensorRT/CUDA/CPU providers), bosdyn SDK 5.2.0, requests, pytest. All already in spot-env.

**Spec:** `docs/superpowers/specs/2026-10-29-vision-upgrade-design.md` (read it first). Spike code: `docs/superpowers/specs/2026-10-29-vision-spikes/`.

**Where:** worktree `/home/spotdog/spot/spot_vision`, branch `vision/perception-v2`. Never edit the main checkout `/home/spotdog/spot/dartmouth_spot_capstone` (demo code). Run tests with the main checkout's venv from the worktree root:
`/home/spotdog/spot/dartmouth_spot_capstone/spot-env/bin/python -m pytest tests/voice_control/vision -q`
Robot scripts: run from the **main** checkout dir so `.env` loads (`load_dotenv(cwd/.env)`), robot at `192.168.50.3` (wired).

## Global Constraints

- No new packages in spot-env; never import `ultralytics` from runtime code (it auto-installs into the venv). `pin-guardian` applies if `requirements.txt` changes (it should not).
- Models: `yolo26l-seg` (main), `yolo26l-pose` (on demand), `yoloe-26l-seg` (150-noun list baked in). Files under `models/vision/` (gitignored), engines cached in `models/vision/trt_cache/`.
- Upright rotation (deg CCW): back 0, frontleft −78, frontright −102, left 0, right 180, hand_color 0.
- Bearings: degrees in the body frame, **+left** (CCW), range (−180, 180]; 0 = straight ahead. Same convention as `gaze_at_bearing`.
- Directions spoken in eight sectors: in front, front-left, on my left, behind-left, behind me, behind-right, on my right, front-right. Distances in whole metres ("about two metres"); < 1 m: "right next to me".
- Spot never says stop, halt, freeze or e-stop (use "paused" / "holding still"). Existing test `tests/voice_control/test_spoken_no_stop_words.py` pattern applies to every new template.
- VLM calls: `chat_template_kwargs: {"enable_thinking": false}`; one VLM call per view; never several labelled views in one call.
- GPU budget: one-shot unlimited burst; look-at-speaker one check per gaze request; go-to/follow ≤ 20 images/s total (tuning value).
- Stop distances: 1.5 m person, 1.0 m object; arrived = within 0.3 m of goal distance; lost after 3 s unseen; "something's in the way" after 4 s without progress; follow distance 2.0 m.
- Speaker match window ±30°; detection freshness 1 s; "person in red" ≤ 3 candidates; raised hand = wrist above same-side shoulder (upright coords), keypoint conf ≥ 0.5, 3 consecutive frames, ≤ 5 s.
- Obstacles: Spot's built-in obstacle avoidance (never disable it in mobility params) + existing `capped_mobility_params()`; `_forward_collision_blocks` not relied on.
- Follow-ups are single-turn: Spot asks and ends the action; answers come back as fresh commands.
- Every motion command re-checks the dispatch safety generation (`_aborted(gen)`) immediately before sending. Any task touching motion or gaze gets a `safety-reviewer` pass before commit.
- Commit messages: one line, no trailers (repo convention).
- Fixtures with people in them are gitignored (`tests/fixtures/vision/private/`); people-free frames may be committed.

## Review Focus

1. **Front cameras are cross-mounted** — `frontleft_fisheye_image` looks front-*right*. A detection must get its bearing from the transform, never the source name. → Task 1 test `test_bearing_uses_transform_not_name`.
2. **Detections on the upright image mapped back wrongly** (rotation by −78/−102 is not a 90° swap). A pixel round-trip must land on the same raw pixel. → Task 1 test `test_upright_to_raw_roundtrip_tilted`.
3. **Depth missing / zero inside the mask** (black, too far, reflective) must fall back to ground plane, and report `distance_method`, not return 0 m and walk into someone. → Task 2 test `test_distance_zero_depth_falls_back_to_ground`.
4. **A stop arriving mid-walk** must prevent any later motion command from the go-to/follow loop, even one computed just before the stop. → Task 11/12 tests `test_goto_no_command_after_abort`, `test_follow_no_command_after_abort`.
5. **GPU unavailable at startup** (TensorRT build fails, CUDA busy): one-shot actions still answer on CPU; follow/look-at-speaker refuse politely instead of crashing the voice loop. → Task 5 test `test_runtime_falls_back_to_cpu`, Task 12 test `test_follow_refuses_on_cpu`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voice_control/vision/__init__.py` | public API re-exports: `describe`, `find`, `refine_speaker_bearing`, `goto`, `follow`, `set_speaker_bearing_getter` |
| `vision/types.py` | dataclasses `CameraFrame`, `Detection`, `ActionResult` |
| `vision/cameras.py` | rotation, pixel mapping, bearing, sector/distance wording, capture, depth, ground plane |
| `vision/stitch.py` | front stitch (port of spike `cv_stitch.py`) |
| `vision/runtime.py` | ORT sessions, provider fallback, letterbox, decode seg/pose |
| `vision/vocab.py` | COCO names, YOLOE noun list, synonym table, `resolve_label()` |
| `vision/world.py` | `snapshot()` → detections with bearing + distance |
| `vision/vlm.py` | multi-image thinking-off VLM call |
| `vision/describe.py`, `find.py` | one-shot answer actions |
| `vision/speaker.py` | bearing getter registry, `refine_speaker_bearing()` |
| `vision/people.py` | candidate selection, raised hand, clothing check |
| `vision/navio.py` | `NavIO` protocol + `SpotNavIO` adapter (real robot) |
| `vision/goto.py`, `follow.py` | motion actions |
| `vision/tracker.py` | per-camera IoU+Kalman tracker, cross-camera hand-off |
| `scripts/setup_vision_models.py` | export env, ONNX export, TRT engine warm-build |
| `scripts/capture_vision_fixture.py` | save `GetImageResponse` frames for tests |
| `tests/voice_control/vision/test_*.py` | one test file per module |

---

### Task 1: Types and camera geometry (pure)

**Files:**
- Create: `src/voice_control/vision/__init__.py` (empty for now), `src/voice_control/vision/types.py`, `src/voice_control/vision/cameras.py`
- Test: `tests/voice_control/vision/__init__.py`, `tests/voice_control/vision/test_geometry.py`

**Interfaces:**
- Produces (`types.py`):
  - `@dataclass CameraFrame: source: str; image: np.ndarray` (upright, HxW or HxWx3 uint8) `; raw_rows: int; raw_cols: int; rot_deg: float; fx: float; fy: float; cx: float; cy: float; body_T_sensor: np.ndarray` (4x4) `; depth: np.ndarray | None` (raw-frame metres) `; jpeg: bytes; t: float; raw: np.ndarray | None = None` (raw-orientation image, used by the stitch)
  - `@dataclass Detection: label: str; conf: float; source: str; box: tuple[float,float,float,float]` (upright px) `; mask: np.ndarray | None` (upright, bool) `; keypoints: np.ndarray | None` (17x3 upright x,y,conf) `; bearing_deg: float; distance_m: float | None; distance_method: str | None` ("depth"|"ground"|None) `; track_id: int | None = None`
  - `@dataclass ActionResult: text: str; ok: bool; data: dict = field(default_factory=dict)`
- Produces (`cameras.py`):
  - `ROTATION_DEG: dict[str, float]` (Global Constraints values), `BODY_SOURCES` (5 fisheyes), `HAND_SOURCE = "hand_color_image"`
  - `upright(img: np.ndarray, source: str) -> np.ndarray` — `cv2.warpAffine` rotation with expanded canvas, same as PIL `rotate(angle, expand=True)`
  - `upright_to_raw(x: float, y: float, raw_rows: int, raw_cols: int, rot_deg: float) -> tuple[float, float]` and inverse `raw_to_upright(...)`
  - `pixel_bearing(frame: CameraFrame, x_up: float, y_up: float) -> float` — map to raw, unproject with pinhole K to a sensor ray, rotate by `body_T_sensor`, `atan2(y, x)` in degrees
  - `pixel_ray_body(frame, x_up, y_up) -> tuple[np.ndarray, np.ndarray]` (origin, unit direction in body frame) — shared by bearing and ground-plane code
  - `sector(bearing_deg: float) -> str` (eight sectors, 45° each, centred on 0, ±45, ±90, ±135, 180)
  - `distance_words(d: float | None) -> str` ("" for None, "right next to me" < 1.0, else "about N metres", N rounded, "one metre" singular)

- [ ] **Step 1: Write failing tests** in `test_geometry.py`:
  - `test_upright_to_raw_roundtrip_tilted`: for each source in `ROTATION_DEG`, for raw points (0,0), (639,479), (320,240), (100,400): `upright_to_raw(*raw_to_upright(p, 480, 640, rot), 480, 640, rot)` equals p within 1e-6.
  - `test_upright_shape`: `upright(np.zeros((480,640)), "frontleft_fisheye_image").shape == (726, 604)`; right → (480, 640) and pixel (0,0) of raw ends at (479,639).
  - `test_bearing_uses_transform_not_name`: synthetic `CameraFrame` named `"frontleft_fisheye_image"` with `body_T_sensor` = camera at origin, optical axis (sensor +z) pointing to body −y (right): centre pixel bearing ≈ −90 (± 0.5). Same frame with axis to +x: ≈ 0.
  - `test_sector_boundaries`: 0→"in front", 30→"front-left", 90→"on my left", −135→"behind-right", 180 and −179→"behind me", 22.4→"in front", 22.6→"front-left".
  - `test_distance_words`: None→"", 0.6→"right next to me", 1.2→"about one metre", 2.4→"about two metres".
- [ ] **Step 2: Run** `…/spot-env/bin/python -m pytest tests/voice_control/vision/test_geometry.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `types.py` and the `cameras.py` functions above. Raw↔upright: rotate about image centre by `rot_deg` CCW, then offset into the expanded canvas `(W', H') = (ceil(|w cos|+|h sin|), ceil(|w sin|+|h cos|))` — exactly the transform `upright()` uses (build the 2x3 matrix once in a helper `_rot_matrix(rows, cols, rot_deg)` and invert it for the reverse map). Salvage nothing else from the old branch.
- [ ] **Step 4: Run** tests → PASS.
- [ ] **Step 5: Commit** `git add src/voice_control/vision tests/voice_control/vision && git commit -m "vision: types, upright rotation, pixel->body bearing, sector wording"`

### Task 2: Capture, depth and ground-plane distance

**Files:**
- Modify: `src/voice_control/vision/cameras.py`
- Create: `scripts/capture_vision_fixture.py`
- Modify: `.gitignore` (add `models/vision/`, `tests/fixtures/vision/private/`)
- Test: `tests/voice_control/vision/test_capture.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `frame_from_response(resp, depth_resp=None) -> CameraFrame` (decodes JPEG, rotates, reads `source.pinhole.intrinsics`, `get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name_image_sensor).to_matrix()`; depth `PIXEL_FORMAT_DEPTH_U16` / `depth_scale` → metres, 0 → NaN)
  - `capture(robot, sources: list[str], with_depth: bool = False) -> dict[str, CameraFrame]` — one `get_image` call; depth source name = `source.replace("_fisheye_image", "_depth_in_visual_frame")` (verify names with `scripts/probe_image_sources.py` on the robot in Step 5; record actual names in a constant `DEPTH_SOURCE`)
  - `mask_distance(frame: CameraFrame, mask_up: np.ndarray | None, box_up) -> tuple[float | None, str | None]` — erode mask 3 px, map to raw, median of finite depth inside; < 20 valid pixels → ground plane from the box's bottom-centre pixel via `pixel_ray_body` intersected with plane z = `GROUND_Z` (body-frame floor height read from the snapshot: `get_a_tform_b(snapshot, BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME)` z; fall back to −0.5 m if missing); ray not hitting the floor within 15 m → (None, None)
- `scripts/capture_vision_fixture.py --out PATH [--host 192.168.50.3] [--with-depth]` writes a serialized `GetImageResponse` for all body sources + hand (read-only, no lease).

- [ ] **Step 1: Write failing tests** (fixture-free, synthetic frames):
  - `test_mask_distance_uses_depth_median`: depth array 2.0 m everywhere inside mask, 5.0 outside → (2.0, "depth").
  - `test_distance_zero_depth_falls_back_to_ground`: depth all NaN; camera 0.5 m above floor looking 30° down; box bottom at centre → ≈ 0.5/tan(30°) horizontal distance (± 0.05), method "ground".
  - `test_ground_ray_above_horizon_returns_none`: same camera looking up → (None, None).
  - `test_frame_from_saved_response` (skip if `tests/fixtures/vision/front_pair.bin` missing): loads, `image.shape == (726, 604)` for frontleft, `fx > 0`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the functions and the capture script.
- [ ] **Step 4: Run** → PASS (fixture test skipped until Step 5).
- [ ] **Step 5: On robot (operator or controller, read-only):** from the main checkout run `spot-env/bin/python /home/spotdog/spot/spot_vision/scripts/capture_vision_fixture.py --out /home/spotdog/spot/spot_vision/tests/fixtures/vision/front_pair.bin --with-depth` with the cameras pointed at an area with **no people**; re-run tests → fixture test PASS. Confirm depth source names and fix `DEPTH_SOURCE` if needed.
- [ ] **Step 6: Commit** (fixture included only if people-free) `git commit -m "vision: one-call capture, mask depth distance with ground-plane fallback"`

### Task 3: Front stitch

**Files:**
- Create: `src/voice_control/vision/stitch.py`
- Test: `tests/voice_control/vision/test_stitch.py`, fixture `tests/fixtures/vision/stitch_ref.png` (BD OpenGL output for `front_pair.bin`, generated with spike `bd_stitch_once.py` in its isolated venv)

**Interfaces:**
- Consumes: Task 2 `CameraFrame` (`raw` field, filled by `frame_from_response`).
- Produces: `stitch_front(fr: CameraFrame, fl: CameraFrame, size=(1080, 720)) -> np.ndarray` (uint8 grey); maps cached by a key of both sources' intrinsics + `body_T_sensor` rounded to 1 mm / 0.1°.

- [ ] **Step 1: Failing tests:** `test_stitch_matches_bd_reference` (skip without fixtures): mean abs diff < 1.0 and identical zero-pixel count vs `stitch_ref.png`; `test_stitch_maps_cached`: second call with same frames does not rebuild (patch the builder, assert called once); `test_stitch_time`: second call < 60 ms.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** by porting `docs/superpowers/specs/2026-10-29-vision-spikes/cv_stitch.py` (`build_maps`, `stitch`, `score`), keeping BD's sum-of-positions eye, `DIST=2.0`, `FOVY=110`, the −0.5 sampling offset, `BORDER_REPLICATE` and exact division (no clamp).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "vision: OpenCV front stitch matching BD stitch_front_images"`

### Task 4: Vocabulary and model setup script

**Files:**
- Create: `src/voice_control/vision/vocab.py`, `scripts/setup_vision_models.py`
- Test: `tests/voice_control/vision/test_vocab.py`

**Interfaces:**
- Produces:
  - `COCO_NAMES: list[str]` (80, Ultralytics order), `YOLOE_NAMES: list[str]` (the 150 nouns from spike `export_yoloe.py`, same order — the ONNX class index depends on it)
  - `resolve_label(phrase: str) -> tuple[str, str] | None` → `("coco"|"yoloe", canonical_name)`; lower-case, strip articles ("a/an/the/some"), singularise trailing "s"/"es", synonym table `SYNONYMS = {"bag": "backpack", "rucksack": "backpack", "tv": "tv", "television": "tv", "telly": "tv", "mobile": "cell phone", "phone": "cell phone", "sofa": "couch", "bin": "trash can", "rubbish bin": "trash can", "people": "person", "someone": "person", "man": "person", "woman": "person", "kid": "person", "child": "person"}`; COCO preferred when a name is in both.
- `scripts/setup_vision_models.py [--env /mnt/ssd/vision-export-env] [--out models/vision]`: creates the export venv if missing (`python3 -m venv`, `pip install ultralytics onnx onnxslim onnxruntime`), downloads `.pt`, exports `yolo26l-seg`, `yolo26l-pose`, and `yoloe-26l-seg` with `set_classes(YOLOE_NAMES)` (import the list from `vocab.py` by path, not by package import), `imgsz=640, opset=17, simplify=True`; then, **in spot-env**, opens each ONNX once with the TRT provider (Task 5 `load_model`) to build engines into `models/vision/trt_cache/`. Idempotent: skip any file that exists. Prints total time.

- [ ] **Step 1: Failing tests:** `test_resolve_coco`: "a backpack"→("coco","backpack"), "Bags"→("coco","backpack"), "the people"→("coco","person"); `test_resolve_yoloe`: "whiteboard"→("yoloe","whiteboard"), "posters"→("yoloe","poster"); `test_resolve_unknown`: "unicorn"→None; `test_yoloe_names_match_spike`: `len(YOLOE_NAMES)==150`, first three `["person","door","stairs"]`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Run the setup script** (`spot-env/bin/python scripts/setup_vision_models.py`); expect 3 ONNX files + engines; re-run prints "skip" for all.
- [ ] **Step 6: Commit** `git commit -m "vision: vocabulary/synonyms and model export + TRT warm-build script"`

### Task 5: GPU detector runtime

**Files:**
- Create: `src/voice_control/vision/runtime.py`
- Test: `tests/voice_control/vision/test_runtime.py`

**Interfaces:**
- Consumes: Task 4 file layout.
- Produces:
  - `class Model: name: str; provider: str` ("trt"|"cuda"|"cpu"); `detect(img: np.ndarray, conf: float = 0.35) -> list[RawDet]`
  - `@dataclass RawDet: cls: int; conf: float; box: tuple` (input-image px) `; mask: np.ndarray | None; keypoints: np.ndarray | None`
  - `load_model(name: str, model_dir: Path = MODEL_DIR) -> Model` — providers tried in order TRT (fp16, engine cache + timing cache in `trt_cache/`) → CUDA → CPU; first that creates a session wins; result cached per name (thread-safe, `threading.Lock`).
  - `gpu_available() -> bool` (any loaded model on trt/cuda)
  - `letterbox(img, size=640) -> tuple[np.ndarray, float, tuple[int,int]]` (NCHW float32 0–1, RGB; grey images expanded to 3 channels; scale; pad)
  - Decode: YOLO26 end-to-end output `out0[0]` rows `[x1,y1,x2,y2,conf,cls, …extra]`; seg extra = 32 mask coeffs with `out1[0]` protos `(32,160,160)` → `sigmoid(coeffs @ protos)` upsampled to 640, cropped to box, > 0.5, un-letterboxed; pose extra = 17×3 keypoints in letterbox px. **Verify the column layout first** by printing `session.get_outputs()` shapes for each exported model and one decoded row against Ultralytics' own prediction on the same image in the export venv; write the confirmed layout as a comment.
  - Inference serialised per model with a lock (ORT sessions are thread-safe but TRT contexts are shared; keep it simple).

- [ ] **Step 1: Failing tests:**
  - `test_letterbox_grey_to_rgb`: (480,640) grey → shape (1,3,640,640), scale 1.0, pad (0,80)… values of padding = 114/255.
  - `test_decode_det_rows`: synthetic `out0` with two rows, conf 0.9 and 0.2, threshold 0.35 → one `RawDet`, box un-letterboxed correctly for scale 0.5 / pad (0,80).
  - `test_decode_seg_mask`: protos/coeffs constructed so mask is 1 inside a known square → mask True inside box only.
  - `test_decode_pose_keypoints`: keypoints scaled back like boxes.
  - `test_runtime_falls_back_to_cpu`: patch `ort.InferenceSession` to raise for TRT and CUDA providers → `load_model("yolo26l-seg").provider == "cpu"`, `gpu_available()` False.
  - `test_gpu_models_agree` (skip unless `models/vision/*.onnx` exist, the TRT/CUDA provider is present, and `tests/fixtures/vision/front_pair.bin` exists): on the fixture's `hand_color_image` frame, TRT vs CPU boxes within 1 px and the same classes, for each of the three models.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `git commit -m "vision: onnxruntime TensorRT runtime with seg/pose decode and CPU fallback"`

### Task 6: World snapshot

**Files:**
- Create: `src/voice_control/vision/world.py`
- Test: `tests/voice_control/vision/test_world.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5.
- Produces:
  - `detect_frames(frames: dict[str, CameraFrame], model: str = "yolo26l-seg", labels: set[str] | None = None, conf: float = 0.35) -> list[Detection]` — runs the model on each frame's upright image, names via `COCO_NAMES`/`YOLOE_NAMES` by model, filters `labels`, fills `bearing_deg` (box centre, or mask centroid when present), `distance_m/method` via `mask_distance`. Sorted by conf desc.
  - `snapshot(robot, sources: list[str] = BODY_SOURCES, model="yolo26l-seg", labels=None, with_depth=True) -> list[Detection]` = `capture` + `detect_frames`.
  - `dedupe(dets: list[Detection], bearing_tol=10.0, dist_tol=0.7) -> list[Detection]` — same label seen by two overlapping cameras collapses to the higher-conf one.

- [ ] **Step 1: Failing tests** with a fake `Model` (patch `load_model`) returning fixed `RawDet`s: `test_detect_frames_fills_bearing_and_distance`; `test_labels_filter`; `test_dedupe_overlapping_front_cameras` (person at bearing 2° from both front cams → one).
- [ ] **Step 2–4:** run FAIL → implement → PASS.
- [ ] **Step 5: Commit** `git commit -m "vision: world snapshot with bearings, distances and cross-camera dedupe"`

### Task 7: VLM client (multi-image, thinking off)

**Files:**
- Create: `src/voice_control/vision/vlm.py`
- Test: `tests/voice_control/vision/test_vlm.py`

**Interfaces:**
- Produces: `ask_vlm(prompt: str, images: list[bytes], system: str = SYSTEM_PROMPT, max_tokens: int = 200, timeout: float = 30.0) -> str | None` — POST to `f"{LLAMACPP_URL}/v1/chat/completions"` (import `LLAMACPP_URL` from `src.voice_control.brain.llamacpp_backend`), one user message with text + each image as a JPEG `image_url` data URI, `temperature 0.5`, `chat_template_kwargs {"enable_thinking": False}`, `cache_prompt True`; returns stripped content, None on error/empty. `SYSTEM_PROMPT` = the first-person rules of `LLMBrain.query_vlm`'s system prompt (copy text, including the stop-word rule). `encode_jpeg(img: np.ndarray, quality=85) -> bytes`.
- Hard rule in docstring: one view per call; describe sends stitch + gripper as two images of the **same** view (front), which is allowed.

- [ ] **Step 1: Failing tests** (patch `requests.post`): payload has `enable_thinking: False`, N images → N `image_url` parts, returns content; HTTP error → None; empty content → None.
- [ ] **Step 2–4.** **Step 5: Commit** `git commit -m "vision: thinking-off multi-image VLM client"`

### Task 8: Describe and find

**Files:**
- Create: `src/voice_control/vision/describe.py`, `src/voice_control/vision/find.py`
- Modify: `src/voice_control/vision/__init__.py`
- Test: `tests/voice_control/vision/test_describe.py`, `test_find.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `describe(robot, side: str = "front", question: str = "") -> ActionResult` — `side ∈ {"front","left","right","back","around"}`. Front: capture body + hand; stitch; include gripper image only if `gripper_faces_forward(frame)` (camera optical axis yaw within ±60° of body +x and pitch above −60°); one `ask_vlm` with YOLO front hints; concurrently `detect_frames` on left/right/back → `also_line(dets)` = at most 3, people first, by conf: "I can also spot a backpack behind me and a chair on my left." or "I can look around if you like." Side: one VLM call on that upright fisheye. Around: front + left + right + back sequentially, joined "In front, … On my left, …".
  - `gripper_faces_forward(frame: CameraFrame) -> bool`
  - `also_line(dets: list[Detection], max_items=3) -> str`
  - `find(robot, phrase: str) -> ActionResult` — `resolve_label`; COCO or YOLOE `snapshot` on all body sources + hand; found → "Yes, there's a {label} {sector}{, about N metres away}." (`data={"detections": [...]}`); not resolvable or not found → VLM yes/no on front stitch + gripper ("Answer yes or no first: do you see a {phrase}?"); VLM "no"/None → "I don't see one right now."; VLM yes → "I think I see a {phrase} in front of me, but I'm not certain."
  - Text templates collected in module-level constants so the stop-word test can import them.

- [ ] **Step 1: Failing tests** (fake capture/model/vlm):
  - `test_describe_front_sends_stitch_and_gripper` (2 images in one call) and `test_describe_front_drops_gripper_when_facing_down` (1 image).
  - `test_also_line_people_first_max_three`, `test_also_line_empty_offer`.
  - `test_describe_around_four_calls`.
  - `test_find_coco_hit_says_direction_and_distance`: backpack at bearing 170°, 2.1 m → "Yes, there's a backpack behind me, about two metres away."
  - `test_find_falls_back_to_vlm_no` → "I don't see one right now."
  - `test_templates_have_no_stop_words` (reuse the regex from `tests/voice_control/test_spoken_no_stop_words.py`).
- [ ] **Step 2–4.** **Step 5: Commit** `git commit -m "vision: describe (front stitch + gripper, also-spot line, sides, around) and find"`

### Task 9: Look at speaker

**Files:**
- Create: `src/voice_control/vision/speaker.py`
- Test: `tests/voice_control/vision/test_speaker.py`

**Interfaces:**
- Produces:
  - `set_speaker_bearing_getter(fn: Callable[[], float | None]) -> None`; `speaker_bearing() -> float | None` (None if no getter or getter raises).
  - `cameras_covering(bearing_deg: float) -> list[str]` — body sources whose horizontal FOV (from the last-seen frame transforms; static default table if none yet: front pair ±60° about 0, left 90, right −90, back 180, each ±50°) contains the bearing; at most 2.
  - `refine_speaker_bearing(robot, doa_deg: float | None, window_deg: float = 30.0) -> float | None` — None in → None out; no GPU (`gpu_available()` False) → doa unchanged; else `snapshot(robot, cameras_covering(doa), labels={"person"})`, people within ±window → bearing of the closest-in-bearing one; none → doa. Never raises (log and return doa).

- [ ] **Step 1: Failing tests:** `test_refine_picks_person_nearest_doa` (people at 12° and 40°, doa 25° → 12°), `test_refine_tie_prefers_higher_conf` (10° conf 0.6, 40° conf 0.9, doa 25° → 40°), `test_refine_no_person_returns_doa`, `test_refine_outside_window_returns_doa`, `test_refine_never_raises` (snapshot raises → doa), `test_getter_none`.
- [ ] **Step 2–4.** **Step 5: Commit** `git commit -m "vision: speaker bearing refinement from mic DoA + person detections"`

### Task 10: People selection (come to me, raised hand, clothing, side)

**Files:**
- Create: `src/voice_control/vision/people.py`
- Test: `tests/voice_control/vision/test_people.py`

**Interfaces:**
- Consumes: Tasks 6, 7, 9.
- Produces:
  - `parse_person_description(phrase: str) -> tuple[str | None, str | None]` → (clothing phrase, side sector) e.g. "person in the red shirt on my left" → ("red shirt", "on my left"); "me" → (None, None).
  - `hand_raised(kps: np.ndarray, min_conf: float = 0.5) -> bool` — COCO keypoints: left shoulder 5, right 6, left wrist 9, right wrist 10; true if either wrist y < same-side shoulder y (upright coords, smaller y = higher) with both conf ≥ min_conf.
  - `wait_for_raised_hand(robot, sources, timeout_s=5.0, frames_needed=3) -> Detection | None` — loop pose snapshots; per-person track by nearest bearing (≤ 10°); exactly one person with 3 consecutive raised frames → that person; several → None.
  - `wearing(robot, person: Detection, clothing: str, gaze: Callable[[float], bool]) -> bool | None` — `gaze(person.bearing_deg)`; False → None; wait 2.5 s; capture hand; crop person box (gripper YOLO person nearest image centre); `ask_vlm("Answer yes or no: is this person wearing {clothing}?", [crop])` → True/False/None.
  - `select_person(robot, phrase: str, speak: Callable[[str], None], gaze: Callable[[float], bool]) -> tuple[Detection | None, str | None]` → (target, message to speak if ending). Implements spec 6.4: "me" → candidates within ±30° of `speaker_bearing()`; 1 → it; >1 → speak "Can you raise your hand?" + `wait_for_raised_hand`; still ambiguous → (None, "I can see a few people. Tell me what you're wearing and I'll come to you."); none → (None, "I can't see you. Could you step in front of me?"). Clothing → `wearing` over ≤ 3 candidates nearest the speaker bearing (or sector if given); exactly one True → it; else (None, "I'm not sure who you mean."). Side only → candidates in that sector, nearest; >1 → (None, "Do you mean the person on my left or on my right?") with the two sectors filled in.

- [ ] **Step 1: Failing tests:** `test_hand_raised_rules` (conf below 0.5 → False; right wrist above right shoulder → True; wrist above opposite shoulder only → False); `test_select_me_single_candidate_no_question` (speak not called); `test_select_me_two_candidates_asks_hand` (speak called with "Can you raise your hand?"); `test_select_me_ambiguous_ends_with_clothing_prompt`; `test_select_clothing_max_three_gazes`; `test_select_clothing_arm_disabled` (gaze returns False → asks raised hand); `test_parse_person_description` (4 cases).
- [ ] **Step 2–4.** **Step 5: Commit** `git commit -m "vision: person selection (speaker bearing, raised hand, clothing, side)"`

### Task 11: NavIO and go-to

**Files:**
- Create: `src/voice_control/vision/navio.py`, `src/voice_control/vision/goto.py`
- Test: `tests/voice_control/vision/test_goto.py`

**Interfaces:**
- Produces (`navio.py`):
  - `class NavIO(Protocol): def snapshot(self, sources: list[str], labels: set[str] | None, model: str) -> list[Detection]; def odom_pose(self) -> tuple[float, float, float]` (x, y, yaw rad in odom) `; def move_body(self, x: float, y: float, yaw: float) -> bool` (body-frame goal, trajectory, returns False if aborted/failed) `; def velocity(self, vx: float, vy: float, vrot: float) -> bool; def halt(self) -> None; def aborted(self) -> bool; def speak(self, text: str) -> None`
  - `class SpotNavIO(NavIO)`: built by the dispatcher hook with `(session, gen, speak)`; every `move_body`/`velocity` checks `spot_dispatch._aborted(gen)` **immediately before** `robot_command` and returns False if aborted; uses `RobotCommandBuilder.synchro_trajectory_command_in_body_frame` / `synchro_velocity_command` with `capped_mobility_params()` (obstacle avoidance left on); `halt()` sends `stop_command`.
- Produces (`goto.py`):
  - `goto(io: NavIO, target: Detection, label: str, is_person: bool, stop_event: threading.Event, rate_hz: float = 2.0) -> ActionResult`
  - Loop (spec 6.4): turn to bearing; goal = distance − (1.5 if person else 1.0); `move_body` toward goal; every 0.5 s re-`snapshot` on `cameras_covering(bearing)` for `label`, nearest to predicted bearing (≤ 25°) updates target; outcomes with exact texts: arrived (|distance − stop| ≤ 0.3) → "I'm here." ok=True; unseen 3 s → "I lost sight of it."; odom progress < 0.1 m over 4 s while goal > 0.3 m away → "Something's in the way."; `stop_event` set or `io.aborted()` → "I paused." (and no further `move_body`/`velocity` call). Distance None (no depth, no ground) → walk 1 m steps toward bearing and re-measure, max 8 steps.
  - Does **not** speak; returns text. The caller speaks (Task 13).

- [ ] **Step 1: Failing tests** with a `FakeNavIO` (scripted detections per call, simulated odom advancing on `move_body`):
  - `test_goto_arrives_object_stops_one_metre` → ok True, text "I'm here.", final commanded goal leaves 1.0 m.
  - `test_goto_person_stops_one_and_half`.
  - `test_goto_lost_after_three_seconds` (fake clock).
  - `test_goto_no_progress_says_in_the_way`.
  - `test_goto_no_command_after_abort`: `aborted()` flips True after the 2nd `move_body` → no 3rd motion call, text "I paused.", ok False.
  - `test_goto_texts_no_stop_words`.
- [ ] **Step 2–4.**
- [ ] **Step 5: safety-reviewer** on `navio.py` + `goto.py` (abort check placement, halt path, obstacle avoidance not disabled, speed caps). Fix findings, re-run tests.
- [ ] **Step 6: Commit** `git commit -m "vision: NavIO adapter and go-to loop with honest outcomes"`

### Task 12: Tracker and follow

**Files:**
- Create: `src/voice_control/vision/tracker.py`, `src/voice_control/vision/follow.py`
- Test: `tests/voice_control/vision/test_tracker.py`, `test_follow.py`

**Interfaces:**
- Produces (`tracker.py`): `class CameraTracker: update(dets: list[Detection], t: float) -> list[Detection]` (assigns `track_id`; IoU ≥ 0.3 match + constant-velocity Kalman on box centre; tracks die after 1 s unseen). `handoff(last: Detection, candidates: list[Detection], max_dist=1.0, max_bearing=30.0) -> Detection | None` (body-frame position from bearing+distance; nearest within both limits). Builder may wrap Ultralytics' `BYTETracker` instead only if it imports without torch; otherwise this tracker.
- Produces (`follow.py`): `follow(io: NavIO, target: Detection, stop_event, rate_hz=10.0, keep_m=2.0, max_images_per_s=20) -> ActionResult` — velocity control toward keeping `keep_m` (forward speed ∝ (d − keep), yaw rate ∝ bearing, both clipped by existing caps), detections on front pair + the camera the target was last in, budget-limited; lost 3 s → `io.halt()`, "I lost you. Say follow me to begin again."; abort → "I paused."; refuses immediately with "My vision processor isn't ready, so I can't follow right now." when `gpu_available()` is False.

- [ ] **Step 1: Failing tests:** tracker keeps ID across 5 frames of a moving box; ID dies after 1 s; `test_handoff_picks_nearest_within_limits`; `test_follow_keeps_distance` (fake person at 3 m → positive vx; at 2 m → ~0); `test_follow_lost_after_three_seconds`; `test_follow_no_command_after_abort`; `test_follow_refuses_on_cpu`; `test_follow_respects_image_budget` (count snapshot images per simulated second ≤ 20).
- [ ] **Step 2–4.**
- [ ] **Step 5: safety-reviewer** on `follow.py` (same checklist as Task 11 + velocity command expiry: each velocity command `end_time` ≤ 0.6 s so a dead loop stops the robot).
- [ ] **Step 6: Commit** `git commit -m "vision: per-camera tracker, hand-off and follow loop"`

### Task 13: Hooks into the voice pipeline

**Files:**
- Modify: `src/voice_control/grammar/spot_action.gbnf` (add `"\"find\""` and `"\"look_around\""` to `action-name`)
- Modify: `src/voice_control/llm_brain.py` (action list: `describe` gains `side` param `"front"|"left"|"right"|"back"`; new `find` {"what": "<object or person>"}; new `look_around` {}; rule at line ~138 changed: "Do you see X?/Is there a X?/Where is the X?" → `find`; examples for single-turn answers: "I'm the one in the red shirt" → `go_to_object` {"description": "person in red shirt"}, "the one on the left" → `go_to_object` {"description": "person on my left"}, "come to me" → `go_to_object` {"description": "me"})
- Modify: `src/voice_control/client_mic.py`:
  - describe special case (~2736–2748): match any of `describe`, `find`, `look_around` (stock acknowledgement, Gemma's reply not spoken)
  - describe execution branch (~2805): replace `capture_frame` + `detect_in_image` + `query_vlm` with `vision.describe(robot, side, question)` / `vision.find(robot, what)` / `vision.describe(robot, "around")`; speak `result.text`; robot from `spot_dispatch.ensure_spot_session()["robot"]`
  - `_gaze_loop` (~2398): `body = vision.refine_speaker_bearing(robot, body)` before `gaze_at_bearing` (wrapped in try; on error use the raw bearing)
  - startup: `vision.set_speaker_bearing_getter(lambda: _status.get("locked_deg"))`
  - chain runner (~2872): after `_wait_for_nav_complete()` for `go_to_object`/`follow_me`, if `spot_dispatch.last_nav_ok() is False` → print and `break`
- Modify: `src/voice_control/spot_dispatch.py`:
  - `go_to_object` step 3: description → if person phrase ("me", "person…", clothing/side words) use `people.select_person`, else `resolve_label` + `world.snapshot`; not found → speak the message and return False; found → `_spawn_nav(lambda ev: _thread_then_restore(lambda: _vision_nav(goto…), ev), mode="visual_nav", gen=_dispatch_gen)` where `_vision_nav` runs `goto`, speaks `result.text` via the `speak` passed into `dispatch_intent`, stores `_last_nav_ok = result.ok`, returns `result.ok`
  - `follow_me`: same with `select_person("me")` + `follow`, mode "follow"
  - `last_nav_ok() -> bool | None`
  - `find`/`look_around` registered as vision-answer actions (no motion; dispatcher returns True so client_mic handles them like describe)
- Test: `tests/voice_control/vision/test_hooks.py`; extend `tests/voice_control/test_prompt_layout.py` only if it asserts the action list.

**Interfaces:**
- Consumes: `vision.describe`, `vision.find`, `vision.refine_speaker_bearing`, `vision.set_speaker_bearing_getter`, `people.select_person`, `goto.goto`, `follow.follow`, `navio.SpotNavIO`.

- [ ] **Step 1: Failing tests:** grammar contains `"find"` and `"look_around"` (parse the GBNF text); `llm_brain` prompt lists `find`; `test_find_reply_not_spoken` (client_mic describe-special-case helper returns the stock acknowledgement for a `find` action — extract the matching into a small function `_vision_answer_action(actions)` so it is testable); `test_chain_breaks_on_failed_goto` (patch `last_nav_ok` False → following action not executed); `test_gaze_loop_uses_refined_bearing` (patch `refine_speaker_bearing` → `gaze_at_bearing` called with refined value; refine raising → raw value).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the hook edits (each a few lines; no logic beyond calling the package).
- [ ] **Step 4: Run the full suite** `…/spot-env/bin/python -m pytest -q` → all pass (baseline 615 + new).
- [ ] **Step 5: safety-reviewer** on the `spot_dispatch.py` and `client_mic.py` diffs (gen passed at request time, `_spawn_nav` used so stop cancels, no vision call on the audio thread).
- [ ] **Step 6: Commit** `git commit -m "vision: wire describe/find/look_around, go-to, follow and speaker gaze into the voice pipeline"`

### Task 14: On-robot verification and docs

**Files:**
- Create: `docs/project/vision-robot-checklist.md`
- Modify: `docs/project/status.md` (vision rows), `docs/project/bugs.md` (strike D1, D2, D3, D5, D6, B9, B10 when verified)

- [ ] **Step 1:** Write the checklist from spec section 11 (one line each, expected spoken text and pass condition).
- [ ] **Step 2:** Controller runs read-only checks on the robot (describe front/sides/around, find COCO + YOLOE object, directions per side). Record results in the checklist.
- [ ] **Step 3:** Operator runs motion checks (go-to object 1.0 m, come-to-me one/two people, person in red, follow + lost message, stop/freeze during go-to and follow). Record results.
- [ ] **Step 4:** Update status/bugs for verified items only.
- [ ] **Step 5: Commit** `git commit -m "docs: vision robot checklist results, status and bug register"`
- [ ] **Step 6:** Whole-branch review (`/code-review high` against `81fedfe`) before any transplant.
