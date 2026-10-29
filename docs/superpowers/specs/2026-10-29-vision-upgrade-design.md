# Vision upgrade: design

Date: 2026-10-29. Branch: `vision/perception-v2` (from demo HEAD `81fedfe`), worktree `/home/spotdog/spot/spot_vision`.

## 1. Intent

Spot's vision is the weakest part of the voice robot. "What do you see" sends one sideways fisheye image to Gemma. "Go to X" and "follow me" run old YOLO models on the CPU, turn in fixed steps, and report success when a thread starts. The operator wants vision to be dependable and clearly defined: a small set of named vision actions, each with a clear behaviour, running the best current YOLO models on the Jetson GPU, wired into the voice pipeline.

The work is built as a **self-contained vision package** with **thin hooks** into the existing code. The operator is restructuring the codebase separately; once that lands, the package is transplanted into the new layout. So the package must not depend on the internals of `client_mic.py` or `spot_dispatch.py` beyond a few named call points (section 9).

## 2. Goals

1. Five clearly defined vision actions: **describe**, **find**, **look at speaker**, **go to** (object or person), **follow**.
2. All YOLO inference on the GPU through TensorRT, using the YOLO26 family (large size).
3. Every image upright and every detection placed in the robot's body frame (direction, and distance where possible), so Spot can say "behind me on the left" correctly.
4. Spoken, honest outcomes: Spot says what it found, where, and whether it arrived or lost the target.
5. No regression in safety: stop/freeze, obstacle checks, speed caps and the dispatch generation checks keep working exactly as on the demo branch.

## 3. Non-goals

- Semantic segmentation, classification and OBB models (benchmarked reasoning in section 5).
- Cross-camera person re-identification by appearance (follow hands off between cameras by bearing and distance only).
- Following on stairs. Follow stops if the person goes out of reach; stairs remain the operator's responsibility.
- Changing the speech, wake, speaker-lock or LLM text paths, except the hooks listed in section 9.
- A JetPack upgrade or any change to spot-env's pinned packages.
- Mapping objects into GraphNav or publishing them to the tablet.

## 4. Anti-goals (would be a failure even if it works)

- Any vision path that can move the robot after a stop/freeze, or with Spot's built-in obstacle avoidance turned off (see 8).
- Spot guessing: claiming to see something YOLO and the VLM did not see, or saying "I'm there" when it is not.
- Gemma replies becoming noticeably slower during conversation because of background vision load.
- Installing packages into spot-env (numpy/onnxruntime-gpu/opencv/kokoro pins are fragile; `pin-guardian` applies).
- Wrong directions: an object on the left reported as on the right (the front fisheyes are cross-mounted; use the camera transforms, never name-based guesses).

## 5. Decisions already made (with evidence)

### 5.1 Models

| Role | Model | Measured on Orin (TensorRT FP16, alone) |
|---|---|---|
| Main detector: people + 80 COCO classes, with outlines | `yolo26l-seg` | 13.6 ms/image |
| Raised hand, body orientation (on demand) | `yolo26l-pose` | 10.1 ms/image |
| Open-vocabulary "find X" beyond COCO | `yoloe-26l-seg`, fixed noun list baked in at export | not yet measured; expect ~15 ms |

- Large size chosen by the operator; `l` costs ~4 ms more than `s` per image.
- Instance segmentation is used for **distance**: depth is read only inside the object's outline, not its whole box.
- YOLOE prompt-free (4,585 built-in names) and visual-prompt variants are optional later additions, not in this build.

### 5.2 Runtime: our own onnxruntime + TensorRT

- spot-env already has `onnxruntime-gpu` 1.23 with the TensorRT and CUDA providers, and TensorRT 10.3 is on the Jetson. Verified: TensorRT, CUDA and CPU give identical detections (boxes within 0.3 px).
- Ultralytics itself cannot use the GPU in spot-env (its torch is CPU-only; it picks providers from `torch.cuda.is_available()`), and loading an `.onnx` through Ultralytics auto-installs packages into the venv. **The runtime must not import Ultralytics for inference.** Pre-processing (letterbox) and decoding (boxes, masks, keypoints) are our code. YOLO26 is NMS-free (fixed `[1,300,…]` output), so decoding is simple.
- Native Ultralytics on NVIDIA's Jetson torch also works (isolated venv) but is 3–4× slower (l-seg 50 ms). Rejected.
- Export `.pt` → `.onnx` happens **off spot-env**, in a separate throwaway venv with current Ultralytics + `onnx` + `onnxslim` (method proven in the spike). YOLOE classes are set with `set_classes()` before export; changing the noun list means re-exporting.
- TensorRT engines are built on first use and cached on disk. A first build takes 1–8 minutes per model, so a setup script builds them ahead of time (section 10).

### 5.3 GPU budget

Measured with Gemma (llama-server) generating at the same time:

| YOLO images/s | Gemma speed |
|---|---|
| 0 | 22.5 tok/s |
| 10 | 20.5 (−9%) |
| 30 | 15.0 (−33%) |
| 50+ | 11.5 (−49%) |

- **One-shot work** (describe, find, choosing a go-to target): unlimited burst, e.g. 5 images ≈ 130 ms.
- **Continuous work** while Spot may be talking (look at speaker): at most **10 images/s total**.
- **Follow and go-to tracking**: up to 20 images/s total (Spot talks little while moving). Configurable.
- With Gemma busy, YOLO takes ~25 ms/image instead of 10–14 ms.

### 5.4 Images and geometry

- Capture all body cameras (5 greyscale fisheyes) and the gripper colour camera in **one** `get_image` call (measured 0.02–0.03 s), JPEG, plus depth where needed.
- **Upright rotation** uses Boston Dynamics' own table (degrees CCW): back 0, frontleft −78, frontright −102, left 0, right 180, hand 0. Verified on the robot: all six images upright.
- **Front stitch**: OpenCV re-implementation of BD's `stitch_front_images` example (virtual camera between the two front lenses, plane 2 m ahead, 110° vertical FOV, BD blend weights, GL-style sampling). Verified identical to BD's OpenGL output on the same frames (mean difference 0.1 grey level, identical uncovered area). 20 ms per frame; lookup maps built once (165 ms) and rebuilt only if calibration changes. Spike code: `2026-10-29-vision-spikes/cv_stitch.py` next to this doc (copy into the package). Benchmarks and other spikes are in the same folder.
- **Direction of a detection**: computed from the camera's intrinsics (the robot reports pinhole intrinsics for every source) and its body transform from the image's `transforms_snapshot`, never from the camera's name. Run detection on the **upright** image and map pixels back to the original image before projecting (see the `fisheye-coords` skill).
- **Distance**: median of body-camera depth pixels inside the segmentation mask. If depth is missing or the object is beyond depth range, fall back to the **ground-plane** estimate (ray through the bottom of the mask intersected with the floor plane from the body transform). Report which method was used.
- Body cameras are greyscale: **colour words only come from the gripper camera**.

### 5.5 VLM calls

- Vision VLM calls must disable Gemma's hidden thinking (`chat_template_kwargs: {"enable_thinking": false}`). Measured: with thinking on, an 80-token budget returned nothing; with it off, 1.1–1.9 s per image.
- One VLM call per view. Sending six labelled images in one call took 10.9 s and described only one direction.
- Gemma sees each image at ~150 tokens, so its descriptions are coarse; YOLO supplies precise object names and directions as hints.

## 6. The five actions

Each action is a function in the vision package returning a result object (spoken text + structured data). The dispatcher/voice loop speaks the text.

### 6.1 Describe ("what do you see", "what's behind you", "look around")

- **Plain "what do you see"**:
  1. Capture all cameras.
  2. One VLM call with the **stitched front image** plus the **plain gripper image** (no gripper opening). Include the gripper image only if its camera points roughly forward, judged from its transform: yaw within ±60° of the body's forward axis and not pointing at the robot's own body or straight down. Otherwise send the stitch alone.
  3. In parallel, YOLO on the left, right and back cameras. Add one short sentence naming at most three detections, people first, highest confidence, with directions: "I can also spot a backpack behind me and a chair on my left." If nothing is found: "I can look around if you like."
  4. YOLO results on the front views are passed to the VLM as a hint (existing `yolo_hint` pattern) so names are precise.
- **"What do you see in front / on your left / behind you"**: one VLM call on that side only (front = stitch + gripper; left/right/back = that fisheye, upright).
- **"Look around"**: VLM on front (stitch + gripper), left, right and back in sequence, ~7 s. One combined spoken answer, organised by direction.
- **"Do you see X?"** routes to **find** (6.2), not describe.
- Keep the existing first-person prompt rules (never "image/photo", never stop words).

### 6.2 Find ("do you see a backpack?", "where is the door?")

- No movement.
- Run the COCO detector on all cameras; if X is not a COCO class, run YOLOE on all cameras.
- **YOLOE does not block the build.** If its ONNX/TensorRT export is not working yet, `find` ships with COCO classes + the VLM fallback, and YOLOE is added when proven.
- If found: answer with direction and, when known, distance: "Yes, there's a backpack behind me on the right, about two metres away."
- If not found by either model: VLM fallback on the stitched front + gripper image with the question. If the VLM also says no: "I don't see one right now."
- Map the spoken noun to a model class with a small synonym table (e.g. "bag" → backpack/handbag, "TV" → tv) plus case/plural normalisation.

### 6.3 Look at speaker (fuses mic and camera)

- Today: the mic's direction-of-arrival (DoA) bearing drives the folded arm's shoulder yaw (`gaze_at_bearing`, clamps and safety already in the demo code).
- New: when a DoA bearing arrives, run the COCO detector on the cameras covering that bearing (1–2 cameras, within the 10 images/s budget). Among people whose bearing is within ±30° of the DoA bearing, pick the closest in bearing. Send the gaze **that person's bearing** instead of the raw DoA bearing.
- No person within ±30°: use the DoA bearing unchanged (today's behaviour).
- Detections older than 1 s are not used. Keep the existing gaze rate limits, clamps (±60°) and the request-time safety generation.
- Pose (shoulders) to judge which way a person faces is optional and not in this build.

### 6.4 Go to (object or person)

Common flow:
1. **Pick the target** (below). Not found: "I can't see that from here" and do nothing (no wandering).
2. **Turn** toward the target's bearing.
3. **Measure** distance (5.4). Goal = point on the line to the target, **1.0 m short for objects, 1.5 m short for people**.
4. **Walk** using the existing trajectory/velocity approach with Spot's built-in obstacle avoidance and the existing speed caps (see 8).
5. **Re-check** with YOLO about every 0.5 s on the cameras covering the target (tracking budget, 5.3) and update the goal.
6. **Stop and speak** the outcome:
   - "I'm here." (within 0.3 m of the goal distance)
   - "I lost sight of it." (target not seen for 3 s)
   - "Something's in the way." (no progress toward the goal for 4 s while walking; Spot's obstacle avoidance is holding it back)
   - "I paused." (stop/freeze/cancel; never say the word "stop")
   - **Threading (fixes bug B10):** go-to and follow keep running on the existing nav thread (`_spawn_nav`), so the voice loop and stop/freeze stay responsive. The nav thread speaks the outcome itself when it ends, through the same speak path the dispatcher already uses (queued behind any speech in progress, like other nav announcements). The thread records success/failure; a command chain ("go to the chair and sit") checks it after `_wait_for_nav_complete()` and drops the remaining actions if the go-to failed.

**Target selection**
- Saved map location or fiducial/dock: unchanged (existing `go_to_object` steps 1–2).
- Object: as in **find**. If several match, take the closest; say which ("the chair in front of me").
- "Come to me": candidates = people within ±30° of the locked speaker's DoA bearing.
  - Exactly one: go.
  - More than one: Spot asks "Can you raise your hand?" and runs `yolo26l-pose` on those cameras for up to 5 s (no spoken reply needed; this stays inside the action). A raised hand = wrist keypoint above the same-side shoulder keypoint, in **upright** image coordinates, keypoint confidence ≥ 0.5, for 3 consecutive frames. One person raising: go. Nobody or several: Spot ends the action with "I can see a few people. Tell me what you're wearing and I'll come to you."
  - No person seen: "I can't see you. Could you step in front of me?"
- "The person in the red shirt" (also how the clothing answer comes back, see below):
  1. Aim the gripper camera at each candidate in turn using the existing `gaze_at_bearing` (folded arm, ±60° shoulder yaw clamp; candidates beyond the clamp are skipped). Its rate limit makes each aim ~2 s, so at most **3 candidates**, nearest the speaker's bearing first. If the arm is disabled (`--no-arm`) or the gaze refuses, skip this step and ask for a raised hand instead.
  2. Crop the person from the gripper image and ask the VLM a yes/no question ("Is this person wearing a red shirt?").
  3. Exactly one yes: go. Otherwise: "I'm not sure who you mean."
- **Follow-up questions are single-turn (operator decision).** When Spot needs more information ("Tell me what you're wearing", "Do you mean the person on my left or on my right?"), it asks and **ends the action**. The user's answer comes back as a fresh command through Gemma, e.g. "I'm the one in the red shirt" → `go_to_object` with description "person in red shirt"; "the one on the left" → `go_to_object` with description "person on my left". No dialogue state is added to the voice loop. This needs: prompt examples in `llm_brain.py` for these answers, and `go_to_object` accepting a person description with a clothing phrase and/or a side ("left/right/in front/behind"), which narrows candidates to that sector.
- **Confirmation**: only when more than one candidate remains after selection does Spot ask (single-turn, as above). With one candidate, it just goes.

### 6.5 Follow ("follow me")

- Target = the requesting speaker, selected exactly as "come to me" (6.4).
- Track with ByteTrack, one tracker per camera, on our own detections. Ultralytics' `BYTETracker` class can be reused if it works on plain numpy detections without importing torch-backed inference; otherwise a small IoU + Kalman tracker. Builder's choice.
- Hand-offs between cameras: when the tracked person leaves one camera, take the person in another camera whose body-frame bearing and distance are closest to the last known position (within 1 m and 30°).
- Keep **2.0 m** behind (velocity control at 10 Hz as today), with existing speed caps and Spot's built-in obstacle avoidance (see 8).
- Lost for **3 s**: halt, say "I lost you. Say follow me to begin again," and end follow mode. Outcome spoken by the nav thread as in 6.4.
- Detection rate: up to 20 images/s total across the cameras that matter (front pair + the camera the person was last in).

## 7. Speech and wording rules

- First person, short, no "image/camera/photo".
- Never the words stop, halt, freeze or e-stop in anything Spot says (the safety listener hears Spot's own voice). "Paused" / "I'm holding still" instead. Applies to all new templates and to the VLM system prompt (already in place).
- Directions use eight sectors relative to Spot's body: in front, front-left, on my left, behind-left, behind me, behind-right, on my right, front-right.
- Distances in whole metres ("about two metres"); under 1 m: "right next to me".

## 8. Safety

- The safety-reviewer agent must review every change that touches motion (go-to, follow, gaze changes).
- Every motion step checks the dispatch generation (`_dispatch_gen` / the gen passed in) before sending a command, as the demo code does; a stop/freeze invalidates it.
- **Obstacles (operator decision): same as today.** Go-to and follow rely on Spot's built-in obstacle avoidance (on by default in trajectory and velocity commands; never disable it in mobility params) plus the existing speed caps. The repo's own forward check (`_forward_collision_blocks`, gated by `SPOT_COLLISION_PRECHECK`, fail-open) stays as it is and is not relied on. "Something's in the way" is detected as lack of progress (6.4). No new motion primitives.
- Gaze for "person in red" uses the existing `gaze_at_bearing` path (clamps, rate limits, request-time generation).
- If the GPU runtime fails to start, vision falls back to CPU for one-shot actions (describe/find, ~0.2 s/image for `s`); **follow and continuous speaker tracking are disabled on CPU** and Spot says so.

## 9. Package layout and hooks

New package `src/voice_control/vision/` (name final at transplant time). Suggested modules, builder may adjust:

| Module | Responsibility |
|---|---|
| `cameras.py` | capture all sources in one call, upright rotation, pixel ↔ original mapping, pixel → body bearing, depth lookup, ground-plane fallback |
| `stitch.py` | front stitch (maps cached per calibration) |
| `runtime.py` | onnxruntime TensorRT sessions, engine cache, letterbox, decode seg/pose/YOLOE, CPU fallback |
| `world.py` | snapshot: detections from N cameras → list of objects with class, confidence, bearing, distance, camera |
| `describe.py`, `find.py`, `speaker.py`, `goto.py`, `follow.py` | the five actions |
| `vocab.py` | synonym table, YOLOE noun list |

**Hooks into existing code (kept as small as possible, each a few lines):**
1. `client_mic.py` vision-answer branch: the existing describe special case (~lines 2736–2748 drop Gemma's own reply and speak a stock acknowledgement; ~2805 runs the VLM) is extended to **all vision-answer actions: `describe`, `find`, `look_around`**, each calling the vision package. Gemma's own reply text is never spoken for these (it would be a guess).
2. `client_mic.py` `_gaze_loop` (the gaze worker thread, **not** the audio path): pass the DoA bearing through `vision.speaker.refine(bearing)` before calling `gaze_at_bearing`. This is a **one-off** YOLO check per gaze request (~50–100 ms on the worker), not a background detector, so there is no constant GPU load during conversation.
6. A getter for the locked speaker's bearing that the vision package can call from the dispatcher side (e.g. `client_mic` registers a callable with the vision package at startup). It returns the **latest** lock angle (the lock is nudged on each accepted utterance), or None when no speaker is locked.
7. Nav outcome: `vision.goto` / `vision.follow` return success/failure from the nav thread; the chain runner checks it after `_wait_for_nav_complete()` (6.4).
3. `spot_dispatch.py` `go_to_object` step 3 and `follow_me`: call `vision.goto` / `vision.follow` instead of `visual_nav`.
4. New action `find` (and `look_around`, or a `side` parameter on `describe`): add to `grammar/spot_action.gbnf`, the action list and examples in `llm_brain.py`, and the dispatcher. The GBNF grammar **must** be updated or the LLM silently misfires. Update the prompt rule that routes "do you see X" to describe so it routes to `find`.
5. VLM: a multi-image, thinking-off VLM call. Add it in the vision package, posting to the llama-server URL taken from the existing backend, so `llamacpp_backend.py` stays unchanged. The live describe path also gets thinking off through this.

`visual_nav.py` stays in place (unused by the new paths) until the transplant, then is deleted.

## 10. Setup and files

- `scripts/setup_vision_models.py`:
  1. Creates the throwaway export venv (outside the repo, e.g. `/mnt/ssd/vision-export-env`).
  2. Downloads the `.pt` files and exports the ONNX files (YOLOE with the noun list).
  3. Builds and caches the TensorRT engines by running each model once in spot-env.
  4. Idempotent: skips existing files.
- Models and engine cache live in `models/vision/` (gitignored). No `.pt`/`.onnx` in git.
- Salvage from branch `vision/w5-w6-capture-gaze`: the rotation table and the `arm_is_deployed` check from `vision_camera.py`. Its `speaker_gaze.py` and gaze changes are superseded by the demo branch's gaze and are not brought over.

## 11. Testing

- Unit tests (no robot, no GPU) for: rotation and pixel mapping, bearing from intrinsics + transform, stitch maps against a saved BD reference frame, decoding of seg/pose/YOLOE outputs from saved model outputs, raised-hand rule, sector wording, synonym mapping, follow hand-off choice, outcome wording without stop words.
- Fixtures: saved `GetImageResponse` frames without people may be committed. Frames with people stay gitignored, matching how operator audio is handled.
- GPU integration test (skipped without GPU): each model loads, decodes a saved frame, and agrees with CPU output.
- On-robot checklist (operator): describe front/side/look-around; find a COCO and a non-COCO object; directions correct for each side; look-at-speaker with two people; come-to-me with one and two people (raised hand); person in red; go-to object stops at 1.0 m; follow at 2 m and loses-you message; stop/freeze during go-to and follow.

## 12. Left to the builder

- Exact module split and names inside the package.
- ByteTrack reuse vs a small own tracker.
- Letterbox/decode implementation details, ONNX opset, input size (640 default).
- Whether engines warm up at startup or on first use (startup must not block the voice loop).
- Synonym table contents and the YOLOE noun list (start with ~150–200 common indoor/lab/event nouns: door, stairs, poster, table, chair, backpack, laptop, bottle, cup, sign, whiteboard, plant, robot…).
- Thresholds not fixed above (detection confidence, mask erosion for depth sampling), tuned on saved frames.

## 13. Calls made by Claude (operator to check)

- Defaults: stop 1.5 m from a person and 1.0 m from an object; confirm only with multiple candidates; follow at 2.0 m; lost after 3 s.
- New branch `vision/perception-v2` from demo HEAD instead of continuing the old vision branch.
- OpenCV stitch over BD's OpenGL stitch.
- Gripper image included only when its camera points roughly forward.
- ±30° window to match a person to the mic bearing; 1 s detection freshness; 10 images/s budget while talking, 20 while moving.
- CPU fallback for one-shot actions only.
- YOLOE with a fixed noun list in this build; prompt-free and visual-prompt variants later. YOLOE does not block the build.
- Go-to/follow outcomes spoken by the nav thread; chains drop remaining actions after a failed go-to.
- `find` and `look_around` get describe's treatment (Gemma's own reply never spoken).
- Look-at-speaker refinement is a one-off check per gaze request on the gaze worker, not a background detector.
- Locked-speaker bearing comes from a registered getter returning the latest lock angle.
- "Person in red" uses the existing gaze path: ~2 s per aim, at most 3 candidates; arm disabled → raised-hand question instead.

## 14. Operator decisions (2026-10-29)

- Obstacles: option (a), Spot's built-in obstacle avoidance as today; "something's in the way" from lack of progress.
- Follow-up questions: option (a), single-turn; the answer returns as a fresh command through Gemma.
- Large models (`l`) for all YOLO roles.
- "What do you see" starts with the front (stitch + plain gripper image) and mentions what YOLO spots elsewhere.
