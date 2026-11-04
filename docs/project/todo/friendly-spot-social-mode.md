# Friendly Spot social mode: validation and next steps

Context: commit `a3c2a16` on `vision/friendly-social` (worktree `../spot_vision`, branched from `vision/perception-v2`). Not merged, not on the demo branch, **never run on the robot**. Started only by the voice intent `social_mode` ("social mode" / "friendly mode"); nothing changes unless someone says that.

What it is: the friendly_spot class project (`~/projects/friendly_spot`) ported onto the capstone vision stack.
- `vision/comfort.py`: comfort model copied unchanged (same weights and thresholds).
- `vision/social_pose.py`: the rule-based pose labels re-written for YOLO26-pose COCO-17 keypoints. Three original bugs fixed: motion compared a frame with itself (walking/running never fired), torso angle read ~180 deg when upright (standing never fired, bending fired for anyone far away), arms-crossed fired whenever the hands were level.
- `vision/social.py`: the 2 Hz loop. Surround fisheyes + `hand_color_image` with depth, YOLO26-pose, pick and track one person, yaw the folded arm toward them (the gripper replaces the Spot CAM PTZ), score, debounce (2 cycles), act.
- `vision/social_models.py`: YuNet face (CPU) + HSEmotion `enet_b0_8_va_mtl` emotion (replaces DeepFace) + HaGRIDv2 YOLOv10n gestures (replaces MediaPipe), TensorRT, models in `models/vision/social/` (gitignored). Loaded at startup warm-up, never inside a dispatch.
- Dispatcher: `social_mode` is a locomotion intent with arm policy `("talk", "talk")`; `gaze_at_bearing(track=True)` is allowed only inside the `"social"` nav thread.

Behaviour map: GO_CLOSE / GO_CLOSE_SLOWLY = step <= 0.5 / 0.3 m toward the person, stop at 1.0 / 1.5 m (never closer than 0.8 m to anyone within 30 deg, minus a 0.2 m body margin, torso-keypoint depth only). BACK_AWAY / _SLOWLY = reverse ~0.3 m @ 0.2 m/s / ~0.2 m @ 0.12 m/s, only if the back camera's depth is clear to 1.5 m and it sees no one. STAY = hold. SIT = crouch to -0.2 m (arm stays up). Ends after 600 s or ~2.5-3 m from the start.

---

## 1. Validate on the robot (in this order)

Run from the worktree, but load the main checkout's env file explicitly (the worktree has none). Spot standing, arm in the talk pose, e-stop in hand, a clear 3 m circle.

### 1.1 Where does the gripper camera look in the talk pose? (go / no-go)
Capture one `hand_color_image` with the arm folded, then again after `gaze_at_bearing(+40)` and `(-40)`.
- **Pass:** a standing adult 1.5-3 m in front fills a useful part of the frame (head and torso visible).
- **Fail:** it sees ceiling, floor or Spot's own body. Then the gripper-as-PTZ idea is dead in this pose: either find a talk-pose variant with the camera level (arm joint tweak in `src/session.py` poses, re-test gaze reach) or run social mode on fisheyes only (set `SOURCES = BODY_SOURCES`) and accept low-res faces.

### 1.2 Sensing latency (blocks all motion if too slow)
Time `_social_sense(robot)` 20 times (6 images + 6 depth, YOLO26-pose on each).
- **Pass:** median < 0.6 s, p95 < 0.9 s. `STALE_S = 1.0` is measured from when the images were taken, so a slower cycle means `fresh` is never true and Spot only turns its arm, never steps or reverses.
- **If slow:** drop to front pair + hand + back (back is needed for reverse), or batch the frames into one inference, before touching `STALE_S`.
- Also log emotion + gesture time per cycle (expected ~10 ms face on CPU + ~6 ms emotion + ~7 ms gesture on TRT).

### 1.3 Startup warm-up
Start `client_mic`; look for `[Vision] social emotion model ready` and `... gesture model ready`. Engines are cached in `models/vision/trt_cache` of the worktree (first build took ~13 min). If the main checkout runs it, copy `models/vision/social/` and the cache, or the first start rebuilds.
- **Pass:** both ready in < 30 s on a warm cache. If `social_mode` logs `emotion=off gesture=off`, the loop still runs on pose + distance only.

### 1.4 Dry behaviour: nobody in front
Say "social mode" with no one within 5 m.
- **Pass:** no body motion, arm stays centred, "stop" ends it, mode does not crash on an empty scene.

### 1.5 One person, gaze tracking
One person walks a slow arc 2 m around the front (+/-60 deg).
- **Pass:** the arm follows within ~1 s, no jerky back-and-forth between two people, body turns to face them past ~50 deg. Arm never moves while a body turn is in flight (watch for it).

### 1.6 Each behaviour once, with a person acting it out
Log lines `social: <source> <bearing> <dist> pose=.. emo=.. gest=.. -> <comfort> <behaviour>` show the reasoning.
| Act | Expected |
|---|---|
| Calm, neutral, 1.2 m | ~0.56 -> crouch (SIT). Strangers default to crouch; this is the original design. |
| Smile + thumbs up, 2.5 m | GO_CLOSE / SLOWLY: one step, stops >= 1.0 m from them |
| Frown + fist, 0.6 m | BACK_AWAY: ~0.3 m slow reverse, then stops |
| Same, with a second person 1 m behind Spot | No reverse (log: "someone/something behind me") |
| Same, with a wall 1 m behind | No reverse |
| Wave / raise hand | pose label changes in the log (`waving`, `raising_hand`) |
- **Pass:** each row matches, measured stop distances >= 0.8 m from the nearest person, reverse <= 0.35 m (tape on the floor).

### 1.7 Stops
Say "stop" during: a forward step, a reverse, a crouch, and an arm swing. Then "freeze" once during a step.
- **Pass:** motion stops within ~0.5 s every time, no motion or arm move afterwards.
- **Record:** body height after a stop mid-crouch. Expected: stays crouched until the next stand/walk (deliberate). If the operator wants it to stand back up, add that to the stop path for `_nav_mode == "social"`.

### 1.8 Limits
Let it run with a person slowly leading it away: it must end on its own near 2.5-3 m from the start and say so. Leave it 10 min: it ends at 600 s.

### Before merging
- Re-run the safety reviewer on `a3c2a16`: the fixes from the first review (rear depth check, odometry-bounded reverse, keypoint range, no gaze during moves, arm policy) were applied after it and not re-reviewed.
- Decide about the gesture model's provenance: the model agent was blocked from loading the HaGRID `.pt` and instead extracted only the weight tensors with a restricted unpickler and rebuilt the model. Safe method, but it went around a block without asking. Originals are in `models/vision/social/hagrid_pt/`. Weights license: HaGRID (CC BY-SA 4.0 variant); the ONNX "AGPL-3.0" tag is from the exporter.
- Transplant onto `restructure-modular` (vision package + thin hooks in client_mic / spot_dispatch / llm_brain / gbnf), same as the rest of the vision upgrade.

---

## 2. Improvement directions (ranked)

### A. Familiar faces (biggest behaviour change)
`face_label` is always "unknown", so every person gets the -0.18 unfamiliar penalty and the default reaction is a crouch. Options: treat the voice-locked speaker (wake word + WeSpeaker voiceprint) as familiar while the lock holds; or add a face embedding (e.g. SFace in opencv_zoo, pairs with YuNet) with a small enrolment ("remember me"). Voice lock is the cheap one and needs no new model.

### B. Setup script and model packaging
`scripts/setup_vision_models.py` knows nothing about `models/vision/social/`. Add a download + checksum step for YuNet / HSEmotion and the converted HaGRID ONNX (host the converted file ourselves; don't re-run the conversion on every machine), plus the TRT warm-up.

### C. Smooth the soft signals
Emotion and gesture are single-frame reads and flicker; the 2-cycle debounce on the behaviour is the only smoothing. Keep a short EMA of emotion scores (HSEmotion also gives valence/arousal, currently unused, a cleaner continuous signal than the argmax label) and require a gesture on 2 of the last 3 frames.

### D. Speak and emote
The loop never speaks. Short lines on behaviour changes ("Hi!", "I'll give you some space") and a tail-wag emote on GO_CLOSE would make it read as social. Must respect the mic mute / speaker-lock flow.

### E. Faster, cheaper sensing
Batch all frames through one YOLO26-pose call, skip side/back cameras when the target is in front (keep back for reverse), run face + gesture only on the gripper view. Lets `RATE_HZ` rise toward the original 5 Hz.

### F. Re-tune the comfort weights for the new signals
Weights were hand-set for DeepFace and MediaPipe. HSEmotion and HaGRID have different confidence profiles, and "iloveyou" maps only approximately to HaGRID `rock`. Collect a short labelled session and use the original `ComfortTuner` (ported, unused) to adjust, or re-weight by hand from the logs.

### G. Better pose actions
The rules are fragile (thresholds in person-box units, motion speed changes with frame rate). A small learned classifier over keypoint sequences, or just raising the wave/clap thresholds after real logs, would help. Low priority until 1.6 shows which labels misfire.

### H. Stair / edge awareness for reverse
Rear depth blocks walls and people, not drop-offs. If social mode is ever used near stairs or a stage edge, add a floor-present check in the back depth (lower rows must see floor within ~1 m) or forbid reverse.
