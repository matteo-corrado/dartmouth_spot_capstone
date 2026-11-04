# Arm gaze turn: next steps

Context: operator request 2026-11-04, on `restructure-modular`. Goal: upgrade how the folded arm turns toward the locked speaker.

Today (`src/spot/features/gaze/handlers.py` `gaze_at_bearing`, called from the gaze worker in `src/spot/conversation/loop.py`):
- Moves **shoulder yaw (sh0) only**, via a slow joint move (`joints_pose_command`). No pitch, so the gripper camera does not tilt to face height.
- Clamped to `gaze_max_deg` (tuning default **20**, code cap `GAZE_HARD_MAX_DEG = 60`). A speaker more than 150 deg behind is ignored.
- Rate-limited: re-aims only if the target moved > 15 deg **and** >= 2 s since the last gaze. In a fast back-and-forth it lags or skips turns.
- Only in the `talk` pose, standing and powered, not navigating, after the arm settle time. Each send re-checks the safety gen under `ARM._lock`.
- `gaze_sh0_sign` is **unverified on the robot**: the arm may turn away from the speaker.
- No body turn: a speaker past the limit just gets the arm at its stop.
- `src/spot/features/gaze/math.py` (`gaze_target`, body-frame point for `arm_gaze_command`) is unused.

Robot findings 2026-10-28 (`scripts/gaze_robot_check.py`, old branch `vision/w5-w6-capture-gaze`): `arm_gaze_command` from the ready pose reaches +/-60 deg within ~6 deg; +/-90 saturates; direct swings > 60 deg (e.g. +60 -> -60) stall or flail unless routed through the ready pose first; SDK `BodyAssistForManipulation` did **not** move the body for gaze; mouth (claw-only) commands do not disturb a held gaze.

---

## 1. Validate what exists (robot, e-stop in hand)

1. **Sign check.** Talk from ~45 deg left. **Pass:** arm turns left. If it turns away, set `gaze_sh0_sign: -1` in `config/tuning.yaml`.
2. **Widen the limit.** Raise `gaze_max_deg` 20 -> 40 -> 60, talking from +/-30, +/-50, +/-70. **Pass:** reaches the clamp each time, no stall, never past 60, nothing hits the body.
3. **Mic yaw offset.** The DoA-to-body offset is uncalibrated (`scripts/xvf_probe.py clap`). **Pass:** with the speaker straight ahead, the gaze log shows a bearing within ~10 deg of 0.

## 2. Upgrades (ranked)

1. **Faster, smoother turns.** Cut the 2 s / 15 deg rate limit (e.g. 0.7 s / 10 deg) and send a velocity-capped trajectory instead of one slow jump. Keep the C3 velocity cap.
   - **Pass:** two people ~60 deg apart taking turns: the arm lands on each talker within ~1 s, no jitter while one person talks.
2. **Big swings via centre.** A move across > 60 deg (left speaker -> right speaker) goes through 0 first, like the ready-pose routing that fixed the stall.
   - **Pass:** +50 -> -50 swings 10/10 times with no stall.
3. **Tilt to face height.** Switch from sh0-only to `arm_gaze_command` with `math.py` `gaze_target` (body-frame point at ~1.6 m height, 1.5-2 m out), or add a fixed sh1/wrist pitch. Check that the talk pose and mouth still work.
   - **Pass:** the `hand_color_image` frame shows the speaker's face, not ceiling, floor or the body. (Shares the camera-framing check with `friendly-spot-social-mode.md` 1.1.)
4. **Body turn past the limit.** If the speaker sits beyond `gaze_max_deg` for > ~2 s, turn the body in place (small yaw, velocity-capped) and re-centre the arm. The SDK body assist is not an option (no-op, see above).
   - Only while standing still, never during nav, a dance or social mode; a stop wins via the safety gen.
   - **Pass:** speaker at 90 deg: body turns to face them within ~3 s, arm ends near centre; "stop" during the turn halts it.
5. **G11 safety checks** (`docs/project/bugs.md` G11): before the send, refuse if e-stopped, a behavior fault is active, or body velocity > ~0.05 m/s / 0.1 rad/s. Needed before item 4.

## 3. Safety

- Re-run the `safety-reviewer` agent on any change here (arm motion path).
- "stop" / "freeze" during a gaze swing and during a body turn: **pass** = motion stops within ~0.5 s, no gaze command after the stop.
