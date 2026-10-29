# Arm joints pose / mouth / auto-fold / gaze report (2026-10-28)

Base: d5558ed (demo-poster-2026-10-29). Commits: e070caa (code + tests + daily-operations doc), then a follow-up adding the gaze settle guard and this report.

## What changed

1. **One pose** (`src/session.py`): `WALK_POSE = "joints"`. New `joints_pose_command(sh0_rad=0.0)`
   builds WALK_JOINTS (sh0 0, sh1 -2.6, el0 2.6, el1 0, wr0 0, wr1 0) with sh0 replaced, 2.5 s,
   `max_vel=ARM_MAX_ANGULAR_VEL`. `"talk"` and `"walk"` both send it. Pose names stay
   "stowed"/"talk"/"walk" (state machine unchanged); only the command behind "talk" changed.
   `block_until_arm_arrives` handles `arm_joint_move_feedback` (checked in the venv SDK).
2. **Walking keeps the mouth** (`spot_dispatch.py`): `arm_policy_for(locomotion, "talk")` →
   `("talk","talk")` when talk == walk pose, so no mouth suspend and no tuck. If a gaze left sh0
   off-centre, the walk first sends a centred joints pose (no wait). New `_pose_during_walk`
   is used as `require_pose` in `_restore_after_motion`, so "disable mouth" mid-walk still
   stows after the walk ends and a stop's stow still blocks any restore. Emotes: stow +
   mouth off, then restore joints + mouth. Stop/freeze/estop/sit/power_off/shutdown paths unchanged.
3. **Auto fold + mouth**: `set_auto_arm(on)` (client_mic startup), `auto_arm_if_ready(gen, wait_s)`,
   `start_auto_arm` (one daemon thread at a time). Fires after a successful `stand` (spotctl or
   voice, waits up to 8 s for standing), after session bring-up, at the start of every
   motion-allowed turn, and from `set_auto_arm` if a session already exists (--map). Requires
   live session + `_standing_and_powered` + not navigating + arm stowed + no pending halt
   resume + no stop since `gen` (checked under `_dispatch_lock` and again under `_arm_lock`).
   E-stop clears the pending fold (only an explicit stand re-arms it). "disable mouth" and
   `spotctl stow` turn auto off for the session; "enable mouth" turns it back on.
   `enable_mouth` now also refuses unless standing + powered, and enables the mouth under
   `_arm_lock` only if no stop landed while the arm was rising.
   Opt-out: `--no-arm` (client_mic, run_voice_control, wakespot forward it) or tuning
   `auto_arm: false`. `--enable-arm` is kept as a no-op.
4. **Gaze** (`gaze_at_bearing(body_deg, gen=None)` rewritten): joints pose with sh0 = clamp(bearing,
   ±`gaze_max_deg`=45) × `gaze_sh0_sign`; bearings > 150° behind are ignored; None re-centres.
   Only if: session, not closing, pose "talk", not navigating (walk/dance/follow), standing +
   powered, gaze_max_deg > 0. Rate limit: target moved > 15° AND ≥ 2 s since last gaze.
   Sent under `_arm_lock` after re-checking pose + `_aborted(gen)` + navigating; no arrival wait;
   errors are logged. Gaze memory resets on every pose change. **Settle guard:** no gaze for
   JOINT_MOVE_SECS + 0.5 s after any pose send, so a gaze can't take over a fold or halt-resume
   that is still rising (that would leave the mouth off). spotctl stand → `_on_immediate` →
   `execute_on_spot` → `dispatch_intent(stand)`, so the re-fold after a sit runs (code checked). client_mic hooks: wake lock
   (`_on_wake`), accepted-utterance lock nudge, `_release` → None. Requests go to one daemon
   worker with a size-1 queue (newest only), so the audio loop never blocks on an RPC. No-op in dry run.
   `gaze.py` (old arm_gaze_command math) is now unused but left in place.
5. **Tuning keys**: `auto_arm: true`, `gaze_max_deg: 45.0`, `gaze_sh0_sign: 1`.

## Tests

Full suite: **555 passed, 3 skipped** (`spot-env/bin/python -m pytest tests -p no:warnings`).
Arm/gaze/emote/halt files run 4x: 124 passed every time.

- New `tests/voice_control/test_auto_arm.py` (17): talk == walk joints, auto fold only when
  standing + powered, never without a session, opt-out (flag/tuning/dry-run), stand → fold,
  stand gives up if never standing, stop/sit since gen blocks the fold, estop cancels it,
  not while moving, halt resume owns the arm, disable-mouth opt-out until enable,
  enable_mouth refused unless standing, walk after the fold keeps the mouth on.
- `test_gaze_dispatch.py` rewritten (19, incl. no gaze while a pose move is in flight): clamp/wrap, joints sh0 move at 1 rad/s, tuning limit and
  sign, rate limit, re-centre, behind ignored, talk pose only, not while navigating / not
  standing, gaze off at 0, blocked after stop, stale gen refused, never raises.
- `test_arm_policy.py`: joints-walk tests added (no arm command during the walk, gaze re-centre
  without a wait, mouth stays on, stop mid-walk stows + mouth off + arms the halt resume,
  disable_mouth mid-walk stows after). The old tuck tests now run with `WALK_POSE="stow"`
  patched, so that path stays covered. `arm_ready` → `arm_joints` where talk is restored
  (emote, halt resume). Every stop/freeze/estop/sit command sequence assertion is unchanged.

## Operator must verify on the robot

1. **sh0 sign:** speak from Spot's left; the folded arm should turn left. If it turns the
   wrong way: `gaze_sh0_sign: -1` in `config/tuning.yaml`.
2. **Gaze clearance:** shipped at `gaze_max_deg: 20` (default and tuning.yaml). Check ±20° clears
   the body/payload and the claw still opens there, then widen (code cap 60).
3. **Fold timing:** after `spotctl stand` the arm should fold within ~3 s of standing and
   the mouth should move on the next reply. With `--map` it folds right after launch bring-up.
4. **Mouth while walking:** claw animation during a walk (gripper-only commands must not
   disturb the mobility command).
5. A stop mid-gaze, mid-fold and mid-walk: arm stows, claw closed, nothing moves after.

## Notes / not done

- Gaze's "standing" check accepts STANDING or STEPPING (shared `_standing_and_powered`). A
  tablet/joystick walk is not `is_navigating()`, so gaze could fire during one. Low risk at the demo.

- pipeline-explainer.md updated in the safety-review round (below).
- If gaze sends within 2 s of the lock release, the re-centre is skipped. The next walk
  re-centres anyway.

## Safety-review fixes (round 2)

- **I1:** client_mic `_gaze_at_speaker` takes the safety gen at request time and queues
  `(body_deg, gen)`; the worker calls `gaze_at_bearing(body, gen)`. If there is no gen
  (dry run / no robot), no gaze is queued. Test: a stop between request and pickup → no arm command.
- **M1:** `_hold_gaze()` blocks gaze for JOINT_MOVE_SECS at the start of `_dispatch_locomotion`
  (before `_cancel_nav`) and again just before `_dispatch_core`.
- **M2:** the gaze RPC timeout is 0.5 s (< SAFETY_LOCK_TIMEOUT_S 1.0 s).
- **M3:** `_cancel_auto_arm()` in sit, power_off and close_spot_session.
- **M6:** gaze is hard-capped at 60° in code. `config/tuning.yaml` now has explicit `auto_arm: true`,
  `gaze_max_deg: 20`, `gaze_sh0_sign: 1` (marked unverified). The code default for
  `gaze_max_deg` is now also 20, so `test_shipped_yaml_is_all_defaults` still holds.
- pipeline-explainer.md: startup, flags, gaze, arm policy, mouth and tunables are updated.

Full suite after round 2: **563 passed, 3 skipped**. Arm/gaze/emote/halt/tuning files run 3x: 137 passed each time.
