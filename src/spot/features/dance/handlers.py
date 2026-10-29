"""Emotes / dances (Task 15): the `emote` action. Moved from
the old spot_dispatch.py (Task 11), behaviour unchanged.

The clearance scan stays in perception (obstacle_query.clearance_scan); this
feature only calls it. The arm stow/restore and the stop plumbing stay in the
robot core (spec 7.1b). The robot owners are reached through
`import src.spot.robot as _r` at call time only (never at import: the
feature registry loads this module while src.spot.robot may be importing).
"""
import time

_last_routine = None  # last routine run, so "dance again" never repeats it
EMOTE_TIMEOUT_S = 60.0
EMOTE_STALE_GRACE_S = 4.0
# ChoreographyStatusResponse statuses that mean "still going"
_CHOREO_RUNNING = {1, 3, 4, 5}  # DANCING, PREPPING, WAITING_FOR_START_TIME, VALIDATING
_CHOREO_COMPLETED = 2           # STATUS_COMPLETED_SEQUENCE
NO_ROOM_LINE = "I need a little more room to dance."
NO_GRID_LINE = "I can't check my surroundings right now, so I'll skip the dance."


def _clearance_refusal(robot):
    """None if the floor around Spot is clear, else the line to speak.
    Fails closed: no usable local grid -> refuse."""
    from src.spot.core.tuning import get_tuning
    from src.spot.perception.obstacle_query import clearance_scan
    clearance = get_tuning().dance_clearance_m
    try:
        usable, hit = clearance_scan(robot, max_radius_m=clearance)
    except Exception as e:
        print(f"[Spot] emote: obstacle check failed: {e}")
        return NO_GRID_LINE
    if not usable:
        print("[Spot] emote: no usable local grid")
        return NO_GRID_LINE
    if hit is not None and hit.distance_m < clearance:
        print(f"[Spot] emote: obstacle at {hit.distance_m:.2f} m < {clearance} m")
        return NO_ROOM_LINE
    return None


def _emote_routine(session, routine, prev_pose, stop_event, speak=None, gen=None):
    """Background thread: stow arm (mouth latched off), re-check clearance,
    run the routine, restore the arm only after it completed. Every robot
    command is gated on stop_event under _arm_lock, so a stop/freeze/estop
    (which sets the event before its own stop) is never followed by a routine
    start or an arm restore from here."""
    global _last_routine
    import src.spot.robot as _r
    from src.spot.robot import motion as _motion
    from src.spot.robot.arm import RPC_TIMEOUT_S
    from src.spot.robot.dispatcher import _say
    from bosdyn.choreography.client.choreography import ChoreographyClient
    cmd = session["cmd"]
    outcome = "aborted"  # "done" | "timeout" | "abnormal" | "aborted"
    started = False
    try:
        _r.ARM.mouth_suspend()  # the claw never moves during a routine
        if prev_pose != "stowed":
            if not _r.ARM.send(cmd, "stowed", stop_event, timeout_s=5.0):
                print("[Spot] emote: arm did not stow — not dancing")
                outcome = "done"  # restore the previous pose explicitly
                return
        refusal = _clearance_refusal(session["robot"])  # re-check right before
        if refusal is not None:
            _say(speak, refusal)
            outcome = "done"  # nothing ran; put the arm back
            return
        choreo = session["robot"].ensure_client(ChoreographyClient.default_service_name)
        with _r.ARM._lock:
            if stop_event.is_set() or _r.SAFETY_GEN.aborted(gen):
                return
            started = True
            choreo.execute_choreography(routine, client_start_time=time.time() + 1.0,
                                        choreography_starting_slice=0, timeout=RPC_TIMEOUT_S)
        _last_routine = routine  # only a routine that actually started counts as "last"
        print(f"[Spot] ✓ Routine '{routine}' started")
        t0 = time.time()
        deadline = t0 + EMOTE_TIMEOUT_S
        seen_running = False
        outcome = "timeout"
        stop_event.wait(1.0)
        while not stop_event.is_set() and time.time() < deadline:
            try:
                status, _ = choreo.get_choreography_status()
                st = status.status
                if st in _CHOREO_RUNNING:
                    seen_running = True
                elif seen_running:
                    outcome = "done" if st == _CHOREO_COMPLETED else "abnormal"
                    break
                elif time.time() > t0 + EMOTE_STALE_GRACE_S:
                    outcome = "abnormal"  # never saw it run: don't trust the status
                    break
            except Exception as e:
                print(f"[Spot] emote status poll failed: {e}")
            stop_event.wait(0.5)
        if stop_event.is_set():
            outcome = "aborted"
        print(f"[Spot] Routine '{routine}' ended: {outcome}")
    except Exception as e:
        print(f"[Spot] ✗ emote '{routine}' failed: {e}")
        outcome = "abnormal" if started else "done"
    finally:
        if stop_event.is_set():
            return  # the stop owns the robot and the arm
        try:
            if outcome in ("timeout", "abnormal") and started:
                _motion.send_stop(cmd)
            if outcome in ("done", "timeout") and prev_pose != "stowed":
                # also covers a failed stow: put the arm back in a known pose
                _r.ARM.send(cmd, prev_pose, stop_event, timeout_s=5.0)
            _r.ARM.mouth_restore()
        except Exception as e:
            print(f"[Spot] ⚠️  Arm restore after emote failed: {e}")


def _handle_emote(params, session, speak=None):
    """emote {"name": str}: refuse while moving, sitting or crowded, then run
    the routine on a background thread registered as the nav thread (so stop
    cancels it)."""
    import src.spot.robot as _r
    from src.spot.robot import state as _state
    from src.spot.robot.dispatcher import _say
    from src.spot.features.dance.emotes import resolve_routine
    if _r.NAV._mode == "emote" and _r.NAV.is_navigating():
        _say(speak, "One dance at a time!")
        return False
    if _r.NAV.is_navigating() or _r.ARM._pose == "walk":
        _say(speak, "I can't dance while I'm on the move.")
        return False
    try:
        name = (params or {}).get("name")
        if isinstance(name, str):
            name = name.strip().lower().replace(" ", "_")  # "Happy Wiggle" -> happy_wiggle
        routine = resolve_routine(name or None, _last_routine)
    except ValueError:
        _say(speak, "I don't know that one yet.")
        return False
    if not _state.standing_and_powered(session):
        _say(speak, "Ask me to stand up first.")
        return False
    refusal = _clearance_refusal(session["robot"])
    if refusal is not None:
        _say(speak, refusal)
        return False
    gen = _r.SAFETY_GEN.dispatch_current()
    if _r.SAFETY_GEN.aborted(gen):
        return False
    _r.NAV.cancel()
    prev = _r.ARM._pose
    if not _r.NAV.spawn(lambda ev: _emote_routine(session, routine, prev, ev, speak, gen),
                      mode="emote", gen=gen):
        return False
    print(f"[Spot] emote: running '{routine}'")
    return True


def emote(params, ctx):
    """Registry handler for `emote`. Same session bring-up as the legacy
    branch (session=None there: emote is not an aborting intent); the
    dispatcher's catch-all handles a raise."""
    import src.spot.robot as _r
    session = _r.CONNECTION.ensure()
    session["cmd"]  # legacy branch read cmd_client before dispatching
    return _handle_emote(params, session, ctx.speak)
