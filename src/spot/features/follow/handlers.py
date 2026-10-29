"""follow_me: the branch body moved from spot_dispatch._dispatch_core
(Task 13), unchanged. Reached through dispatcher._dispatch_locomotion ->
_dispatch_core -> _dispatch_feature, the point the legacy branch was reached
from. Imports are call-time only (see walk/handlers.py)."""


def handle_follow_me(params, ctx):
    import src.spot.robot as _r
    from src.spot.robot import motion as _motion
    session = _r.CONNECTION.ensure()
    session["cmd"]  # the legacy chain read "cmd" before every branch

    # Follow the nearest person using YOLO person detection
    print("[Spot] Starting follow mode...")

    def _follow_thread(ev):
        try:
            from src.voice_control.visual_nav import follow_person
            result = follow_person(session, ev)
            if result:
                print("[Spot] ✓ Follow mode ended cleanly")
            else:
                print("[Spot] Lost the person — follow mode ended")
            return bool(result)
        except ImportError:
            print("[Spot] ✗ ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"[Spot] ✗ Follow failed: {e}")

    _r.NAV.cancel()
    return _r.NAV.spawn(
        lambda ev: _motion.thread_then_restore(lambda: _follow_thread(ev), ev),
        mode="follow", gen=_r.SAFETY_GEN.dispatch_current())
