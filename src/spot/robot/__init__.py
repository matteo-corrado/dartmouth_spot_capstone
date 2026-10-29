"""Spot robot core: connection, arm, motion, state, dispatcher.

The owner singletons below hold the state that used to be module globals in
the old spot_dispatch module. They are created once here; ``main``
passes them on and tests patch attributes on them. This is a deliberate,
temporary exception to spec section 5 ("no singletons reached by import"):
about 260 test monkeypatch sites depend on one shared object per state.

Every module (the owner classes included) reaches the singletons ONLY through
the module at call time (``import src.spot.robot as _r``; ``_r.NAV``), never
by binding them at import or storing peers, so a rebind can never split them.
"""
from src.spot.safety.gen import SafetyGen
from src.spot.robot.connection import Connection
from src.spot.robot.arm import Arm
from src.spot.nav.runner import NavRunner

SAFETY_GEN = SafetyGen()
CONNECTION = Connection()
ARM = Arm()
NAV = NavRunner()


# Feature callbacks the core calls on its halt / stow / stand / close paths
# (_cancel_auto_arm, _handle_disable_mouth, _auto_arm_after_stand,
# start_auto_arm, AUTO_ARM_STAND_WAIT_S). Bound ONCE here, at import, after the
# singletons (so a feature reaching src.spot.robot during this import finds
# them): the core reads ``_r.mouth_handlers.<name>`` -- a plain attribute
# lookup, no import on a stop path -- so a test patch on the module is seen.
# The module is imported whether or not "mouth" is in ENABLED: with the
# feature disabled these calls still run (no-ops while nothing is owed).
# Guarded: if the features package or the mouth modules raise at import,
# src.spot.robot must still import (estop_run, spotctl, the stop path), so the
# failure is printed loudly and a no-op stand-in is bound instead. In robot
# mode main's feature check (C6) still refuses to start on such a failure.


class _NoMouth:
    """Stand-in when the mouth handlers failed to import: every core call
    is a no-op that never raises."""
    AUTO_ARM_STAND_WAIT_S = 0.0

    def __init__(self, error):
        self.error = error

    def _cancel_auto_arm(self):
        return None

    def _auto_arm_after_stand(self):
        return None

    def start_auto_arm(self, gen, wait_s=0.0):
        return None

    def _handle_disable_mouth(self, params, session=None):
        """The real handler's arm part only (ARM / NAV): clear the mouth and
        talk-resume latches, then stow (or stow when the walk ends). Still
        False (the mouth itself is unavailable); never raises."""
        print(f"[Spot] disable_mouth: mouth handlers unavailable ({self.error}); stowing the arm only")
        try:
            ARM._mouth_resume = False
            ARM.set_talk_resume(False)
            if NAV.is_navigating():
                ARM._pose_before_walk = "stowed"  # arm is tucked; stow when the walk ends
            elif session is not None:
                ARM.send(session["cmd"], "stowed", timeout_s=5.0)
        except Exception as e:  # noqa: BLE001
            print(f"[Spot] ✗ disable_mouth (stand-in) failed: {e}")
        return False


try:
    from src.spot.features.mouth import handlers as mouth_handlers  # noqa: E402
except Exception as _e:  # noqa: BLE001
    import traceback as _tb
    print(f"[Spot] ⚠️  MOUTH HANDLERS FAILED TO IMPORT ({type(_e).__name__}: {_e}); "
          "auto-arm / disable_mouth are no-ops", flush=True)
    _tb.print_exc()
    mouth_handlers = _NoMouth(f"{type(_e).__name__}: {_e}")
