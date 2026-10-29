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
from src.spot.features.mouth import handlers as mouth_handlers  # noqa: E402
