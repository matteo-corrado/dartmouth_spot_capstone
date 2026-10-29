"""Spot robot core: connection, arm, motion, state, dispatcher.

The owner singletons below hold the state that used to be module globals in
``src/voice_control/spot_dispatch.py``. They are created once here; ``main``
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


def legacy_features():
    """The feature functions still in spot_dispatch (Wave 4 moves them).
    Core calls them at call time through the module object, so a test patch
    on spot_dispatch is seen. The module is already loaded by then, so this
    import is a sys.modules lookup and takes no lock."""
    import src.voice_control.spot_dispatch as sd
    return sd
