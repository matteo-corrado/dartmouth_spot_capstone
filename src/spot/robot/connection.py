"""The live Spot session (moved out of spot_dispatch globals ``_spot_session``,
``_session_context``, ``_session_lock`` and ``_closing``)."""
import threading
import time

from src.spot.robot.session import spot_session
import src.spot.robot as _r
from src.spot.robot import motion as _motion


class Connection:
    def __init__(self):
        self._session = None    # dict of live clients (robot, lease_client, cmd, power, state)
        self._context = None    # the spot_session() context manager
        self._lock = threading.Lock()  # guards ensure() bring-up
        self._closing = False   # set by close(): no new bring-up after that

    @property
    def session(self):
        return self._session

    def cmd_client(self):
        s = self._session
        return None if s is None else s["cmd"]

    def ensure(self, map_path=None):
        """Ensure Spot session is initialized and return session dict.

        Args:
            map_path: Optional GraphNav map directory to upload during session
                bring-up. When provided AND the session is not yet active, the
                session is created with ``upload_map=True``, ``map_path=...``, and
                ``auto_localize=True`` so the robot uploads the map and attempts
                fiducial localization as part of stand-up.

                When provided AND a session is ALREADY active, the existing
                session is returned silently — the assumption is that the caller
                wants the session, not necessarily the map. To switch maps at
                runtime, use the ``load_map`` voice handler instead, which
                re-uploads on a live session.

        Returns:
            dict with the live session clients (robot, lease_client, cmd, power,
            state).
        """
        if self._closing:  # checked first: nothing may use the live session during close
            raise RuntimeError("Spot session is shutting down; not bringing it up")
        if self._session is not None:
            return self._session
        with self._lock:  # HTTP threads + main loop must not both bring it up
            if self._session is not None:
                return self._session
            if self._closing:
                raise RuntimeError("Spot session is shutting down; not bringing it up")
            try:
                self._context = spot_session(
                    stand_on_enter=True,
                    sit_on_exit=False,  # keep standing for voice commands
                    upload_map=bool(map_path),
                    map_path=str(map_path) if map_path else None,
                    auto_localize=bool(map_path),
                )
                self._session = self._context.__enter__()
                print("[Spot] Connected and standing ready for voice commands.")
                sd = _legacy()
                sd.start_auto_arm(_r.SAFETY_GEN.current(), wait_s=sd.AUTO_ARM_STAND_WAIT_S)  # waits for this dispatch
            except Exception as e:
                print(f"[Spot] Failed to establish session: {e}")
                # Clear the context so we can retry
                self._context = None
                raise

        return self._session

    def close(self):
        """Close the Spot session (sit and power off). After this no command can
        bring a new session up (ensure() raises while closing)."""
        self._closing = True
        _r.SAFETY_GEN.bump()  # abort any locomotion still being set up
        try:  # nothing fallible may block the send_stop below (R18)
            _legacy()._cancel_auto_arm()
        except Exception as e:
            print(f"[Spot] ⚠️  auto-arm cancel failed during close: {e}")

        if self._session is not None and self._context is not None:
            _motion.send_stop(self._session["cmd"])  # stop first, then stow, then sit
            _r.ARM.close_mouth()
            _r.NAV.cancel()
            _r.ARM.stow_now(self._session["cmd"], force=True)
            time.sleep(1.5)  # let the stow finish before sitting
            try:
                from bosdyn.client.robot_command import blocking_sit
                blocking_sit(self._session["cmd"], timeout_sec=15)
            except Exception as e:
                print(f"[Spot] Warning during sit: {e}")
            try:
                self._session["robot"].power_off(cut_immediately=False, timeout_sec=20)
            except Exception as e:
                print(f"[Spot] Warning during power off: {e}")
            self._context.__exit__(None, None, None)
            self._session = None
            self._context = None
            print("[Spot] Session closed.")


def _legacy():
    return _r.legacy_features()


def _conn():
    return _r.CONNECTION


def ensure_spot_session(map_path=None):
    """Old spot_dispatch name for CONNECTION.ensure()."""
    return _conn().ensure(map_path)


def close_spot_session():
    """Old spot_dispatch name for CONNECTION.close()."""
    return _conn().close()
