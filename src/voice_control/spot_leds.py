# src/voice_control/spot_leds.py
"""Onboard RGB LED state indicator for the voice pipeline (Stage 2F P7).

Drives Spot's 5 body LED groups via AudioVisualClient. One pulse behavior is
registered per VoiceState; set_state() runs the matching one. A daemon thread
re-runs the active behavior every REFRESH_S so the LED auto-extinguishes if
this process dies (run_behavior's end_time expires). Pulse (not solid) avoids
the AV-board overheat the SDK warns about.

DEGRADES TO NO-OP, never raises into the caller, on any of: no robot, robot
without an AV system, AV client init failure, or any RPC error. In those cases
chimes remain the only state indicator. This module is import-safe without a
robot connection.
"""
from __future__ import annotations

import threading
import time

from src.voice_control.voice_state import VoiceState
from src.voice_control.state_feedback import led_color_for, LED_PERIOD_S

REFRESH_S = 2.0          # re-run cadence; end_time = now + REFRESH_S + margin
_END_MARGIN_S = 0.5
_PRIORITY = 10


def _behavior_name(state: VoiceState) -> str:
    return f"spotvoice_{state.name.lower()}"


def _build_pulse_behavior(r: int, g: int, b: int, period_s: float):
    """Build an AudioVisualBehavior that pulses all 5 LED groups one color."""
    from bosdyn.api import audio_visual_pb2 as av  # local import: optional dep
    grp = av.LedSequenceGroup()
    for led in (grp.front_center, grp.front_left, grp.front_right,
                grp.hind_left, grp.hind_right):
        led.pulse_sequence.color.rgb.r = int(r)
        led.pulse_sequence.color.rgb.g = int(g)
        led.pulse_sequence.color.rgb.b = int(b)
        led.pulse_sequence.period.seconds = int(period_s)
        led.pulse_sequence.period.nanos = int((period_s % 1) * 1e9)
    return av.AudioVisualBehavior(enabled=True, priority=_PRIORITY,
                                  led_sequence_group=grp)


class StateLeds:
    """Maps VoiceState -> onboard LED pulse. No-op when unavailable."""

    def __init__(self, get_robot, start_thread: bool = True):
        self._get_robot = get_robot
        self.enabled = False
        self._av = None
        self._initialized = False
        self._failed = False          # once True, never retry (permanent degrade)
        self._active = None           # active behavior name
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    # -- public ----------------------------------------------------------
    def set_state(self, state: VoiceState) -> None:
        """Show `state` on the LEDs. Swallows ALL errors (degrade) — a LED
        failure must never propagate into the voice loop."""
        try:
            if self._failed:
                return
            if not self._ensure():
                return
            name = _behavior_name(state)
            with self._lock:
                self._active = name
            self._run(name)
        except Exception as e:
            print(f"[LEDs] Disabled (set_state error: {type(e).__name__}: {e}).")
            self._failed = True

    def shutdown(self) -> None:
        self._stop.set()
        if self._av is not None and self._active is not None:
            try:
                self._av.stop_behavior(self._active)
            except Exception:
                pass

    # -- internal --------------------------------------------------------
    def _ensure(self) -> bool:
        """Lazily init the AV client + register behaviors. Degrades on failure."""
        if self._initialized:
            return self.enabled
        try:
            robot = self._get_robot()
            if robot is None:
                # Session not up yet (before the first command connects). Do NOT
                # mark initialized/failed — retry on a later state change so LEDs
                # activate once a real command has brought the session up.
                return False
            self._initialized = True
            hw = robot.get_cached_hardware_hardware_configuration()
            if not getattr(hw, "has_audio_visual_system", False):
                print("[LEDs] Robot has no audio-visual system — LED indicator off.")
                self._failed = True
                return False
            from bosdyn.client.audio_visual import AudioVisualClient
            self._av = robot.ensure_client(AudioVisualClient.default_service_name)
            for st in VoiceState:
                r, g, b = led_color_for(st)
                beh = _build_pulse_behavior(r, g, b, LED_PERIOD_S[st])
                self._av.add_or_modify_behavior(_behavior_name(st), beh)
            self.enabled = True
            print("[LEDs] Onboard RGB state indicator active.")
            return True
        except Exception as e:
            print(f"[LEDs] Disabled (init failed: {type(e).__name__}: {e}).")
            self._failed = True
            self.enabled = False
            return False

    def _run(self, name: str) -> None:
        try:
            from bosdyn.util import now_sec
            self._av.run_behavior(name, end_time_secs=now_sec() + REFRESH_S + _END_MARGIN_S)
        except Exception as e:
            print(f"[LEDs] Disabled (run failed: {type(e).__name__}: {e}).")
            self._failed = True
            self.enabled = False

    def _refresh_loop(self) -> None:
        while not self._stop.wait(REFRESH_S):
            if not self.enabled or self._failed:
                continue
            with self._lock:
                name = self._active
            if name is not None:
                self._run(name)
