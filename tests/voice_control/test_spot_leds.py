# tests/voice_control/test_spot_leds.py
"""Unit tests for the StateLeds AV wrapper degrade + dispatch (Stage 2F P7)."""
import pytest
from src.voice_control.client_mic import VoiceState
from src.voice_control.spot_leds import StateLeds


class FakeAvClient:
    def __init__(self):
        self.added = []
        self.ran = []
        self.stopped = []
    def add_or_modify_behavior(self, name, behavior):
        self.added.append((name, behavior))
    def run_behavior(self, name, end_time_secs, **kw):
        self.ran.append(name)
    def stop_behavior(self, name, **kw):
        self.stopped.append(name)


class FakeHw:
    def __init__(self, has_av):
        self.has_audio_visual_system = has_av


class FakeRobot:
    def __init__(self, has_av=True, av_client=None, raise_on_client=False):
        self._hw = FakeHw(has_av)
        self._av = av_client
        self._raise = raise_on_client
    def get_cached_hardware_hardware_configuration(self):
        return self._hw
    def ensure_client(self, name):
        if self._raise:
            raise RuntimeError("no AV service")
        return self._av


def test_no_robot_is_noop():
    leds = StateLeds(get_robot=lambda: None, start_thread=False)
    leds.set_state(VoiceState.LISTENING)  # must not raise
    assert leds.enabled is False


def test_robot_without_av_system_degrades():
    robot = FakeRobot(has_av=False, av_client=FakeAvClient())
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    assert leds.enabled is False


def test_client_init_failure_degrades_permanently():
    robot = FakeRobot(has_av=True, raise_on_client=True)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    assert leds.enabled is False
    # a later call must not retry/raise
    leds.set_state(VoiceState.THINKING)


def test_registers_one_behavior_per_state_and_runs_active():
    av = FakeAvClient()
    robot = FakeRobot(has_av=True, av_client=av)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.THINKING)
    assert leds.enabled is True
    # one behavior registered per VoiceState
    assert {n for n, _ in av.added} == {f"spotvoice_{s.name.lower()}" for s in VoiceState}
    # the active state's behavior was run
    assert av.ran[-1] == "spotvoice_thinking"


def test_set_state_runs_new_behavior():
    av = FakeAvClient()
    robot = FakeRobot(has_av=True, av_client=av)
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)
    leds.set_state(VoiceState.RESPONDING)
    assert av.ran[-1] == "spotvoice_responding"


def test_rpc_error_during_run_degrades_not_raises():
    class BoomClient(FakeAvClient):
        def run_behavior(self, name, end_time_secs, **kw):
            raise RuntimeError("rpc down")
    robot = FakeRobot(has_av=True, av_client=BoomClient())
    leds = StateLeds(get_robot=lambda: robot, start_thread=False)
    leds.set_state(VoiceState.LISTENING)  # must swallow
    assert leds.enabled is False
