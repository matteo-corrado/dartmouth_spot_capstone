"""Demo 2026-10-29 gate fix: loudest-150 ms gate statistic, short-utterance
commands-only path + safety rescue, live voiceprint threshold."""
import numpy as np
import pytest

import src.voice_control.client_mic as cm
from src.voice_control.audio_frontend import UtteranceStats, passes_gate
from src.voice_control.control_server import parse_command
from src.voice_control.intent import short_utterance_ok
from src.voice_control.tuning import Tuning


# ---- gate statistic --------------------------------------------------------
def _one_word_utterance(window_frames=5):
    # ~2.4 s utterance: pre-roll + hangover at -58 dBFS, a 0.18 s word at -38.
    s = UtteranceStats(frame_s=0.03, window_frames=window_frames)
    [s.add(-58.0, True) for _ in range(30)]
    [s.add(-38.0, True) for _ in range(6)]
    [s.add(-58.0, True) for _ in range(44)]
    [s.add(-60.0, False) for _ in range(20)]
    return s


def test_peak_scores_a_short_word_in_a_long_utterance():
    s = _one_word_utterance()
    assert s.speech_db_p90 - (-60.0) < 8.0          # old p90 statistic: dropped
    assert abs(s.snr_db(-60.0) - 22.0) < 1e-6       # loudest 150 ms over the floor
    assert passes_gate(s, floor_db=-60, min_speech_dbfs=-70, min_snr_db=8.0) is True


def test_peak_still_rejects_floor_level_noise():
    s = UtteranceStats(window_frames=5)
    [s.add(-57.0 + 0.1 * (i % 5), True) for i in range(80)]
    assert passes_gate(s, floor_db=-60, min_speech_dbfs=-70, min_snr_db=8.0) is False


def test_peak_window_is_a_mean_of_the_top_frames():
    s = UtteranceStats(window_frames=2)
    for db in (-50, -40, -30, -45):
        s.add(db, True)
    s.add(0.0, False)  # non-speech frames never count
    assert s.speech_db_peak == pytest.approx(-35.0)
    assert UtteranceStats().speech_db_peak == -120.0


def test_tuning_defaults():
    t = Tuning()
    assert t.min_snr_db == 8.0 and t.gate_window_ms == 150.0
    assert t.short_utt_s == 1.5
    assert t.voiceprint_threshold == 0.25 and t.voiceprint_enabled is False


def test_short_utterance_is_by_vad_speech_not_wall_clock():
    s = _one_word_utterance()            # 2.4 s of speech-flagged frames
    assert cm.is_short_utterance(s, 1.5) is False
    w = UtteranceStats()
    [w.add(-40.0, True) for _ in range(30)]   # 0.9 s of VAD speech
    [w.add(-60.0, False) for _ in range(60)]  # end silence doesn't count
    assert cm.is_short_utterance(w, 1.5) is True
    assert cm.is_short_utterance(UtteranceStats(), 1.5) is False


# ---- short-path transcript filter ------------------------------------------
@pytest.mark.parametrize("text,conv,ok", [
    ("Stop.", False, True), ("freeze", False, True), ("Sit.", False, True),
    ("stand up", False, True), ("Dance!", False, True), ("stow the arm", False, True),
    ("turn left", False, True),
    ("Yes.", True, True), ("okay", True, True), ("yes", False, False),
    ("what's your name", True, False), ("the", True, False), ("", True, False),
    ("I like to dance a lot", False, False), ("Thank you.", True, False), ("you", True, False),
])
def test_short_utterance_ok(text, conv, ok):
    assert short_utterance_ok(text, conv) is ok


def _run_short(monkeypatch, transcript, in_conversation=True, safety_only=False):
    halts, llm = [], []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: transcript)
    monkeypatch.setattr(cm, "_dispatch_safety", lambda intent: halts.append(intent["intent"]) or True)
    monkeypatch.setattr(cm, "_start_hush", lambda: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "_resume_talk_after_halt", lambda gen: None)
    monkeypatch.setattr(cm, "_auto_arm_at_turn", lambda gen: None)
    monkeypatch.setattr(cm, "parse_intent", lambda t: llm.append(t) or None)
    info = {}
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], info=info,
                         has_wake_detector=True, safety_only=safety_only,
                         short_filter=True, in_conversation=in_conversation)
    cm._abort_chain.clear()
    return halts, llm, info


def test_short_path_executes_quiet_stop(monkeypatch):
    halts, llm, info = _run_short(monkeypatch, "Stop.")
    assert halts == ["stop"] and llm == [] and not info.get("reached_llm")


def test_short_path_executes_stop_in_wake_word_state(monkeypatch):
    halts, llm, info = _run_short(monkeypatch, "freeze", in_conversation=False, safety_only=True)
    assert halts == ["freeze"] and not info.get("reached_llm")


def test_short_path_drops_non_commands(monkeypatch):
    halts, llm, info = _run_short(monkeypatch, "what's your name")
    assert halts == [] and llm == [] and not info.get("reached_llm")


def test_short_path_lets_direct_commands_through(monkeypatch):
    halts, llm, info = _run_short(monkeypatch, "Sit.")
    assert halts == [] and info.get("reached_llm") and llm == ["Sit"]


def test_short_path_answers_only_mid_conversation(monkeypatch):
    _, llm, info = _run_short(monkeypatch, "Yes.", in_conversation=True)
    assert info.get("reached_llm")
    _, llm, info = _run_short(monkeypatch, "Yes.", in_conversation=False)
    assert not info.get("reached_llm")


def test_short_path_commands_only_in_wake_word_state(monkeypatch):
    _, llm, info = _run_short(monkeypatch, "Sit.", in_conversation=False, safety_only=True)
    assert not info.get("reached_llm") and llm == []


# ---- safety rescue for lock/voice-rejected short utterances ---------------
def test_rescue_executes_stop(monkeypatch):
    halts = []
    monkeypatch.setattr(cm, "_dispatch_safety", lambda intent: halts.append(intent["intent"]) or True)
    monkeypatch.setattr(cm, "_start_hush", lambda: None)
    text, done = cm.rescue_safety(None, b"\0\0" * 1600, asr=lambda stub, pcm: "Stop.")
    assert done is True and text == "Stop." and halts == ["stop"]
    cm._abort_chain.clear()


def test_rescue_ignores_non_safety_text(monkeypatch):
    calls = []
    text, done = cm.rescue_safety(None, b"\0\0" * 1600, asr=lambda stub, pcm: "sit down",
                                  run_safety=lambda t: calls.append(t) or False)
    assert done is False and text == "sit down" and calls  # checked, nothing executed


def test_rescue_survives_asr_failure():
    def boom(stub, pcm):
        raise RuntimeError("asr down")
    assert cm.rescue_safety(None, b"\0\0" * 1600, asr=boom) == ("", False)
    assert cm.rescue_safety(None, b"", asr=boom) == ("", False)


# ---- voiceprint threshold ---------------------------------------------------
def test_voiceprint_threshold_command():
    assert cm.apply_voiceprint_arg("threshold 0.3", False, 0.25) == (False, 0.3)
    assert cm.apply_voiceprint_arg("threshold 1.5", True, 0.25) == (True, 0.9)
    assert cm.apply_voiceprint_arg("threshold -2", True, 0.25) == (True, 0.0)
    assert cm.apply_voiceprint_arg("on", False, 0.25) == (True, 0.25)
    assert cm.apply_voiceprint_arg(None, True, 0.25) == (False, 0.25)
    for bad in ("threshold", "threshold abc", "threshold nan"):
        with pytest.raises(ValueError):
            cm.apply_voiceprint_arg(bad, False, 0.25)


def test_voiceprint_threshold_arg_reaches_the_queue():
    assert parse_command("/cmd/voiceprint", {"arg": "threshold 0.3"}) == ("voiceprint", "threshold 0.3")


def test_status_shows_voiceprint_threshold(monkeypatch):
    monkeypatch.setattr(cm, "_voiceprint_threshold", 0.42)
    assert cm._status_snapshot()["voiceprint_threshold"] == 0.42
