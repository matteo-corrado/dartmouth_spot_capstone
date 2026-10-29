"""client_mic front-end wiring helpers (Task 10): DoA filtering, timeout,
trailing trim, metrics records + privacy, endpointer frame math."""
import json
import math

import src.spot.audio.capture as _cap
import src.spot.conversation.gate as conv_gate
import src.spot.conversation.utterance_log as _ul
import src.spot.audio.capture as _cap
from src.spot.audio.frontend import UtteranceStats
from src.spot.conversation.speaker_lock import SpeakerLock


def test_speech_degs_filters_non_speech_and_nan():
    samples = [(1.0, 10, True), (1.1, 200, False), (1.2, float("nan"), True), (1.3, 15, True)]
    assert conv_gate.speech_degs(samples) == [10.0, 15.0]


def test_dbfs_from_rms():
    assert _cap.dbfs_from_rms(1.0) == 0.0
    assert math.isclose(_cap.dbfs_from_rms(0.01), -40.0)
    assert _cap.dbfs_from_rms(0.0) == -120.0


def test_endpointer_frames_from_tuning():
    assert _cap._frames(1.5) == 50
    assert _cap._frames(20.0) == 667
    assert _cap._frames(0.0) == 1


def test_conversation_expired_locked_uses_lock():
    lock = SpeakerLock(35, 20, 0.3)
    lock.lock([90.0], now=100.0)
    assert not conv_gate.conversation_expired(lock, listening_since=0.0, now=110.0, timeout_s=20)
    assert conv_gate.conversation_expired(lock, listening_since=0.0, now=121.0, timeout_s=20)


def test_conversation_expired_unlocked_falls_back_to_wall_clock():
    lock = SpeakerLock(35, 20, 0.3)
    assert not conv_gate.conversation_expired(lock, listening_since=100.0, now=115.0, timeout_s=20)
    assert conv_gate.conversation_expired(lock, listening_since=100.0, now=121.0, timeout_s=20)


def _utt(flags):
    return [(i * 0.03, b"", -30.0, f) for i, f in enumerate(flags)]


def test_trim_trailing_keeps_n_after_last_speech():
    u = _utt([False, True, True] + [False] * 50)
    out = conv_gate.trim_trailing(u, 10)
    assert len(out) == 3 + 10
    assert conv_gate.trim_trailing(_utt([False] * 5), 10) == _utt([False] * 5)


def test_last_speech_t():
    u = _utt([True, True, False, False])
    assert conv_gate.last_speech_t(u) == 0.03


def _stats():
    s = UtteranceStats()
    for _ in range(20):
        s.add(-30.0, True)
    return s


def test_utterance_record_has_all_keys():
    r = _ul.build_utterance_record("end", 1.2, 2.0, -60.0, _stats(), 90.0, 95.0,
                                  accepted=True, transcript="hi", reached_llm=True,
                                  latency_s=1.234)
    for k in ("ts", "floor_dbfs", "speech_dbfs_median", "speech_dbfs_p90", "snr_db",
              "endpoint", "end_silence_s", "duration_s", "lock_mic_deg", "utt_mic_deg",
              "lock_body_deg", "utt_body_deg", "accepted", "transcript", "reached_llm"):
        assert k in r
    assert r["snr_db"] == 30.0 and r["latency_s"] == 1.23
    json.dumps(r)


def test_utterance_log_privacy_nulls_transcript(tmp_path):
    log = _ul.UtteranceLog(tmp_path, store_transcript=False)
    log.write(_ul.build_utterance_record("end", 1.2, 2.0, -60.0, _stats(), None, None,
                                        accepted=True, transcript="secret"))
    log.write(_ul.build_idle_record("WAKE_WORD", -60.0, -55.0, None))
    lines = [json.loads(x) for f in tmp_path.iterdir() for x in f.read_text().splitlines()]
    assert lines[0]["transcript"] is None and lines[0]["lock_body_deg"] is None
    assert lines[1]["kind"] == "idle"


def test_utterance_log_io_error_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = _ul.UtteranceLog(blocker / "sub")  # mkdir under a file fails
    log.write({"kind": "idle"})
    log.write({"kind": "idle"})  # disabled after first failure, still no raise


def test_mic_absent_degrades(monkeypatch):
    monkeypatch.setattr(_cap.CAPTURE, "_mic", None)
    assert _cap._mic_usable() is False
    assert _cap._mic_speech_degs(0, 1) == []


def test_dead_mic_degrades(monkeypatch):
    class Dead:
        dead = True

        def samples_between(self, a, b):
            raise AssertionError("must not read a dead mic")

    monkeypatch.setattr(_cap.CAPTURE, "_mic", Dead())
    monkeypatch.setattr(_cap.CAPTURE, "_mic_dead_logged", False)
    assert _cap._mic_speech_degs(0, 1) == []
    _cap._check_mic_alive()
    assert _cap.CAPTURE._mic_dead_logged is True


def test_conversation_clock_resets_only_for_turns_that_reached_llm():
    from src.spot.conversation.speaker_lock import SpeakerLock
    L = SpeakerLock(35, 20, 0.3); L.lock([10], 0)
    # loud noise passed the gate but never reached intent/LLM: no refresh
    assert conv_gate.refresh_conversation(L, 0.0, 15.0, reached_llm=False) == 0.0
    assert L.expired(20.5) is True
    # a processed turn restarts both clocks
    assert conv_gate.refresh_conversation(L, 0.0, 15.0, reached_llm=True) == 15.0
    assert L.expired(20.5) is False
    # unlocked fallback uses the returned listening_since
    U = SpeakerLock(35, 20, 0.3)
    assert conv_gate.conversation_expired(U, conv_gate.refresh_conversation(U, 0.0, 15.0, False), 21.0, 20) is True
    assert conv_gate.conversation_expired(U, conv_gate.refresh_conversation(U, 0.0, 15.0, True), 21.0, 20) is False
