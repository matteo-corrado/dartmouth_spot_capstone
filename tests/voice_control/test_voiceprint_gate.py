"""Voiceprint wiring (Task 18 steps 5-6): VoiceprintGate lifecycle + accept rule,
fail-open paths, warm-up, and the spotctl voiceprint toggle / status fields."""
import queue

import numpy as np
import pytest

from src.voice_control import speaker_embed as se
from src.voice_control.control_server import ControlServer, parse_command
from src.voice_control.speaker_embed import VoiceprintGate, warm_up_async


def _vec(*xs):
    v = np.zeros(256, np.float32)
    v[:len(xs)] = xs
    return v


class FakeEmbedder:
    """Maps the first byte of the PCM to a fixed vector; records calls."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def embed(self, pcm16):
        self.calls += 1
        return self.table[pcm16[:1]]


A, B = b"A" * 960, b"B" * 960            # speaker A, speaker B
A2 = b"a" * 960                          # speaker A again (slightly different)
TABLE = {b"A": _vec(1, 0), b"a": _vec(0.9, 0.1), b"B": _vec(0, 1)}


def _gate(log=None):
    return VoiceprintGate(FakeEmbedder(TABLE), log_fn=(log.append if log is not None else lambda m: None))


def test_enrol_and_clear_lifecycle():
    g = _gate()
    assert not g.enrolled and g.last_sim is None
    g.set_wake(A)
    assert not g.enrolled and g.wake_emb is not None
    ok, sim = g.decide(A2, None, enabled=True, threshold=0.5)   # first utterance: direction-only, enrols
    assert ok and g.enrolled
    assert np.linalg.norm(g.enrol) == pytest.approx(1.0, abs=1e-5)
    # enrolment = normalised mean of wake + first utterance
    want = _vec(1, 0) + _vec(0.9, 0.1) / np.linalg.norm(_vec(0.9, 0.1))
    assert se.cosine(g.enrol, want) == pytest.approx(1.0, abs=1e-5)
    first_enrol = g.enrol.copy()
    g.decide(A, None, True, 0.5)
    assert np.array_equal(g.enrol, first_enrol)                 # later utterances don't re-enrol
    g.clear()                                                   # lock release
    assert not g.enrolled and g.wake_emb is None
    assert g.last_sim is not None                               # score survives for spotctl status


def test_takeover_wake_restarts_enrolment():
    g = _gate()
    g.set_wake(A)
    g.decide(A, None, True, 0.5)
    assert g.enrolled
    g.set_wake(B)                                               # new speaker says the wake phrase
    assert not g.enrolled
    ok, _ = g.decide(B, None, True, 0.5)
    assert ok and se.cosine(g.enrol, _vec(0, 1)) == pytest.approx(1.0, abs=1e-5)
    ok, sim = g.decide(A, None, True, 0.5)                      # the old speaker is now rejected
    assert not ok and sim == pytest.approx(0.0, abs=1e-5)


def test_running_enrolment_folds_in_confident_accepts():
    g = _gate()
    g.set_wake(A)
    g.decide(A2, None, True, 0.25, update_threshold=0.35)       # completes enrolment (wake + first)
    assert g.n_enrol == 2
    g.decide(A, None, True, 0.25, update_threshold=0.35)        # sim ~1 >= 0.35 -> folded in
    assert g.n_enrol == 3
    want = 2 * _vec(1, 0) + _vec(0.9, 0.1) / np.linalg.norm(_vec(0.9, 0.1))
    assert se.cosine(g.enrol, want) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(g.enrol) == pytest.approx(1.0, abs=1e-5)
    g.clear()
    assert g.n_enrol == 0


def test_running_enrolment_never_learns_a_low_score_voice():
    g = _gate()
    g.set_wake(A)
    g.decide(A, True, True, 0.25, update_threshold=0.35)
    before = g.enrol.copy()
    ok, _ = g.decide(B, True, False, 0.25, update_threshold=0.35)   # gate OFF: accepted by direction
    assert ok and np.array_equal(g.enrol, before) and g.n_enrol == 2
    ok, _ = g.decide(B, True, True, 0.25, update_threshold=0.35)    # gate ON: rejected
    assert not ok and np.array_equal(g.enrol, before)
    ok, _ = g.decide(A, False, True, 0.25, update_threshold=0.35)   # direction veto: no update
    assert not ok and np.array_equal(g.enrol, before)


def test_update_bar_is_never_below_accept_threshold():
    g = _gate()
    g.set_wake(A)
    g.decide(A, None, True, 0.5)
    sim = se.cosine(g.enrol, _vec(0.9, 0.1))                      # ~0.99, so use a bar above it
    g.decide(A2, None, False, sim + 0.005, update_threshold=0.0)
    assert g.n_enrol == 2


def test_takeover_resets_running_enrolment():
    g = _gate()
    g.set_wake(A)
    g.decide(A, None, True, 0.25, update_threshold=0.35)
    g.decide(A2, None, True, 0.25, update_threshold=0.35)
    assert g.n_enrol == 3
    g.set_wake(B)
    assert g.n_enrol == 0
    g.decide(B, None, True, 0.25, update_threshold=0.35)
    assert g.n_enrol == 2 and se.cosine(g.enrol, _vec(0, 1)) == pytest.approx(1.0, abs=1e-5)


def test_accept_with_voiceprint_on():
    g = _gate()
    g.set_wake(A)
    g.decide(A, True, True, 0.5)
    assert g.decide(A2, True, True, 0.5)[0] is True             # same voice, right direction
    assert g.decide(B, True, True, 0.5)[0] is False             # other voice, right direction
    assert g.decide(A, False, True, 0.5)[0] is False            # direction still vetoes
    assert g.decide(A, None, True, 0.5)[0] is True              # no direction: voice decides
    assert g.decide(B, None, True, 0.5)[0] is False
    assert g.last_sim == pytest.approx(0.0, abs=1e-5)


def test_voiceprint_off_is_direction_only_but_still_scores():
    g = _gate()
    g.set_wake(A)
    g.decide(A, True, False, 0.5)
    ok, sim = g.decide(B, True, False, 0.5)
    assert ok is True and sim == pytest.approx(0.0, abs=1e-5)   # sim logged, not gating
    assert g.decide(A, False, False, 0.5)[0] is False
    assert g.decide(B, None, False, 0.5)[0] is True


def test_first_utterance_rejected_by_direction_does_not_enrol():
    g = _gate()
    g.set_wake(A)
    ok, _ = g.decide(B, False, True, 0.5)
    assert not ok and not g.enrolled


def test_no_wake_audio_enrols_from_first_utterance():
    g = _gate()                                                 # spotctl listen: no wake
    ok, sim = g.decide(A, None, True, 0.5)
    assert ok and sim is None and g.enrolled
    assert g.decide(B, None, True, 0.5)[0] is False


def test_model_missing_fails_open(tmp_path):
    log = []
    g = VoiceprintGate(None, log_fn=log.append)
    assert warm_up_async(g, tmp_path / "nope.onnx", log_fn=log.append) is None
    assert g.embedder is None and any("missing" in m for m in log)
    g.set_wake(A)
    assert g.decide(B, True, True, 0.5) == (True, None)
    assert g.decide(B, None, True, 0.5) == (True, None)
    assert g.decide(B, False, True, 0.5) == (False, None)
    g.decide(A, True, True, 0.5)
    assert sum("not loaded" in m for m in log) == 1             # logged once


def test_embedding_error_fails_open_and_logs_once():
    class Boom:
        def embed(self, pcm16):
            raise ValueError("audio too short")

    log = []
    g = VoiceprintGate(Boom(), log_fn=log.append)
    g.set_wake(A)
    for _ in range(3):
        assert g.decide(B, True, True, 0.5) == (True, None)
    assert g.decide(B, False, True, 0.5) == (False, None)
    assert sum("embedding failed" in m for m in log) == 1


def test_embedding_error_after_enrolment_fails_open():
    class Flaky(FakeEmbedder):
        fail = False

        def embed(self, pcm16):
            if self.fail:
                raise RuntimeError("cuda")
            return super().embed(pcm16)

    e = Flaky(TABLE)
    g = VoiceprintGate(e, log_fn=lambda m: None)
    g.set_wake(A)
    g.decide(A, True, True, 0.5)
    e.fail = True
    assert g.decide(B, True, True, 0.5) == (True, None)         # direction-only
    assert g.enrolled


def test_warm_up_assigns_embedder_last(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    seen = []

    class F:
        def __init__(self, path):
            seen.append(path)

        def embed(self, pcm16):
            assert g.embedder is None                           # not visible until warmed
            return _vec(1)

    g = VoiceprintGate(None, log_fn=lambda m: None)
    t = warm_up_async(g, model, log_fn=lambda m: None, factory=F)
    t.join(5)
    assert isinstance(g.embedder, F) and seen == [model]


def test_warm_up_failure_leaves_none(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    def bad(path):
        raise RuntimeError("bad onnx")

    log = []
    g = VoiceprintGate(None, log_fn=log.append)
    warm_up_async(g, model, log_fn=log.append, factory=bad).join(5)
    assert g.embedder is None and any("failed to load" in m for m in log)


def test_gate_never_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = _gate()
    g.set_wake(A)
    g.decide(A, True, True, 0.5)
    g.decide(B, True, True, 0.5)
    g.clear()
    assert list(tmp_path.iterdir()) == []


# ---- spotctl voiceprint on|off + status --------------------------------------
def test_voiceprint_toggle_arg_is_forwarded():
    assert parse_command("/cmd/voiceprint", {"arg": "on"}) == ("voiceprint", "on")
    assert parse_command("/cmd/voiceprint", {"arg": "off"}) == ("voiceprint", "off")
    assert parse_command("/cmd/voiceprint", {}) == ("voiceprint", None)
    assert parse_command("/cmd/crowd", {"arg": "on"}) == ("crowd", "on")
    assert parse_command("/cmd/hush", {"arg": "on"}) == ("hush", None)


def test_spotctl_voiceprint_on_reaches_queue():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "spotctl.py"
    spec = importlib.util.spec_from_file_location("spotctl_vp", p)
    spotctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spotctl)
    q = queue.Queue()
    srv = ControlServer(port=0, on_immediate=lambda n, a: {"ok": True}, queue=q)
    srv.start()
    try:
        assert spotctl.main(["--port", str(srv.port), "voiceprint", "on"]) == 0
        assert q.get(timeout=2) == ("voiceprint", "on")
        assert spotctl.main(["--port", str(srv.port), "voiceprint", "off"]) == 0
        assert q.get(timeout=2) == ("voiceprint", "off")
    finally:
        srv.stop()


def test_client_toggle_and_status_fields(monkeypatch):
    import src.voice_control.client_mic as cm
    assert cm._toggle(False, "on") is True and cm._toggle(True, "off") is False
    assert cm._toggle(False, None) is True
    g = VoiceprintGate(FakeEmbedder(TABLE), log_fn=lambda m: None)
    monkeypatch.setattr(cm, "_voice", g)
    monkeypatch.setattr(cm, "_voiceprint_mode", True)
    st = cm._on_immediate("status", None)
    assert st["voiceprint"] is True and st["voiceprint_enrolled"] is False
    assert st["voiceprint_last_sim"] is None and st["voiceprint_ready"] is True
    g.set_wake(A)
    g.decide(A, True, True, 0.5)
    g.decide(A2, True, True, 0.5)
    st = cm._on_immediate("status", None)
    assert st["voiceprint_enrolled"] is True
    assert 0.9 < st["voiceprint_last_sim"] <= 1.0


def test_voiceprint_default_comes_from_tuning():
    import src.voice_control.client_mic as cm
    src = open(cm.__file__).read()
    assert "_voiceprint_mode = _tuning.voiceprint_enabled" in src
