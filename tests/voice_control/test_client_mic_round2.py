"""Review fixes: chime mask, hush silences the reply, transcript privacy."""
from collections import deque

import src.voice_control.client_mic as cm
from src.voice_control.audio_frontend import Endpointer


def _simulate(frames, tail_s=0.15):
    """Mirror of the loop's per-frame path: (t, is_speech, chime_busy)
    -> list of utterances (each a list of frame ids incl. pre-roll)."""
    mask = cm.ChimeMask(tail_s)
    ep = Endpointer(3, 10, 600)
    pre = deque(maxlen=cm.PREROLL_FRAMES + 3)
    utts, cur, in_utt = [], None, False
    for i, (t, sp, chime) in enumerate(frames):
        masked, _ended = mask.update(chime, t)
        if masked:
            sp = False
        pre.append((i, sp))
        ev = ep.feed(sp, -30.0 if sp else -70.0, -70.0)
        if ev == "start":
            in_utt, cur = True, [f for f, _ in pre]
        elif in_utt:
            cur.append(i)
        if ev in ("end", "cap") and in_utt:
            in_utt = False
            utts.append(cur)
    return utts


def test_chime_only_frames_make_no_utterance():
    # VAD flags the chime itself as speech; it must not start an utterance.
    frames = [(i * 0.03, True, i < 8) for i in range(10)] + \
             [(0.3 + i * 0.03, False, False) for i in range(40)]
    assert _simulate(frames) == []


def test_speech_right_after_chime_kept_with_first_frames():
    frames = [(i * 0.03, True, True) for i in range(5)]            # chime 0..4
    frames += [(0.15 + i * 0.03, True, False) for i in range(20)]  # speech 5..24
    frames += [(0.75 + i * 0.03, False, False) for i in range(30)]
    utts = _simulate(frames)
    assert len(utts) == 1
    assert 9 in utts[0]  # first unmasked speech frame (t=0.27) is kept via pre-roll


def test_chime_mask_edges():
    m = cm.ChimeMask(0.15)
    assert m.update(False, 0.0) == (False, False)
    assert m.update(True, 1.0) == (True, False)
    assert m.update(False, 1.1) == (True, False)    # within tail
    assert m.update(False, 1.2) == (False, True)    # lifts: ended once
    assert m.update(False, 1.3) == (False, False)


class _Chunker:
    def __init__(self):
        self.suppressed = False

    def accept(self, d):
        pass


class _Filler:
    cancelled = False

    def cancel(self):
        self.cancelled = True


def test_hush_mid_stream_suppresses_rest_of_reply():
    c, f = _Chunker(), _Filler()
    w = cm._TokenWatch(c, f)
    w('{"actions": [], "response": "Hello')
    assert c.suppressed is False
    cm._abort_chain.set()
    try:
        w(' there."}')
        assert c.suppressed is True and f.cancelled
    finally:
        cm._abort_chain.clear()


def test_say_guarded_by_abort():
    said = []

    class T:
        def speak(self, text, voice=None):
            said.append(text)

    cm._abort_chain.clear()
    cm._say(T(), "one")
    cm._abort_chain.set()
    cm._say(T(), "two")
    cm._abort_chain.clear()
    assert said == ["one"]


def test_heard_redacts_without_transcript_log(monkeypatch):
    monkeypatch.setattr(cm, "_store_transcript", True)
    assert cm._heard("go to the lab") == '"go to the lab"'
    monkeypatch.setattr(cm, "_store_transcript", False)
    assert cm._heard("go to the lab") == "<heard 4 words>"


def test_enter_state_updates_status(monkeypatch):
    monkeypatch.setattr(cm, "_mic", None)
    monkeypatch.setattr(cm, "_state_leds", None)
    monkeypatch.setattr(cm, "chime_for", lambda s: None)
    cm._enter_state(cm.VoiceState.THINKING)
    assert cm._status["state"] == "THINKING"
