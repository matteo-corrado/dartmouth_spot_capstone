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


class _LagVAD:
    """Fake Silero: after reset() it needs `lag` speech frames before it flags."""

    def __init__(self, lag=8):
        self.lag, self._n = lag, lag

    def reset(self):
        self._n = 0

    def feed(self, speech):
        if not speech:
            return False
        self._n += 1
        return self._n > self.lag


def test_one_breath_wake_keeps_command_onset_under_listen_cue():
    # "hey spot" (0..9) -> wake at 9, listen_cue + tail mask the next frames
    # while the user keeps going: "stand up" starts at frame 10, under the cue.
    tail_s, cue_frames = 0.15, 5           # listen_cue 140 ms ~ 5 frames
    mask, vad = cm.ChimeMask(tail_s), _LagVAD()
    pre = deque(maxlen=cm.PREROLL_FRAMES + 3)
    ep, utt, in_utt = None, None, False
    for i in range(70):
        t = i * 0.03
        spoken = i < 40                    # one breath, no pause
        chime = 10 <= i < 10 + cue_frames
        sp = vad.feed(spoken)
        masked, ended = mask.update(chime, t, sp)
        if masked:
            sp = False
        elif mask.should_reset_vad(ended, in_utt):
            vad.reset()
        pre.append(i)
        if i == 9:
            ep = Endpointer(3, 10, 600)    # wake: fresh endpointer, pre-roll kept
            continue
        if ep is None:
            continue
        ev = ep.feed(sp, -30.0 if sp else -70.0, -70.0)
        if ev == "start":
            in_utt, utt = True, list(pre)
    assert utt is not None and 10 in utt   # the "stand" onset survived
    assert cm.PREROLL_FRAMES * cm.FRAME_S >= 0.14 + tail_s + 0.25  # cue + tail + Silero lag


def test_vad_reset_after_idle_chime_but_not_mid_speech():
    m = cm.ChimeMask(0.15)
    m.update(False, 0.0, False)
    m.update(True, 0.03, False)
    _, ended = m.update(False, 0.3, False)
    assert m.should_reset_vad(ended, in_utt=False) is True   # idle: clear hangover
    m = cm.ChimeMask(0.15)
    m.update(False, 0.0, True)                               # user already talking
    m.update(True, 0.03, True)
    _, ended = m.update(False, 0.3, True)
    assert ended and m.should_reset_vad(ended, in_utt=False) is False


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
