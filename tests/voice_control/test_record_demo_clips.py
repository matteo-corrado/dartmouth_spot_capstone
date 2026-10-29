import csv
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[2] / "scripts" / "record_demo_clips.py"
_spec = importlib.util.spec_from_file_location("record_demo_clips", _p)
rdc = importlib.util.module_from_spec(_spec)
sys.modules["record_demo_clips"] = rdc
_spec.loader.exec_module(rdc)
build_prompts = rdc.build_prompts


def test_build_prompts_counts_and_names():
    ps = build_prompts(["Hey Spot", "Hello Spotty"])
    names = [p.clip for p in ps]
    assert "wake_hey_spot_left" in names and "pause_hello_spotty" in names
    assert sum(p.kind == "wake" for p in ps) == 8
    assert sum(p.kind == "background" for p in ps) == 1
    assert {p.angle_deg for p in ps if p.kind == "wake"} == {0, 90, 180, 270}
    assert sum(p.kind == "quick_reply" for p in ps) == 3
    assert sum(p.kind == "overlap" for p in ps) == 3
    assert [p.duration_s for p in ps if p.kind == "background"] == [120.0]


def test_readable_casing():
    ps = build_prompts(["LISTEN UP SPOT"])
    assert ps[0].say == "Listen up Spot, what is this poster about?"
    assert ps[0].clip == "wake_listen_up_spot_front"


def test_load_phrases(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("HEY SPOT\n\nEXCUSE ME SPOT\n")
    assert rdc.load_phrases(None, f) == ["HEY SPOT", "EXCUSE ME SPOT"]
    assert rdc.load_phrases("A|B", f) == ["A", "B"]
    assert rdc.load_phrases(None, tmp_path / "missing.txt") == ["HEY SPOT"]


def test_collect_doa_merges_chunks():
    class Mic:
        h = [(1.0, 10, True), (2.0, 20, False), (3.0, 30, True)]

        def samples_between(self, a, b):
            return [s for s in self.h if a <= s[0] <= b]

    ticks = iter([0.0, 1.5, 3.5, 5.0, 5.0])
    got = rdc.collect_doa(Mic(), 0.0, 4.0, clock=lambda: next(ticks), sleep=lambda _: None)
    assert [s[1] for s in got] == [10, 20, 30]


def _seq(items):
    it = iter(items)
    return lambda _: next(it)


def _rec(calls=None, doa=True):
    def rec(p):
        if calls is not None:
            calls.append(p.clip)
        d = [(10.5, 90, True), (11.0, 95, False)] if doa else []
        return np.full((1600, 2), 1000, dtype=np.int16), 10.0, d, None
    return rec


def test_run_session_writes_files_and_handles_skip_redo_quit(tmp_path):
    ps = build_prompts(["Hey Spot"])[:3]
    calls = []
    # clip1: record, redo, record, keep; clip2: skip; clip3: quit
    answers = iter(["", "r", "", "", "s", "q"])
    n = rdc.run_session(ps, tmp_path, _rec(calls), input_fn=lambda _: next(answers))
    assert n == 2 and calls == [ps[0].clip, ps[0].clip]
    with wave.open(str(tmp_path / f"{ps[0].clip}.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 16000, 2)
    rows = list(csv.reader(open(tmp_path / f"{ps[0].clip}.doa.csv")))
    assert rows[0] == ["t_rel_s", "deg", "speech"] and rows[1] == ["0.500", "90", "1"]
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert len(m) == 1 and set(m[0]) == {"clip", "kind", "phrase", "expected_text", "angle_deg"}
    assert not (tmp_path / f"{ps[1].clip}.wav").exists()


def test_existing_clip_defaults_to_skip_and_redo_removes_stale_csv(tmp_path):
    ps = build_prompts(["Hey Spot"])[:1]
    rdc.run_session(ps, tmp_path, _rec(), input_fn=_seq(["", ""]))
    csvp = tmp_path / f"{ps[0].clip}.doa.csv"
    assert csvp.exists()
    calls = []
    rdc.run_session(ps, tmp_path, _rec(calls), input_fn=_seq([""]))  # default skip
    assert calls == []
    (tmp_path / f"{ps[0].clip}.chime.json").write_text("{}")
    rdc.run_session(ps, tmp_path, _rec(calls, doa=False), input_fn=_seq(["r", ""]))
    assert calls and not csvp.exists()
    assert not (tmp_path / f"{ps[0].clip}.chime.json").exists()


def test_record_exception_returns_to_prompt(tmp_path):
    ps = build_prompts(["Hey Spot"])[:1]
    state = {"n": 0}

    def rec(p):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("usb glitch")
        return _rec()(p)

    n = rdc.run_session(ps, tmp_path, rec, input_fn=_seq(["", "", ""]))
    assert n == 1 and (tmp_path / f"{ps[0].clip}.wav").exists()


def test_eof_exits_cleanly_keeping_manifest(tmp_path):
    ps = build_prompts(["Hey Spot"])[:2]
    answers = iter(["", ""])  # record clip 1, keep, then input raises StopIteration->EOF
    def inp(_):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError
    assert rdc.run_session(ps, tmp_path, _rec(), input_fn=inp) == 1
    assert len(json.loads((tmp_path / "manifest.json").read_text())) == 1


def test_manifest_backup_on_corrupt(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text("{not json")
    rdc.save_manifest_entry(m, build_prompts(["Hey Spot"])[0])
    assert len(json.loads(m.read_text())) == 1
    assert list(tmp_path.glob("manifest.bad.*.json"))


def test_quick_reply_records_chime_sidecar(tmp_path, monkeypatch):
    import types
    played = {}

    class FakeOut:
        def __init__(self, **kw):
            played["dev"] = kw.get("device")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, x):
            played["n"] = len(x)
            end = rdc.time.monotonic() + 0.02
            while rdc.time.monotonic() < end:
                pass

    fake_sd = types.SimpleNamespace(
        rec=lambda n, **kw: np.zeros((n, 2), dtype=np.int16),
        wait=lambda: None, OutputStream=FakeOut)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(rdc, "CHIME_LEAD_S", 0.05)
    monkeypatch.setattr(rdc.time, "sleep", lambda s: None)
    p = [x for x in build_prompts(["Hey Spot"]) if x.kind == "quick_reply"][0]
    rdc_audio, t0, doa, chime, beams = rdc.record_clip(0, p, None, output_device=7)
    assert beams == []  # no MicArray -> no beam log
    assert played["dev"] == 7 and played["n"] > 0
    assert 0 <= chime["chime_start_s"] < chime["chime_end_s"] <= p.duration_s
    assert chime["source"] in ("audio_feedback.listen_cue", "audio_feedback.ready", "generated_tone")
    # sidecar written through run_session
    rdc.run_session([p], tmp_path, lambda q: (rdc_audio, t0, doa, chime), input_fn=_seq(["", ""]))
    side = json.loads((tmp_path / f"{p.clip}.chime.json").read_text())
    assert side["chime_start_s"] < side["chime_end_s"]


def test_body_to_mic_inverts_mic_to_body():
    from src.voice_control.speaker_lock import mic_to_body_deg
    for body in (0, 90, 180, 270, 33.5):
        for off, sign in ((179.0, -1), (0.0, 1), (45.0, 1)):
            mic = rdc.body_to_mic_deg(body, off, sign)
            assert mic_to_body_deg(mic, off, sign) == pytest.approx(body % 360.0)


def test_run_session_writes_beams_csv_and_manifest_extra(tmp_path):
    import math
    ps = build_prompts(["Hey Spot"])[:1]
    beams = [(10.5, [0.0, 0.0, math.pi / 2, 0.0], [0.1, 0.1, 0.0, 0.1])]

    def rec(p):
        return np.full((1600, 2), 1000, dtype=np.int16), 10.0, [], None, beams

    n = rdc.run_session(ps, tmp_path, rec, input_fn=_seq(["", ""]),
                        manifest_extra={"beam_steered": True, "interferer_body_deg": 90.0})
    assert n == 1
    rows = list(csv.reader(open(tmp_path / f"{ps[0].clip}.beams.csv")))
    assert rows[0] == ["t_rel_s", *rdc.BEAM_COLS]
    assert rows[1][:4] == ["0.500", "0.0", "0.0", "90.0"]
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m[0]["beam_steered"] is True and m[0]["interferer_body_deg"] == 90.0


def test_collect_beams_skips_read_errors():
    t = iter([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3])
    reads = iter([RuntimeError("usb"), ([1.0] * 4, [0.0] * 4)])

    class M:
        def read_beams(self):
            r = next(reads)
            if isinstance(r, Exception):
                raise r
            return r

    got = rdc.collect_beams(M(), 0.0, 0.25, clock=lambda: next(t), sleep=lambda s: None)
    assert len(got) == 1 and got[0][1] == [1.0] * 4
