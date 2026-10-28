import csv
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np

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


def test_run_session_writes_files_and_handles_skip_redo_quit(tmp_path):
    ps = build_prompts(["Hey Spot"])[:3]
    calls = []

    def rec(p):
        calls.append(p.clip)
        return np.full((1600, 2), 1000, dtype=np.int16), 10.0, [(10.5, 90, True), (11.0, 95, False)]

    # clip1: record, redo, record, keep; clip2: skip; clip3: quit
    answers = iter(["", "r", "", "", "s", "q"])
    n = rdc.run_session(ps, tmp_path, rec, input_fn=lambda _: next(answers))
    assert n == 2 and calls == [ps[0].clip, ps[0].clip]
    with wave.open(str(tmp_path / f"{ps[0].clip}.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 16000, 2)
    rows = list(csv.reader(open(tmp_path / f"{ps[0].clip}.doa.csv")))
    assert rows[0] == ["t_rel_s", "deg", "speech"] and rows[1] == ["0.500", "90", "1"]
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert len(m) == 1 and set(m[0]) == {"clip", "kind", "phrase", "expected_text", "angle_deg"}
    assert not (tmp_path / f"{ps[1].clip}.wav").exists()
