#!/usr/bin/env python3
"""Beam-steering A/B: compare a beams-off and a beams-on recording of the same room.

Record both back to back with scripts/record_demo_clips.py (--set overlap, same
--interferer-deg, the second run with --beam), then:

    spot-env/bin/python tests/audio/beam_ab.py --off ab_off --on ab_on

Per condition, on the overlap clips (visitor at the front, second talker keeps going):
  interferer dBFS  level of the podcast-only tail after the visitor stops (lower = suppressed)
  S+I / I dB       visitor window level minus tail level (higher = visitor stands out more)
  voiceprint       same-speaker vs other-voice cosine medians (replay_demo calibration)
  DoA in tail      where DOA_VALUE and the free-running beam point while only the
                   second talker speaks: near the beam (front) means DoA follows the
                   fixed beam (self-confirming lock); near the interferer means it still
                   hears them, which the speaker lock needs.
Wake / lock / ASR per condition: run replay_demo.py on both (printed at the end).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import replay_demo as R  # noqa: E402
from src.voice_control.speaker_lock import ang_diff, circular_median_deg, mic_to_body_deg  # noqa: E402
from src.voice_control.tuning import load_tuning  # noqa: E402


def dbfs(pcm: np.ndarray) -> float:
    x = pcm.astype(np.float64) / 32768.0
    return float(10 * np.log10(np.mean(x * x) + 1e-12))


def read_beams(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def tail_span(pcm: np.ndarray, dur: float):
    """(visitor window, podcast-only tail) in seconds, or None if no speech found."""
    ref = R.reference_speech(pcm)
    if not ref:
        return None
    on = ref[0][0]
    v_end = on + R.VP_VISITOR_S
    t0 = v_end + R.VP_PAD_S
    if dur - t0 < 0.5:
        return None
    return (on, v_end), (t0, dur)


def analyse(root: Path, cond: str, ch: int, cfg, model: Path | None) -> dict:
    off, sign = cfg.mic_yaw_offset_deg, cfg.mic_yaw_sign
    body = lambda d: mic_to_body_deg(d, off, sign)  # noqa: E731
    rows = []
    for c in R.load_condition(root, cond):
        if c["kind"] != "overlap":
            continue
        pcm = c["audio"][:, ch]
        dur = len(pcm) / R.SAMPLE_RATE
        spans = tail_span(pcm, dur)
        if spans is None:
            continue
        (v0, v1), (t0, t1) = spans
        sl = lambda a, b: pcm[int(a * R.SAMPLE_RATE):int(b * R.SAMPLE_RATE)]  # noqa: E731
        i_db, v_db = dbfs(sl(t0, t1)), dbfs(sl(v0, v1))
        doa = [d for t, d, s in c["doa"] if s and t0 <= t <= t1]
        beams = [b for b in read_beams(root / cond / f"{c['clip']}.beams.csv") if t0 <= b["t_rel_s"] <= t1]
        free = [b["az_free"] for b in beams if b["en_free"] > 0]
        rows.append(dict(
            clip=c["clip"], interferer_dbfs=round(i_db, 1), sir_db=round(v_db - i_db, 1),
            doa_tail_body=None if not doa else round(body(circular_median_deg(doa)), 0),
            free_tail_body=None if not free else round(body(circular_median_deg(free)), 0),
            en_free=None if not beams else round(float(np.median([b["en_free"] for b in beams])), 4),
            en_beam=None if not beams else round(float(np.median([b["en1"] for b in beams])), 4),
            interferer_body_deg=c.get("interferer_body_deg"), beam=c.get("beam_steered"),
            target_body_deg=c["angle_deg"]))
    res = dict(cond=cond, clips=rows)
    if rows:
        res["interferer_dbfs"] = round(float(np.median([r["interferer_dbfs"] for r in rows])), 1)
        res["sir_db"] = round(float(np.median([r["sir_db"] for r in rows])), 1)
    if model is not None and model.exists():
        vp = R.voiceprint_calibration(root, [cond], ch, model)
        res["vp_same"] = vp["same"].get("median")
        res["vp_other"] = vp["other"].get("median")
    return res


def verdict_doa(r) -> str:
    tgt, intf = r["target_body_deg"], r["interferer_body_deg"]
    d = r["doa_tail_body"]
    if d is None or intf is None:
        return "-"
    return "follows beam" if ang_diff(d, tgt) < ang_diff(d, intf) else "hears interferer"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(R.DEFAULT_ROOT))
    ap.add_argument("--off", required=True, help="beams-off condition dir")
    ap.add_argument("--on", required=True, help="beams-on condition dir")
    ap.add_argument("--channel", type=int, default=None, help="default: Tuning.mic_channel")
    ap.add_argument("--speaker-model", default=str(R.MAIN_CHECKOUT / "models" / "speaker"
                                                   / "voxceleb_resnet34_LM.onnx"))
    a = ap.parse_args(argv)
    cfg = load_tuning(REPO / "config" / "tuning.yaml")
    ch = cfg.mic_channel if a.channel is None else a.channel
    root, model = Path(a.root), Path(a.speaker_model)

    res = [analyse(root, c, ch, cfg, model) for c in (a.off, a.on)]
    print(f"## Beam-steering A/B (ch{ch})\n")
    print("| condition | overlap clips | interferer dBFS (lower=better) | S+I/I dB (higher=better) "
          "| voiceprint same | voiceprint other |")
    print("|---|---|---|---|---|---|")
    for r in res:
        print(f"| {r['cond']} | {len(r['clips'])} | {r.get('interferer_dbfs')} | {r.get('sir_db')} "
              f"| {r.get('vp_same')} | {r.get('vp_other')} |")
    print("\nPer clip (tail = second talker only):\n")
    print("| condition | clip | interferer dBFS | S+I/I dB | DoA tail (body) | free beam tail (body) "
          "| energy free / beam | DoA verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res:
        for c in r["clips"]:
            print(f"| {r['cond']} | {c['clip']} | {c['interferer_dbfs']} | {c['sir_db']} | {c['doa_tail_body']} "
                  f"| {c['free_tail_body']} | {c['en_free']} / {c['en_beam']} | {verdict_doa(c)} |")
    print(f"\nWake / lock / ASR: spot-env/bin/python tests/audio/replay_demo.py "
          f"--conditions {a.off} {a.on} --channel {ch} --asr")


if __name__ == "__main__":
    main()
