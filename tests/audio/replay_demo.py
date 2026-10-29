#!/usr/bin/env python3
"""Replay the operator's demo recordings through the real voice front end.

Mirrors client_mic.py's per-frame loop (Silero VAD -> ChimeMask -> FloorTracker
-> pre-roll -> wake KWS -> Endpointer -> SNR gate -> speaker lock) offline, so
thresholds, mic channel and wake phrase can be chosen from data. Nothing here
opens the mic, talks to the robot or writes into models/.

    spot-env/bin/python tests/audio/replay_demo.py --channel 0 1 --json out.json
    spot-env/bin/python tests/audio/replay_demo.py --kws-sweep --gain-ab --endpoint-sweep
    spot-env/bin/python tests/audio/replay_demo.py --voiceprint --channel 1   # voiceprint only

Recordings: <root>/<condition>/{manifest.json, <clip>.wav (stereo int16 16 kHz,
ch0 processed / ch1 ASR beam), <clip>.doa.csv, <clip>.chime.json}.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import tempfile
import wave
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.voice_control.audio_frontend import (  # noqa: E402
    Endpointer, FloorTracker, UtteranceStats, frame_dbfs, passes_gate)
from src.voice_control.speaker_lock import (  # noqa: E402
    SpeakerLock, ang_diff, circular_median_deg, mic_to_body_deg)
from src.voice_control.tuning import load_tuning  # noqa: E402

MAIN_CHECKOUT = Path("/home/spotdog/spot/dartmouth_spot_capstone")
DEFAULT_ROOT = MAIN_CHECKOUT / "tests" / "audio" / "demo"
KWS_DIR = MAIN_CHECKOUT / "models" / "kws"
SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"   # = client_mic.SILERO_VAD_PATH

# Mirrors of client_mic constants (client_mic is not imported: it loads .env).
SAMPLE_RATE = 16000
FRAME_S = 0.03
FRAME_SAMPLES = 480
PREROLL_FRAMES = 25          # = client_mic.PREROLL_FRAMES (0.75 s) since the Gate 2 fix
SILENCE_CROSSOVER = 2.0
LOCK_LOOKBACK_S = 1.5        # _on_wake: speech DoA in [t_wake - 1.5, t_wake]
LOCK_ACCEPT_MAX_SAMPLES = 100
PAD_S = 2.5                  # clip's own pre-speech lead, tiled after the clip so the endpointer can end
LEAD_S = 0.4                 # every clip starts with >= 0.5 s before the visitor speaks
KWS_MODEL_FILES = ("encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
                   "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
                   "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
                   "tokens.txt")
CHIME_RATE = 48000


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/voice_control/test_replay_demo.py)
# ---------------------------------------------------------------------------
class ChimeMask:
    """Copy of client_mic.ChimeMask (25b25fc): masks VAD speech while a chime
    plays and for tail_s after it; keeps the VAD state if the user was already
    talking when the chime began."""

    def __init__(self, tail_s: float):
        self.tail_s = tail_s
        self._last = None
        self._was = False
        self._prev_speech = False
        self.speech_before = False

    def update(self, chime_busy: bool, now: float, speech: bool = False):
        if chime_busy:
            self._last = now
        masked = self._last is not None and (chime_busy or now - self._last < self.tail_s)
        if masked and not self._was:
            self.speech_before = self._prev_speech or speech
        ended = self._was and not masked
        self._was = masked
        self._prev_speech = speech
        return masked, ended

    def should_reset_vad(self, ended: bool, in_utt: bool) -> bool:
        return ended and not in_utt and not self.speech_before


def frames_of(seconds: float) -> int:
    """= client_mic._frames."""
    return max(1, int(round(seconds / FRAME_S)))


def split_frames(pcm: np.ndarray) -> list:
    """int16 mono -> list of 480-sample PCM16 byte frames (partial tail dropped)."""
    n = len(pcm) // FRAME_SAMPLES
    return [pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes() for i in range(n)]


def pad_from_lead(pcm: np.ndarray, lead_s: float = LEAD_S, pad_s: float = PAD_S) -> np.ndarray:
    """The clip's own lead (room noise before the visitor speaks) tiled to pad_s.
    The background take can't be used: ch0's AGC ramps it ~10 dB above clip silence."""
    lead = pcm[:int(lead_s * SAMPLE_RATE)]
    if len(lead) == 0:
        return np.zeros(int(pad_s * SAMPLE_RATE), dtype=pcm.dtype)
    reps = int(math.ceil(pad_s * SAMPLE_RATE / len(lead)))
    return np.tile(lead, reps)[:int(pad_s * SAMPLE_RATE)]


def in_windows(t: float, windows) -> bool:
    return any(a <= t < b for a, b in windows)


def overlap_s(spans, lo: float, hi: float) -> float:
    """Total length of `spans` [(a, b)] inside [lo, hi]."""
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in spans)


def coverage(ref_spans, after_t: float, utt_t0: float, utt_t1: float) -> float | None:
    """Fraction of reference speech after `after_t` that falls inside the
    utterance [utt_t0, utt_t1]. None when there is no reference speech."""
    total = overlap_s(ref_spans, after_t, float("inf"))
    if total <= 0:
        return None
    return overlap_s(ref_spans, max(after_t, utt_t0), utt_t1) / total


def norm_kw(text: str) -> str:
    """KWS result ('HEY_SPOT' label or 'HEY SPOT') -> 'HEY SPOT'."""
    return " ".join(text.replace("_", " ").replace("@", " ").upper().split())


def speech_degs_between(doa, t0: float, t1: float) -> list:
    """Speech-flagged degrees in [t0, t1] from doa rows (t, deg, speech)."""
    return [float(d) for t, d, s in doa if s and t0 <= t <= t1 and math.isfinite(float(d))]


def lock_result(degs, angle_deg, offset, sign, window):
    """(mic_deg, body_deg, correct) for a lock from `degs`; (None, None, None) if no readings."""
    m = circular_median_deg(degs)
    if m is None:
        return None, None, None
    body = mic_to_body_deg(m, offset, sign)
    return m, body, ang_diff(body, angle_deg) <= window


def per_min(count: int, seconds: float) -> float:
    return count / (seconds / 60.0) if seconds > 0 else 0.0


def median(xs):
    return float(np.median(xs)) if xs else None


def last_word_heard(expected: str, transcript: str) -> bool:
    """ASR cut-off rule: expected_text's last word appears in the transcript."""
    words = [w.strip("?.,!'").lower() for w in expected.split()]
    words = [w for w in words if w]
    if not words:
        return True
    heard = [w.strip("?.,!'").lower() for w in (transcript or "").split()]
    return words[-1] in heard


def threshold_midway(same, other):
    """Voiceprint threshold midway between the same-speaker and other-voice medians."""
    return (float(np.median(same)) + float(np.median(other))) / 2.0


def accept_rates(same, other, thr):
    """(true-accept, false-accept) fractions at `thr` (sim >= thr accepts)."""
    ta = sum(x >= thr for x in same) / len(same) if same else None
    fa = sum(x >= thr for x in other) / len(other) if other else None
    return ta, fa


def pcm_slice(pcm: np.ndarray, t0: float, t1: float) -> bytes:
    """int16 mono clip [t0, t1) seconds (clamped) -> PCM16 bytes."""
    a = max(0, int(t0 * SAMPLE_RATE))
    b = min(len(pcm), int(t1 * SAMPLE_RATE))
    return pcm[a:b].tobytes() if b > a else b""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return x.reshape(-1, w.getnchannels())


def read_doa(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [(float(r["t_rel_s"]), float(r["deg"]), r["speech"] == "1")
                for r in csv.DictReader(f)]


def load_condition(root: Path, cond: str) -> list:
    d = root / cond
    clips = []
    for e in json.loads((d / "manifest.json").read_text()):
        wav = d / f"{e['clip']}.wav"
        if not wav.exists():
            continue
        chime_p = d / f"{e['clip']}.chime.json"
        clips.append(dict(e, audio=read_wav(wav), doa=read_doa(d / f"{e['clip']}.doa.csv"),
                          chime=json.loads(chime_p.read_text()) if chime_p.exists() else None))
    return clips


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_vad():
    from sherpa_onnx import SileroVadModelConfig, VadModelConfig, VoiceActivityDetector
    cfg = VadModelConfig(  # = client_mic.py VAD config
        silero_vad=SileroVadModelConfig(model=SILERO_VAD_PATH, threshold=0.5,
                                        min_silence_duration=0.5, min_speech_duration=0.2,
                                        window_size=512),
        sample_rate=SAMPLE_RATE, debug=False)
    return VoiceActivityDetector(cfg, buffer_size_in_seconds=10)


def reference_speech(pcm: np.ndarray) -> list:
    """Offline Silero segments [(t0, t1)] over a whole mono clip (no mask)."""
    vad = make_vad()
    x = pcm.astype(np.float32) / 32768.0
    out = []
    for i in range(0, len(x), 512):
        vad.accept_waveform(x[i:i + 512])
        while not vad.empty():
            s = vad.front
            out.append((s.start / SAMPLE_RATE, (s.start + len(s.samples)) / SAMPLE_RATE))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        out.append((s.start / SAMPLE_RATE, (s.start + len(s.samples)) / SAMPLE_RATE))
        vad.pop()
    return out


def load_phrases():
    from src.voice_control.wake.phrases import load_wake_phrases
    return load_wake_phrases(REPO / "config" / "wake_phrases.txt")


def write_kws_dir(tmp_root: Path, phrases, score: float, threshold: float) -> Path:
    """Temp model dir: symlinks to the real KWS model + a TEMPORARY keywords.txt."""
    sys.path.insert(0, str(REPO / "scripts"))
    from setup_kws import _bpe_tokenizer, build_keyword_lines
    d = Path(tempfile.mkdtemp(dir=tmp_root, prefix=f"kws_s{score}_t{threshold}_"))
    for f in KWS_MODEL_FILES:
        os.symlink(KWS_DIR / f, d / f)
    import setup_kws
    setup_kws.MODEL_DIR = KWS_DIR  # tokenizer reads tokens/bpe from the real (read-only) dir
    lines = build_keyword_lines(phrases, _bpe_tokenizer(), score, threshold)
    (d / "keywords.txt").write_text("\n".join(lines) + "\n")
    return d


class _LabelSpy:
    """Wraps the sherpa KeywordSpotter so the harness learns WHICH keyword fired."""

    def __init__(self, inner):
        self._inner = inner
        self.last = ""

    def get_result(self, stream):
        r = self._inner.get_result(stream)
        if r.strip():
            self.last = r.strip()
        return r

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_detector(model_dir: Path, peak_target: float):
    import contextlib
    import io
    from src.voice_control.wake.sherpa_onnx import WakeWordDetector
    with contextlib.redirect_stdout(io.StringIO()):
        det = WakeWordDetector(model_dir=model_dir, peak_target=peak_target)
    if not det.is_available():
        raise RuntimeError(f"KWS not available from {model_dir}")
    det._spotter = _LabelSpy(det._spotter)
    return det


def det_fire(det, frame: bytes):
    """process_frame (silenced); returns the fired keyword or None."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        hit = det.process_frame(frame)
    return norm_kw(det._spotter.last) if hit else None


# ---------------------------------------------------------------------------
# Front-end simulation (mirror of client_mic main loop, per 30 ms frame)
# ---------------------------------------------------------------------------
def simulate(frames, cfg, vad, det=None, start_state="wake", chime_windows=(),
             seed_db=-60.0, preroll_frames=PREROLL_FRAMES, mask=True, listen_cue_s=0.0):
    """Returns {"wakes": [(t, kw)], "utts": [dict]}. Clip-relative times."""
    vad.reset()
    if det is not None:
        det.reset()
    floor = FloorTracker(cfg.floor_window_s, cfg.floor_percentile, cfg.floor_rise_db_per_s, FRAME_S)
    floor.seed(seed_db)
    new_ep = lambda: Endpointer(cfg.onset_frames, frames_of(cfg.end_silence_s_short),  # noqa: E731
                                frames_of(cfg.max_utterance_s))
    ep = new_ep()
    cm = ChimeMask(cfg.speaker_tail_s)
    windows = list(chime_windows)
    preroll = deque(maxlen=preroll_frames + cfg.onset_frames)
    state = start_state
    in_utt, utt, stats, end_sil = False, [], None, cfg.end_silence_s_short
    wakes, utts = [], []
    for i, frame in enumerate(frames):
        t = i * FRAME_S
        db = frame_dbfs(frame)
        vad.accept_waveform(np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0)
        is_speech = vad.is_speech_detected()
        while not vad.empty():
            vad.pop()
        masked, ended = cm.update(mask and in_windows(t, windows), t, is_speech)
        if masked:
            is_speech = False
        elif cm.should_reset_vad(ended, in_utt):
            vad.reset()
        else:
            floor.update(db, is_speech)
        preroll.append((t, frame, db, is_speech))
        if det is not None and state in ("wake", "listen"):
            kw = det_fire(det, frame)
            if kw:
                was_idle = state == "wake"
                wakes.append((t, kw))
                state = "listen"
                if listen_cue_s > 0:  # _enter_state(LISTENING) plays listen_cue
                    windows.append((t, t + listen_cue_s))
                if was_idle:
                    ep, in_utt, utt = new_ep(), False, []
        if state == "wake" and det is not None:
            continue
        ev = ep.feed(is_speech, db, floor.floor_db)
        if ev == "start":
            in_utt = True
            utt = list(preroll)
            stats = UtteranceStats(frame_s=FRAME_S)
            for _, _, d, s in utt:
                stats.add(d, s)
            end_sil = cfg.end_silence_s_short
            ep.set_end_silence_frames(frames_of(end_sil))
        elif in_utt:
            utt.append((t, frame, db, is_speech))
            stats.add(db, is_speech)
            if len(utt) * FRAME_S >= SILENCE_CROSSOVER and end_sil != cfg.end_silence_s:
                end_sil = cfg.end_silence_s
                ep.set_end_silence_frames(frames_of(end_sil))
        if ev in ("end", "cap") and in_utt:
            in_utt = False
            t_last = next((u[0] for u in reversed(utt) if u[3]), utt[-1][0])
            utts.append(dict(
                t0=utt[0][0], t_last=t_last + FRAME_S, t_end=t + FRAME_S, endpoint=ev,
                floor_db=floor.floor_db, speech_db=stats.speech_db_median,
                snr_db=stats.snr_db(floor.floor_db), snr_p90=stats.speech_db_p90 - floor.floor_db,
                passed=passes_gate(stats, floor.floor_db, cfg.min_speech_dbfs, cfg.min_snr_db),
                pcm=b"".join(u[1] for u in utt)))
            utt, stats = [], None
    if in_utt:  # still open at the end of the padded clip
        t_last = next((u[0] for u in reversed(utt) if u[3]), utt[-1][0])
        utts.append(dict(t0=utt[0][0], t_last=t_last + FRAME_S, t_end=len(frames) * FRAME_S,
                         endpoint="open", floor_db=floor.floor_db, speech_db=stats.speech_db_median,
                         snr_db=stats.snr_db(floor.floor_db),
                         snr_p90=stats.speech_db_p90 - floor.floor_db,
                         passed=passes_gate(stats, floor.floor_db, cfg.min_speech_dbfs,
                                            cfg.min_snr_db),
                         pcm=b"".join(u[1] for u in utt)))
    return {"wakes": wakes, "utts": utts}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def asr_transcribe(stub, pcm: bytes) -> str:
    from src.voice_control.asr_pb2 import AudioChunk, StreamingConfig, StreamingRequest

    def gen():
        yield StreamingRequest(config=StreamingConfig(language_code="en", sample_rate_hz=SAMPLE_RATE,
                                                      enable_punctuation=True))
        for i in range(0, len(pcm), FRAME_SAMPLES * 2):
            yield StreamingRequest(audio=AudioChunk(pcm16=pcm[i:i + FRAME_SAMPLES * 2]))
    for r in stub.StreamingRecognize(gen()):
        if r.is_final:
            return r.transcript.strip()
    return ""


def make_asr_stub(addr="localhost:50055"):
    import grpc
    from src.voice_control.asr_pb2_grpc import ASRStub
    return ASRStub(grpc.insecure_channel(addr))


class Ctx:
    """Per (condition, channel) data + models."""

    def __init__(self, clips, ch, cfg, det, vad, preroll_frames, asr=None, mask=True):
        self.clips, self.ch, self.cfg, self.det, self.vad = clips, ch, cfg, det, vad
        self.preroll_frames, self.asr, self.mask = preroll_frames, asr, mask
        bg = next((c for c in clips if c["kind"] == "background"), None)
        self.bg = bg["audio"][:, ch] if bg is not None else np.zeros(SAMPLE_RATE * 4, np.int16)
        cal = self.bg[:2 * SAMPLE_RATE].astype(np.float64) / 32768.0  # 2 s calibration
        rms = float(np.sqrt(np.mean(cal * cal))) if len(cal) else 0.0
        self.seed_db = -120.0 if rms <= 1e-6 else max(-120.0, 20 * math.log10(rms))
        self._listen_cue_s = len(render_chimes()["listen_cue"]) / CHIME_RATE


def run_clip(ctx: Ctx, clip, start_state):
    pcm = clip["audio"][:, ctx.ch]
    windows = []
    if clip.get("chime"):
        c = clip["chime"]
        windows.append((c["chime_start_s"], c["chime_end_s"]))
    frames = split_frames(np.concatenate([pcm, pad_from_lead(pcm)]))
    return simulate(frames, ctx.cfg, ctx.vad, ctx.det if start_state == "wake" else None,
                    start_state, windows, ctx.seed_db, ctx.preroll_frames, ctx.mask,
                    ctx._listen_cue_s)


def eval_speech_clip(ctx: Ctx, clip) -> dict:
    """wake / wake_pause / overlap clip, full loop from WAKE_WORD."""
    cfg = ctx.cfg
    pcm = clip["audio"][:, ctx.ch]
    dur = len(pcm) / SAMPLE_RATE
    sim = run_clip(ctx, clip, "wake")
    want = norm_kw(clip["phrase"])
    hit = next((w for w in sim["wakes"] if w[1] == want), None)
    r = dict(clip=clip["clip"], kind=clip["kind"], phrase=want, angle_deg=clip["angle_deg"],
             wake_hit=hit is not None, wakes=[w[1] for w in sim["wakes"]],
             t_wake=None if not sim["wakes"] else round(sim["wakes"][0][0], 2))
    # lock (_on_wake): speech DoA in the 1.5 s before the wake fired; else whole clip
    if sim["wakes"]:
        tw = sim["wakes"][0][0]
        degs = speech_degs_between(clip["doa"], tw - LOCK_LOOKBACK_S, tw)
        r["lock_src"] = "wake"
    else:
        degs = speech_degs_between(clip["doa"], 0.0, dur)
        r["lock_src"] = "clip"
    mic, body, ok = lock_result(degs, clip["angle_deg"], cfg.mic_yaw_offset_deg,
                                cfg.mic_yaw_sign, cfg.lock_window_deg)
    r.update(lock_mic=_r(mic), lock_body=_r(body), lock_ok=ok)
    utts = [u for u in sim["utts"] if u["t0"] < dur]
    passed = [u for u in utts if u["passed"]]
    r.update(utts=len(utts), passed=len(passed),
             caps=sum(u["endpoint"] == "cap" for u in utts),
             snr=[round(u["snr_db"], 1) for u in utts],
             snr_p90=[round(u["snr_p90"], 1) for u in utts])
    if not sim["wakes"]:
        r.update(cut=None, cov=None, split=None, accept=None, transcript=None)
        return r
    # cut-offs are judged on the first utterance after the wake, gate or not;
    # the SNR gate verdict is reported separately (gate_drop)
    tw = sim["wakes"][0][0]
    after = [u for u in utts if u["t_end"] > tw]
    first = after[0] if after else None
    ref = reference_speech(pcm)
    if first is None:
        r.update(cut=True, cov=0.0, split=False, accept=None, transcript=None, no_utt=True,
                 gate_drop=None)
        return r
    r["gate_drop"] = not first["passed"]
    cov = coverage(ref, tw, first["t0"], first["t_end"])
    split = any(u["t0"] >= first["t_end"] and overlap_s(ref, u["t0"], u["t_end"]) > 0.15
                for u in utts[utts.index(first) + 1:])
    r.update(cov=None if cov is None else round(cov, 2), split=split,
             start_lag=round(first["t0"] - tw, 2))
    if ctx.asr is not None:
        r["transcript"] = asr_transcribe(ctx.asr, first["pcm"])
        r["cut"] = not last_word_heard(clip["expected_text"], r["transcript"])
    else:
        r["transcript"] = None
        r["cut"] = bool(split or (cov is not None and cov < 0.8))
    if r["lock_mic"] is not None:
        lock = SpeakerLock(cfg.lock_window_deg, cfg.lock_timeout_s, cfg.lock_nudge)
        lock.angle = mic
        udegs = speech_degs_between(clip["doa"], first["t0"], first["t_last"])
        r["accept"] = lock.accept(udegs[-LOCK_ACCEPT_MAX_SAMPLES:], 0.0)
    else:
        r["accept"] = None
    return r


def eval_quick_reply(ctx: Ctx, clip) -> dict:
    """LISTENING from t=0, recorded chime masked (+speaker_tail_s); first-word capture."""
    pcm = clip["audio"][:, ctx.ch]
    dur = len(pcm) / SAMPLE_RATE
    ch = clip.get("chime") or {}
    c0, c1 = ch.get("chime_start_s", 0.0), ch.get("chime_end_s", 0.0)
    sim = run_clip(ctx, clip, "listen")
    ref = [s for s in reference_speech(pcm) if s[1] > c1]  # user speech after the cue
    onset = ref[0][0] if ref else None
    utts = [u for u in sim["utts"] if u["t0"] < dur]
    first = next((u for u in utts if onset is None or u["t_end"] > onset), None)
    in_chime = sum(1 for u in utts if u["t0"] < c1 + ctx.cfg.speaker_tail_s and u["t_end"] < (onset or dur))
    r = dict(clip=clip["clip"], kind="quick_reply", chime=[c0, c1],
             onset=_r(onset, 2), utts=len(utts), passed=sum(u["passed"] for u in utts),
             chime_utts=in_chime, snr=[round(u["snr_db"], 1) for u in utts],
             snr_p90=[round(u["snr_p90"], 1) for u in utts])
    if first is None or onset is None:
        r.update(utt_t0=None, first_word=False, margin_s=None, cut=None, gate_drop=None)
        return r
    r.update(utt_t0=round(first["t0"], 2), margin_s=round(onset - first["t0"], 2),
             gate_drop=not first["passed"],
             first_word=first["t0"] <= onset + 0.03)
    cov = coverage(ref, c1, first["t0"], first["t_end"])
    r["cov"] = None if cov is None else round(cov, 2)
    if ctx.asr is not None:
        r["transcript"] = asr_transcribe(ctx.asr, first["pcm"])
        r["cut"] = not last_word_heard(clip["expected_text"], r["transcript"])
    else:
        r["cut"] = cov is not None and cov < 0.8
    return r


def eval_background(ctx: Ctx, clip) -> dict:
    pcm = clip["audio"][:, ctx.ch]
    dur = len(pcm) / SAMPLE_RATE
    frames = split_frames(pcm)
    # (a) idle: wake KWS false fires (detector reset after each fire, like the loop)
    ctx.det.reset()
    fires = []
    for i, f in enumerate(frames):
        kw = det_fire(ctx.det, f)
        if kw:
            fires.append((round(i * FRAME_S, 2), kw))
    # (b) LISTENING with nobody addressing Spot: utterances started / passing the gate
    sim = simulate(frames, ctx.cfg, ctx.vad, None, "listen", (), ctx.seed_db,
                   ctx.preroll_frames, ctx.mask)
    by_kw = {}
    for _, kw in fires:
        by_kw[kw] = by_kw.get(kw, 0) + 1
    return dict(clip="background", kind="background", dur_s=dur, fires=fires,
                ff_per_min=round(per_min(len(fires), dur), 2),
                ff_by_kw={k: round(per_min(v, dur), 2) for k, v in by_kw.items()},
                listen_utts=len(sim["utts"]), listen_passed=sum(u["passed"] for u in sim["utts"]),
                listen_snr=[round(u["snr_db"], 1) for u in sim["utts"]],
                listen_snr_p90=[round(u["snr_p90"], 1) for u in sim["utts"]])


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------
# Chime self-trigger
# ---------------------------------------------------------------------------
_CHIMES = None


def render_chimes() -> dict:
    """Every AudioFeedback cue rendered at 48 kHz via a capture player."""
    global _CHIMES
    if _CHIMES is None:
        from src.voice_control.audio_feedback import AudioFeedback

        class _Cap:
            got = None

            def is_available(self):
                return True

            def enqueue_raw(self, samples, rate, label=""):
                self.got = np.asarray(samples, dtype=np.float32)
        out = {}
        for name in ("wake_detected", "listening", "command_ok", "error", "chain_next", "thinking",
                     "listen_cue", "think_cue", "speak_cue", "released_cue", "ready"):
            cap = _Cap()
            getattr(AudioFeedback(player=cap), name)()
            out[name] = cap.got
        _CHIMES = out
    return _CHIMES


def recorded_chime_gain(clips, ch) -> float | None:
    """Mic-level / digital-level RMS ratio of the recorded 'ready'/listen cue."""
    chimes = render_chimes()
    ratios = []
    for c in clips:
        info = c.get("chime")
        if not info:
            continue
        src = info.get("source", "audio_feedback.ready").split(".")[-1]
        dig = chimes.get(src)
        if dig is None:
            continue
        x = c["audio"][:, ch].astype(np.float64) / 32768.0
        a, b = int(info["chime_start_s"] * SAMPLE_RATE), int(info["chime_end_s"] * SAMPLE_RATE)
        pre = x[max(0, a - SAMPLE_RATE // 4):a]
        seg = x[a:b]
        if len(seg) == 0:
            continue
        p_seg, p_pre = float(np.mean(seg ** 2)), float(np.mean(pre ** 2)) if len(pre) else 0.0
        tone = math.sqrt(max(p_seg - p_pre, 1e-12))
        n = int(np.argmax(np.abs(dig) > 0)) if np.any(dig) else 0
        d = dig[n:len(dig) - int(0.08 * CHIME_RATE)].astype(np.float64)
        ratios.append(tone / math.sqrt(float(np.mean(d ** 2))))
    return float(np.median(ratios)) if ratios else None


def chime_self_trigger(ctx: Ctx, gain: float) -> dict:
    """Each cue at 16 kHz on floor-level noise, mask ON (real loop) and OFF."""
    from scipy.signal import resample_poly
    rng = np.random.default_rng(0)
    noise_rms = 10 ** (ctx.seed_db / 20.0)
    res = {}
    for name, s48 in render_chimes().items():
        s16 = resample_poly(s48.astype(np.float64), 1, 3) * gain
        lead, tail = int(1.0 * SAMPLE_RATE), int(2.0 * SAMPLE_RATE)
        x = rng.normal(0, noise_rms, lead + len(s16) + tail)
        x[lead:lead + len(s16)] += s16
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        frames = split_frames(pcm)
        win = [(lead / SAMPLE_RATE, (lead + len(s16)) / SAMPLE_RATE)]
        row = {}
        for mask in (True, False):
            sim = simulate(frames, ctx.cfg, ctx.vad, None, "listen", win, ctx.seed_db,
                           ctx.preroll_frames, mask)
            row["mask" if mask else "nomask"] = len(sim["utts"])
        ctx.det.reset()
        row["kws"] = sum(1 for f in frames if det_fire(ctx.det, f))
        res[name] = row
    return res


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_condition(clips, ch, cfg, det, vad, preroll_frames, asr=None) -> dict:
    ctx = Ctx(clips, ch, cfg, det, vad, preroll_frames, asr)
    rows, quick, bg = [], [], None
    for c in clips:
        if c["kind"] in ("wake", "wake_pause", "overlap"):
            rows.append(eval_speech_clip(ctx, c))
        elif c["kind"] == "quick_reply":
            quick.append(eval_quick_reply(ctx, c))
        elif c["kind"] == "background":
            bg = eval_background(ctx, c)
    return dict(channel=ch, seed_db=round(ctx.seed_db, 1), clips=rows, quick=quick, background=bg)


def kws_only(clips, ch, det) -> dict:
    """Wake hits per phrase + background false fires (KWS alone)."""
    hits, n, wrong = {}, {}, 0
    ff, bg_dur = {}, 0.0
    for c in clips:
        frames = split_frames(c["audio"][:, ch])
        det.reset()
        fired = [kw for kw in (det_fire(det, f) for f in frames) if kw]
        if c["kind"] in ("wake", "wake_pause", "overlap"):
            want = norm_kw(c["phrase"])
            n[want] = n.get(want, 0) + 1
            hits[want] = hits.get(want, 0) + (want in fired)
            wrong += sum(1 for kw in fired if kw != want)
        elif c["kind"] == "background":
            bg_dur += len(frames) * FRAME_S
            for kw in fired:
                ff[kw] = ff.get(kw, 0) + 1
    return dict(hits=hits, n=n, wrong=wrong, bg_s=bg_dur,
                ff={k: round(per_min(v, bg_dur), 2) for k, v in ff.items()},
                ff_total=round(per_min(sum(ff.values()), bg_dur), 2))


def _job(args):
    kind, root, cond, ch, over, extra = args
    clips = load_condition(Path(root), cond)
    cfg = dataclasses.replace(load_tuning(REPO / "config" / "tuning.yaml"), **over)
    tmp = Path(tempfile.mkdtemp(prefix="replay_kws_"))
    kdir = write_kws_dir(tmp, extra.get("phrases") or load_phrases(), cfg.kws_score, cfg.kws_threshold)
    det = make_detector(kdir, cfg.kws_peak_target)
    vad = make_vad()
    if kind == "kws":
        return dict(cond=cond, ch=ch, over=over, **kws_only(clips, ch, det))
    asr = make_asr_stub() if extra.get("asr") else None
    out = run_condition(clips, ch, cfg, det, vad, extra.get("preroll", PREROLL_FRAMES), asr)
    out.update(cond=cond, over=over)
    if extra.get("chimes"):
        ctx = Ctx(clips, ch, cfg, det, vad, extra.get("preroll", PREROLL_FRAMES))
        g = recorded_chime_gain(clips, ch)
        out["chime_gain"] = g
        out["chime_self_trigger"] = {"digital": chime_self_trigger(ctx, 1.0)}
        if g:
            out["chime_self_trigger"]["recorded_level"] = chime_self_trigger(ctx, g)
        ctx.mask = False  # acoustic recorded chime, mask disabled
        out["quick_nomask"] = [eval_quick_reply(ctx, c) for c in clips if c["kind"] == "quick_reply"]
    for c in out.get("clips", []):
        c.pop("pcm", None)
    return out


def pool_map(jobs, n):
    if n <= 1:
        return [_job(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(n) as p:
        return p.map(_job, jobs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def frac(a, b):
    return f"{a}/{b}" if b else "-"


def table_main(res) -> str:
    """One table per condition x channel: per phrase rows."""
    out = []
    bg = res["background"] or {}
    out.append(f"#### {res['cond']} / ch{res['channel']} (floor seed {res['seed_db']} dBFS)\n")
    out.append("| phrase | wake hit | false fires/min | utts | cut-offs | gate drops | caps | SNR med (p90) | lock ok | overlap accept |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    phrases = sorted({c["phrase"] for c in res["clips"]})
    for p in phrases:
        rows = [c for c in res["clips"] if c["phrase"] == p]
        woken = [c for c in rows if c["wake_hit"]]
        cuts = [c for c in rows if c.get("cut")]
        snr = median([s for c in rows for s in c["snr"]])
        snr9 = median([s for c in rows for s in c.get("snr_p90", [])])
        drops = sum(1 for c in woken if c.get("gate_drop"))
        locks = [c for c in rows if c["lock_ok"] is not None]
        ovl = [c for c in rows if c["kind"] == "overlap" and c.get("accept") is not None]
        out.append(f"| {p} | {frac(len(woken), len(rows))} | {bg.get('ff_by_kw', {}).get(p, 0.0)} | "
                   f"{sum(c['utts'] for c in rows)} | {frac(len(cuts), len(woken))} | "
                   f"{drops} | {sum(c['caps'] for c in rows)} | "
                   f"{'-' if snr is None else round(snr, 1)} ({'-' if snr9 is None else round(snr9, 1)}) | "
                   f"{frac(sum(c['lock_ok'] for c in locks), len(locks))} | "
                   f"{frac(sum(c['accept'] for c in ovl), len(ovl)) if ovl else '-'} |")
    if bg:
        out.append(f"\nBackground {bg['dur_s']:.0f} s: wake false fires {bg['ff_per_min']}/min "
                   f"{bg['fires']}; LISTENING utterances started {bg['listen_utts']} "
                   f"(gate-passing {bg['listen_passed']}), SNR med {bg['listen_snr']} p90 {bg.get('listen_snr_p90')}")
    return "\n".join(out)


def table_clips(res) -> str:
    out = ["| clip | wake | t_wake | lock body (src) | ok | utts | start lag | cov | split | cut | gate drop | SNR med | SNR p90 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["clips"]:
        out.append(f"| {c['clip']} | {','.join(c['wakes']) or '-'} | {c['t_wake']} | "
                   f"{c['lock_body']} ({c['lock_src']}) | {c['lock_ok']} | {c['utts']} | {c.get('start_lag')} | "
                   f"{c.get('cov')} | {c.get('split')} | {c.get('cut')} | {c.get('gate_drop')} | {c['snr']} | "
                   f"{c.get('snr_p90')} |")
    return "\n".join(out)


def table_quick(res, key="quick") -> str:
    out = ["| clip | chime s | speech onset s | utt t0 s | margin s | first word | cov | cut | gate drop | SNR med / p90 | chime utts | utts |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in res.get(key, []):
        out.append(f"| {q['clip']} | {q['chime'][0]:.2f}-{q['chime'][1]:.2f} | {q['onset']} | "
                   f"{q['utt_t0']} | {q['margin_s']} | {q['first_word']} | {q.get('cov')} | "
                   f"{q['cut']} | {q.get('gate_drop')} | {q['snr']} / {q.get('snr_p90')} | {q['chime_utts']} | {q['utts']} |")
    return "\n".join(out)


def table_chimes(res) -> str:
    st = res.get("chime_self_trigger") or {}
    levels = list(st)
    head = "| cue | " + " | ".join(f"{lv} mask / nomask / kws" for lv in levels) + " |"
    out = [f"chime mic/digital gain (recorded cue): {res.get('chime_gain')}", "", head,
           "|---|" + "---|" * len(levels)]
    for name in (st.get("digital") or {}):
        cells = [f"{st[lv][name]['mask']} / {st[lv][name]['nomask']} / {st[lv][name]['kws']}"
                 for lv in levels]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_kws(rows) -> str:
    phrases = sorted({p for r in rows for p in r["n"]})
    out = ["| cond | ch | score | thr | peak | " + " | ".join(f"{p} hit / ff" for p in phrases)
           + " | all ff/min | wrong-kw |",
           "|---|---|---|---|---|" + "---|" * len(phrases) + "---|---|"]
    for r in rows:
        o = r["over"]
        cells = [f"{frac(r['hits'].get(p, 0), r['n'].get(p, 0))} / {r['ff'].get(p, 0.0)}" for p in phrases]
        out.append(f"| {r['cond']} | {r['ch']} | {o.get('kws_score')} | {o.get('kws_threshold')} | "
                   f"{o.get('kws_peak_target')} | " + " | ".join(cells)
                   + f" | {r['ff_total']} | {r['wrong']} |")
    return "\n".join(out)


def table_endpoint(rows) -> str:
    out = ["| cond | ch | min_snr_db | end_silence_s | short | pause cut-offs | wake cut-offs | "
           "gate drops (woken) | bg LISTENING passes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r["over"]
        woken = [c for c in r["clips"] if c["wake_hit"]]
        pause = [c for c in woken if c["kind"] == "wake_pause"]
        wake = [c for c in woken if c["kind"] != "wake_pause"]
        drops = sum(1 for c in woken if c.get("gate_drop"))
        bg = r["background"] or {}
        out.append(f"| {r['cond']} | {r['channel']} | {o['min_snr_db']} | {o['end_silence_s']} | "
                   f"{o['end_silence_s_short']} | {frac(sum(bool(c['cut']) for c in pause), len(pause))} | "
                   f"{frac(sum(bool(c['cut']) for c in wake), len(wake))} | {drops} | "
                   f"{bg.get('listen_passed')} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Voiceprint calibration (Task 18 step 5)
# ---------------------------------------------------------------------------
VP_WAKE_FIRE_S = 1.1     # proxy KWS fire time after Silero onset (replay: fires 0.8-1.4 s in)
VP_WAKE_AUDIO_S = 1.5    # = client_mic.WAKE_AUDIO_S (audio kept before the fire)
VP_VISITOR_S = 2.6       # "hey spot what is this poster about" length (wake clips, Silero)
VP_PAD_S = 0.3           # pre-roll / trailing keep around a Silero segment
VP_MIN_S = 0.8           # shortest segment embedded


def _summ(xs):
    if not xs:
        return dict(n=0)
    a = np.asarray(xs)
    return dict(n=len(xs), min=round(float(a.min()), 3), p10=round(float(np.percentile(a, 10)), 3),
                median=round(float(np.median(a)), 3), p90=round(float(np.percentile(a, 90)), 3),
                max=round(float(a.max()), 3))


def voiceprint_calibration(root: Path, conds, ch: int, model_path: Path) -> dict:
    """Same-speaker vs other-voice cosine scores against live-style enrolments.

    Enrolment per wake / pause / overlap clip = L2-normalised mean of
    embed(wake audio) and embed(first utterance), as client_mic does:
      wake audio  = the VP_WAKE_AUDIO_S before a proxy fire at onset + VP_WAKE_FIRE_S
      first utt   = the visitor's Silero speech (+/- VP_PAD_S; overlap clips
                    capped at onset + VP_VISITOR_S, the podcast keeps going).
    Trials (the recordings are all one operator):
      same    = enrolment vs every OTHER clip's first utterance + quick replies (same condition)
      other   = enrolment vs the podcast tail of each overlap clip (second voice, off to the side)
      unknown = enrolment vs talking in the fan background take (speaker unknown)
      noise   = enrolment vs 2 s chunks of the babble background (no Silero speech: noise only)
    """
    from src.voice_control.speaker_embed import SpeakerEmbedder, cosine
    emb = SpeakerEmbedder(model_path)

    def e(b):
        return emb.embed(b) if len(b) >= int(VP_MIN_S * SAMPLE_RATE) * 2 else None

    enrols, utts, others, unknown, noise = [], [], [], [], []
    for cond in conds:
        for c in load_condition(root, cond):
            pcm = c["audio"][:, ch]
            dur = len(pcm) / SAMPLE_RATE
            ref = reference_speech(pcm)
            kind = c["kind"]
            if kind in ("wake", "wake_pause", "overlap") and ref:
                on = ref[0][0]
                end = ref[-1][1]
                if kind == "overlap":
                    end = min(end, on + VP_VISITOR_S)
                fire = on + VP_WAKE_FIRE_S
                wv = e(pcm_slice(pcm, fire - VP_WAKE_AUDIO_S, fire))
                fv = e(pcm_slice(pcm, on - VP_PAD_S, end + VP_PAD_S))
                if wv is None or fv is None:
                    continue
                m = (wv + fv) / 2.0
                enrols.append(dict(cond=cond, clip=c["clip"], v=m / np.linalg.norm(m)))
                utts.append(dict(cond=cond, clip=c["clip"], v=fv))
                if kind == "overlap":
                    tail = [(max(a, end + VP_PAD_S), b) for a, b in ref if b > end + VP_PAD_S]
                    tail = [(a, b) for a, b in tail if b - a >= VP_MIN_S]
                    if tail:
                        ov = e(pcm_slice(pcm, tail[0][0], tail[-1][1]))
                        if ov is not None:
                            others.append(dict(cond=cond, clip=c["clip"], v=ov,
                                               span=(round(tail[0][0], 2), round(tail[-1][1], 2))))
            elif kind == "quick_reply":
                c1 = (c.get("chime") or {}).get("chime_end_s", 0.0)
                segs = [s for s in ref if s[1] > c1]
                if segs:
                    qv = e(pcm_slice(pcm, max(c1, segs[0][0] - VP_PAD_S), segs[-1][1] + VP_PAD_S))
                    if qv is not None:
                        utts.append(dict(cond=cond, clip=c["clip"], v=qv))
            elif kind == "background":
                if ref:
                    for a, b in ref:
                        if b - a >= VP_MIN_S:
                            v = e(pcm_slice(pcm, a, b))
                            if v is not None:
                                unknown.append(dict(cond=cond, clip=f"bg@{a:.0f}s", v=v))
                else:
                    for t in np.arange(0.0, min(dur, 40.0) - 2.0, 4.0):
                        v = e(pcm_slice(pcm, t, t + 2.0))
                        if v is not None:
                            noise.append(dict(cond=cond, clip=f"bg@{t:.0f}s", v=v))

    same = [cosine(en["v"], u["v"]) for en in enrols for u in utts
            if u["cond"] == en["cond"] and u["clip"] != en["clip"]]
    same_quick = [cosine(en["v"], u["v"]) for en in enrols for u in utts
                  if u["cond"] == en["cond"] and u["clip"].startswith("quick_reply")]
    other = [cosine(en["v"], o["v"]) for en in enrols for o in others]
    unk = [cosine(en["v"], o["v"]) for en in enrols for o in unknown]
    noi = [cosine(en["v"], o["v"]) for en in enrols for o in noise]
    # Per-utterance medians (pairs share segments, so these are the honest n).
    other_by_seg = {o["clip"]: round(float(np.median([cosine(en["v"], o["v"]) for en in enrols])), 3)
                    for o in others}
    res = dict(channel=ch, conds=list(conds), n_enrol=len(enrols), n_utts=len(utts),
               n_other_segs=len(others), n_unknown_segs=len(unknown), n_noise_segs=len(noise),
               other_spans={o["clip"]: o["span"] for o in others}, other_by_seg=other_by_seg,
               same=_summ(same), same_quick=_summ(same_quick), other=_summ(other), unknown=_summ(unk), noise=_summ(noi))
    if same and other:
        thr = threshold_midway(same, other)
        ta, fa = accept_rates(same, other, thr)
        res.update(threshold=round(thr, 3), true_accept=round(ta, 3), false_accept=round(fa, 3),
                   unknown_accept=None if not unk else round(accept_rates(unk, unk, thr)[0], 3),
                   noise_accept=None if not noi else round(accept_rates(noi, noi, thr)[0], 3))
        res["quick_accept"] = None if not same_quick else round(accept_rates(same_quick, [], thr)[0], 3)
        res["sweep"] = [dict(thr=t, ta=round(accept_rates(same, other, t)[0], 3),
                             fa=round(accept_rates(same, other, t)[1], 3),
                             ta_quick=(None if not same_quick
                                       else round(accept_rates(same_quick, [], t)[0], 3)))
                        for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)]
    return res


def table_voiceprint(r) -> str:
    rows = ["| set | n pairs | min | p10 | median | p90 | max |", "|---|---|---|---|---|---|---|"]
    for k in ("same", "same_quick", "other", "unknown", "noise"):
        d = r[k]
        if d.get("n"):
            rows.append(f"| {k} | {d['n']} | {d['min']} | {d['p10']} | {d['median']} | {d['p90']} | {d['max']} |")
    out = "\n".join(rows)
    out += (f"\n\nch{r['channel']}: {r['n_enrol']} enrolments, {r['n_utts']} same-speaker utterances, "
            f"{r['n_other_segs']} other-voice segments {r['other_spans']}, "
            f"{r['n_unknown_segs']} unknown-talker segments, {r['n_noise_segs']} noise chunks")
    out += f"\nother-voice median sim per segment: {r['other_by_seg']}"
    if "threshold" in r:
        out += (f"\nthreshold (midway between medians) = {r['threshold']}: true-accept "
                f"{r['true_accept']:.0%} (quick replies {r['quick_accept']}), false-accept {r['false_accept']:.0%}, "
                f"unknown-talker accept {r['unknown_accept']}, noise accept {r['noise_accept']}")
        out += "\nsweep: " + ", ".join(f"{d['thr']}: TA {d['ta']:.0%} (quick {d['ta_quick']}) "
                                       f"FA {d['fa']:.0%}" for d in r["sweep"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--conditions", nargs="*", default=None, help="default: every dir with a manifest")
    ap.add_argument("--channel", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    ap.add_argument("--asr", action="store_true", help="cut-off by ASR (needs gRPC server on :50055)")
    ap.add_argument("--preroll-frames", type=int, default=PREROLL_FRAMES)
    ap.add_argument("--clips", action="store_true", help="also print per-clip tables")
    ap.add_argument("--kws-sweep", action="store_true", help="score x threshold sweep (KWS only)")
    ap.add_argument("--gain-ab", action="store_true", help="kws_peak_target 0.9 vs legacy x8 (KWS only)")
    ap.add_argument("--endpoint-sweep", action="store_true", help="min_snr_db x end_silence_s sweep")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--json", default=None, help="save all results here")
    ap.add_argument("--voiceprint", action="store_true",
                    help="voiceprint calibration only (same-speaker vs other-voice similarity)")
    ap.add_argument("--speaker-model", default=str(MAIN_CHECKOUT / "models" / "speaker"
                                                   / "voxceleb_resnet34_LM.onnx"))
    a = ap.parse_args(argv)

    root = Path(a.root)
    conds = a.conditions or sorted(p.parent.name for p in root.glob("*/manifest.json"))
    base = load_tuning(REPO / "config" / "tuning.yaml")
    extra = dict(asr=a.asr, preroll=a.preroll_frames, chimes=True)
    results = {"tuning": dataclasses.asdict(base), "preroll_frames": a.preroll_frames}

    if a.voiceprint:
        results["voiceprint"] = []
        for ch in a.channel:
            r = voiceprint_calibration(root, conds, ch, Path(a.speaker_model))
            results["voiceprint"].append(r)
            print(f"## Voiceprint calibration ch{ch} ({', '.join(conds)})\n")
            print(table_voiceprint(r), "\n")
        if a.json:
            Path(a.json).write_text(json.dumps(results, indent=1, default=str))
            print(f"saved {a.json}")
        return

    main_jobs = [("main", str(root), c, ch, {}, extra) for c in conds for ch in a.channel]
    results["main"] = pool_map(main_jobs, a.jobs)
    print(f"## Replay (preroll {a.preroll_frames} frames, ASR {'on' if a.asr else 'off'})\n")
    for r in results["main"]:
        print(table_main(r), "\n")
        if a.clips:
            print(table_clips(r), "\n")
        print(f"Quick reply ({r['cond']} ch{r['channel']}), chime masked:\n" + table_quick(r) + "\n")
        print("Quick reply, mask OFF (acoustic chime self-trigger):\n" + table_quick(r, "quick_nomask") + "\n")
        print("Chime self-trigger (utterances started; must be 0):\n" + table_chimes(r) + "\n")

    if a.kws_sweep or a.gain_ab:
        overs = []
        if a.kws_sweep:
            overs += [dict(kws_score=s, kws_threshold=t, kws_peak_target=base.kws_peak_target)
                      for s in (1.0, 1.5, 2.0) for t in (0.05, 0.1, 0.2, 0.3)]
        if a.gain_ab:
            for pk in (base.kws_peak_target, 0.0):
                o = dict(kws_score=base.kws_score, kws_threshold=base.kws_threshold, kws_peak_target=pk)
                if o not in overs:
                    overs.append(o)
        jobs = [("kws", str(root), c, ch, o, {}) for o in overs for c in conds for ch in a.channel]
        results["kws"] = pool_map(jobs, a.jobs)
        print("## KWS sweep (hit = right keyword fired on wake/pause/overlap clips; ff = per min on background)\n")
        print(table_kws(results["kws"]), "\n")

    if a.endpoint_sweep:
        overs = [dict(min_snr_db=snr, end_silence_s=es, end_silence_s_short=min(es, 1.2))
                 for snr in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0) for es in (1.0, 1.2, 1.5)]
        overs += [dict(min_snr_db=4.0, end_silence_s=1.5, end_silence_s_short=sh) for sh in (0.9, 1.5)]
        ex = dict(extra, chimes=False)
        jobs = [("main", str(root), c, ch, o, ex) for o in overs for c in conds for ch in a.channel]
        results["endpoint"] = pool_map(jobs, a.jobs)
        print("## Endpoint sweep\n")
        print(table_endpoint(results["endpoint"]), "\n")

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, default=str))
        print(f"saved {a.json}")


if __name__ == "__main__":
    main()
