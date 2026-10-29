"""Golden conversation harness: drive the real client_mic.main() on recorded clips.

run_pipeline(clips) runs today's conversation pipeline end to end on recorded
audio and returns what it decided, as ordered event records:

    {"kind": "utterance", ...}   the real build_utterance_record() output
    {"kind": "dispatch", "intent": ..., "params": ..., "via": ...}
    {"kind": "state", "to": ...}
    {"kind": "say", "text": ...}

Every record also carries "clip" (the clip being fed when it happened; None
at startup). The real wake KWS, safety KWS, Silero VAD, FloorTracker,
endpointer, SNR gate, speaker lock and voiceprint run. Faked, test-side only
(no change under src/):

  * sounddevice: a fake InputStream feeds the clip PCM into the real
    audio_callback in 30 ms frames AT REAL TIME, lock-stepped: the next frame
    is handed over only once the main loop, then the safety-KWS thread, have
    fully processed the last one.
  * clock: client_mic's time.monotonic() (endpointer, ChimeMask, RingThrottle,
    gate reopen, lock/conversation timeout, DoA window) reads a virtual clock
    that advances exactly 30 ms per fed frame. A frame dropped by the callback
    (mic gated) still drives one loop iteration, so the gate reopens on a
    fixed frame. Without this, wall-clock jitter moved the reopen (and its
    VAD reset, i.e. Silero's 512-sample window phase) by a few frames and
    two runs on the same commit differed in utterance boundaries.
  * ASR: returns tests/integration/golden/transcripts.json[clip].
  * robot: execute_on_spot / get_spot_state / _dispatch_safety record and never
    reach spot_dispatch; main() runs with --dry-run (no TCP probe, no LEDs,
    no gaze, no auto-arm).
  * TTS + AudioPlayer: record text, no sound, never busy (so ChimeMask never
    masks and the mic is never muted for playback: deterministic).
  * XVF3800: a fake mic giving one fixed DoA per clip (circular median of the
    clip's .doa.csv speech rows; 179 deg = FRONT for clips without one).
  * LLM: --no-brain and _brain_ok=False -> regex path.
  * control server: not started (port 8770 may be held by live voice control);
    its queue is used to send spotctl "listen" before conversation-mode clips
    and "reset" after every clip, so each clip starts from a clean state.

main() runs in a child process: it mutates many client_mic module globals and
os.environ, starts daemon threads and installs signal handlers, none of which
may leak into the rest of the pytest session.

Between clips the feeder pads with the clip's own quietest 0.4 s, tiled (as
tests/audio/replay_demo.py does). Never zeros: FloorTracker would fall to
-120 dBFS and every SNR gate would pass.

    spot-env/bin/python tests/integration/harness.py generate OUT.json
"""
from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
TRANSCRIPTS_PATH = GOLDEN_DIR / "transcripts.json"
GOLDEN_PATH = GOLDEN_DIR / "conversation.json"
LAB_DIR = REPO / "tests" / "audio"
DEFAULT_DEMO_ROOT = Path("/home/spotdog/spot/dartmouth_spot_capstone/tests/audio/demo")

RATE = 16000                 # = client_mic.SAMPLE_RATE
FRAME = 480                  # = client_mic.FRAME_SAMPLES (30 ms)
FRAME_S = FRAME / RATE
QUIET_WIN_S = 0.4            # pad source: the clip's quietest 0.4 s
TAIL_S = 2.0                 # fixed pad after every clip (> end_silence_s 1.2 + reopen 0.5)
MAX_EXTRA_S = 30.0           # extra pad (frame by frame) until settled; > max_utterance_s 20 s
CALIB_S = 3.0                # calibration audio (main() measures 2 s of it)
DEFAULT_MIC_DEG = 179.0      # FRONT in the mic frame (config/tuning.yaml mic_yaw_offset_deg)
CHILD_TIMEOUT_S = 1200

# Fields that cannot be reproduced run to run (see report). Everything else
# in a record is compared.
EXCLUDED_FIELDS = (
    "ts",          # wall-clock time.time() at record build
    "latency_s",   # wall-clock processing time of the turn
)


# ---------------------------------------------------------------------------
# Clip naming
# ---------------------------------------------------------------------------
def demo_root() -> Path:
    return Path(os.environ.get("SPOT_DEMO_AUDIO", str(DEFAULT_DEMO_ROOT)))


def resolve_clip(name: str) -> Path:
    """'lab/<file>.wav' -> tests/audio/<file>.wav; '<cond>/<file>.wav' -> demo root."""
    cond, fname = name.split("/", 1)
    if cond == "lab":
        return LAB_DIR / fname
    return demo_root() / cond / fname


def clip_key(path: Path) -> str:
    path = Path(path)
    if path.resolve().parent == LAB_DIR.resolve():
        return f"lab/{path.name}"
    return f"{path.parent.name}/{path.name}"


def clip_mode(key: str) -> str:
    """'wake': the clip starts in WAKE_WORD (wake phrases, idle, safety).
    'listen': spotctl listen first, so a bare command reaches the regex path
    (lab command clips, quick replies)."""
    cond, fname = key.split("/", 1)
    stem = fname.rsplit(".", 1)[0]
    if cond == "lab":
        return "wake" if stem.startswith(("hey_spot", "adv_", "idle_")) else "listen"
    return "listen" if stem.startswith("quick_reply") else "wake"


def default_clips() -> list[str]:
    """Golden clip list (one run must stay near 5 min of real-time audio):
    all 14 safety clips, every hey_spot wake clip, a spread of adversarial
    near-wake, idle and direct-command _lab clips, and fan-noise demo clips
    (wake, wake-pause, quick reply, overlap)."""
    lab = ["adv_hey_dog", "adv_hey_stop", "adv_hot_spot",
           "battery_status", "come_here", "describe_surroundings", "find_a_chair",
           "go_to_the_kitchen",
           "hey_spot_close", "hey_spot_far", "hey_spot_mid", "hey_spot_stand_up",
           "hey_spot_walk_forward",
           "idle_01", "idle_03", "idle_05",
           "sit_down", "stand_up", "stop", "turn_left_90", "walk_forward_1m"]
    safety = sorted(f"safety/{p.name}" for p in (demo_root() / "safety").glob("*.wav"))
    fan = ["fan/wake_hey_spot_front.wav", "fan/pause_hey_spot.wav",
           "fan/quick_reply_1.wav", "fan/overlap_1.wav"]
    return [f"lab/{s}_lab.wav" for s in lab] + safety + fan


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def load_pcm(path: Path) -> np.ndarray:
    """(N, channels) float32 at 16 kHz, native channel count."""
    with wave.open(str(path)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected int16, got {8 * width}-bit")
    pcm = np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
    if rate != RATE:  # same helper replay_demo.py uses for 48 kHz audio
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(RATE, rate)
        pcm = resample_poly(pcm.astype(np.float64), RATE // g, rate // g, axis=0)
        pcm = pcm.astype(np.float32)
    return pcm


def quiet_tile(pcm: np.ndarray, seconds: float) -> np.ndarray:
    """The clip's quietest QUIET_WIN_S (frame-aligned) tiled to `seconds`."""
    win = int(QUIET_WIN_S * RATE)
    if len(pcm) <= win:
        seg = pcm
    else:
        mono = pcm[:, -1].astype(np.float64)
        best, best_e = 0, None
        for s in range(0, len(pcm) - win + 1, FRAME):
            e = float(np.mean(mono[s:s + win] ** 2))
            if best_e is None or e < best_e:
                best, best_e = s, e
        seg = pcm[best:best + win]
    n = int(round(seconds * RATE))
    reps = -(-n // len(seg))
    return np.tile(seg, (reps, 1))[:n]


def clip_deg(path: Path) -> float:
    """Fixed mic-frame DoA for a clip: circular median of its .doa.csv."""
    from src.voice_control.speaker_lock import circular_median_deg
    csv_path = Path(path).with_suffix(".doa.csv")
    if not csv_path.exists():
        return DEFAULT_MIC_DEG
    with open(csv_path) as fh:
        rows = [(float(r["deg"]), r["speech"] == "1") for r in csv.DictReader(fh)
                if r.get("deg") not in (None, "")]
    degs = [d for d, s in rows if s] or [d for d, _ in rows]
    m = circular_median_deg(degs)
    return DEFAULT_MIC_DEG if m is None else float(round(m))


# ---------------------------------------------------------------------------
# Normalisation (golden write + compare use the same function)
# ---------------------------------------------------------------------------
def _round(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    return v


def normalize(records: list[dict]) -> list[dict]:
    return [_round({k: v for k, v in r.items() if k not in EXCLUDED_FIELDS})
            for r in records]


# ---------------------------------------------------------------------------
# Public entry: run in a child process
# ---------------------------------------------------------------------------
def run_pipeline(clips: list[Path]) -> list[dict]:
    """Run client_mic.main() on `clips` (in order) and return its event records."""
    clips = [Path(c) for c in clips]
    with tempfile.TemporaryDirectory(prefix="golden-") as td:
        out = Path(td) / "records.json"
        log = Path(td) / "child.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "_child", str(out)]
        cmd += [str(c) for c in clips]
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=CHILD_TIMEOUT_S)
        if proc.returncode != 0 or not out.exists():
            tail = log.read_text()[-4000:]
            raise RuntimeError(f"harness child failed (rc={proc.returncode}):\n{tail}")
        data = json.loads(out.read_text())
        keep = os.environ.get("GOLDEN_CHILD_LOG")
        if keep:
            Path(keep).write_text(log.read_text())
    return data["records"]


# ---------------------------------------------------------------------------
# Child process: fakes + feeder + main()
# ---------------------------------------------------------------------------
class _H:
    """Harness state shared by the fakes, the feeder thread and main()."""
    lock = threading.Lock()
    events: list = []
    clip = None
    deg = DEFAULT_MIC_DEG
    transcripts: dict = {}
    ctl_q = None
    waiting = False      # main loop blocked in audio_queue.get()
    gets = 0             # blocking gets returned (frames + ticks)
    vbase = 0.0          # virtual monotonic clock: vbase + frames fed * FRAME_S
    vnow = 0.0
    ep = None
    done = threading.Event()
    calib = None
    plan: list = []
    meta: dict = {}
    error = None

    @classmethod
    def add(cls, rec: dict) -> None:
        with cls.lock:
            cls.events.append(dict(rec, clip=cls.clip))


def _via() -> str:
    return "safety_kws" if threading.current_thread().name == "safety-kws" else "main"


class FakeMic:
    """XVF3800 stand-in: one fixed DoA for the current clip, no-op LEDs/beams."""
    dead = False

    def samples_between(self, t0, t1):
        n = max(1, int((t1 - t0) / 0.05) + 1)  # 20 Hz poller
        deg = _H.deg
        return [(t0 + i * 0.05, deg, 1) for i in range(n)]

    def set_fixed_beams(self, az):
        pass

    def set_led_effect(self, e):
        pass

    def set_led_color(self, c):
        pass

    def set_ring(self, r):
        pass

    def start_poller(self, hz=20.0):
        pass

    def add_restore_listener(self, fn):
        pass

    def close(self):
        pass


class FakePlayer:
    """AudioPlayer stand-in: accepts everything, plays nothing, never busy."""

    def __init__(self, *a, **k):
        pass

    def is_available(self):
        return True

    def is_busy(self):
        return False

    def is_speech_busy(self):
        return False

    def enqueue_raw(self, *a, **k):
        pass

    def enqueue_render(self, *a, **k):
        pass

    def wait(self):
        pass

    def shutdown(self):
        pass

    def force_reset(self):
        pass


class FakeTTS:
    """SpotTTS stand-in: records the text it is asked to speak."""

    def __init__(self, *a, **k):
        pass

    def is_available(self):
        return True

    def speak(self, text, voice=None, **k):
        _H.add({"kind": "say", "text": text})

    def wait(self):
        pass


class FakeUtteranceLog:
    """UtteranceLog stand-in: keeps the real build_utterance_record() output
    (idle lines are 1 Hz wall-clock samples and are dropped); writes no file."""

    def __init__(self, log_dir, store_transcript=True):
        self.store_transcript = store_transcript

    def write(self, record):
        if record.get("kind") != "utterance":
            return
        if not self.store_transcript and "transcript" in record:
            record = dict(record, transcript=None)
        _H.add(record)


def _sleep_until(t):
    d = t - time.monotonic()
    if d > 0:
        time.sleep(d)


def _feed(cb, pcm, stop=None):
    """Calibration stream: frames of `pcm` into `cb` at real time."""
    n = len(pcm) // FRAME
    t0 = time.monotonic()
    for i in range(n):
        if stop is not None and stop.is_set():
            return i
        _sleep_until(t0 + i * FRAME_S)
        cb(pcm[i * FRAME:(i + 1) * FRAME], FRAME, None, None)
    return n


def _wait(cond, timeout, what):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError(f"harness: timed out waiting for {what}")
        time.sleep(0.001)


_TICK = object()  # wakes the main loop for one iteration without audio


class LockstepQueue(queue.Queue):
    """client_mic.audio_queue stand-in (same maxsize). Counts the callback's
    puts and the main loop's blocking gets so the feeder can hand over one
    frame and wait until the loop has fully processed it. A _TICK item makes
    the loop's get() raise queue.Empty: exactly the loop's own timeout path."""

    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.puts = 0

    def put_nowait(self, item):
        super().put_nowait(item)
        self.puts += 1

    def get(self, block=True, timeout=None):
        if not block:  # _drain_audio_queue's get_nowait: plain behaviour
            return super().get(block, timeout)
        _H.waiting = True
        try:
            item = super().get(block, timeout)
        finally:
            _H.waiting = False
        _H.gets += 1
        if item is _TICK:
            raise queue.Empty
        return item


class SafetyLockstepQueue(queue.Queue):
    """client_mic._safety_q stand-in (same maxsize). The safety-KWS thread
    may take a frame only after the feeder releases it, i.e. after the main
    loop has finished that same frame, so per frame the order is always
    main loop first, then safety KWS, and no frame is ever dropped."""

    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.permits = threading.Semaphore(0)
        self.taken = 0
        self.in_get = False

    def get(self, block=True, timeout=None):
        if not block:
            return super().get(block, timeout)
        self.in_get = True
        if not self.permits.acquire(timeout=timeout):
            raise queue.Empty
        self.in_get = False
        item = super().get_nowait()
        self.taken += 1
        return item

    def step(self):
        """Release one queued frame and wait until the thread is back in get()."""
        if self.empty():
            return
        target = self.taken + 1
        self.permits.release()
        _wait(lambda: self.taken >= target and self.in_get, 30.0, "safety KWS frame")


def _one_iteration(q, action):
    """Run `action` (feed a frame / queue a command), make sure exactly one
    main-loop iteration follows, and wait until the loop is idle again."""
    before_gets, before_puts = _H.gets, q.puts
    action()
    if q.puts == before_puts:  # frame dropped (gated / muted) or no frame
        queue.Queue.put_nowait(q, _TICK)
    _wait(lambda: _H.gets > before_gets and _H.waiting and q.empty(), 30.0,
          "main loop iteration")


class _Feeder:
    """Main-stream feeder: real-time paced, lock-stepped with the main loop,
    and the virtual clock advances exactly FRAME_S per frame."""

    def __init__(self, cm, cb):
        self.cm, self.cb = cm, cb
        self.n = 0
        self.t_real0 = None

    def frame(self, f):
        if self.t_real0 is None:
            self.t_real0 = time.monotonic()
        _sleep_until(self.t_real0 + self.n * FRAME_S)
        _H.vnow = _H.vbase + self.n * FRAME_S  # stamp seen by audio_callback
        self.n += 1
        _one_iteration(self.cm.audio_queue, lambda: self.cb(f, FRAME, None, None))
        self.cm._safety_q.step()

    def feed(self, pcm):
        for i in range(len(pcm) // FRAME):
            self.frame(pcm[i * FRAME:(i + 1) * FRAME])

    def control(self, name, state):
        """spotctl command, applied on the next loop iteration (virtual time
        does not advance; the real gap re-anchors the real-time pacing)."""
        _one_iteration(self.cm.audio_queue, lambda: _H.ctl_q.put((name, None)))
        if self.cm._status["state"] != state or not _H.ctl_q.empty():
            raise RuntimeError(f"harness: spotctl {name} left state "
                               f"{self.cm._status['state']} (want {state})")
        self.t_real0 = time.monotonic() - self.n * FRAME_S

    def settled(self) -> bool:
        cm, ep = self.cm, _H.ep
        return (not cm._mic_gated and cm._mic_reopen_at is None
                and not (ep is not None and ep._in_speech)
                and cm._status["state"] in ("WAKE_WORD", "LISTENING"))


def _feed_main(cm, cb):
    fd = _Feeder(cm, cb)
    try:
        _wait(lambda: _H.waiting and _H.ctl_q is not None, 120.0, "main loop start")
        for key, path in _H.plan:
            pcm = load_pcm(path)
            with _H.lock:
                _H.clip, _H.deg = key, clip_deg(path)
            if clip_mode(key) == "listen":
                fd.control("listen", "LISTENING")
            fd.feed(np.concatenate([pcm, quiet_tile(pcm, TAIL_S)]))
            pad = quiet_tile(pcm, MAX_EXTRA_S)
            extra = 0
            while not fd.settled():  # deterministic: a function of the frames fed
                if extra * FRAME >= len(pad):
                    raise TimeoutError(f"harness: {key} never settled")
                fd.frame(pad[extra * FRAME:(extra + 1) * FRAME])
                extra += 1
            _H.meta.setdefault("extra_tail_frames", {})[key] = extra
            fd.control("reset", "WAKE_WORD")
        with _H.lock:
            _H.clip = None
    except BaseException as e:  # noqa: BLE001 - surfaced by the child's exit code
        _H.error = e
        import traceback
        traceback.print_exc()
    finally:
        _H.done.set()
        try:
            queue.Queue.put_nowait(cm.audio_queue, _TICK)  # wake the loop to exit
        except Exception:
            pass


class FakeInputStream:
    """sounddevice.InputStream stand-in. The main stream (callback is the
    real audio_callback) gets the clip plan; any other stream (the noise
    calibration) gets the first clip's quiet pad on a loop."""

    def __init__(self, device=None, channels=2, samplerate=RATE, callback=None,
                 blocksize=FRAME, **k):
        import src.voice_control.client_mic as cm
        self._cm, self._cb = cm, callback
        self._stop = threading.Event()
        self._thread = None
        self._main = callback is cm.audio_callback

    def start(self):
        if self._main:
            target = lambda: _feed_main(self._cm, self._cb)  # noqa: E731
        else:
            def target():
                while not self._stop.is_set():
                    _feed(self._cb, _H.calib, self._stop)
        self._thread = threading.Thread(target=target, daemon=True,
                                        name="fake-mic-main" if self._main else "fake-mic-cal")
        self._thread.start()

    def stop(self):
        if not self._main:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)

    def close(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        self.close()


class _VirtualTime:
    """client_mic's `time` module with monotonic() replaced by the feeder's clock."""

    def monotonic(self):
        return _H.vnow

    def __getattr__(self, name):
        return getattr(time, name)


def _forbidden(*a, **k):
    raise RuntimeError("golden harness: real audio/mic hardware must not be opened")


def _child(out_path: str, clip_paths: list[str]) -> int:
    for p in (str(REPO), str(REPO / "src" / "voice_control")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pytest
    import sounddevice

    with pytest.MonkeyPatch.context() as mp:
        # Hardware first, before client_mic or anything else can open it.
        mp.setattr(sounddevice, "InputStream", _forbidden)
        mp.setattr(sounddevice, "OutputStream", _forbidden)
        mp.setattr(sounddevice, "play", _forbidden)
        mp.setattr(sounddevice, "rec", _forbidden)
        from src.voice_control import mic_array
        mp.setattr(mic_array.MicArray, "open", classmethod(_forbidden))

        import src.voice_control.client_mic as cm
        import src.voice_control.control_server as control_server

        # client_mic loads the project's dotenv file at import: pin the env after it.
        mp.setenv("SPOT_DRY_RUN", "1")
        mp.setenv("SPOT_WAKE_BACKEND", "sherpa_onnx")
        mp.delenv("SPOT_NO_TRANSCRIPT_LOG", raising=False)
        mp.setattr(sys, "argv", ["client_mic.py", "--no-brain", "--dry-run",
                                 "--device", "0", "--output-device", "0"])

        _H.transcripts = json.loads(TRANSCRIPTS_PATH.read_text())
        _H.plan = [(clip_key(Path(p)), Path(p)) for p in clip_paths]
        missing = [k for k, _ in _H.plan if k not in _H.transcripts]
        if missing:
            raise KeyError(f"no transcript for {missing} in {TRANSCRIPTS_PATH}")
        _H.calib = quiet_tile(load_pcm(_H.plan[0][1]), CALIB_S)

        mp.setattr(cm, "sd", SimpleNamespace(
            InputStream=FakeInputStream, query_devices=lambda *a, **k: [],
            default=SimpleNamespace(device=[0, 0])))
        mp.setattr(cm, "_open_mic_array", lambda: FakeMic())
        mp.setattr(cm, "_make_ring_animator", lambda mic: None)
        mp.setattr(cm, "AudioPlayer", FakePlayer)
        mp.setattr(cm, "SpotTTS", FakeTTS)
        mp.setattr(cm, "UtteranceLog", FakeUtteranceLog)
        mp.setattr(cm, "_brain_ok", False)

        # Virtual monotonic clock for client_mic only: frame k of the main
        # stream is stamped vbase + k * 30 ms and every main-loop decision
        # (endpointer, ChimeMask, gate reopen, lock/conversation timeout) reads
        # that, so wall-clock jitter cannot move a decision by a frame. Audio
        # is still fed at real time (the safety-KWS thread runs on its own
        # real clock); time.time()/sleep stay real.
        _H.vbase = _H.vnow = time.monotonic()
        mp.setattr(cm, "time", _VirtualTime())
        mp.setattr(cm, "audio_queue", LockstepQueue(cm.AUDIO_QUEUE_MAX))
        # Safety KWS: lock-stepped behind the main loop, and its 1.5 s
        # refractory window reads the same virtual clock (it defaults to the
        # real one). Run 2 of the un-lock-stepped harness missed/added a
        # safety hit after a refractory window that ended a few frames
        # earlier or later.
        mp.setattr(cm, "_safety_q", SafetyLockstepQueue(cm.SAFETY_QUEUE_MAX))
        real_safety_loop = cm.safety_kws_loop

        def safety_loop(*a, **k):
            k["clock"] = _VirtualTime().monotonic
            return real_safety_loop(*a, **k)
        mp.setattr(cm, "safety_kws_loop", safety_loop)

        def fake_asr(stub, pcm_bytes, timeout=None):
            return _H.transcripts.get(_H.clip, "") if _H.clip else ""
        mp.setattr(cm, "send_to_asr", fake_asr)

        def fake_execute(intent, brain=None, speak=None):
            _H.add({"kind": "dispatch", "intent": intent.get("intent"),
                    "params": intent.get("params", {}), "via": _via()})
            return True
        mp.setattr(cm, "execute_on_spot", fake_execute)
        mp.setattr(cm, "get_spot_state", lambda: {})

        # Dry-run's _dispatch_safety returns before execute_on_spot: record
        # here, then run the real one (it only prints in dry-run).
        real_dispatch_safety = cm._dispatch_safety

        def dispatch_safety(intent):
            _H.add({"kind": "dispatch", "intent": intent.get("intent"),
                    "params": intent.get("params", {}), "via": _via(), "safety": True})
            return real_dispatch_safety(intent)
        mp.setattr(cm, "_dispatch_safety", dispatch_safety)

        real_enter_state = cm._enter_state

        def enter_state(new_state):
            _H.add({"kind": "state", "to": new_state.name})
            return real_enter_state(new_state)
        mp.setattr(cm, "_enter_state", enter_state)

        real_new_ep = cm._new_endpointer

        def new_endpointer():
            _H.ep = real_new_ep()  # observed (in_speech) by the feeder, never driven
            return _H.ep
        mp.setattr(cm, "_new_endpointer", new_endpointer)

        mp.setattr(cm, "_start_control_server", lambda q: None)
        real_drain = control_server.drain

        def drain(q, handler):
            # First statement of every main-loop iteration (main() imports
            # drain locally, so it is patched on control_server).
            _H.ctl_q = q
            if _H.done.is_set():
                raise KeyboardInterrupt  # main()'s own shutdown path
            return real_drain(q, handler)
        mp.setattr(control_server, "drain", drain)

        t0 = time.monotonic()
        cm.main()
        _H.meta["run_s"] = round(time.monotonic() - t0, 1)

    if _H.error is not None or not _H.done.is_set():
        print(f"harness: feeder failed: {_H.error!r}")
        return 1
    Path(out_path).write_text(json.dumps({"records": _H.events, "meta": _H.meta}, indent=1))
    print(f"harness: {len(_H.events)} records, meta={_H.meta}")
    return 0


def _generate(out_path: str) -> int:
    clips = default_clips()
    t0 = time.monotonic()
    records = normalize(run_pipeline([resolve_clip(c) for c in clips]))
    Path(out_path).write_text(json.dumps({"clips": clips, "records": records}, indent=1) + "\n")
    print(f"wrote {out_path}: {len(clips)} clips, {len(records)} records, "
          f"{time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_child":
        rc = _child(sys.argv[2], sys.argv[3:])
        sys.stdout.flush()
        os._exit(rc)  # daemon threads (safety KWS, voiceprint) must not block exit
    if len(sys.argv) == 3 and sys.argv[1] == "generate":
        sys.exit(_generate(sys.argv[2]))
    print(__doc__)
    sys.exit(2)
