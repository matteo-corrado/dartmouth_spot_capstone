"""Per-utterance latency tracing for the voice pipeline.

Records timestamps at every stage of the request path (VAD onset → ASR →
LLM → optional VLM → action dispatch → TTS render → playback) so the user
can analyze pipeline performance offline. Designed for the Jetson AGX Orin
(memory- and disk-constrained), zero new dependencies.

USAGE
=====

1. Enable via CLI flag (off by default — zero overhead when disabled):

       python scripts/run_voice_control.py --latency-metrics MODE [options]

   Modes:
       off       (default) — recorder is None, all callsites short-circuit
       summary   — in-memory ring + percentile table on shutdown (Ctrl-C)
       file      — JSONL output only, no shutdown table
       all       — both summary and JSONL output

2. Per-trace JSONL output (when mode is file or all) lands in
       runs/latency-{YYYYMMDD-HHMMSS}.jsonl
   alongside a one-shot
       runs/startup-{YYYYMMDD-HHMMSS}.json
   capturing boot-time phases (Riva connect, brain warmup, VLM warmup, etc.)

3. Live snapshot without stopping the pipeline:
       kill -USR1 <pid>
   dumps the current ring buffer to /tmp/spot_latency_snapshot_{pid}.jsonl
   and prints the path to stdout.

4. Offline analysis with polars (already in env via ultralytics):
       import polars as pl
       df = pl.read_ndjson("runs/latency-*.jsonl")
       df.select(
           ((pl.col("t_llm_response_ns") - pl.col("t_llm_request_ns")) / 1e6)
           .alias("llm_ms")
           .quantile(0.95)
       )

JSONL TRACE SCHEMA (per utterance)
==================================

   utt_id              str   — increasing per-process counter
   t0_wall             float — time.time() at vad_onset for human correlation
   t_*_ns              int   — time.monotonic_ns() at each stage boundary
   asr_text            str   — transcribed text (omit with --latency-no-transcript)
   intent_action       str   — name of dispatched action (or null)
   audio_samples       int   — len(speech_buffer) for normalizing by utterance length
   safety_path         bool  — True if utterance hit the regex safety fast-path
   vlm_used            bool  — True if a describe action ran the VLM
   error               str   — exception string if processing failed (else null)

Stage names recorded by t_*_ns fields:
   vad_onset, vad_offset, safety_check,
   asr_request, asr_response,
   llm_request, llm_response,
   vlm_request, vlm_response,
   intent_dispatch, dispatch_complete,
   tts_enqueue, tts_render_start, tts_render_end,
   tts_play_start, tts_play_end

MEMORY AND DISK
===============

- In-memory ring: 1000 traces × ~400 B = ~400 KB (configurable via
  --latency-ring-size). Bounded — old traces are auto-evicted.
- JSONL file: capped at --latency-max-file-mb (default 50 MB) per file,
  rotated to .1/.2/.3 with at most 3 files retained (200 MB total worst case).
- A startup warning fires if the configured cap exceeds 10% of free disk.
- Recording happens on the main thread *after* player.wait() returns, so
  the realtime audio path (PortAudio callback at ~33 Hz) is never blocked.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import statistics
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RING_SIZE = 1000
DEFAULT_MAX_FILE_MB = 50
MAX_ROTATED_FILES = 3
SNAPSHOT_DIR = "/tmp"
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[2] / "runs"

VALID_MODES = ("off", "summary", "file", "all")


# ---------------------------------------------------------------------------
# Trace dataclass — slots for memory efficiency
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Trace:
    """One utterance's worth of timing data.

    All ``t_*_ns`` fields are filled in via ``mark()`` or ``span()`` from
    instrumentation points across the pipeline. Fields stay None if their
    stage didn't run (e.g. ``t_vlm_*_ns`` is None unless a describe action
    fired). The recorder skips missing fields when computing percentiles.
    """
    utt_id: str
    t0_wall: float
    # Stage timestamps (monotonic ns) — None until the stage runs
    t_vad_onset_ns: Optional[int] = None
    t_vad_offset_ns: Optional[int] = None
    t_safety_check_ns: Optional[int] = None
    t_asr_request_ns: Optional[int] = None
    t_asr_response_ns: Optional[int] = None
    t_llm_request_ns: Optional[int] = None
    t_llm_response_ns: Optional[int] = None
    t_vlm_request_ns: Optional[int] = None
    t_vlm_response_ns: Optional[int] = None
    t_intent_dispatch_ns: Optional[int] = None
    t_dispatch_complete_ns: Optional[int] = None
    t_tts_enqueue_ns: Optional[int] = None
    t_tts_render_start_ns: Optional[int] = None
    t_tts_render_end_ns: Optional[int] = None
    t_tts_play_start_ns: Optional[int] = None
    t_tts_play_end_ns: Optional[int] = None
    # Non-time fields
    audio_samples: int = 0
    safety_path: bool = False
    vlm_used: bool = False
    intent_action: Optional[str] = None
    asr_text: Optional[str] = None
    error: Optional[str] = None

    def mark(self, stage: str) -> None:
        """Stamp ``t_{stage}_ns`` with the current monotonic_ns timestamp.

        Silently ignores unknown stage names — keeps callsites tolerant of
        drift between this module and the rest of the pipeline.
        """
        attr = f"t_{stage}_ns"
        if hasattr(self, attr):
            setattr(self, attr, time.monotonic_ns())

    def set(self, field_name: str, value: Any) -> None:
        """Set a non-time field on the trace (asr_text, intent_action, ...)."""
        if hasattr(self, field_name):
            setattr(self, field_name, value)

    def span(self, stage: str) -> "_Span":
        """Context manager that marks ``{stage}_request`` on enter and
        ``{stage}_response`` on exit. Usage::

            with trace.span("asr"):
                transcript = send_to_asr(...)
        """
        return _Span(self, stage)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict; drops None fields to keep JSONL compact."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class _Span:
    """Marks ``{stage}_request_ns`` on enter, ``{stage}_response_ns`` on exit."""
    __slots__ = ("trace", "stage")

    def __init__(self, trace: Trace, stage: str):
        self.trace = trace
        self.stage = stage

    def __enter__(self) -> "_Span":
        self.trace.mark(f"{self.stage}_request")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.trace.mark(f"{self.stage}_response")


# ---------------------------------------------------------------------------
# StartupTrace — single one-off boot phases trace
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class StartupTrace:
    """Captured once per process boot via ``recorder.startup_phase('name')``.

    Each ``t_*_ns`` field stores ELAPSED ns for that phase, not a wall
    clock. ``t_total_boot_ns`` is set in ``finalize_boot()`` from the
    monotonic delta since the recorder was created.
    """
    t_argparse_done_ns: Optional[int] = None
    t_riva_connect_ns: Optional[int] = None
    t_audio_player_init_ns: Optional[int] = None
    t_brain_warmup_ns: Optional[int] = None
    t_vlm_warmup_ns: Optional[int] = None
    t_tts_init_ns: Optional[int] = None
    t_kws_load_ns: Optional[int] = None
    t_total_boot_ns: Optional[int] = None
    started_at_wall: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# LatencyRecorder
# ---------------------------------------------------------------------------
class LatencyRecorder:
    """Singleton-style recorder. Created via ``init_recorder()``.

    Thread safety: per-trace mutations come from the main thread (ASR,
    LLM, dispatch) and the AudioPlayer worker thread (TTS render/play
    callbacks). These run in *sequential phases*, not concurrently —
    ``player.wait()`` provides the happens-before barrier so ``complete()``
    sees all worker writes. Individual attribute writes are GIL-atomic.
    No locks needed on any path.
    """

    def __init__(
        self,
        mode: str,
        file_path: Optional[Path] = None,
        ring_size: int = DEFAULT_RING_SIZE,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        include_transcript: bool = True,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.ring_size = ring_size
        self.max_file_bytes = max_file_mb * 1024 * 1024
        self.include_transcript = include_transcript
        self._traces: Deque[Trace] = deque(maxlen=ring_size)
        self._current: Optional[Trace] = None
        self._utt_counter = 0
        self.startup = StartupTrace()
        self._t0_boot_ns = time.monotonic_ns()

        # File output
        self._file_path: Optional[Path] = None
        self._file = None
        if mode in ("file", "all") and file_path is not None:
            self._open_file(file_path)

    # ------------------------------------------------------------------
    # Per-utterance lifecycle
    # ------------------------------------------------------------------
    def begin_utterance(self) -> Trace:
        """Create a fresh Trace, store it as the 'current' utterance."""
        self._utt_counter += 1
        utt_id = f"{int(time.time())}_{self._utt_counter:06d}"
        trace = Trace(utt_id=utt_id, t0_wall=time.time())
        self._current = trace
        return trace

    def current(self) -> Optional[Trace]:
        """Return the current in-flight Trace, if any."""
        return self._current

    def complete(self, trace: Trace) -> None:
        """Append the trace to the ring buffer and (if enabled) the JSONL file."""
        if not self.include_transcript:
            trace.asr_text = None
        self._traces.append(trace)
        if self._file is not None:
            try:
                line = json.dumps(trace.to_dict())
                self._file.write(line + "\n")
                # Cheap rotation check — every 50 writes is enough
                if self._utt_counter % 50 == 0:
                    self._maybe_rotate()
            except Exception as e:
                print(f"[Latency] file write failed: {e}")
        if self._current is trace:
            self._current = None

    # ------------------------------------------------------------------
    # Startup phases
    # ------------------------------------------------------------------
    def startup_phase(self, name: str) -> "_StartupPhase":
        """Context manager: ``with rec.startup_phase('brain_warmup'): ...``."""
        return _StartupPhase(self, name)

    def finalize_boot(self) -> None:
        """Stamp t_total_boot_ns and write the startup metrics file (if enabled)."""
        self.startup.t_total_boot_ns = time.monotonic_ns() - self._t0_boot_ns
        if self.mode in ("file", "all") and self._file_path is not None:
            # runs/latency-{ts}.jsonl  →  runs/startup-{ts}.json
            stem = self._file_path.stem
            ts = stem.split("-", 1)[1] if "-" in stem else stem
            startup_path = self._file_path.parent / f"startup-{ts}.json"
            try:
                startup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(startup_path, "w") as f:
                    json.dump(self.startup.to_dict(), f, indent=2)
                print(f"[Latency] Wrote startup metrics to {startup_path}")
            except Exception as e:
                print(f"[Latency] startup metrics write failed: {e}")

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _open_file(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", buffering=1)  # line-buffered
            self._file_path = path
            print(f"[Latency] Writing JSONL to {path}")
            # Disk-budget warning
            try:
                import shutil
                free = shutil.disk_usage(path.parent).free
                budget = self.max_file_bytes * MAX_ROTATED_FILES
                if budget > 0.10 * free:
                    print(f"[Latency] WARNING: configured budget "
                          f"{budget / 1e6:.0f} MB exceeds 10% of free disk "
                          f"{free / 1e9:.1f} GB")
            except Exception:
                pass
        except Exception as e:
            print(f"[Latency] failed to open {path}: {e}")
            self._file = None

    def _maybe_rotate(self) -> None:
        if self._file is None or self._file_path is None:
            return
        try:
            size = self._file_path.stat().st_size
        except OSError:
            return
        if size < self.max_file_bytes:
            return
        # Close, rotate, reopen
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass
        # Shift .2 -> .3, .1 -> .2, base -> .1
        base = self._file_path
        for i in range(MAX_ROTATED_FILES, 0, -1):
            src = base if i == 1 else base.with_suffix(base.suffix + f".{i - 1}")
            dst = base.with_suffix(base.suffix + f".{i}")
            if src.exists():
                try:
                    if i == MAX_ROTATED_FILES and dst.exists():
                        dst.unlink()
                    src.rename(dst)
                except Exception as e:
                    print(f"[Latency] rotation failed at .{i}: {e}")
        try:
            self._file = open(base, "a", buffering=1)
            print(f"[Latency] rotated; new file at {base}")
        except Exception as e:
            print(f"[Latency] reopen after rotation failed: {e}")
            self._file = None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Compute count/mean/p50/p95/p99/max per stage from the current ring."""
        stages = {
            "asr_dur":             ("t_asr_request_ns",      "t_asr_response_ns"),
            "llm_dur":             ("t_llm_request_ns",      "t_llm_response_ns"),
            "vlm_dur":             ("t_vlm_request_ns",      "t_vlm_response_ns"),
            "dispatch_dur":        ("t_intent_dispatch_ns",  "t_dispatch_complete_ns"),
            "tts_render_dur":      ("t_tts_render_start_ns", "t_tts_render_end_ns"),
            "tts_play_dur":        ("t_tts_play_start_ns",   "t_tts_play_end_ns"),
            "tts_queue_wait":      ("t_tts_enqueue_ns",      "t_tts_render_start_ns"),
            "time_to_first_audio": ("t_vad_onset_ns",        "t_tts_play_start_ns"),
            "total":               ("t_vad_onset_ns",        "t_tts_play_end_ns"),
        }
        out: Dict[str, Dict[str, float]] = {}
        for label, (start_field, end_field) in stages.items():
            samples: List[float] = []
            for tr in self._traces:
                a = getattr(tr, start_field)
                b = getattr(tr, end_field)
                if a is not None and b is not None and b >= a:
                    samples.append((b - a) / 1e6)  # ns → ms
            out[label] = self._summarize(samples)
        return out

    @staticmethod
    def _summarize(samples: List[float]) -> Dict[str, float]:
        n = len(samples)
        if n == 0:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0,
                    "p99": 0.0, "max": 0.0}
        s = sorted(samples)
        mean = sum(s) / n
        p50 = s[n // 2]
        if n >= 2:
            qs = statistics.quantiles(s, n=100, method="inclusive")
            p95 = qs[94]
            p99 = qs[98]
        else:
            p95 = s[0]
            p99 = s[0]
        return {"count": n, "mean": mean, "p50": p50, "p95": p95,
                "p99": p99, "max": s[-1]}

    def dump_summary(self, stream=None) -> None:
        """Print a fixed-width summary table to ``stream`` (default stdout)."""
        import sys
        if stream is None:
            stream = sys.stdout
        snap = self.snapshot()

        # Startup section
        sd = self.startup.to_dict()
        if sd:
            stream.write("\n========== STARTUP PHASES ==========\n")
            for k, v in sd.items():
                if k.endswith("_ns") and isinstance(v, (int, float)):
                    name = k[2:-3]
                    stream.write(f"  {name:<24} {v / 1e6:>10.1f} ms\n")
            stream.write("\n")

        # Per-stage table
        stream.write("========== LATENCY SUMMARY (ms) ==========\n")
        header = (f"  {'stage':<22} {'count':>6} {'mean':>9} {'p50':>9} "
                  f"{'p95':>9} {'p99':>9} {'max':>9}\n")
        stream.write(header)
        stream.write("  " + "-" * (len(header) - 4) + "\n")
        for label, stats in snap.items():
            stream.write(
                f"  {label:<22} {int(stats['count']):>6d} "
                f"{stats['mean']:>9.1f} {stats['p50']:>9.1f} "
                f"{stats['p95']:>9.1f} {stats['p99']:>9.1f} "
                f"{stats['max']:>9.1f}\n"
            )
        stream.write("\n")

    def snapshot_to_file(self, path: Optional[Path] = None) -> Path:
        """Dump the current ring contents as JSONL to a snapshot file."""
        if path is None:
            path = Path(SNAPSHOT_DIR) / f"spot_latency_snapshot_{os.getpid()}.jsonl"
        try:
            with open(path, "w") as f:
                for tr in self._traces:
                    f.write(json.dumps(tr.to_dict()) + "\n")
            print(f"[Latency] snapshot written to {path}")
        except Exception as e:
            print(f"[Latency] snapshot write failed: {e}")
        return path

    def shutdown(self) -> None:
        """Print summary (if mode includes it) and close any open file."""
        if self.mode in ("summary", "all"):
            try:
                self.dump_summary()
            except Exception as e:
                print(f"[Latency] summary print failed: {e}")
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass


class _StartupPhase:
    """Context manager that records elapsed ns into a StartupTrace field."""
    __slots__ = ("recorder", "name", "_t0")

    def __init__(self, recorder: LatencyRecorder, name: str):
        self.recorder = recorder
        self.name = name
        self._t0 = 0

    def __enter__(self) -> "_StartupPhase":
        self._t0 = time.monotonic_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic_ns() - self._t0
        attr = f"t_{self.name}_ns"
        if hasattr(self.recorder.startup, attr):
            setattr(self.recorder.startup, attr, elapsed)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------
_RECORDER: Optional[LatencyRecorder] = None


def init_recorder(
    mode: str = "off",
    file_path: Optional[str] = None,
    ring_size: int = DEFAULT_RING_SIZE,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
    include_transcript: bool = True,
) -> Optional[LatencyRecorder]:
    """Initialize the singleton recorder. Idempotent — second call is a no-op.

    When ``mode == 'off'`` returns None and leaves the singleton None, so all
    ``get_recorder()`` callsites short-circuit with zero overhead.
    """
    global _RECORDER
    if mode == "off":
        _RECORDER = None
        return None
    if _RECORDER is not None:
        return _RECORDER
    fp: Optional[Path] = None
    if mode in ("file", "all"):
        if file_path:
            fp = Path(file_path)
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            fp = DEFAULT_RUN_DIR / f"latency-{ts}.jsonl"
    _RECORDER = LatencyRecorder(
        mode=mode,
        file_path=fp,
        ring_size=ring_size,
        max_file_mb=max_file_mb,
        include_transcript=include_transcript,
    )
    atexit.register(_RECORDER.shutdown)
    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
    except (ValueError, OSError):
        # Not main thread, or signal not available — silently skip
        pass
    return _RECORDER


def get_recorder() -> Optional[LatencyRecorder]:
    """Return the singleton recorder, or None if disabled."""
    return _RECORDER


def _sigusr1_handler(signum, frame) -> None:
    rec = _RECORDER
    if rec is not None:
        rec.snapshot_to_file()
