"""Mic input: device lookup, frame helpers, stream open/retry, the PortAudio
audio callback, chime mask, queue drain and noise-floor calibration.

Moved from the old client_mic.py (Task 10). The state that used to
be client_mic module globals lives on the AudioCapture owner, CAPTURE,
created once at import. Other modules reach it (and every other moved name)
through this module at call time (``_cap.CAPTURE``), never by binding it.
"""
import glob
import math
import queue
import subprocess
import time

import numpy as np
import sounddevice as sd

import src.spot.conversation.gate as _gate
import src.spot.conversation.loop as _loop
import src.spot.mic.ring_control as _ring
import src.spot.safety.listener as _sl
from src.spot.audio.frontend import Endpointer
from src.spot.core.tuning import get_tuning

_tuning = get_tuning()


# ============================================================================
# Device Lookup (by name, so index changes don't break things)
# ============================================================================
MIC_DEVICE_NAME = "XVF3800"          # reSpeaker XVF3800 4-Mic Array
SPEAKER_DEVICE_NAME = "UACDemoV1.0"  # USB speaker on Spot CORE I/O

def _find_device_by_name(substring: str, kind: str = "input") -> int | None:
    """Find audio device index by substring match on its name.

    Args:
        substring: Partial name to match (case-insensitive).
        kind: "input" to require input channels, "output" for output channels.

    Returns:
        Device index or None if not found.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if substring.lower() in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] > 0:
                return i
            elif kind == "output" and dev["max_output_channels"] > 0:
                return i
    return None


# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_S = FRAME_MS / 1000.0      # matches FloorTracker/UtteranceStats frame_s=0.03
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes

SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"
VAD_WINDOW_SAMPLES = 512         # Silero default @ 16 kHz = 32 ms; VAD buffers internally
NOISE_CALIBRATION_SECONDS = 2    # Seconds to measure ambient noise


def _frames(seconds: float) -> int:
    """Seconds -> 30 ms frame count (Endpointer is frame-based)."""
    return max(1, int(round(seconds / FRAME_S)))


def dbfs_from_rms(rms: float) -> float:
    """Linear RMS (full scale 1.0) -> dBFS, floored at -120."""
    if rms <= 1e-6:
        return -120.0
    return max(-120.0, 20.0 * math.log10(rms))


# ============================================================================
# Audio Queue (filled by callback)
# ============================================================================
# Bounded so that a wedged main loop or a long mic-mute window can't grow
# the queue without limit. 33 frames at 30ms each = ~1s of headroom, well
# above the 0.15s Tuning.speaker_tail_s grace period. On overflow we
# drop frames in the callback (see AudioCapture.callback) rather than blocking
# PortAudio's high-priority thread.
AUDIO_QUEUE_MAX = 33
MIC_CHANNEL = _tuning.mic_channel  # 0=left (beamformed), 1=right (ASR); --mic-channel


class AudioCapture:
    """Owner of the mic-input state (was client_mic module globals).

    Mic-mute interlock — derived from AudioPlayer state instead of a flag.
    The audio callback discards input frames whenever the player is busy
    (TTS speaking or beep playing) so the mic can never feed Spot's own
    voice back into ASR. Wired up in main() before the input stream opens.
    """

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self._audio_player = None  # AudioPlayer | None
        # Stage 2F P6: hard mic gate during THINKING/RESPONDING + the 0.5s reopen
        # settle. The audio callback drops frames while gated, so no speech uttered
        # from the moment Spot commits to a response until 0.5s after it stops
        # speaking is ever captured (no barge-in, no late phantom commands).
        self._mic_gated = False
        self._mic_reopen_at = None  # monotonic time the post-response settle ends
        # Cooldown after the player goes idle: the OS audio buffer keeps draining
        # for ~tens of ms after stream.write() returns, so the mic would otherwise
        # pick up the tail of Spot's own utterance. Length = Tuning.speaker_tail_s,
        # measured from the last frame where Spot's SPEECH was busy.
        self._player_last_busy_at = 0.0  # monotonic timestamp of the most recent speech-busy frame
        # XVF3800 control (DoA, LED ring, fixed beams). Optional: None when unplugged
        # and treated as absent once `dead` (any USB error) — the loop never stops.
        self._mic = None
        self._mic_dead_logged = False
        # The callback handed to PortAudio, bound ONCE: the stream and the
        # client_mic re-export are the same object (identity is checked).
        self._callback = self.callback

    def callback(self, indata, frames, time_info, status):
        """Sounddevice callback - converts stereo to mono PCM16.

        Mute logic: while the AudioPlayer is busy (TTS speaking or beep playing)
        we discard frames, and we keep discarding for Tuning.speaker_tail_s
        after it goes idle so the OS audio buffer has time to fully drain.
        The watchdog that recovers a wedged player runs in the main loop, not
        here — this callback runs on PortAudio's high-priority thread and must
        not do heavy work or block.
        """
        now = time.monotonic()  # local clock — PortAudio's per-stream clock isn't comparable
        # Select channel: 0=left (beamformed), 1=right (ASR) for stereo mics like XVF3800
        if indata.shape[1] >= 2:
            mono = indata[:, MIC_CHANNEL].astype(np.float32)
        else:
            mono = indata[:, 0].astype(np.float32)
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        # Safety tap FIRST: the safety-KWS thread hears every frame in every
        # state — gated, Spot speaking, operator-muted — so a spoken stop/freeze
        # always works. (Spot saying "stop" itself may halt it: acceptable.)
        if _sl.LISTENER._active:
            if _sl._put_drop_oldest(_sl.LISTENER._q, pcm16):
                try:
                    _sl._note_safety_drop(now)
                except Exception:  # noqa: BLE001 - reporting must never stop capture
                    pass
        if _loop.CONVERSATION._operator_muted:
            return  # operator mute (spotctl mute) — discard everything else
        if _speech_busy():
            self._player_last_busy_at = now
            return  # discard frame — Spot is talking
        if self._mic_gated:
            # Gated for a response. Don't bump the tail clock here: the tail is
            # measured from the last SPEECH frame, so the first frame after the
            # reopen settle is captured (no clipped first word).
            return
        if now - self._player_last_busy_at < _tuning.speaker_tail_s:
            return  # cooldown — speaker buffer still draining
        if status:
            print(f"[Audio: {status}]")
        try:
            self.audio_queue.put_nowait((now, pcm16))  # monotonic stamp (DoA/lock clock)
        except queue.Full:
            # Drop the frame instead of blocking PortAudio's callback thread.
            # ~1s of frames are already buffered; if the main loop is that far
            # behind, the right answer is to keep the audio device responsive.
            pass

    def open(self, device):
        """Open the mic stream (stereo for XVF3800, mono fallback, retry on busy)."""
        return _open_input_stream_with_retry(device, self._callback)

    def drain(self, keep_tail: bool = True):
        """Discard stale audio frames accumulated during processing.

        keep_tail=True keeps the most recent ~1s (a wake fire's in-breath command
        could be in there). keep_tail=False drains everything — used after a
        response so speech uttered while Spot was thinking/speaking is never
        replayed as a phantom command.
        """
        frames = []
        while not self.audio_queue.empty():
            try:
                frames.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        keep_count = max(1000 // FRAME_MS, 1) if keep_tail else 0  # ~33 frames at 30ms
        kept, dropped = partition_drain(frames, keep_count)

        for frame in kept:
            self.audio_queue.put(frame)

        if dropped:
            print(f"[Drained {dropped} stale audio chunks, kept {len(kept)}]")

    def calibrate_floor(self, duration_sec: float, device=None) -> float:
        """
        Measure ambient noise. Returns the MEASURED noise RMS (always — it seeds
        the FloorTracker once, A2); a high reading only warns.
        """
        print(f"\n[Calibrating noise floor for {duration_sec}s - please stay quiet...]")

        samples = []
        samples_needed = int(SAMPLE_RATE * duration_sec)

        def callback(indata, frames, time_info, status):
            if indata.shape[1] >= 2:
                mono = indata[:, MIC_CHANNEL].astype(np.float32)
            else:
                mono = indata[:, 0].astype(np.float32)
            samples.append(mono.copy())

        try:
            stream_ctx = _open_input_stream_with_retry(device, callback)
            with stream_ctx:
                start = time.time()
                while sum(len(s) for s in samples) < samples_needed:
                    time.sleep(0.05)
                    if time.time() - start > duration_sec + 1:
                        break

            noise_audio = np.concatenate(samples)[:samples_needed]
            noise_rms = np.sqrt(np.mean(noise_audio ** 2))
            print(f"[Noise floor RMS: {noise_rms:.5f}]")

            # Suspiciously high noise: robot motors / crowd during calibration.
            # Keep the measured value (the floor tracker falls instantly if the
            # room gets quieter) but tell the operator.
            if noise_rms > 0.003:
                print(f"[WARNING: High noise ({noise_rms:.5f}, {dbfs_from_rms(noise_rms):.1f} dBFS) "
                      f"— robot motors or crowd running during calibration?]")

            return float(noise_rms)

        except Exception as e:
            print(f"[Calibration error: {e}]")
            return 0.01  # Default fallback


def _mic_usable() -> bool:
    return CAPTURE._mic is not None and not getattr(CAPTURE._mic, "dead", True)


def _check_mic_alive() -> None:
    """Log XVF3800 loss/restore transitions once each; lock/ring degrade while
    dead (MicArray reconnects in the background) and repaint on restore."""
    if CAPTURE._mic is None:
        return
    dead = getattr(CAPTURE._mic, "dead", False)
    if dead and not CAPTURE._mic_dead_logged:
        CAPTURE._mic_dead_logged = True
        print("[MicArray] XVF3800 control lost — speaker lock + ring paused, voice continues")
    elif not dead and CAPTURE._mic_dead_logged:
        CAPTURE._mic_dead_logged = False
        _ring.RING._ring_pending = True  # repaint the ring for the current state
        print("[MicArray] XVF3800 control back — speaker lock + ring resumed")


def _mic_speech_degs(t0: float, t1: float) -> list:
    """Speech-flagged DoA degrees (mic frame) in [t0, t1] (monotonic)."""
    if not _mic_usable():
        return []
    try:
        return _gate.speech_degs(CAPTURE._mic.samples_between(t0, t1))
    except Exception as e:
        print(f"[MicArray] DoA read failed: {e}")
        return []


def _clear_fixed_beams(mic) -> None:
    """G10: turn off fixed beams a previous run left on (e.g. an interrupted
    record_demo_clips --beam): the same runtime write the loop uses on release
    and shutdown (set_fixed_beams(None) = AEC_FIXEDBEAMSONOFF 0). Never saves
    the config, reboots or resets the device. Never raises.

    MicArray._write swallows its own USB errors and marks the link dead, so
    a failed write shows up as mic.dead afterwards, not as an exception."""
    try:
        mic.set_fixed_beams(None)
    except Exception as e:  # noqa: BLE001
        print(f"[MicArray] ⚠️  fixed-beams clear failed: {type(e).__name__}: {e}", flush=True)
        return
    if getattr(mic, "dead", False):
        print("[MicArray] ⚠️  fixed-beams clear failed: XVF3800 control link is dead "
              "(beams may still be on until it reconnects or is power-cycled)", flush=True)


def _open_mic_array():
    """Open the XVF3800 control interface + start the DoA poller. Never raises."""
    try:
        from src.spot.mic.xvf3800 import MicArray
        mic = MicArray.open()
    except Exception as e:
        print(f"[MicArray] unavailable: {type(e).__name__}: {e}")
        return None
    if mic is None:
        print("[MicArray] XVF3800 control not found — no speaker lock / ring")
        return None
    _clear_fixed_beams(mic)
    try:
        # Runs on the xvf-reconnect thread (never the audio thread) once a
        # lost link is restored: the same runtime-only beams-off write.
        mic.add_restore_listener(lambda: _clear_fixed_beams(mic))
    except Exception as e:  # noqa: BLE001
        print(f"[MicArray] restore listener not added: {e}")
    try:
        mic.start_poller(20.0)
    except Exception as e:
        print(f"[MicArray] DoA poller failed: {e} — ring only, no speaker lock")
        return mic
    print("[MicArray] XVF3800 control ready (DoA poller 20 Hz)")
    return mic


def _new_endpointer() -> Endpointer:
    return Endpointer(_tuning.onset_frames, _frames(_tuning.end_silence_s_short),
                      _frames(_tuning.max_utterance_s))

# The mic-mute watchdog lives inside AudioPlayer (audio_player.py) as its own
# daemon thread. Putting it there instead of the main loop here means it can
# still recover from a wedge that happens *inside* process_utterance, when
# this loop is blocked on tts.wait() — which was the exact failure shape that
# motivated the refactor.


def _speech_busy() -> bool:
    """True while Spot's own SPEECH is queued/playing. Chimes (beep: labels)
    don't count, so a turn cue never mutes the mic (Task 8)."""
    p = CAPTURE._audio_player
    if p is None:
        return False
    return p.is_speech_busy()


def _chime_busy() -> bool:
    """A beep (not speech) is queued/playing."""
    p = CAPTURE._audio_player
    return p is not None and p.is_busy() and not p.is_speech_busy()


class ChimeMask:
    """Masks VAD speech while a chime plays and for `tail_s` after it.
    update() -> (masked, ended): ended is True on the frame the mask lifts.
    `speech` is the raw VAD flag; speech_before is True if VAD was already
    flagging speech when the mask began (user talking through the chime)."""

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
        """Clear Silero's chime hangover when the mask lifts, unless the user
        was already speaking before the chime (keep their onset)."""
        return ended and not in_utt and not self.speech_before


# ============================================================================
# Mic stream open (with retry on transient device-busy)
# ============================================================================
# PortAudio reports both "wrong channel count" and "device busy" through the
# same generic exception. The original code caught any failure, assumed it
# was a channel mismatch, and printed "Stereo not supported, falling back to
# mono" — which masked the real cause when the actual problem was that a
# previous client_mic.py instance was still releasing the snd_usb_audio
# substream (kernel-release lag after the prior process exited). The helper
# below distinguishes the two cases and retries on busy.

def _is_format_error(err: Exception) -> bool:
    """True if a PortAudio error is a real channel/sample-rate mismatch.

    Anything else (especially -9985 / paDeviceUnavailable) is treated as a
    transient busy condition worth retrying.
    """
    msg = str(err).lower()
    return ("invalid number of channels" in msg
            or "invalid sample rate" in msg
            or "incompatible" in msg)


def _log_pcm_holders() -> None:
    """Print processes holding any ALSA capture PCM. Best-effort, never raises.

    Called when a stream open fails with a non-format error so we can see
    *who* is squatting on the mic. Output is empty when the kernel hasn't
    finished releasing the substream from a just-exited holder — that empty
    case is itself the diagnosis (kernel-release lag).
    """
    try:
        pcm_files = sorted(glob.glob("/dev/snd/pcmC*c"))
        if not pcm_files:
            print("[mic-busy probe] no /dev/snd/pcmC*c devices found")
            return
        result = subprocess.run(
            ["fuser", "-v"] + pcm_files,
            capture_output=True, text=True, timeout=2,
        )
        # fuser writes its table to stderr; combine for safety.
        out = ((result.stderr or "") + (result.stdout or "")).strip()
        if out:
            print(f"[mic-busy probe] capture PCM holders:\n{out}")
        else:
            print("[mic-busy probe] no PCM holders — likely kernel-release lag")
    except FileNotFoundError:
        print("[mic-busy probe] fuser not installed; skipping")
    except Exception as exc:
        print(f"[mic-busy probe failed: {exc}]")


def _open_input_stream_with_retry(device, callback, *,
                                  attempts: int = 6,
                                  backoff: float = 0.5) -> sd.InputStream:
    """Open a stereo InputStream, retrying on transient device-busy errors.

    - Real channel/format mismatches → fall back to mono on the first attempt.
    - Anything else (e.g. -9985 paDeviceUnavailable) → retry with backoff.
      On the first failure we dump the current PCM holders so the next
      occurrence is diagnosable from the log.
    """
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return sd.InputStream(
                device=device, channels=2, samplerate=SAMPLE_RATE,
                callback=callback, blocksize=FRAME_SAMPLES,
            )
        except Exception as e:
            if _is_format_error(e):
                print(f"[Stereo not supported on device {device}, falling back to mono]")
                return sd.InputStream(
                    device=device, channels=1, samplerate=SAMPLE_RATE,
                    callback=callback, blocksize=FRAME_SAMPLES,
                )
            last_err = e
            if attempt == 0:
                print(f"[mic open failed: {e}]")
                _log_pcm_holders()
            if attempt < attempts - 1:
                print(f"[mic busy; retry {attempt + 1}/{attempts - 1} in {backoff}s]")
                time.sleep(backoff)
    assert last_err is not None
    raise last_err


# ============================================================================
# Audio Processing
# ============================================================================
def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 array to PCM16 bytes."""
    return (audio * 32767).astype(np.int16).tobytes()


def partition_drain(frames, keep_count):
    """Split drained frames into (kept, dropped_count).

    keep_count > 0 keeps the most recent `keep_count` frames (used after a
    wake fire so an in-breath command isn't lost); keep_count == 0 keeps
    nothing (used after a response so post-decision speech can't replay).
    """
    if not frames:
        return [], 0
    if keep_count <= 0:
        return [], len(frames)
    dropped = max(0, len(frames) - keep_count)
    return frames[dropped:], dropped


CAPTURE = AudioCapture()
audio_callback = CAPTURE._callback  # the one callable PortAudio is given
calibrate_noise_floor = CAPTURE.calibrate_floor
_drain_audio_queue = CAPTURE.drain
