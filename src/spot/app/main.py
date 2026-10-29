"""
Voice control client for Spot with LLM brain — composition root.

Architecture (Boston Dynamics "Robots That Can Chat" style):
    Mic → VAD → Whisper ASR → LLM Brain (state + history + personality) → Action + Response

Safety commands (stop/estop/freeze) bypass the LLM for zero-latency execution.
Everything else goes through the LLM brain which decides what to do AND what to say.

Moved from src/voice_control/client_mic.py (Task 10): CLI args, startup,
the shutdown signal handler and the shutdown sequence. The voice loop itself
is Conversation.run() (src/spot/conversation/loop.py).
"""
import os
import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Auto-load .env from project root so callers don't need `source .env`
# first. `override=True` so .env wins over stale shell exports (e.g. an old
# SPOT_TTS_BACKEND=kokoro from a prior session). Broad except: dotenv may
# not be installed, .env may be unreadable (PermissionError) or non-UTF-8
# (UnicodeDecodeError); none of those should crash the launcher.
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
except Exception:
    pass

from src.voice_control.chatbot.persona import get_persona, voice_id_for  # noqa: F401

import re  # noqa: F401
import glob  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import time  # noqa: F401
import queue  # noqa: F401
import random  # noqa: F401
import signal
import argparse
import subprocess  # noqa: F401
import threading  # noqa: F401
import numpy as np  # noqa: F401
import sounddevice as sd
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig
import grpc


# Shutdown signal handling.
#
# First SIGINT or SIGTERM → raise KeyboardInterrupt so the main loop's
# finally block runs the graceful shutdown (cancel nav, sit robot down,
# power off, drain audio player).
#
# Second signal → os._exit() immediately. cleanup_spot() can take tens of
# seconds for legitimate reasons (blocking_sit + power_off are bosdyn
# gRPC blocking calls that aren't interruptible from Python while they're
# inside C++). Without an escape hatch, a user pressing Ctrl+C in the
# terminal can find themselves unable to exit at all — we observed this
# in the field, where Ctrl+C fired but the pipeline kept running and the
# user had to manually `kill` the processes. The hard-exit on the second
# signal guarantees they can always escape.
#
# Both SIGINT and SIGTERM share this handler so the web panel's
# SIGINT-then-SIGTERM escalation also gets the hard-exit on the SIGTERM.
_shutdown_requested = False

def _shutdown_signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        # Second signal — escape hatch. cleanup is wedged; bail.
        print("\n[Shutdown] second signal received — force exiting", flush=True)
        os._exit(130 if signum == signal.SIGINT else 143)
    _shutdown_requested = True
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, _shutdown_signal_handler)
signal.signal(signal.SIGTERM, _shutdown_signal_handler)

from contextlib import nullcontext

from src.spot.stt.asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk  # noqa: F401
from src.spot.stt.asr_pb2_grpc import ASRStub
from src.spot.brain.intent import parse_intent  # noqa: F401
from src.spot.brain.llm_brain import LLMBrain, DEFAULT_MODEL
from src.spot.audio.player import AudioPlayer
from src.spot.tts.speaker import SpotTTS
from src.spot.audio.chimes import beep
from src.spot.core.latency import init_recorder, get_recorder
from src.voice_control import startup_status  # noqa: F401 — available for later use
from src.voice_control.response_gate import should_arm_reopen, should_reopen_mic  # noqa: F401
from src.voice_control.tuning import get_tuning
from src.voice_control.barge_in import should_reset_after_player  # noqa: F401
from src.voice_control.audio_frontend import (  # noqa: F401
    frame_dbfs, FloorTracker, Endpointer, UtteranceStats, passes_gate)
from src.voice_control.speaker_lock import (  # noqa: F401
    SpeakerLock, circular_median_deg, mic_to_body_deg)
from src.voice_control.speaker_embed import VoiceprintGate, warm_up_async  # noqa: F401

_tuning = get_tuning()
from src.voice_control.voice_state import VoiceState  # noqa: F401,E402
from src.voice_control.state_feedback import chime_for, ring_for as _ring_for  # noqa: F401,E402

import src.spot.audio.capture as _cap  # noqa: E402
import src.spot.conversation.loop as _loop  # noqa: E402
import src.spot.conversation.utterance_log as _ul  # noqa: E402
import src.spot.mic.ring_control as _ring  # noqa: E402
import src.spot.safety.listener as _sl  # noqa: E402


def cleanup_spot():
    """Clean up Spot session and audio player on exit.

    Order matters:

    1. Cancel any in-progress navigation thread FIRST. follow_me /
       go_to_object / tour / patrol all run in a daemon thread that
       keeps issuing velocity / trajectory commands until its
       ``stop_event`` is set. If we skip this and go straight to
       ``blocking_sit``, the daemon's velocity stream fights the sit
       command — both bosdyn calls hit their full 15s + 20s timeouts
       and the whole shutdown drags on so long that the user thinks
       Ctrl+C didn't work and reaches for ``kill``.

    2. Then close the Spot session (sit + power_off + lease release).

    3. Then shut the AudioPlayer down. Doing this BEFORE the process
       exits keeps the daemon worker thread from being torn down mid
       ``stream.write()``, which would corrupt the heap (the same
       ``malloc(): unaligned tcache chunk`` bug we already fixed for
       force_reset). The shutdown call drains the queue, joins the
       worker, and closes the stream gracefully.
    """
    # In dry-run mode there is no Spot session to close; skip the
    # spot_dispatch imports entirely so cleanup doesn't trigger bosdyn
    # module-load side effects in a path advertised as robot-free.
    if os.environ.get("SPOT_DRY_RUN") != "1":
        try:
            from src.voice_control.spot_dispatch import _cancel_nav
            _cancel_nav()
        except Exception as e:
            print(f"[Spot] cancel_nav error during cleanup: {e}")
        try:
            from src.voice_control.spot_dispatch import close_spot_session
            close_spot_session()
        except Exception:
            pass
    if _cap.CAPTURE._audio_player is not None:
        try:
            _cap.CAPTURE._audio_player.shutdown()
        except Exception as e:
            print(f"[Player] shutdown error during cleanup: {e}")


def _probe_host() -> str:
    """Robot host for the startup TCP probe: SPOT_IP, else src.config's
    BOSDYN_ROBOT_IP (only that name is imported), else the WiFi-AP default."""
    host = os.environ.get("SPOT_IP")
    if not host:
        try:
            from src.config import BOSDYN_ROBOT_IP
            host = BOSDYN_ROBOT_IP
        except Exception:
            host = None
    return host or "192.168.80.3"


def _make_wake_detector():
    """Wake backend from SPOT_WAKE_BACKEND; the sherpa-onnx spotter gets the
    peak-safe gain target from Tuning.kws_peak_target."""
    if os.environ.get("SPOT_WAKE_BACKEND", "sherpa_onnx").lower() == "sherpa_onnx":
        from src.voice_control.wake.sherpa_onnx import WakeWordDetector
        return WakeWordDetector(peak_target=_tuning.kws_peak_target)
    from src.voice_control.wake import make_wake_detector
    return make_wake_detector()


def auto_arm_wanted(args, tuning) -> bool:
    """Automatic fold + mouth unless --no-arm, tuning auto_arm: false or dry-run."""
    return bool(tuning.auto_arm and not getattr(args, "no_arm", False)
                and not getattr(args, "dry_run", False))


def _shutdown_runtime() -> None:
    """Stop the safety thread, ring, beams and mic, then the robot session
    and player. Safe to call on any exit path."""
    _sl.LISTENER.stop()
    _anim = _ring.RING._anim
    if _anim is not None:
        try:
            _anim.stop()  # clears the ring
        except Exception:
            pass
    _mic = _cap.CAPTURE._mic
    if _mic is not None:
        try:
            if not _mic.dead:
                _mic.set_fixed_beams(None)
                if _anim is None:
                    _mic.set_led_effect(0)  # ring off
            _mic.close()
        except Exception:
            pass
    cleanup_spot()


# ============================================================================
# Main Voice Control Loop
# ============================================================================
def main(argv: list[str] | None = None):
    C = _loop.CONVERSATION

    parser = argparse.ArgumentParser(description="Spot Voice Control Client")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Mic input device index (auto-detected from XVF3800)")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker output device index (auto-detected from UACDemoV1.0)")
    parser.add_argument("--no-brain", action="store_true", help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--tts-backend", choices=["kokoro", "elevenlabs"], default=None,
                        help="TTS backend (overrides SPOT_TTS_BACKEND env). "
                             "kokoro=local ONNX, elevenlabs=cloud premium.")
    parser.add_argument("--no-wake-word", action="store_true", help="Always listening (skip wake word)")
    parser.add_argument("--dry-run", action="store_true", help="Skip robot dispatch (transcript+brain only; for mic-verify)")
    parser.add_argument("--debug-audio", action="store_true", help="Print audio levels for mic diagnostics")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="TTS + beep output gain (0.0-1.5, default 1.0)")
    parser.add_argument("--map", type=str, default=None,
                        help="GraphNav map path to upload during session bring-up.")
    parser.add_argument("--enable-arm", action="store_true",
                        help="No-op, kept for old launch lines: the arm folds + the mouth "
                             "turns on automatically once Spot is standing (see --no-arm).")
    parser.add_argument("--no-arm", action="store_true",
                        help="Never fold the arm / turn the gripper mouth on automatically "
                             "(also: tuning auto_arm: false). 'enable mouth' still works.")
    parser.add_argument(
        "--latency",
        choices=["off", "summary", "file", "all"],
        default="off",
        help="Latency telemetry mode: off (default), summary (in-memory only), file (write JSONL), all (both)",
    )
    parser.add_argument(
        "--latency-out",
        type=str,
        default=None,
        help="Latency JSONL file path (default: logs/latency-{timestamp}.jsonl). Only used when --latency=file or all.",
    )
    parser.add_argument("--mic-channel", type=int, default=_tuning.mic_channel,
                        help="XVF3800 input channel (0=beamformed, 1=ASR); default Tuning.mic_channel")
    parser.add_argument("--no-transcript-log", action="store_true",
                        help="Privacy: conversation + utterance logs store transcript: null")
    parser.add_argument("--allow-no-safety-kws", action="store_true",
                        help="Start on the robot even if the always-on safety keyword "
                             "spotter fails to load (default: refuse)")
    args = parser.parse_args(argv)

    _cap.MIC_CHANNEL = args.mic_channel
    if args.no_transcript_log:
        os.environ["SPOT_NO_TRANSCRIPT_LOG"] = "1"  # read by ConversationLog
    store_transcript = os.environ.get("SPOT_NO_TRANSCRIPT_LOG", "").strip().lower() not in (
        "1", "true", "yes", "on")
    utt_log = _ul.UtteranceLog(_ul.UTTERANCE_LOG_DIR, store_transcript=store_transcript)
    C._store_transcript = store_transcript

    if args.dry_run:
        os.environ["SPOT_DRY_RUN"] = "1"
        print("[dry-run] robot dispatch disabled (transcript + brain only)")

    if args.tts_backend is not None:
        os.environ["SPOT_TTS_BACKEND"] = args.tts_backend
        print(f"[TTS] backend override: {args.tts_backend}")

    # Latency telemetry — no-op when --latency=off
    init_recorder(mode=args.latency, file_path=args.latency_out,
                  include_transcript=store_transcript)

    # Auto-detect devices by name if not explicitly specified
    if args.device is None:
        args.device = _cap._find_device_by_name(_cap.MIC_DEVICE_NAME, "input")
        if args.device is None:
            print(f"ERROR: Could not find mic device matching '{_cap.MIC_DEVICE_NAME}'")
            print("Run with --list-devices to see available devices, then pass --device <index>")
            return
        print(f"[Auto-detected mic: device {args.device} ({_cap.MIC_DEVICE_NAME})]")
    if args.output_device is None:
        args.output_device = _cap._find_device_by_name(_cap.SPEAKER_DEVICE_NAME, "output")
        if args.output_device is None:
            print(f"WARNING: Could not find speaker matching '{_cap.SPEAKER_DEVICE_NAME}', using system default")
        else:
            print(f"[Auto-detected speaker: device {args.output_device} ({_cap.SPEAKER_DEVICE_NAME})]")

    if args.list_devices:
        print(sd.query_devices())
        return

    # ========================================================================
    # Startup
    # ========================================================================
    print("=" * 60)
    print("SPOT VOICE CONTROL — LLM Brain Mode")
    print("=" * 60)

    # Show device info
    if args.device is not None:
        print(f"Audio device: {args.device}")
    else:
        try:
            info = sd.query_devices(sd.default.device[0])
            print(f"Audio device: {info['name']} (default)")
        except Exception as e:
            print(f"Audio device: system default (query failed: {e})")

    # Calibrate noise floor (seeds the adaptive noise tracker)
    noise_rms = _cap.CAPTURE.calibrate_floor(_cap.NOISE_CALIBRATION_SECONDS, args.device)
    floor = FloorTracker(_tuning.floor_window_s, _tuning.floor_percentile,
                         _tuning.floor_rise_db_per_s, frame_s=_cap.FRAME_S)
    floor.seed(_cap.dbfs_from_rms(noise_rms))  # seed ONCE (seed clears the window)
    print(f"[Noise floor: {floor.floor_db:.1f} dBFS (gate: speech >= {_tuning.min_speech_dbfs:.0f} dBFS, "
          f"SNR(p90) >= {_tuning.min_snr_db:.1f} dB)]")

    # Connect to ASR
    asr_server = "localhost:50055"
    print(f"\nConnecting to ASR server at {asr_server}...")
    channel = grpc.insecure_channel(asr_server)
    stub = ASRStub(channel)

    # Initialize LLM Brain
    brain = None
    if not args.no_brain:
        print(f"\nInitializing LLM brain (model: {DEFAULT_MODEL})...")
        brain = LLMBrain(model=DEFAULT_MODEL)
        # After a reboot llama-server answers 503 while loading: wait for it
        # instead of disabling the brain for the session.
        print("[Brain] waiting up to 90s for llama-server...")
        ready = brain.wait_available(timeout_s=90.0)
        C._brain_ok = bool(ready)
        if ready:
            print(f"[Brain] Ready — model: {DEFAULT_MODEL}")
            # Warm the single brain model (gemma4:e4b handles both LLM and VLM after
            # Phase 2 collapse). Cold-start ~15-20s blocks AudioPlayer + TTS init, but
            # surfaces model-load errors before the mic opens.
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain.warm_up()
        else:
            # Keep the brain object: process_utterance re-checks before each
            # utterance and switches to the LLM as soon as it answers.
            print("[Brain] llama-server not available yet — regex-only until it answers "
                  "(re-checked every utterance)")
            print("[Brain] Check: systemctl status llama-server (port 11435)")
    else:
        print("\n[Brain] Disabled (--no-brain flag). Using regex-only mode.")

    # Initialize the shared audio player BEFORE TTS / beeps. Owns the single
    # persistent OutputStream that all playback (TTS and beeps) flows through.
    # The mic-mute interlock in the audio callback reads its is_busy() state.
    print("\nInitializing audio player...")
    _cap.CAPTURE._audio_player = AudioPlayer(output_device=args.output_device)

    # Initialize TTS — submits playback through the shared player.
    tts = None
    if not args.no_tts:
        print("\nInitializing TTS...")
        tts = SpotTTS(player=_cap.CAPTURE._audio_player, volume=args.volume)
        if tts.is_available():
            print(f"[TTS] Ready (output device: {args.output_device}, "
                  f"volume={args.volume}, mic will mute during playback)")
        else:
            print("[TTS] Not available (kokoro-onnx model missing). Continuing without speech output.")
            tts = None
        # Register the live TTS instance as the module singleton so the LLM
        # set_volume action (in spot_dispatch._handle_set_volume) can find it.
        if tts is not None:
            import src.voice_control.spot_tts as _spot_tts_mod
            _spot_tts_mod._tts_instance = tts
    else:
        print("\n[TTS] Disabled (--no-tts flag)")

    # Wire the audio feedback beeps to the shared player and apply volume.
    # ``beep.device`` is kept as a no-op attribute for backwards compat —
    # device selection is now owned by the player.
    beep.set_player(_cap.CAPTURE._audio_player)
    beep.set_volume(args.volume)

    # Stage 2F P7: onboard RGB LED state indicator. Lazily binds to the Spot
    # AV service on first state change; no-op in dry-run or if the robot has
    # no AV system. Chimes are the fallback indicator either way.
    if os.environ.get("SPOT_DRY_RUN") == "1":
        C._state_leds = None
    else:
        from src.voice_control.spot_leds import StateLeds
        import src.voice_control.spot_dispatch as _spot_dispatch

        def _led_robot():
            # Return the ALREADY-established Spot session's robot, or None.
            # Never call ensure_spot_session() here: it would connect (and
            # possibly stand) the robot synchronously on the mic-loop thread,
            # freezing voice control for ~30s — and could fire on a noise burst.
            # LEDs stay off until a real command brings the session up, then
            # activate on the next state change.
            try:
                sess = getattr(_spot_dispatch, "_spot_session", None)
                return sess["robot"] if sess else None
            except Exception:
                return None

        C._state_leds = StateLeds(get_robot=_led_robot)

    # Initialize VAD (Silero via sherpa-onnx — replaces webrtcvad 2012-era model)
    vad_cfg = VadModelConfig(
        silero_vad=SileroVadModelConfig(
            model=_cap.SILERO_VAD_PATH,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.2,
            window_size=_cap.VAD_WINDOW_SAMPLES,
        ),
        sample_rate=_cap.SAMPLE_RATE,
        debug=False,
    )
    vad = VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=10)

    # Wake word detector (sherpa-onnx keyword spotter)
    wake_detector = None
    use_wake_word = not args.no_wake_word
    if use_wake_word:
        try:
            wake_detector = _make_wake_detector()
            if wake_detector.is_available():
                print("[WakeWord] keyword spotter ready")
            else:
                print("[WakeWord] Detector not available — falling back to ASR-based")
                wake_detector = None
        except Exception as e:
            print(f"[WakeWord] Import failed: {e} — falling back to ASR-based")
            wake_detector = None
    else:
        print("[WakeWord] Disabled (--no-wake-word) — always listening")

    # Stage 2F C1: always-on safety KWS — stop/freeze/estop fire at frame level
    # in EVERY voice state (own thread, own queue fed before any mic gate),
    # with no wake required. Fail-closed on the robot: if it cannot load,
    # startup refuses (below) unless --allow-no-safety-kws is passed. In
    # dry-run it only warns; then safety needs a transcript (wake first).
    safety_detector = None
    try:
        from src.voice_control.wake.safety_kws import make_safety_detector
        safety_detector = make_safety_detector()
        if not safety_detector.is_available():
            print("[SafetyKWS] UNAVAILABLE — spoken stop/freeze/estop only work via ASR "
                  "after a wake; run scripts/setup_safety_kws.py")
            safety_detector = None
    except Exception as e:
        print(f"[SafetyKWS] import failed: {e}")
        safety_detector = None

    # YOLO models (YOLOv8n, YOLO-World) lazy-load on first use.
    # Pre-loading them at startup starves the audio thread (CPU-bound
    # PyTorch init causes PortAudio init overflow and delays wake word).

    # Probe Spot connectivity BEFORE starting the audio loop.
    #
    # Two paths:
    #   --map present: eager full session init (stand_on_enter=True) so
    #     the map uploads as part of the SAME bring-up. This DOES power
    #     on the robot and requires e-stop to be running already. If it
    #     fails (missing SDK, unreachable, lease conflict), we auto-
    #     degrade to dry-run.
    #   no --map: light TCP probe to SPOT_IP:443 only — does NOT power
    #     on, does NOT acquire lease, does NOT require e-stop. The real
    #     session bring-up happens lazily on first dispatch_intent.
    #     If the probe fails, auto-degrade.
    #
    # Caller can opt out by passing --dry-run explicitly (skips everything).
    # Fail closed BEFORE --map powers on / stands the robot.
    if args.map and not _sl.safety_startup_ok(safety_detector is not None, args.dry_run,
                                              args.allow_no_safety_kws):
        cleanup_spot()
        return
    if not args.dry_run:
        if args.map:
            print(f"\n[Spot] Eager session init with map: {args.map}")
            try:
                from src.voice_control.spot_dispatch import ensure_spot_session
                ensure_spot_session(map_path=args.map)
                print("[Spot] Map loaded; session standing.")
            except ModuleNotFoundError as e:
                print(f"[Spot] SDK not installed ({e}); auto-enabling dry-run.")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True
            except Exception as e:
                print(f"[Spot] Eager session init failed "
                      f"({type(e).__name__}: {e}); auto-enabling dry-run.")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True
        else:
            import socket
            spot_host = _probe_host()
            try:
                print(f"\n[Spot] TCP-probing {spot_host}:443 ...")
                with socket.create_connection((spot_host, 443), timeout=2.0):
                    pass
                print(f"[Spot] {spot_host} reachable; full session deferred to first command.")
            except OSError as e:
                print(f"[Spot] {spot_host} not reachable ({type(e).__name__}: {e})")
                print("[Spot] Auto-enabling dry-run mode (voice pipeline "
                      "only; no robot dispatch, no e-stop required).")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True

    # No-map path: the TCP probe above only connects (no power/stand); the
    # check runs after it so an unreachable robot (auto dry-run) still starts.
    if not args.map and not _sl.safety_startup_ok(safety_detector is not None, args.dry_run,
                                                  args.allow_no_safety_kws):
        cleanup_spot()
        return

    # XVF3800 control (DoA for the speaker lock, LED ring). Optional.
    _cap.CAPTURE._mic = _cap._open_mic_array()
    _ring.RING._anim = _ring._make_ring_animator(_cap.CAPTURE._mic)

    if safety_detector is not None:
        _sl.LISTENER.start(safety_detector)
        print("[SafetyKWS] safety thread running (every state, ignores mute/gate)")

    # Voiceprint model warm-up off the main thread (torch import + ORT/CUDA
    # init take seconds); utterances fail open to direction-only until ready.
    # Wait for it BEFORE the mic stream opens: the torch import holds the GIL
    # for ~2.5 s, which starves the PortAudio callback ("input overflow").
    _wt = warm_up_async(C._voice)
    if _wt is not None:
        _wt.join(timeout=15.0)
    print(f"[Voice] voiceprint gating {'ON' if C._voiceprint_mode else 'OFF'} "
          f"(threshold {C._voiceprint_threshold:.2f}; spotctl voiceprint on|off|threshold <x>)")

    # Open audio stream (stereo for XVF3800, mono fallback, retry on busy)
    try:
        stream = _cap.CAPTURE.open(args.device)
        stream.start()
    except Exception as e:
        print(f"Audio error: {e}")
        print("Run with --list-devices to see available devices")
        _shutdown_runtime()
        return

    mode = f"LLM Brain ({DEFAULT_MODEL})" if brain else "Regex-only (legacy)"
    if not use_wake_word:
        ww_status = "OFF"
    elif wake_detector:
        ww_status = "ON (sherpa-onnx keyword spotter)"
    else:
        ww_status = "ON (ASR-based fallback)"
    print("\n" + "=" * 60)
    print(f"LISTENING — Mode: {mode}")
    print(f"  Wake word: {ww_status}")
    print("  Safety commands (stop/freeze/estop) always instant")
    print("  Everything else goes through the LLM brain")
    print("=" * 60 + "\n")

    beep.ready()

    # Operator decision 3 (2026-10-28): the arm folds into the joints pose and
    # the gripper mouth turns on automatically once Spot is standing + powered
    # (session bring-up, a stand, or the next turn) -- never before. Opt out
    # with --no-arm or tuning auto_arm: false. Skipped in dry-run.
    _auto = auto_arm_wanted(args, _tuning)
    print(f"[client_mic] automatic arm fold + mouth: {'on' if _auto else 'off'}")
    if not args.dry_run:
        try:
            from src.voice_control.spot_dispatch import set_auto_arm
            set_auto_arm(_auto)
        except Exception as e:
            print(f"[client_mic] auto arm setup failed: {e}")

    C.run(args=args, stub=stub, brain=brain, tts=tts, vad=vad,
          wake_detector=wake_detector, use_wake_word=use_wake_word, floor=floor,
          utt_log=utt_log, store_transcript=store_transcript, stream=stream,
          shutdown=_shutdown_runtime)


if __name__ == "__main__":
    main()
