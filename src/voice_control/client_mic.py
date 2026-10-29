"""Voice control client for Spot (script shim).

Split in Task 10 into src/spot/{audio/capture, mic/ring_control,
safety/listener, operator/control, stt/client, conversation/{gate,
utterance_log, filler, loop}, app/main}. ``python src/voice_control/client_mic.py``
(launched by scripts/run_voice_control.py) still runs the pipeline:
src.spot.app.main.main().

This module re-exports the moved functions and constants for tests and
scripts. It holds NO state: the old module globals live on owner objects
(CAPTURE, RING, LISTENER, CONVERSATION) or in their new home module; reads
of the old names are forwarded by __getattr__ below. Writes (and test
patches) must go to the owner / home module, never to this shim.
"""
import sys
import pathlib

# Add parent and project root to path for local imports — must run BEFORE
# any `from src.*` import below, otherwise launching client_mic.py as a
# script (e.g. via run_voice_control.py subprocess) fails with
# ModuleNotFoundError because sys.path[0] is src/voice_control/, not the
# project root.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Loads .env and installs the shutdown signal handlers at import, as before.
from src.spot.app.main import (  # noqa: E402,F401
    main,
    _shutdown_signal_handler,
    cleanup_spot,
    _probe_host,
    _make_wake_detector,
    auto_arm_wanted,
    _shutdown_runtime,
)
import src.spot.app.main as _app  # noqa: E402

# Module objects / leaf names client_mic exposed (tests patch attributes on
# these objects, which are shared, e.g. cm.beep.command_ok, cm.threading.Timer).
import os  # noqa: E402,F401
import re  # noqa: E402,F401
import math  # noqa: E402,F401
import time  # noqa: E402,F401
import queue  # noqa: E402,F401
import random  # noqa: E402,F401
import signal  # noqa: E402,F401
import threading  # noqa: E402,F401
import numpy as np  # noqa: E402,F401
from src.spot.audio.chimes import beep  # noqa: E402,F401
from src.spot.brain.intent import (  # noqa: E402,F401
    check_safety_command, is_filler_only, short_utterance_ok, parse_intent)
from src.spot.core.voice_state import VoiceState  # noqa: E402,F401
from src.spot.core.tuning import get_tuning  # noqa: E402,F401
from src.spot.mic.state_feedback import chime_for, ring_for as _ring_for  # noqa: E402,F401
from src.spot.wake.phrases import WAKE_PHRASES, strip_wake_phrase  # noqa: E402,F401

_tuning = get_tuning()  # the shared cached Tuning (never rebound)

import src.spot.audio.capture as _cap  # noqa: E402
from src.spot.audio.capture import (  # noqa: E402,F401
    MIC_DEVICE_NAME,
    SPEAKER_DEVICE_NAME,
    _find_device_by_name,
    SAMPLE_RATE,
    FRAME_MS,
    FRAME_S,
    FRAME_SAMPLES,
    BYTES_PER_FRAME,
    SILERO_VAD_PATH,
    VAD_WINDOW_SAMPLES,
    NOISE_CALIBRATION_SECONDS,
    _frames,
    dbfs_from_rms,
    AUDIO_QUEUE_MAX,
    AudioCapture,
    CAPTURE,
    _mic_usable,
    _check_mic_alive,
    _mic_speech_degs,
    _open_mic_array,
    _new_endpointer,
    _speech_busy,
    _chime_busy,
    ChimeMask,
    audio_callback,
    _is_format_error,
    _log_pcm_holders,
    _open_input_stream_with_retry,
    calibrate_noise_floor,
    pcm16_to_float32,
    float32_to_pcm16,
    partition_drain,
    _drain_audio_queue,
)
import src.spot.mic.ring_control as _ring  # noqa: E402
from src.spot.mic.ring_control import (  # noqa: E402,F401
    REJECT_FLASH_S,
    REJECT_COLOR,
    LED_SOLID,
    RingThrottle,
    Ring,
    RING,
    _make_ring_animator,
    _current_ring_spec,
    _show_ring,
    _update_ring,
    _flush_ring,
    _end_reject_flash,
    _signal_reject,
)
import src.spot.safety.listener as _sl  # noqa: E402
from src.spot.safety.listener import (  # noqa: E402,F401
    SAFETY_QUEUE_MAX,
    SAFETY_REFRACTORY_S,
    SAFETY_ERROR_WARN_N,
    _SAFETY_LABELS,
    SAFETY_DROP_LOG_S,
    RESCUE_ASR_TIMEOUT_S,
    safety_kws_loop,
    _safety_intent_for,
    SafetyListener,
    LISTENER,
    _on_safety_kw,
    _start_hush,
    _mark_safety,
    _hush_player,
    _dispatch_safety,
    safety_startup_ok,
    _start_safety_thread,
    _stop_safety_thread,
    _put_drop_oldest,
    _note_safety_drop,
    _run_safety,
    rescue_safety,
    _hush,
)
from src.spot.operator.control import (  # noqa: E402,F401
    _MOTION_COMMANDS,
    VOICEPRINT_THRESHOLD_MAX,
    _status_snapshot,
    _on_immediate,
    apply_voiceprint_arg,
    _toggle,
    _start_control_server,
)
from src.spot.stt.client import send_to_asr  # noqa: E402,F401
from src.spot.conversation.gate import (  # noqa: E402,F401
    speech_degs,
    last_speech_t,
    trim_trailing,
    conversation_expired,
    refresh_conversation,
    is_short_utterance,
)
from src.spot.conversation.utterance_log import (  # noqa: E402,F401
    UTTERANCE_LOG_DIR,
    _body,
    _rnd,
    build_utterance_record,
    build_idle_record,
    UtteranceLog,
)
from src.spot.conversation.filler import (  # noqa: E402,F401
    THINKING_FILLER_LINE,
    _PERSONA_ACTION_RE,
    _RESPONSE_KEY_RE,
    ThinkingFiller,
    _TokenWatch,
)
import src.spot.conversation.loop as _loop  # noqa: E402
from src.spot.conversation.loop import (  # noqa: E402,F401
    SILENCE_CROSSOVER,
    PREROLL_FRAMES,
    TRAILING_KEEP_FRAMES,
    NAV_SPEECH_DB_BOOST,
    LOCK_ACCEPT_MAX_SAMPLES,
    WAKE_AUDIO_S,
    CHAIN_MOVEMENT_WAIT_S,
    CHAIN_POSTURE_WAIT_S,
    _DESCRIBE_ACKS,
    _PERSONA_INTENTS,
    _MOTION_FREE_INTENTS,
    MIN_SPEECH_DURATION,
    _STATE_ONLY_INTENTS,
    execute_on_spot,
    get_spot_state,
    _is_robot_moving,
    _wait_for_nav_complete,
    _enter_state,
    _heard,
    _safety_gen_now,
    _auto_arm_at_turn,
    _gaze_loop,
    _gaze_at_speaker,
    _resume_talk_after_halt,
    _cue_speaking,
    _say,
    _brain_ready,
    _resolve_voice,
    Conversation,
    CONVERSATION,
    process_utterance,
)

# Old module-global name -> (owner, attribute). Reads only (PEP 562): the
# owner is looked up at call time, so a read always sees the live value.
_FORWARDS = {
    # AudioCapture
    "audio_queue": (_cap, "CAPTURE", "audio_queue"),
    "_audio_player": (_cap, "CAPTURE", "_audio_player"),
    "_mic_gated": (_cap, "CAPTURE", "_mic_gated"),
    "_mic_reopen_at": (_cap, "CAPTURE", "_mic_reopen_at"),
    "_player_last_busy_at": (_cap, "CAPTURE", "_player_last_busy_at"),
    "_mic": (_cap, "CAPTURE", "_mic"),
    "_mic_dead_logged": (_cap, "CAPTURE", "_mic_dead_logged"),
    "MIC_CHANNEL": (_cap, None, "MIC_CHANNEL"),
    # Ring
    "_ring_state": (_ring, "RING", "_ring_state"),
    "_ring_last_write": (_ring, "RING", "_ring_last_write"),
    "_ring_pending": (_ring, "RING", "_ring_pending"),
    "_reject_until": (_ring, "RING", "_reject_until"),
    "_ring_throttle": (_ring, "RING", "_ring_throttle"),
    "_anim": (_ring, "RING", "_anim"),
    # SafetyListener
    "_safety_q": (_sl, "LISTENER", "_q"),
    "_safety_stop": (_sl, "LISTENER", "_stop"),
    "_safety_active": (_sl, "LISTENER", "_active"),
    "_safety_thread": (_sl, "LISTENER", "_thread"),
    "_safety_health": (_sl, "LISTENER", "_health"),
    "_last_safety_at": (_sl, "LISTENER", "_last_safety_at"),
    "_safety_drops": (_sl, "LISTENER", "_drops"),
    # Conversation
    "_voice": (_loop, "CONVERSATION", "_voice"),
    "_status": (_loop, "CONVERSATION", "_status"),
    "_state_leds": (_loop, "CONVERSATION", "_state_leds"),
    "_lock": (_loop, "CONVERSATION", "_lock"),
    "_operator_muted": (_loop, "CONVERSATION", "_operator_muted"),
    "_crowd_mode": (_loop, "CONVERSATION", "_crowd_mode"),
    "_voiceprint_mode": (_loop, "CONVERSATION", "_voiceprint_mode"),
    "_voiceprint_threshold": (_loop, "CONVERSATION", "_voiceprint_threshold"),
    "_abort_chain": (_loop, "CONVERSATION", "_abort_chain"),
    "_brain_ok": (_loop, "CONVERSATION", "_brain_ok"),
    "_store_transcript": (_loop, "CONVERSATION", "_store_transcript"),
    "_speaking_hook": (_loop, "CONVERSATION", "_speaking_hook"),
    # loop module state (gaze worker; moves to features/gaze in Wave 4)
    "_gaze_q": (_loop, None, "_gaze_q"),
    "_gaze_worker": (_loop, None, "_gaze_worker"),
    # app.main module state
    "_shutdown_requested": (_app, None, "_shutdown_requested"),
}


def __getattr__(name):
    try:
        mod, owner, attr = _FORWARDS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    obj = mod if owner is None else getattr(mod, owner)  # owner looked up at call time
    return getattr(obj, attr)


if __name__ == "__main__":
    main()
