"""Voice gate helpers: DoA speech samples, utterance trimming, the
conversation clock, and the short-utterance test.

Moved from the old client_mic.py (Task 10), unchanged. Pure
functions; no state.
"""
import math


def speech_degs(samples) -> list:
    """Mic-frame degrees of the speech-flagged DoA samples (t, deg, speech).
    Non-finite readings are dropped before lock/accept."""
    out = []
    for _t, deg, speech in samples:
        if speech and deg is not None and math.isfinite(float(deg)):
            out.append(float(deg))
    return out


def last_speech_t(utt) -> float:
    """Timestamp of the last speech-flagged frame (span end for DoA, so the
    end-silence tail isn't counted); last frame if none flagged."""
    for t, _f, _d, s in reversed(utt):
        if s:
            return t
    return utt[-1][0]


def trim_trailing(utt, keep_frames: int):
    """Drop end-silence frames beyond `keep_frames` after the last speech
    frame (the endpointer waits ~1.5 s of silence; ASR doesn't need it)."""
    last = None
    for i in range(len(utt) - 1, -1, -1):
        if utt[i][3]:
            last = i
            break
    if last is None:
        return list(utt)
    return list(utt[:last + 1 + keep_frames])


def conversation_expired(lock, listening_since: float, now: float, timeout_s: float) -> bool:
    """Conversation timeout. Locked: SpeakerLock.expired. Unlocked (no mic /
    no direction readings): wall clock since listening started, so an
    unlocked LISTENING state can't stay open forever in a crowd."""
    if lock.locked:
        return lock.expired(now)
    return (now - listening_since) > timeout_s


def refresh_conversation(lock, listening_since: float, now: float, reached_llm: bool) -> float:
    """Restart the conversation clock (lock + unlocked fallback) only for a
    turn that reached intent/LLM; loud noise, empty or ignored transcripts
    leave it running (final-review M1). Returns the new listening_since."""
    if not reached_llm:
        return listening_since
    lock.touch(now)
    return now


def is_short_utterance(stats, max_speech_s: float) -> bool:
    """Short = little VAD-flagged speech (a one-word command is ~0.6-1.0 s
    including Silero's hangover; background noise utterances run 3.5 s+)."""
    return 0.0 < stats.duration_s <= max_speech_s
