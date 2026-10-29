"""Thinking filler (B12) and the brain token watcher around the streaming
TTS chunker (B6 persona hold-back, filler cancel, speaking cue).

Moved from the old client_mic.py (Task 10), unchanged.
"""
import re
import threading

import src.spot.conversation.loop as _loop

THINKING_FILLER_LINE = "Hmm, let me think."
_PERSONA_ACTION_RE = re.compile(r'"action"\s*:\s*"(?:set_persona|add_persona)"')
_RESPONSE_KEY_RE = re.compile(r'"response"\s*:\s*"')


class ThinkingFiller:
    """B12: speak one filler line if the brain is still silent after
    `delay_s`. `cancel()` is idempotent; the line never plays over speech."""

    def __init__(self, speak, delay_s, is_busy, timer_factory=None):
        self._speak, self._is_busy = speak, is_busy
        self._cancelled = threading.Event()
        make = timer_factory or threading.Timer
        self._timer = make(delay_s, self._fire)
        self._timer.daemon = True

    def start(self):
        self._timer.start()

    def cancel(self):
        self._cancelled.set()
        try:
            self._timer.cancel()
        except Exception:
            pass

    def _fire(self):
        if self._cancelled.is_set() or self._is_busy():
            return
        self._cancelled.set()
        try:
            self._speak()
        except Exception as e:
            print(f"[Filler] failed: {e}")


class _TokenWatch:
    """Brain token callback wrapper around the streaming TTSChunker.

    GBNF emits actions before "response", so when the response key arrives
    we know the action list: a persona switch suppresses streamed speech
    (B6 — spoken after the switch in the new voice), and the thinking filler
    is cancelled as soon as the reply text starts (B12)."""

    def __init__(self, chunker, filler=None):
        self._chunker, self._filler = chunker, filler
        self._head = []
        self._seen_response = False

    def __call__(self, delta: str) -> None:
        if _loop.CONVERSATION._abort_chain.is_set():  # hush/stop mid-reply: say nothing more
            self._chunker.suppressed = True
            if self._filler is not None:
                self._filler.cancel()
        if not self._seen_response:
            self._head.append(delta)
            head = "".join(self._head)
            if _RESPONSE_KEY_RE.search(head):
                self._seen_response = True
                if _PERSONA_ACTION_RE.search(head):
                    self._chunker.suppressed = True
                if self._filler is not None:
                    self._filler.cancel()
                if not self._chunker.suppressed and not _loop.CONVERSATION._abort_chain.is_set():
                    # The chunker may still suppress a describe reply on this
                    # very delta; it checks the same key, so ask it first.
                    self._chunker.accept(delta)
                    if not self._chunker.suppressed:
                        _loop._cue_speaking()  # RESPONDING + speak tick before the 1st sentence
                    return
        self._chunker.accept(delta)
