"""Concurrency guards for LLMBrain backend access (Stage 2E.1 crash fix).

Root cause of the "change-voice kills Spot" crash: set_persona spawned an
untracked warm_up thread that fired a SECOND request at the single-slot
llama-server concurrently with the next turn's process() call. These tests
pin the two guards that fix it:

  1. warm_up_async() never stacks threads (rapid persona switches → 1 warm-up).
  2. A shared backend lock serializes process() and warm_up() — they never
     hit the server concurrently.
"""
import threading
import time

import src.voice_control.llm_brain as llm_brain
from src.voice_control.llm_brain import LLMBrain


class _FakeBackend:
    """Records peak concurrency across chat() calls."""

    name = "fake"

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_concurrent = 0
        self.calls = 0

    def chat(self, system, messages, on_token=None, timeout=60.0,
             profile="action", sampling_overrides=None, max_tokens=512):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
        time.sleep(0.05)  # hold the "slot" long enough for a racer to collide
        with self._lock:
            self.active -= 1
        return '{"actions": [], "response": "ok"}'


def _brain(monkeypatch):
    fake = _FakeBackend()
    monkeypatch.setattr(llm_brain, "_BACKEND", fake)
    brain = LLMBrain()
    monkeypatch.setattr(brain, "is_available", lambda: True)
    return brain, fake


def test_warm_up_async_does_not_stack_threads(monkeypatch):
    brain, fake = _brain(monkeypatch)
    # Fire many rapid persona-switch warm-ups in a tight loop.
    for _ in range(8):
        brain.warm_up_async()
    if brain._warm_thread is not None:
        brain._warm_thread.join(timeout=3.0)
    # Only one warm-up may have reached the backend; the rest must be skipped
    # while the first thread is still alive.
    assert fake.calls == 1


def test_process_and_warm_up_never_run_concurrently(monkeypatch):
    brain, fake = _brain(monkeypatch)

    def run_process():
        brain.process("walk forward", {})

    threads = [
        threading.Thread(target=run_process),
        threading.Thread(target=brain.warm_up),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)
    # The single-slot server is never hit by two callers at once.
    assert fake.max_concurrent == 1
