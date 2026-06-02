# tests/voice_control/test_drain_queue.py
"""Unit tests for the audio-queue drain partition (Stage 2F P6)."""
from src.voice_control.client_mic import partition_drain


def test_keep_tail_keeps_last_n_and_reports_dropped():
    frames = list(range(100))
    kept, dropped = partition_drain(frames, keep_count=33)
    assert kept == list(range(67, 100))   # last 33
    assert dropped == 67


def test_keep_tail_when_fewer_than_keep_count_keeps_all():
    frames = [1, 2, 3]
    kept, dropped = partition_drain(frames, keep_count=33)
    assert kept == [1, 2, 3]
    assert dropped == 0


def test_full_drain_keeps_nothing():
    frames = list(range(50))
    kept, dropped = partition_drain(frames, keep_count=0)
    assert kept == []
    assert dropped == 50


def test_empty_input():
    assert partition_drain([], keep_count=33) == ([], 0)
