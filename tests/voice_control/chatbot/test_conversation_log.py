import json
import time
from pathlib import Path
from src.voice_control.chatbot.conversation_log import ConversationLog


def test_log_turn_writes_jsonl_line(tmp_path):
    log = ConversationLog(log_dir=tmp_path)
    log.log_turn(
        transcript="hello",
        response="hi there",
        actions=[],
        persona="tour_guide",
        tts_backend="kokoro",
        voice_id="af_sarah",
        latency_ms={"asr": 200, "llm": 1500, "tts_first_chunk": 300},
    )
    # Daily file expected
    today = time.strftime("%Y-%m-%d")
    log_file = tmp_path / f"{today}.jsonl"
    assert log_file.exists()
    line = log_file.read_text().strip()
    record = json.loads(line)
    assert record["transcript"] == "hello"
    assert record["response"] == "hi there"
    assert record["persona"] == "tour_guide"
    assert record["tts_backend"] == "kokoro"
    assert record["voice_id"] == "af_sarah"
    assert record["latency_ms"]["llm"] == 1500
    assert "ts" in record  # epoch timestamp present


def test_log_appends_multiple_turns(tmp_path):
    log = ConversationLog(log_dir=tmp_path)
    for i in range(3):
        log.log_turn(
            transcript=f"turn {i}", response=f"resp {i}",
            actions=[], persona="pirate", tts_backend="elevenlabs",
            voice_id="Xq2dbIWNPChFB77imiDe", latency_ms={},
        )
    today = time.strftime("%Y-%m-%d")
    log_file = tmp_path / f"{today}.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["transcript"] == f"turn {i}"
