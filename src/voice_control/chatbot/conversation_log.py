"""JSONL per-turn conversation logger with daily rotation.

Each turn writes one JSON line to `logs/conversations/<YYYY-MM-DD>.jsonl`.
Symlinked to /mnt/ssd/spot-logs/conversations/ per Stage 1.5 layout.
"""
import json
import time
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("logs/conversations")


class ConversationLog:
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_today(self) -> Path:
        return self.log_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"

    def log_turn(
        self,
        transcript: str,
        response: str,
        actions: list,
        persona: str,
        tts_backend: str,
        voice_id: str,
        latency_ms: dict,
    ) -> None:
        record = {
            "ts": time.time(),
            "transcript": transcript,
            "response": response,
            "actions": actions,
            "persona": persona,
            "tts_backend": tts_backend,
            "voice_id": voice_id,
            "latency_ms": latency_ms,
        }
        with self._file_for_today().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
