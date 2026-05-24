"""Conversational session state that persists across turns.

Layered with the existing `get_robot_state_dict()` live SDK snapshot
(spot_dispatch.py:80). SessionState carries *across-turn* memory (persona,
turn index, last action, last comment timestamp). Both layers merge into
the single bullet-list block in the LLM system prompt.

Field growth across stage 2E slices:
- 2E.1: current_persona, turn_index, last_action_taken, last_comment_ts
- 2E.2 (future): arm_deployed, gaze_target, person_present, person_count
- 2E.3 (future): last_yolo_classes, last_vlm_caption, last_uttered_phrase
"""
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionState:
    current_persona: str = "tour_guide"
    turn_index: int = 0
    last_action_taken: Optional[str] = None
    last_comment_ts: float = 0.0

    def as_dict(self) -> dict:
        """Render as flat dict for LLM bullet-list block.

        Matches existing get_robot_state_dict() formatting so the brain
        sees a single unified state list.
        """
        return {
            "current_persona": self.current_persona,
            "turn_index": self.turn_index,
            "last_action_taken": self.last_action_taken or "none",
            "seconds_since_last_comment": int(time.time() - self.last_comment_ts),
        }
