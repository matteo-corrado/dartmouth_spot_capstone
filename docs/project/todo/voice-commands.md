# Voice commands and interaction: next steps

Context: operator request 2026-11-04, on `restructure-modular`. Nothing here is built yet.

Today: `strafe` takes `{"direction": "left"|"right", "distance": <meters>}` (default 0.5). The fast-path regex in `src/spot/features/walk/__init__.py` only matches the "move left 2" pattern; everything else goes to the LLM. The speaker lock is a fixed +/-35 deg window (`lock_window_deg` in `config/tuning.yaml`) with no live on/off switch; `spotctl release` only drops the current lock.

---

## 1. Movement commands

1. **Relative moves with distances, in any phrasing:** "move 2 meters to the left", "step right one meter", "go back half a meter".
   - Extend the fast-path regex: word numbers, "to the left/right", feet to metres, forward/back distances.
   - **Pass:** a replay set of ~20 phrasings each maps to the right action + distance, both via fast path and LLM.
2. **Obstacle-relative positioning:** "stay 1 metre from the wall", "back up until you're 2 m from that", "get closer to the table".
   - Needs per-side obstacle distances from depth (`check_obstacles` is a starting point).
   - New grammar action: update `src/spot/brain/grammar/spot_action.gbnf` and the prompt, or the LLM silently misfires.
   - Clamp to a minimum safe distance; refuse and say why if the target is unsafe.
3. **Spoken confirmation:** "moving one metre left" before the move, and a spoken reason when a move is blocked.

## 2. Voice interaction

1. **Short commands get lost** ("sit", "stop", "left").
   - Record failing cases (`scripts/record_demo_clips.py`), replay them, and find which stage drops them: onset frames, the `short_utt_s` gate, VAD endpointing, the voiceprint threshold, or ASR on one-word clips.
   - **Pass:** >= 95% of recorded one-word commands reach dispatch in a quiet room, >= 85% with babble.
2. **Direction lock controls:**
   - Turn the lock off and on, by voice and from `spotctl` / the web panel.
   - Widen, narrow or re-point the accepted direction ("listen to my left", "listen all around").
   - Show the current mode on the LED ring and in `spotctl status`.
   - Safety: stop/freeze must still be heard from every direction, lock on or off.
