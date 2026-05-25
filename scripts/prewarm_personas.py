"""Pre-warm 8 stock personas at install: search ElevenLabs Library, claim, append to yaml.
Run ONCE at setup (or whenever stock catalog is refreshed). Idempotent: skips
entries already in personas.yaml. Result: 'be a cowboy' hits in-memory registry
(zero API calls per-turn)."""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice_control.chatbot.voice_discovery import find_and_claim  # noqa: E402

STOCK = [
    ("cowboy",         "deep gravelly American male cowboy"),
    ("robot",          "monotone synthetic robotic male"),
    ("wizard",         "deep wise older British male wizard"),
    ("surfer",         "laid-back young American male surfer"),
    ("drill_sergeant", "loud commanding American male military drill sergeant"),
    ("narrator",       "warm authoritative middle-aged male documentary narrator"),
    ("scientist",      "thoughtful precise older male scientist"),
    ("news_anchor",    "professional neutral American female news anchor"),
]

KOKORO_BY_GENDER = {"male": "am_fenrir", "female": "af_sarah"}


def main():
    yaml_path = ROOT / "config" / "personas.yaml"
    data = yaml.safe_load(yaml_path.read_text()) or {}
    added_count = 0

    for name, description in STOCK:
        if name in data:
            print(f"[prewarm] {name}: already in yaml, skipping")
            continue
        print(f"[prewarm] {name}: searching ElevenLabs Library for '{description}'...")
        voice_id = find_and_claim(description, name)
        if not voice_id:
            print(f"[prewarm] {name}: NO MATCH — skipping")
            continue
        desc_words = set(re.findall(r"\b\w+\b", description.lower()))
        gender = "male" if desc_words & {"male", "man", "guy", "boy", "men"} else "female"
        kokoro_slug = KOKORO_BY_GENDER[gender]
        data[name] = {
            "prompt_prefix": (
                f"You are Spot, but in {name} mode. Speak like {description}. "
                "Keep responses short — one or two sentences."
            ),
            "voices": {"kokoro_v1": kokoro_slug, "elevenlabs": voice_id},
            "ack_template": f"One moment as I become a {name}...",
        }
        added_count += 1
        print(f"[prewarm] {name}: added (voice={voice_id})")

    if added_count:
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        print(f"[prewarm] wrote {added_count} new personas to {yaml_path}")
    else:
        print("[prewarm] no new personas added (all already present or all failed)")


if __name__ == "__main__":
    main()
