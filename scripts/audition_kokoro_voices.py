"""Audition top-tier Kokoro English voices. Plays one canonical line per
voice, prompts user for grade (a/b/c/d) + free-text tags, writes a local
catalog to docs/project/kokoro-voice-catalog-<date>.md so future persona
voice swaps are data-driven (not name-guessing).

Scope: all C+ and above (per upstream VOICES.md) plus the 4 currently-used
sub-C+ voices (af_nova, am_onyx, bm_george, bm_lewis) so we can confirm
they hold up subjectively. ~14 voices total.

Same sample line for every voice — only variable is the voice itself.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 24000

SAMPLE_TEXT = (
    "Hello, I am a candidate voice. Today is May twenty-fourth, twenty twenty-six. "
    "Tell me, how do I sound?"
)

# (slug, upstream_grade, currently_used_by_persona_or_None)
# Source: hexgrad/Kokoro-82M VOICES.md (fetched 2026-05-24).
VOICES = [
    # American Female — C+ and above
    ("af_heart",   "A",   None),
    ("af_bella",   "A-",  None),
    ("af_nicole",  "B-",  None),       # 🎧 headphones marker = studio quality
    ("af_aoede",   "C+",  None),
    ("af_kore",    "C+",  None),
    ("af_sarah",   "C+",  "tour_guide / news_anchor"),
    # American Female — sub-C+ but currently used
    ("af_nova",    "C",   "gen_z"),
    # American Male — C+ and above (cap of am_ tier)
    ("am_fenrir",  "C+",  "pirate / surfer / narrator"),
    ("am_michael", "C+",  None),
    ("am_puck",    "C+",  None),
    # American Male — sub-C+ but currently used
    ("am_onyx",    "D",   "snarky"),
    # British Female — C+ and above
    ("bf_emma",    "B-",  None),
    # British Male — currently used (none reach C+)
    ("bm_george",  "C",   "butler"),
    ("bm_lewis",   "D+",  "shakespeare"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print catalog rows, skip audio")
    ap.add_argument("--only", default=None,
                    help="Comma-separated voice slugs to audition (default: all)")
    args = ap.parse_args()

    candidates = VOICES
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        candidates = [v for v in VOICES if v[0] in wanted]
        missing = wanted - {v[0] for v in candidates}
        if missing:
            print(f"WARN: requested slugs not in catalog: {sorted(missing)}")

    print(f"Auditioning {len(candidates)} Kokoro voices")
    print(f"Sample text: {SAMPLE_TEXT!r}\n")
    for slug, grade, used in candidates:
        used_str = used or "-"
        print(f"  [{slug:12s}] upstream={grade:3s}  in_use_by={used_str}")

    if args.dry_run:
        print("\n--dry-run: catalog printed, no audio.")
        return 0

    from src.voice_control.tts.kokoro import KokoroBackend
    print("\nWarming KokoroBackend (loads ~85MB ONNX)...")
    backend = KokoroBackend()

    # catalog[slug] = {"grade": "a"|"b"|"c"|"d"|"s"|"ERR:<type>", "tags": "<freeform>"}
    catalog: dict[str, dict[str, str]] = {}

    print("\nPer voice: grade a/b/c/d (your subjective rank), s=skip, "
          "r=replay, q=quit. After grade, you'll be prompted for tags "
          "(adjectives — e.g. 'deep,gravelly,older' — blank = none).\n")
    for slug, upstream_grade, used in candidates:
        used_str = used or "-"
        print(f"\n=== {slug} ===  upstream={upstream_grade}  in_use_by={used_str}")
        while True:
            print(f"  rendering...", flush=True)
            try:
                pcm = backend.synthesize(SAMPLE_TEXT, slug)
                audio = np.frombuffer(pcm, dtype="<i2")
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"    ERROR: {e}")
                catalog[slug] = {"grade": f"ERR:{type(e).__name__}", "tags": ""}
                break
            v = input(f"    [{slug}] grade (a/b/c/d/s/r/q): ").strip().lower()
            if v == "r":
                continue
            if v == "q":
                print("Early quit — writing partial catalog.")
                write_catalog(catalog, candidates)
                return 0
            if v not in ("a", "b", "c", "d", "s"):
                v = "s"
            tags = input(f"    [{slug}] tags (comma-sep, blank=none): ").strip()
            catalog[slug] = {"grade": v, "tags": tags}
            break

    write_catalog(catalog, candidates)
    return 0


def write_catalog(catalog: dict[str, dict[str, str]],
                  candidates: list[tuple[str, str, str | None]]) -> None:
    date = dt.date.today().isoformat()
    out = ROOT / "docs" / "project" / f"kokoro-voice-catalog-{date}.md"
    lines = [
        f"# Kokoro Voice Catalog — {date}",
        "",
        f"Sample text: `{SAMPLE_TEXT}`",
        "",
        "User grades subjective on this single canonical line. Upstream grade is "
        "from hexgrad/Kokoro-82M VOICES.md (training quality + duration).",
        "",
        "| slug | upstream | user | tags | in_use_by |",
        "|---|---|---|---|---|",
    ]
    for slug, upstream_grade, used in candidates:
        row = catalog.get(slug, {})
        user_grade = row.get("grade", "-")
        tags = row.get("tags", "").replace("|", "\\|")
        used_str = used or "-"
        lines.append(f"| {slug} | {upstream_grade} | {user_grade} | {tags} | {used_str} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
