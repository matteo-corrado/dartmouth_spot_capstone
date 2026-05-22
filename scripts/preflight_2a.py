#!/usr/bin/env python3
"""Stage 2A pre-flight audit.

Captures baseline VRAM headroom + current Ollama-path latency so the
post-swap mic-verify can quantify improvement.

Outputs:
  /tmp/stage2a_preflight_<ts>.json   — structured snapshot
  /tmp/stage2a_preflight_<ts>.log    — human-readable log
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("SPOT_OLLAMA_MODEL", "gemma4:e4b")
N_WARMUP = 1
N_TRIALS = 5

PROMPTS = [
    "stand up",
    "walk forward two meters",
    "what do you see",
    "tell me about Dartmouth Hall",
    "battery status",
]


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def tegrastats_snapshot() -> dict:
    try:
        line = subprocess.check_output(
            ["tegrastats", "--interval", "200"], text=True, timeout=1
        ).strip().split("\n")[0]
    except subprocess.TimeoutExpired as e:
        line = (e.output or "").decode("utf-8", "ignore").strip().split("\n")[0]
    return {"raw": line}


def df_ssd() -> dict:
    out = run(["df", "-B1", "/mnt/ssd"])
    parts = out.splitlines()[1].split()
    return {"used_bytes": int(parts[2]), "avail_bytes": int(parts[3])}


def ollama_chat(prompt: str, fmt_json: bool) -> tuple[str, float]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if fmt_json:
        payload["format"] = "json"
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json()["message"]["content"], elapsed


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(f"/tmp/stage2a_preflight_{ts}.json")
    out_log = Path(f"/tmp/stage2a_preflight_{ts}.log")
    snap: dict = {"timestamp": ts}

    print("=== disk ===")
    snap["disk"] = df_ssd()
    print(f"  /mnt/ssd avail: {snap['disk']['avail_bytes'] / 1e9:.2f} GB")

    print("=== tegrastats ===")
    snap["tegrastats_idle"] = tegrastats_snapshot()
    print(f"  {snap['tegrastats_idle']['raw'][:120]}")

    print("=== ollama warmup ===")
    for _ in range(N_WARMUP):
        ollama_chat("hi", fmt_json=False)

    print("=== ollama freeform latency ===")
    freeform = []
    for p in PROMPTS:
        durations = []
        for _ in range(N_TRIALS):
            _, dt = ollama_chat(p, fmt_json=False)
            durations.append(dt)
        durations.sort()
        median = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
        freeform.append({"prompt": p, "median_s": median, "p95_s": p95})
        print(f"  freeform '{p[:30]:30s}' median={median:.2f}s p95={p95:.2f}s")
    snap["ollama_freeform"] = freeform

    print("=== ollama format:json latency ===")
    json_latencies = []
    for p in PROMPTS[:3]:  # action-shaped prompts only
        durations = []
        for _ in range(N_TRIALS):
            _, dt = ollama_chat(p, fmt_json=True)
            durations.append(dt)
        durations.sort()
        median = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
        json_latencies.append({"prompt": p, "median_s": median, "p95_s": p95})
        print(f"  json     '{p[:30]:30s}' median={median:.2f}s p95={p95:.2f}s")
    snap["ollama_format_json"] = json_latencies

    print("=== tegrastats under load ===")
    snap["tegrastats_loaded"] = tegrastats_snapshot()
    print(f"  {snap['tegrastats_loaded']['raw'][:120]}")

    out_json.write_text(json.dumps(snap, indent=2))
    out_log.write_text(
        f"Stage 2A pre-flight {ts}\n"
        f"disk avail: {snap['disk']['avail_bytes'] / 1e9:.2f} GB\n"
        f"freeform medians: {[(r['prompt'][:20], round(r['median_s'], 2)) for r in freeform]}\n"
        f"format:json medians: {[(r['prompt'][:20], round(r['median_s'], 2)) for r in json_latencies]}\n"
    )
    print(f"\nsaved: {out_json}")
    print(f"saved: {out_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
