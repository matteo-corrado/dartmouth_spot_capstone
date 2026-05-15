#!/usr/bin/env bash
# Health probe for the Spot voice-control stack on the Jetson.
# Read-only: never starts/stops/restarts anything.
set -u

ok()    { printf '  \033[32mOK\033[0m       %s\n' "$1"; }
warn()  { printf '  \033[33mDEGRADED\033[0m %s\n' "$1"; }
down()  { printf '  \033[31mDOWN\033[0m     %s\n' "$1"; }
header(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

header "Processes"
for proc_pattern in "estop_run.py" "run_voice_control.py" "web_panel.py"; do
  if pgrep -af "$proc_pattern" >/dev/null 2>&1; then
    ok "$proc_pattern"
  else
    down "$proc_pattern (not running)"
  fi
done

header "Ollama"
if command -v ollama >/dev/null 2>&1; then
  if curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "ollama daemon (localhost:11434)"
    tags_json=$(curl -sf --max-time 2 http://localhost:11434/api/tags 2>/dev/null || echo '{}')
    for model in "qwen2.5:7b" "qwen2.5vl:7b"; do
      if printf '%s' "$tags_json" | grep -q "\"$model\""; then
        ok "model pulled: $model"
      else
        warn "model not pulled: $model"
      fi
    done
    # Check currently-loaded (in VRAM) models
    ps_out=$(curl -sf --max-time 2 http://localhost:11434/api/ps 2>/dev/null || echo '{}')
    loaded=$(printf '%s' "$ps_out" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
    if [[ -n "$loaded" ]]; then
      ok "models currently in VRAM: $loaded"
    else
      warn "no models in VRAM (first call will cold-start)"
    fi
  else
    down "ollama daemon not responding on :11434"
  fi
else
  down "ollama not installed"
fi

header "NVIDIA Riva (ASR)"
if command -v docker >/dev/null 2>&1; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qi riva; then
    name=$(docker ps --format '{{.Names}}' | grep -i riva | head -1)
    ok "riva container: $name"
  else
    down "no riva container running (try: bash scripts/setup_riva.sh)"
  fi
else
  warn "docker not on PATH"
fi

header "Audio devices"
if command -v arecord >/dev/null 2>&1; then
  if arecord -l 2>/dev/null | grep -qi "respeaker\|xvf3800\|XVF"; then
    ok "input: reSpeaker XVF3800 detected"
  else
    warn "reSpeaker XVF3800 not detected in arecord -l"
  fi
else
  warn "alsa-utils not installed (arecord missing)"
fi
if command -v aplay >/dev/null 2>&1; then
  aplay -l >/dev/null 2>&1 && ok "audio output devices present" || warn "no audio output devices"
fi

header "GPU / VRAM"
if command -v nvidia-smi >/dev/null 2>&1; then
  mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$mem" ]]; then
    ok "nvidia-smi: ${mem} MiB used / total"
  else
    warn "nvidia-smi returned no data"
  fi
else
  warn "nvidia-smi not on PATH (tegrastats may be the better tool here)"
fi

header "Web panel"
if curl -sf --max-time 2 http://localhost:8080/ >/dev/null 2>&1 || \
   curl -sf --max-time 2 http://localhost:5000/ >/dev/null 2>&1 || \
   curl -sf --max-time 2 http://localhost:8000/ >/dev/null 2>&1; then
  ok "web panel reachable on a local port"
else
  warn "web panel not reachable on :8080/:5000/:8000"
fi

header "Spot session"
spot_ip="${SPOT_IP:-}"
if [[ -z "$spot_ip" && -f .env ]]; then
  spot_ip=$(grep -E '^SPOT_IP=' .env 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")
fi
if [[ -n "$spot_ip" ]]; then
  if ping -c 1 -W 1 "$spot_ip" >/dev/null 2>&1; then
    ok "Spot reachable at $spot_ip"
  else
    down "Spot NOT reachable at $spot_ip"
  fi
else
  warn "SPOT_IP not set (skipping ping)"
fi

printf '\n'
