# Stage 1 Rollback Runbook

This runbook restores the Jetson + repo to its pre-Stage-1 state.

## What "stable" means (pre-Stage-1 inventory, 2026-05-15)

- **Branch / tag:** `tour_guide_upgrade_matteo` at tag `pre-stage1-stable`.
- **Disk:** 53 GB used / ~787 MB free / 99% on `/dev/mmcblk0p1`.
- **Riva Docker image:** `nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64` resident at `/var/lib/docker` (24.04 GB).
- **Riva quickstart dir:** `~/riva_quickstart_arm64_v2.17.0/` (2.1 GB).
- **Ollama:** version 0.16.1 with `qwen2.5:7b` (4.7 GB) + `qwen2.5vl:7b` (6.0 GB) resident.
- **VS Code CLI versions:** 4 directories under `~/.vscode-server/cli/servers/`. Most-recent per `lru.json`: `Stable-0958016b2af9f09bb4257e0df4a95e2f90590f9f`.

## Disk-pressure guard (run before any rollback that re-acquires artifacts)

```
df -h /
```

If free space is < 30 GB AND the rollback step pulls Riva back, abort and `ollama rm gemma4:e4b` first to free 9.6 GB.

## Layer 1 — Code rollback (cheap, ~1 minute)

Use this if a Stage 1 code commit broke something or you want to reset the branch.

Inspection-only (non-destructive):
```
git checkout pre-stage1-stable
```
This goes into detached-HEAD mode. To return: `git checkout tour_guide_upgrade_matteo`.

Destructive (only with explicit approval):
```
git checkout tour_guide_upgrade_matteo
git reset --hard pre-stage1-stable
git push origin tour_guide_upgrade_matteo --force-with-lease
```

Verification:
```
git rev-parse HEAD
```
Expected: matches `git rev-parse pre-stage1-stable`.

## Layer 2 — Brain model rollback (cheap, ~1 minute)

Use this if gemma4:e4b regresses on action selection or VLM quality.

Edit `src/voice_control/llm_brain.py:28-29`:
```python
DEFAULT_MODEL = "qwen2.5:7b"
VLM_MODEL = "qwen2.5vl:7b"
```

Restart `run_voice_control.py`. The qwen pair takes over instantly (still on disk per Stage 1 design).

Verification:
```
grep -n 'DEFAULT_MODEL\|VLM_MODEL' src/voice_control/llm_brain.py
ollama list   # should still show qwen2.5:7b and qwen2.5vl:7b
```

To free the 9.6 GB the dead gemma4 occupies:
```
ollama rm gemma4:e4b
```

## Layer 3 — Ollama binary downgrade (cheap, ~2 minutes)

Use this if the Ollama 0.20.x upgrade breaks the daemon.

```
sudo systemctl stop ollama
sudo cp ~/ollama-0.16.1.backup /usr/local/bin/ollama
sudo systemctl start ollama
ollama --version
```

Expected: `ollama version is 0.16.1`.

Models on disk (under `/usr/share/ollama/.ollama/models/`) survive the binary swap.

## Layer 4 — Riva re-acquisition (slow, 30-60 minutes, requires NGC auth)

Use only if you genuinely need Riva back (Parakeet broken AND no other STT works).

```
# 1. Pull the Docker image (~24 GB download)
docker pull nvcr.io/nvidia/riva/riva-speech:2.17.0-l4t-aarch64

# 2. Re-download the quickstart tarball (URL captured at runbook-write time)
cd ~ && wget "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64" -O riva_quickstart_arm64_v2.17.0.tar.gz
<!-- TODO: Replace the URL above with the exact 2.17.0 tarball download URL from the NGC catalog; the landing-page URL is a placeholder — the real wget target is a direct .tar.gz link shown on the catalog page. -->
tar -xzf riva_quickstart_arm64_v2.17.0.tar.gz

# 3. Re-init the model_repository (this is bind-mounted into the container)
cd ~/riva_quickstart_arm64_v2.17.0
bash riva_init.sh

# 4. Start the container
bash riva_start.sh
```

Verification:
```
docker images | grep riva-speech   # 24 GB
docker ps | grep riva-speech       # running
nc -zv localhost 50051             # accepting connections
```

NGC authentication: if `docker pull` fails with auth error, run `ngc config set` to configure your NGC API key (https://ngc.nvidia.com/setup).

## Layer 5 — VS Code CLI restoration (no manual action)

Old VS Code CLI versions under `~/.vscode-server/cli/servers/Stable-*/` re-download automatically the next time you connect from a new VS Code remote session that requests an older version. No manual step required.

## Decision tree

| Symptom | Layer to use |
|---|---|
| gemma4 says wrong things / picks wrong action | Layer 2 (brain rollback) |
| Stage 1 code commit broke imports or behavior | Layer 1 (code rollback) |
| Ollama daemon crashes / upgrade misbehaves | Layer 3 (binary downgrade) |
| Need Riva back | Layer 4 (slow, last resort) |
| Wrong VS Code version on next connect | Layer 5 (do nothing) |

## Stage 2 rollback (placeholder — to be written as part of Stage 2)

Stage 2 will add Parakeet model files, openWakeWord/LiveKit Wakeword model, ONNX YOLO exports, and possibly a Tavily API key. A `docs/project/stage2-rollback.md` will be written when those land.
