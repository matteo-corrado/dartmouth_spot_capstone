# Post-Flash Migration: faster-whisper + JetPack 6.2.1

After flashing the Jetson AGX Orin to JetPack 6.2.1, follow this guide to migrate
the ASR server from openai-whisper to faster-whisper (CTranslate2 backend).

## Why flash first?

| Blocker on JetPack 5.1.2 | Fixed by JetPack 6.2.1 |
|---------------------------|------------------------|
| CUDA 11.4 (ct2 4.x needs 12) | CUDA 12.6 |
| glibc 2.31 (ct2 wheels need 2.35) | glibc 2.35 |
| Python 3.8 (EOL) | Python 3.10 |

On JetPack 6.2.1, `pip install faster-whisper` just works. No source build needed.

## Pre-flash backup

Only one file to save (everything else is on GitHub):

```bash
cat /home/spotdog/dartmouth_spot_capstone/.env
# Save the output somewhere safe (robot credentials)
```

## Flash procedure

1. Install [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) on a Windows PC
   - SDK Manager auto-configures WSL2, Ubuntu, and USBIPD-Win
   - Install the APX driver when prompted
2. Connect Jetson USB-C port (next to 40-pin header) to Windows PC
3. Put Jetson in Force Recovery Mode (hold REC button + press power)
4. Flash **JetPack 6.2.1** via SDK Manager
5. Choose **Pre-Config** for OEM setup (set username: `spotdog`, password)
   - This enables headless boot — no monitor needed

## Post-flash environment rebuild

```bash
# Verify flash
nvcc --version        # CUDA 12.x
python3 --version     # 3.10.x
lsb_release -a        # Ubuntu 22.04

# Clone repo
cd /home/spotdog
git clone https://github.com/matteo-corrado/dartmouth_spot_capstone.git
cd dartmouth_spot_capstone
git checkout feature/llm-brain

# Restore .env
nano .env  # paste saved credentials

# Create virtualenv
python3 -m venv spot-env
source spot-env/bin/activate
pip install --upgrade pip

# PyTorch (JetPack 6 wheel from NVIDIA)
pip install numpy
pip install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v62

# Project deps
pip install -r requirements.txt

# faster-whisper (now works natively)
pip install faster-whisper

# Wake word + other ML deps
pip install openwakeword

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b

# ReSpeaker config tool (optional)
cd /home/spotdog
git clone https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY.git
```

### Verify stack

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
python3 -c "import ctranslate2; print(f'CTranslate2 {ctranslate2.__version__}')"
python3 -c "import openwakeword; print('openwakeword OK')"
```

## Migrate server.py from openai-whisper to faster-whisper

### API changes

| Aspect | openai-whisper (current) | faster-whisper (new) |
|--------|--------------------------|----------------------|
| Import | `import whisper` | `from faster_whisper import WhisperModel` |
| Load | `whisper.load_model(name, device="cuda")` | `WhisperModel(name, device="cuda", compute_type="float16")` |
| Transcribe | returns `dict` with `result["text"]` | returns `(segments_gen, info)` |
| Text | `result["text"].strip()` | `" ".join(s.text for s in segments).strip()` |
| Segments | `result["segments"][i]["no_speech_prob"]` | `segment.no_speech_prob` (attribute) |
| beam_size | `1` (too slow with PyTorch) | `5` (affordable with CTranslate2) |

### New features to enable

- `vad_filter=True` + `vad_parameters=dict(min_silence_duration_ms=500)`
  Built-in Silero VAD pre-filters fan/motor noise before decoding

- `hotwords` parameter
  Load location names from `locations.json` dynamically. This biases Whisper
  toward known location names WITHOUT causing hallucinations like initial_prompt did.

- `beam_size=5`
  Affordable with CTranslate2. Fixes "couch" -> "touch" disambiguation.

### Config changes after migration

| Setting | Before (openai-whisper) | After (faster-whisper) |
|---------|------------------------|----------------------|
| `MIN_AUDIO_DURATION` | 0.5s | 0.15s (VAD filter catches noise) |
| `no_speech_threshold` | 0.7 | 0.6 (safe with vad_filter) |
| `beam_size` | 1 | 5 |
| `HALLUCINATION_PHRASES` | current set | add `"."` and `"..."` |

### Remove after migration
- `import torch` (no longer needed)
- `import whisper` (replaced by faster_whisper)

### Add: load_hotwords() function

```python
def load_hotwords():
    """Load location names from locations.json for ASR hotword biasing."""
    locations_path = os.path.join(os.path.dirname(__file__), "..", "..", "locations.json")
    try:
        with open(locations_path) as f:
            locs = json.load(f)
        names = [name.replace("_", " ") for name in locs
                 if not name.startswith("waypoint_") and not name.startswith("localize")]
        return " ".join(names)
    except Exception:
        return ""
```

### Expected performance improvement

| Metric | openai-whisper | faster-whisper |
|--------|---------------|----------------|
| ASR latency (1s audio) | 0.3-0.5s | 0.1-0.2s |
| beam_size=5 penalty | +2x slower | negligible |
| Memory | ~5GB (large-v3) | ~3GB (float16) |
| Fan noise rejection | manual filters | built-in Silero VAD |
