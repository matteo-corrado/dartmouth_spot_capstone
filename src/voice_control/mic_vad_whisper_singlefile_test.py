import time
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Config
SAMPLE_RATE = 16000          # Whisper expects 16 kHz mono
CHUNK_SECONDS = 2.0          # How often to run inference (seconds)
DEVICE = "auto"              # "auto" chooses GPU if available
COMPUTE_TYPE = "int8_float16" # great for RTX 3050 Ti
MODEL_NAME = "large-v3-turbo"


# initialize whisper model
print("Loading Faster-Whisper model ...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"Loaded {MODEL_NAME}")

# audio queue for streaming microphone input
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # Convert stereo to mono and add to queue
    mono = indata.mean(axis=1).astype(np.float32)
    pcm16 = (mono * 32767).astype(np.int16)
    audio_q.put(pcm16)

# \start audio stream
stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        callback=audio_callback,
                        blocksize=int(SAMPLE_RATE * CHUNK_SECONDS))
stream.start()
print("Listening... press Ctrl+C to stop")

buffer = np.zeros(0, dtype=np.int16)
last_transcription = time.time()

try:
    while True:
        # build up buffer from audio queue
        while not audio_q.empty():
            buffer = np.append(buffer, audio_q.get())

        # every CHUNK_SECONDS, process current buffer
        if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS:
            # convert to float32 [-1,1]
            audio_f32 = buffer.astype(np.float32) / 32768.0
            buffer = np.zeros(0, dtype=np.int16)  # clear buffer

            # run Whisper with built-in VAD
            segments, _ = model.transcribe(
                audio_f32,
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text = "".join(seg.text for seg in segments).strip()
            if text:
                print(f"You said: {text}")
                last_transcription = time.time()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop()
    stream.close()
