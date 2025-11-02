import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import time, queue, numpy as np, sounddevice as sd, webrtcvad, grpc
from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk
from asr_pb2_grpc import ASRStub
from intent import parse_intent

# audio config
SAMPLE_RATE = 16000
FRAME_MS = 30                              # webrtcvad supports 10/20/30 ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
VAD_LEVEL = 2                              # 0-3; 3 is most aggressive
SILENCE_TIMEOUT = 0.6                      # sec after last speech to cut utterance

audio_q = queue.Queue()
vad = webrtcvad.Vad(VAD_LEVEL)

def audio_callback(indata, frames, time_info, status):
    mono = indata.mean(axis=1).astype(np.float32)
    pcm16 = (mono * 32767).astype(np.int16).tobytes()
    audio_q.put(pcm16)

def stream_utterance(stub, pcm_bytes: bytes):
    def req_iter():
        # send config first
        yield StreamingRequest(config=StreamingConfig(language_code="en", sample_rate_hz=SAMPLE_RATE, enable_punctuation=True))
        # then audio in chunks (optional; here we send as-is)
        CHUNK = FRAME_SAMPLES * 2  # bytes (int16)
        for i in range(0, len(pcm_bytes), CHUNK):
            yield StreamingRequest(audio=AudioChunk(pcm16=pcm_bytes[i:i+CHUNK]))
    responses = stub.StreamingRecognize(req_iter())
    final_text = ""
    for r in responses:
        final_text = r.transcript
        if r.is_final:
            break
    return final_text.strip()

def main():
    # connect to local ASR server
    chan = grpc.insecure_channel("localhost:50055")
    stub = ASRStub(chan)

    stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=FRAME_SAMPLES)
    stream.start()
    print("Listening… say a command. Ctrl+C to stop.")

    window = b""
    speech_buf = bytearray()
    last_voice_ts = time.time()

    try:
        while True:
            pcm = audio_q.get()
            window += pcm
            # process in FRAME-sized steps
            while len(window) >= FRAME_SAMPLES*2:
                frame = window[:FRAME_SAMPLES*2]
                window = window[FRAME_SAMPLES*2:]
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                if is_speech:
                    speech_buf.extend(frame)
                    last_voice_ts = time.time()
                else:
                    if speech_buf and (time.time() - last_voice_ts) > SILENCE_TIMEOUT:
                        # flush utterance to ASR
                        wav_bytes = bytes(speech_buf)
                        speech_buf.clear()
                        text = stream_utterance(stub, wav_bytes)
                        if not text:
                            print("[no speech detected]")
                            continue
                        print("Transcript:", text)
                        intent = parse_intent(text)
                        if intent:
                            print("→ Intent:", intent)
                            # Execute the intent on Spot
                            # Import here with proper path setup
                            import sys
                            import pathlib
                            project_root = pathlib.Path(__file__).resolve().parents[2]
                            if str(project_root) not in sys.path:
                                sys.path.insert(0, str(project_root))
                            from src.voice_control.spot_dispatch import dispatch_intent
                            success = dispatch_intent(intent)
                            if success:
                                print("✓ Command executed successfully")
                            else:
                                print("✗ Command failed or not implemented")
                        else:
                            print("(no intent matched)")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(); stream.close()
        # Gracefully close Spot session
        try:
            # Import here to ensure path is set up
            import sys
            import pathlib
            project_root = pathlib.Path(__file__).resolve().parents[2]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from src.voice_control.spot_dispatch import close_spot_session
            close_spot_session()
        except Exception as e:
            print(f"Error closing Spot session: {e}")

if __name__ == "__main__":
    main()
