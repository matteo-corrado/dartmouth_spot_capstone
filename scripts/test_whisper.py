#!/usr/bin/env python3
"""Simple test script to verify Faster Whisper is working.

This script tests the speech-to-text functionality without needing the full gRPC setup.
It records a short audio sample and transcribes it.
"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

def record_audio(duration_sec=3, sample_rate=16000):
    """Record audio from default microphone."""
    print(f"\n Recording for {duration_sec} seconds...")
    print("Speak now!")
    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("   Recording complete. Processing...\n")
    return audio.flatten()

def main():
    print("=" * 60)
    print("Whisper Speech-to-Text Test")
    print("=" * 60)
    print("\nThis will:")
    print("1. Load the Whisper model (may take a minute on first run)")
    print("2. Record 3 seconds of audio from your microphone")
    print("3. Transcribe what you said")
    print("\nWARNING: Make sure your microphone is enabled and working!")
    print("\nStarting in 2 seconds...")
    import time
    time.sleep(2)
    
    # Load model
    print("\n[1/3] Loading Whisper model...")
    print("   (This may take a minute on first run as it downloads the model)")
    try:
        model = WhisperModel(
            "base",  # Use smaller model for faster testing (can change to "large-v3-turbo" later)
            device="cpu",  # Force CPU for compatibility
            compute_type="int8"  # Works on CPU
        )
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model: {e}")
        return 1
    
    # Record audio
    print("\n[2/3] Recording audio...")
    try:
        audio = record_audio(duration_sec=3, sample_rate=16000)
        print(f"   Recorded {len(audio)} samples ({len(audio)/16000:.2f} seconds)")
    except Exception as e:
        print(f"   Error recording audio: {e}")
        print("   Make sure your microphone is connected and permissions are granted.")
        return 1
    
    # Transcribe
    print("\n[3/3] Transcribing...")
    try:
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            vad_filter=True,
            language="en"
        )
        
        text = "".join(segment.text for segment in segments).strip()
        
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        if text:
            print(f"   Transcript: \"{text}\"")
            print(f"\n   Speech-to-text is working!")
        else:
            print("   (No speech detected)")
            print("   Try speaking louder or closer to the microphone.")
        print("=" * 60)
        
    except Exception as e:
        print(f"   Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

