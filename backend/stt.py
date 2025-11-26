"""
stt.py — Local Whisper-based Speech-to-Text
Compatible with Python 3.12
"""

import whisper
import torch

def transcribe_local_whisper(audio_path: str, model_size: str = "turbo") -> dict:
    """
    Transcribe speech from an audio file using local OpenAI Whisper model.
    Automatically detects language and returns text + metadata.
    """
    model = whisper.load_model("tiny")
    result =model.transcribe("output.wav")

    print("🗣️ Transcription complete.")
    return {
        "text": result.text.strip()
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stt.py <audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    output = transcribe_local_whisper(path)
    print("\n--- TRANSCRIPTION ---")
    print(output["text"])
    print(f"(Language: {output['language']})")
