# backend.py
import os
import uuid
import shutil
import tempfile
import subprocess
import asyncio
from typing import Set
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

# External model functions you implement
import whisper
from modelinference import predict_emotion_from_wav
from LLMinference import generate_supportive_reply
from spectrogram_utils import (
    generate_raw_spectrogram_base64,
    generate_emotion_gradient_spectrogram_base64
)


# -------------------------
# Config & global state
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# queue that holds wav file paths (strings)
audio_queue: asyncio.Queue[str] = asyncio.Queue()

# set of pipeline subscriber websockets
pipeline_subscribers: Set[WebSocket] = set()

# whisper model lazy cache + lock
_whisper_model = None
_whisper_model_lock = asyncio.Lock()


# -------------------------
# Utility functions
# -------------------------
def make_output_wav_path() -> Path:
    fname = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    return MEDIA_DIR / fname


def ffmpeg_convert_webm_to_wav(input_webm: str, output_wav: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", input_webm, "-ar", "16000", "-ac", "1", output_wav],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, proc.stderr.decode(errors="ignore")


async def broadcast_json(payload: dict):
    """
    Broadcast JSON to all pipeline subscribers. Remove disconnected ones.
    """
    if not pipeline_subscribers:
        return

    text = json.dumps(payload)
    to_remove = []
    for ws in list(pipeline_subscribers):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove.append(ws)

    for ws in to_remove:
        pipeline_subscribers.discard(ws)


async def get_whisper_model():
    """
    Lazily load the whisper model once. Load in executor to avoid blocking.
    """
    global _whisper_model
    if _whisper_model is None:
        async with _whisper_model_lock:
            if _whisper_model is None:
                loop = asyncio.get_running_loop()
                _whisper_model = await loop.run_in_executor(None, lambda: whisper.load_model("tiny"))
    return _whisper_model


# -------------------------
# WebSocket: receive audio and enqueue WAV
# -------------------------
app = FastAPI()  # will be replaced with lifespan wrapper below


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    """
    Receives binary audio chunks (webm) over WebSocket, writes a temp .webm,
    converts to permanent .wav, and enqueues the wav path for processing.
    """
    await websocket.accept()
    client = websocket.client.host if websocket.client else "unknown"
    print(f"🎙️ Audio WS connected from {client}")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_webm = os.path.join(temp_dir, "recording.webm")

        # Receive binary chunks
        try:
            with open(input_webm, "wb") as f:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        print("⚡ Client disconnected during upload")
                        break

                    # client sends text "END" to signal end of stream
                    if msg.get("text") == "END":
                        print("📥 Received END signal")
                        break

                    if msg.get("bytes"):
                        f.write(msg["bytes"])

        except WebSocketDisconnect:
            print("⚡ WebSocketDisconnect while receiving audio")
        except Exception as e:
            print("💥 Error receiving audio:", e)

        # Convert temp webm -> permanent wav in MEDIA_DIR
        output_wav = make_output_wav_path()
        print(f"🎧 Converting {input_webm} -> {output_wav}")
        ret, ff_err = ffmpeg_convert_webm_to_wav(input_webm, str(output_wav))

        if ret == 0 and output_wav.exists():
            print(f"✅ WAV saved: {output_wav}")
            # enqueue for processing by the background processor
            await audio_queue.put(str(output_wav))
            print("📌 Enqueued for processing")
        else:
            print("❌ ffmpeg conversion failed")
            print(ff_err)

        # Close audio websocket
        try:
            await websocket.close()
        except Exception:
            pass

    print("🔒 Audio WS closed")


async def processor_loop(stop_event: asyncio.Event):
    """
    Background loop that consumes audio_queue and processes items.
    Creates transcript, emotion, spectrograms, LLM reply,
    then deletes WAV + JSON files.
    """
    print("🧠 Processor loop started")
    while not stop_event.is_set():
        try:
            # Wait for next item with cancellation awareness
            wav_path = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print("❌ processor_loop queue error:", e)
            continue

        try:
            session_id = Path(wav_path).stem

            # ------------------------------------------------------
            # BROADCAST SESSION ID
            # ------------------------------------------------------
            await broadcast_json({"type": "session_id", "session_id": session_id})
            await broadcast_json({"type": "status", "stage": "transcribing"})

            # ------------------------------------------------------
            # 1) TRANSCRIBE USING WHISPER
            # ------------------------------------------------------
            transcript = ""
            try:
                whisper_model = await get_whisper_model()
                loop = asyncio.get_running_loop()
                whisper_result = await loop.run_in_executor(
                    None, lambda: whisper_model.transcribe(wav_path)
                )
                transcript = whisper_result.get("text", "") or ""
            except Exception as e:
                print("❌ Whisper error:", e)

            await broadcast_json({"type": "transcript", "text": transcript})

            # ------------------------------------------------------
            # 2) EMOTION PREDICTION
            # ------------------------------------------------------
            await broadcast_json({"type": "status", "stage": "emotion"})
            emotion = None
            windows = []

            try:
                loop = asyncio.get_running_loop()
                emotion = await loop.run_in_executor(
                    None, lambda: predict_emotion_from_wav(wav_path)
                )
                print("Emotion output:", emotion)

                if isinstance(emotion, (list, tuple)) and len(emotion) > 1:
                    windows = emotion[1]

                # save windows for spectrogram use (temp)
                if windows:
                    windows_path = str(wav_path) + "_windows.json"
                    with open(windows_path, "w") as f:
                        json.dump(windows, f)
                    print(f"📁 Saved windows: {windows_path}")

            except Exception as e:
                print("❌ Emotion inference error:", e)

            # broadcast final predicted emotion (string)
            await broadcast_json({"type": "emotion", "payload": emotion[0]})

            # ------------------------------------------------------
            # 3) GENERATE RAW SPECTROGRAM AND SEND OVER WS
            # ------------------------------------------------------
            await broadcast_json({"type": "status", "stage": "spectrogram_raw"})
            try:
                raw_img = generate_raw_spectrogram_base64(wav_path)
                await broadcast_json({
                    "type": "spectrogram_raw",
                    "image": raw_img
                })
            except Exception as e:
                print("❌ Raw spectrogram error:", e)

            # ------------------------------------------------------
            # 4) GENERATE EMOTION-GRADIENT SPECTROGRAM AND SEND WS
            # ------------------------------------------------------
            await broadcast_json({"type": "status", "stage": "spectrogram_emotion"})
            try:
                windows_path = str(wav_path) + "_windows.json"
                if windows:
                    emo_img = generate_emotion_gradient_spectrogram_base64(
                        wav_path, windows
                    )
                    await broadcast_json({
                        "type": "spectrogram_emotion",
                        "image": emo_img
                    })
            except Exception as e:
                print("❌ Emotion spectrogram error:", e)

            # ------------------------------------------------------
            # 5) GENERATE LLM REPLY
            # ------------------------------------------------------
            await broadcast_json({"type": "status", "stage": "reply"})
            reply = ""
            try:
                loop = asyncio.get_running_loop()
                reply = await loop.run_in_executor(
                    None, lambda: generate_supportive_reply(transcript, emotion)
                )
            except Exception as e:
                print("❌ LLM reply generation error:", e)

            await broadcast_json({"type": "reply", "text": reply})

            # ------------------------------------------------------
            # 6) DONE + CLEANUP
            # ------------------------------------------------------
            await broadcast_json({"type": "status", "stage": "done"})
            await broadcast_json({"type": "ready"})

            # 🔥 DELETE WAV + WINDOWS JSON AFTER SENDING EVERYTHING
            try:
                os.remove(wav_path)
                windows_json = wav_path + "_windows.json"
                if os.path.exists(windows_json):
                    os.remove(windows_json)
                print(f"🧹 Cleaned up session files for {session_id}")
            except Exception as e:
                print(f"⚠️ Cleanup failed: {e}")

        except Exception as e:
            print("❌ Unexpected processor error:", e)





# -------------------------
# Pipeline WebSocket (subscribers receive results)
# -------------------------
@app.websocket("/ws/pipeline")
async def pipeline_ws(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Pipeline WS connected")
    pipeline_subscribers.add(websocket)

    # notify client that pipeline is ready
    try:
        await websocket.send_json({"type": "ready"})
    except Exception:
        pass

    try:
        while True:
            # keep connection alive; client doesn't need to send messages
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("⚠️ Pipeline client disconnected")
    except Exception as e:
        print("⚠️ Pipeline WS error:", e)
    finally:
        pipeline_subscribers.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
        print("🔌 Pipeline WS closed")


# -------------------------
# Lifespan: start/stop processor loop
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    processor_task = asyncio.create_task(processor_loop(stop_event))
    print("🚀 Lifespan started: processor running")
    try:
        yield
    finally:
        # signal the background processor to stop and wait for it
        stop_event.set()
        processor_task.cancel()
        try:
            await processor_task
        except Exception:
            pass
        print("🛑 Lifespan stopping: processor stopped")


# replace app with one wired to lifespan
app = FastAPI(lifespan=lifespan)
app.add_api_websocket_route("/ws/audio", audio_stream)
app.add_api_websocket_route("/ws/pipeline", pipeline_ws)

@app.get("/spectrogram")
async def spectrogram(wav: str, mode: str = "raw"):
    """
    wav: absolute path to a WAV file in MEDIA_DIR
    mode: 'raw' or 'emotion'
    """
    wav_path = Path(wav)
    if not wav_path.exists():
        return {"error": "WAV file not found", "path": wav}

    if mode == "raw":
        img64 = generate_raw_spectrogram_base64(str(wav_path))
        return {"spectrogram": img64}

    # emotion mode → need windows JSON
    windows_path = str(wav_path) + "_windows.json"
    if not Path(windows_path).exists():
        return {"error": "No window data found, run inference first"}

    with open(windows_path, "r") as f:
        windows = json.load(f)

    img64 = generate_emotion_gradient_spectrogram_base64(str(wav_path), windows)
    return {"spectrogram": img64}




# A simple health endpoint
@app.get("/")
async def root():
    return {"message": "AudioEmotion backend (lifespan) running"}
