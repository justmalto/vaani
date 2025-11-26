from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import tempfile
import subprocess
import os
from datetime import datetime
import traceback

# Import project modules
from stt import transcribe_local_whisper
from modelinference import predict_emotion_from_wav
from LLMinference import generate_supportive_reply

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "🎧 AudioEmotion Backend running with /ws/audio, /ws/speechtranscriber, /ws/chatresponse"}


# 🎙️ 1️⃣ Combined endpoint: audio → emotion + text + LLM reply
@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Receives audio chunks, converts to WAV, runs:
    - Emotion Recognition (model_inference)
    - Speech-to-Text (stt)
    - LLM Supportive Reply (TinyLlama)
    Sends all results as JSON back to frontend.
    """
    await websocket.accept()
    print("🎙️ [audio] Client connected.")

    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "recording.webm")
    wav_file = os.path.join(temp_dir, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

    try:
        # Receive audio stream
        with open(temp_file, "wb") as f:
            while True:
                msg = await websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()

                if "bytes" in msg and msg["bytes"] is not None:
                    f.write(msg["bytes"])
                elif "text" in msg and msg["text"] == "END":
                    print("📥 Received END signal for /ws/audio.")
                    break

        # Convert WebM → WAV
        print("🎧 Converting WebM → WAV...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, "-ar", "16000", "-ac", "1", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(wav_file):
            await websocket.send_json({"error": "Conversion failed"})
            return

        # 1️⃣ Emotion Detection
        await websocket.send_json({"type": "status", "message": "predicting_emotion"})
        emotion_result = predict_emotion_from_wav(wav_file)
        emotion = emotion_result.get("emotion", "Neutral")
        await websocket.send_json({"type": "emotion", "payload": emotion_result})

        # 2️⃣ Speech-to-Text (Whisper)
        await websocket.send_json({"type": "status", "message": "transcribing"})
        stt_result = transcribe_local_whisper(wav_file)
        transcript = stt_result["text"] if isinstance(stt_result, dict) else stt_result
        await websocket.send_json({"type": "transcript", "text": transcript})

        # 3️⃣ LLM Supportive Reply
        await websocket.send_json({"type": "status", "message": "generating_reply"})
        reply = generate_supportive_reply(transcript, emotion)
        await websocket.send_json({"type": "llm_reply", "text": reply})

        # ✅ Final payload
        await websocket.send_json({
            "type": "result",
            "transcript": transcript,
            "emotion": emotion,
            "probabilities": emotion_result.get("probabilities", {}),
            "llm_reply": reply
        })

    except WebSocketDisconnect:
        print("⚡ [audio] Client disconnected.")
    except Exception as e:
        print("💥 [audio] Error:", e)
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Cleanup temp files
        for f in [temp_file, wav_file]:
            if os.path.exists(f): os.remove(f)
        if os.path.isdir(temp_dir): os.rmdir(temp_dir)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("🔒 [audio] Closed.")


# 🗣️ 2️⃣ Speech Transcriber — audio → text + emotion
@app.websocket("/ws/speechtranscriber")
async def ws_transcriber(websocket: WebSocket):
    await websocket.accept()
    print("🗣️ [speechtranscriber] Connected.")

    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "recording.webm")
    wav_file = os.path.join(temp_dir, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

    try:
        with open(temp_file, "wb") as f:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if "bytes" in msg and msg["bytes"] is not None:
                    f.write(msg["bytes"])
                elif "text" in msg and msg["text"] == "END":
                    break

        # Convert → WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, "-ar", "16000", "-ac", "1", wav_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Emotion Detection
        emotion_result = predict_emotion_from_wav(wav_file)
        # STT
        stt_result = transcribe_local_whisper(wav_file)
        transcript = stt_result["text"] if isinstance(stt_result, dict) else stt_result

        await websocket.send_json({
            "type": "result",
            "transcript": transcript,
            "emotion": emotion_result.get("emotion", "Neutral"),
            "probabilities": emotion_result.get("probabilities", {})
        })

    except WebSocketDisconnect:
        print("⚡ [speechtranscriber] Disconnected.")
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        for f in [temp_file, wav_file]:
            if os.path.exists(f): os.remove(f)
        if os.path.isdir(temp_dir): os.rmdir(temp_dir)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("🔒 [speechtranscriber] Closed.")


# 💬 3️⃣ Chat Response — text + emotion → LLM
@app.websocket("/ws/chatresponse")
async def ws_chatresponse(websocket: WebSocket):
    await websocket.accept()
    print("💬 [chatresponse] Connected.")

    try:
        while True:
            msg = await websocket.receive_json()
            if not msg:
                continue
            text = msg.get("text")
            emotion = msg.get("emotion", "Neutral")
            if not text:
                await websocket.send_json({"error": "Missing text"})
                continue

            print(f"🧠 [chatresponse] Generating for emotion={emotion}")
            reply = generate_supportive_reply(text, emotion)
            await websocket.send_json({"reply": reply, "emotion": emotion, "input": text})

    except WebSocketDisconnect:
        print("⚡ [chatresponse] Disconnected.")
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        print("🔒 [chatresponse] Closed.")
