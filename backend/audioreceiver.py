from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import tempfile
import subprocess
import os
from datetime import datetime

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "AudioEmotion backend is running!"}


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    """
    Receives audio chunks via WebSocket, writes to temporary WebM,
    converts to WAV using ffmpeg, and prepares for downstream STT/LLM/TTS.
    """
    await websocket.accept()
    print("🎙️ Client connected...")

    # Temporary file setup
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "recording.webm")
    wav_file = os.path.join(temp_dir, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

    try:
        # Receive and write audio chunks
        with open(temp_file, "wb") as f:
            while True:
                try:
                    msg = await websocket.receive()

                    # Handle disconnect
                    if msg.get("type") == "websocket.disconnect":
                        print("⚡ Client disconnected.")
                        break

                    # Handle binary audio chunks
                    if "bytes" in msg and msg["bytes"] is not None:
                        f.write(msg["bytes"])

                    # Handle text messages (like "END")
                    elif "text" in msg and msg["text"] == "END":
                        print("📥 Received END signal — closing stream.")
                        break

                except WebSocketDisconnect:
                    print("⚡ WebSocket forcibly closed by client.")
                    break
                except Exception as e:
                    print("⚠️ Error receiving audio chunk:", e)
                    break

        # Convert WebM → WAV
        print("🎧 Converting WebM to WAV...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file, "-ar", "16000", "-ac", "1", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Verify conversion success
        if os.path.exists(wav_file):
            print(f"✅ Saved audio: {wav_file}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("✅ Audio received and converted.")
        else:
            print("❌ Conversion failed.")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("❌ Audio conversion failed.")

    except WebSocketDisconnect:
        print("⚡ Client disconnected before conversion finished.")
    except Exception as e:
        print("💥 Unexpected error in audio stream:", e)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(f"Error: {e}")
            except Exception:
                pass
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(wav_file):
                print(f"🧹 Cleaned up {wav_file}")
            if os.path.isdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

        # Close socket safely
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
                print("🔒 WebSocket closed cleanly.")
            except RuntimeError:
                print("⚡ Attempted to close an already closed socket.")
        else:
            print("🔒 WebSocket already closed.")
