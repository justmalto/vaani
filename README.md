
---

# **VAANI – Real-Time Voice Emotion & AI Conversation System**

*Vaani* is a real-time AI application that listens to your voice, understands your emotions, converts speech into text, and generates intelligent responses — all running **locally** using **FastAPI**, **WebSockets**, **Whisper**, **TinyLlama**, and a custom **Emotion Recognition** model.

It is designed for fast, low-latency, privacy-preserving voice intelligence with a responsive React frontend.

---

## 🌟 **Features**

### 🎙️ Real-Time Speech-to-Text

* Local **Whisper Tiny** model
* Fast transcription of 1-second audio chunks
* Supports noisy input with configurable thresholds

### 😃 Emotion Recognition

* Custom PyTorch classifier
* Takes raw audio → Mel spectrogram → emotion label
* Outputs emotion + probability scores

### 🧠 TinyLlama LLM Response Generation

* Small, fast local LLM (GGUF supported)
* Generates emotionally-aware replies
* Consumes combined transcript + emotion context

### ⚡ True Realtime Pipeline via WebSockets

Frontend ↔ Backend live communication:

* Audio stream
* Transcript updates
* Emotion predictions
* AI response

### 🧵 Parallel Model Execution

Each model runs in its own worker process:

* Whisper Worker
* Emotion Worker
* LLM Worker
  Prevents blocking and improves throughput.

### 🎛️ React Frontend

* Waveform visualizer
* Transcription rendering
* Emotion display
* Chat UI with system replies

---

# 🏗️ **Architecture Overview**

![Architecture]<img width="1494" height="539" alt="diagram-export-11-26-2025-1_08_02-AM" src="https://github.com/user-attachments/assets/83c2784c-4bb9-4187-902c-d7a2f6a686a9" />


**Summary of Architecture:**

* The **React Frontend** captures microphone input every 1 second.
* Audio chunks are sent to the **FastAPI WebSocket server**.
* The server sends the audio simultaneously to:

  * **Whisper Worker** for speech-to-text
  * **Emotion Worker** for emotion classification
* The WebSocket handler collects partial STT + emotion in real-time.
* A combined context is sent to the **TinyLlama Worker** to generate the AI reply.
* The frontend continuously receives:

  * Live transcript updates
  * Emotion predictions
  * Final AI chat response
* Temporary audio files are stored in `/tmp/vaani/session_id/`.
* `multiprocessing.Queue` enables seamless communication across workers.

---

# 🖼️ **Screenshots**
![Landing UI]<img width="1795" height="936" alt="Screenshot from 2025-11-26 12-32-13" src="https://github.com/user-attachments/assets/16af67d7-1ea4-4be8-b2ae-3f32532dfb72" />
![Working UI]<img width="1795" height="936" alt="Screenshot from 2025-11-26 12-31-59" src="https://github.com/user-attachments/assets/d3dad95c-f45d-4220-978c-918e8e2415ce" />


# 📦 **Backend Dependencies**

Add these to `backend/requirements.txt`:

```
fastapi
uvicorn
python-multipart
websockets
pydantic
soundfile
librosa
numpy
scipy
torch
transformers
sentencepiece
accelerate
openai-whisper
ctranslate2
pydub
```

---

# 💻 **Frontend Dependencies**

Installed via `npm install`:


---

# ⚙️ **Local Setup Instructions**

## **1. Clone the repo**

```
git clone https://github.com/justmalto/vaani.git
cd vaani
```

---

# 🚀 **Backend Setup**

```
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run backend:

```
uvicorn main:app --port 8000 --reload
```

---

# 💻 **Frontend Setup**

```
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

# 🔄 **Pipeline Flow Summary**

1. User speaks → microphone captures audio
2. Frontend sends 1-second chunks via WebSocket
3. Backend forwards audio to Whisper & Emotion workers
4. Workers return partial transcript + emotion
5. Aggregator enriches context → TinyLlama worker
6. Frontend receives:

   * Real-time STT
   * Emotion predictions
   * Final AI response
   * Visualization waveform data

---

# 🧪 **Troubleshooting**

### Whisper not transcribing?

Ensure audio format:

```
16 kHz  
16-bit PCM  
mono  
```

### Static noise?

Lower microphone gain or change device.

### LLM too slow?

Use TinyLlama GGUF + llama.cpp or CTranslate.

---

# 📜 **License**

MIT

