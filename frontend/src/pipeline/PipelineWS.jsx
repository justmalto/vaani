import { createContext, useContext, useEffect, useRef, useState } from "react";

const PipelineContext = createContext(null);

export function PipelineProvider({ children }) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [reply, setReply] = useState("");
  const [stage, setStage] = useState("idle");

  // NEW: spectrograms from backend
  const [spectrogramRaw, setSpectrogramRaw] = useState(null);
  const [spectrogramEmotion, setSpectrogramEmotion] = useState(null);

  const connectWS = () => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/pipeline");
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      clearTimeout(reconnectTimer.current);
      console.log("Pipeline WS OPEN");
    };

    ws.onclose = () => {
      setIsConnected(false);
      reconnectTimer.current = setTimeout(connectWS, 1500);
      console.log("Pipeline WS CLOSED");
    };

    ws.onerror = () => ws.close();

    // -------------------------
    // HANDLERS FOR WS MESSAGES
    // -------------------------
    const handlers = {
      session_id: (msg) => {
        setSessionId(msg.session_id);

        // reset spectrograms for new session
        setSpectrogramRaw(null);
        setSpectrogramEmotion(null);
      },

      ready: () => setStage("ready"),

      status: (msg) => setStage(msg.stage),

      transcript: (msg) => setTranscript(msg.text),

      emotion: (msg) => {
        console.log("WS EMOTION RECEIVED:", msg);
        setEmotion(msg.payload);
      },

      reply: (msg) => setReply(msg.text),

      // NEW: raw spectrogram base64 from processor loop
      spectrogram_raw: (msg) => {
        setSpectrogramRaw(msg.image);
      },

      // NEW: emotion spectrogram base64
      spectrogram_emotion: (msg) => {
        setSpectrogramEmotion(msg.image);
      },
    };

    // -------------------------
    // MESSAGE ROUTER
    // -------------------------
    ws.onmessage = (event) => {
      if (typeof event.data !== "string") return;

      try {
        const msg = JSON.parse(event.data);
        if (handlers[msg.type]) handlers[msg.type](msg);
      } catch (err) {
        console.error("Pipeline WS parse error:", err);
      }
    };
  };

  useEffect(() => {
    connectWS();
    return () => {
      wsRef.current?.close();
      clearTimeout(reconnectTimer.current);
    };
  }, []);

  return (
    <PipelineContext.Provider
      value={{
        isConnected,
        sessionId,
        transcript,
        emotion,
        reply,
        stage,
        spectrogramRaw,        // NEW
        spectrogramEmotion,    // NEW
      }}
    >
      {children}
    </PipelineContext.Provider>
  );
}

export const usePipeline = () => useContext(PipelineContext);
