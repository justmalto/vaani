 import { createContext, useContext, useEffect, useRef, useState } from "react";

const PipelineContext = createContext(null);

export function PipelineProvider({ children }) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const isClosingRef = useRef(false);

  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [reply, setReply] = useState("");
  const [stage, setStage] = useState("idle");

  const [spectrogramRaw, setSpectrogramRaw] = useState(null);
  const [spectrogramEmotion, setSpectrogramEmotion] = useState(null);

  const connectWS = () => {
    // -----------------------------
    // PREVENT MULTIPLE WS INSTANCES
    // -----------------------------
    if (
      wsRef.current &&
      wsRef.current.readyState !== WebSocket.CLOSED &&
      wsRef.current.readyState !== WebSocket.CLOSING
    ) {
      console.log("WS already open → skipping new connection");
      return;
    }

    const ws = new WebSocket("wss://vaani-backend.whiteriver-ff52acfc.centralindia.azurecontainerapps.io/ws/pipeline");
    wsRef.current = ws;
    isClosingRef.current = false;

    ws.onopen = () => {
      setIsConnected(true);
      console.log("Pipeline WS OPEN");
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("Pipeline WS CLOSED");

      if (!isClosingRef.current) {
        reconnectTimer.current = setTimeout(connectWS, 1500);
      }
    };

    ws.onerror = () => {
      console.log("Pipeline WS ERROR → closing");
      ws.close();
    };

    const handlers = {
      session_id: (msg) => {
        setSessionId(msg.session_id);
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

      spectrogram_raw: (msg) => setSpectrogramRaw(msg.image),
      spectrogram_emotion: (msg) => setSpectrogramEmotion(msg.image),
    };

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
      isClosingRef.current = true;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
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
        spectrogramRaw,
        spectrogramEmotion,
      }}
    >
      {children}
    </PipelineContext.Provider>
  );
}

export const usePipeline = () => useContext(PipelineContext);
