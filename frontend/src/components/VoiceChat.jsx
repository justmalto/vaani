import React, { useEffect, useState, useRef } from "react";
import "./VoiceChat.css";

export default function VoiceChat() {
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const websocketRef = useRef(null);
  const retryTimeoutRef = useRef(null);
  const manualStopRef = useRef(false);

  const numBars = 6;

  const handleToggleMic = async () => {
    if (isActive) stopMic();
    else await startMic();
  };

  const startMic = async () => {
    try {
      setIsConnecting(true);

      // 1. Mic access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // 2. Start WebSocket FIRST (fix)
      const ws = new WebSocket("wss://vaani-backend.whiteriver-ff52acfc.centralindia.azurecontainerapps.io/ws/audio");
      websocketRef.current = ws;

      ws.onopen = () => {
        console.log("🎧 /ws/audio connected");
        setIsConnecting(false);
        setIsActive(true);

        // 3. Create AudioContext AFTER handshake (fix)
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
        drawVisualizer(true);

        // 4. Setup MediaRecorder
        let options = { mimeType: "audio/webm;codecs=opus" };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
          options = {}; // fallback
        }

        const mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };

        mediaRecorder.start(300);
      };

      ws.onerror = (e) => {
        console.log("❌ audio ws error", e);
        ws.close();
      };

      ws.onclose = () => {
        console.log("🔒 audio ws closed");
        setIsActive(false);
        setIsConnecting(false);

        if (!manualStopRef.current) {
          clearTimeout(retryTimeoutRef.current);
          retryTimeoutRef.current = setTimeout(startMic, 3000);
        } else {
          manualStopRef.current = false;
        }
      };
    } catch (err) {
      console.error(err);
      setIsConnecting(false);
      setIsActive(false);
    }
  };

  const stopMic = () => {
    manualStopRef.current = true;
    setIsActive(false);

    mediaRecorderRef.current?.stop();
    audioContextRef.current?.close();

    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send("END");
      websocketRef.current.close();
    }

    cancelAnimationFrame(animationRef.current);
  };

  const roundedRect = (ctx, x, y, width, height, radius) => {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  };

  const drawVisualizer = (live = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(256);

    const barWidth = 30;
    const barGap = 10;
    const baseHeight = 60;
    let smoothedHeights = Array(numBars).fill(baseHeight);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (live && analyser) analyser.getByteTimeDomainData(dataArray);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const step = Math.floor(dataArray.length / numBars);

      for (let i = 0; i < numBars; i++) {
        let targetHeight = live
          ? Math.max(baseHeight, (dataArray[i * step] / 255) * canvas.height * 1.1)
          : baseHeight;
        smoothedHeights[i] += (targetHeight - smoothedHeights[i]) * 0.4;
      }

      ctx.fillStyle = "#E5E7EB";
      const radius = barWidth / 2;

      for (let i = 0; i < numBars; i++) {
        const height = smoothedHeights[i];
        const offset = (barWidth + barGap) * i;

        roundedRect(ctx, centerX - offset - barWidth - barGap / 2, centerY - height / 2, barWidth, height, radius);
        ctx.fill();

        roundedRect(ctx, centerX + offset + barGap / 2, centerY - height / 2, barWidth, height, radius);
        ctx.fill();
      }
    };

    draw();
  };

  useEffect(() => {
    drawVisualizer(false);
    return stopMic;
  }, []);

  return (
    <div className="voicechat-screen">
      <canvas ref={canvasRef} width={800} height={400} className="voicechat-canvas" />
      <div
        className={`mic-dot ${isActive ? "active" : ""} ${isConnecting ? "connecting" : ""}`}
        onClick={handleToggleMic}
      />
    </div>
  );
}
