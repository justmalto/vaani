import React, { useState } from "react";
import { usePipeline } from "../pipeline/PipelineWS";
import "./SpectrogramViewer.css"; // ← CSS file

export default function SpectrogramViewer() {
  const {
    spectrogramRaw,
    spectrogramEmotion,
    stage,
    sessionId,
  } = usePipeline();

  const [showEmotion, setShowEmotion] = useState(false);

  const activeImage = showEmotion ? spectrogramEmotion : spectrogramRaw;

  const isProcessingSpectrogram =
    stage === "spectrogram_raw" ||
    stage === "spectrogram_emotion" ||
    stage === "emotion";

  return (
    <div className="spectro-container">
      <h2 className="spectro-title">🎵 Spectrogram Viewer</h2>

      {/* Toggle */}
      <div className="spectro-toggle">
        <label>Emotion Gradient:</label>
        <input
          type="checkbox"
          checked={showEmotion}
          onChange={() => setShowEmotion(!showEmotion)}
        />
      </div>

      {/* Status */}
      {isProcessingSpectrogram && (
        <div className="spectro-status">
          ⏳ Generating {showEmotion ? "emotion-gradient" : "raw"} spectrogram...
        </div>
      )}

      {/* No Image */}
      {!activeImage && !isProcessingSpectrogram && (
        <div className="spectro-placeholder">
          No spectrogram yet.<br />
          Record a message to generate one.
        </div>
      )}

      {/* Image Viewer */}
      {activeImage && (
        <div className="spectro-image-wrapper">
          <img
            src={`data:image/png;base64,${activeImage}`}
            alt="Spectrogram"
            className="spectro-image"
          />
        </div>
      )}

      {sessionId && (
        <p className="spectro-session">session: {sessionId}</p>
      )}
    </div>
  );
}
