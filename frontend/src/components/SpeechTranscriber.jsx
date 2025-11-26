import React from "react";
import "./SpeechTranscriber.css";
import { usePipeline } from "../pipeline/PipelineWS";

export default function SpeechTranscriber() {
  const { transcript, emotion, stage } = usePipeline();

  return (
    <div className="speech-container">
      <h2 className="speech-title">🗣️</h2>
      <p className="speech-status">Stage: {stage}</p>

      <div className="speech-box">
        <h3>Transcript</h3>
        <p>{transcript || "..."}</p>
      </div>
    </div>
  );
}
