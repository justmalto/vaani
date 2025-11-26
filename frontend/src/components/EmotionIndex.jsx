// EmotionIndex.jsx
import React from "react";
import "./EmotionIndex.css";
import { usePipeline } from "../pipeline/PipelineWS"; // <-- adjust path if needed

const EMOTION_COLORS = {
  Angry: "rgb(255, 0, 0)",
  Disgust: "rgb(34, 139, 34)",
  Fear: "rgb(138, 43, 226)",
  Happy: "rgb(255, 215, 0)",
  Neutral: "rgb(200, 200, 200)",
  Pleasant_surprise: "rgb(255, 140, 0)",
  Sad: "rgb(70, 130, 180)",
};

export default function EmotionIndex() {
  const { emotion } = usePipeline();     // <-- gets your emotion like "Neutral"

  return (
    <div className="emotion-index-container">
      {Object.entries(EMOTION_COLORS).map(([name, rgb]) => {
        const isActive = emotion === name;

        return (
          <div key={name} className={`emotion-item ${isActive ? "active" : ""}`}>
            <span className="emotion-name">{name}</span>
            <div
              className="emotion-color-box"
              style={{ backgroundColor: rgb }}
            />
          </div>
        );
      })}
    </div>
  );
}
