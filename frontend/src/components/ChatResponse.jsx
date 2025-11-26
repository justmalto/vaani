import React from "react";
import "./ChatResponse.css";
import { usePipeline } from "../pipeline/PipelineWS";

export default function ChatResponse() {
  const { reply, stage } = usePipeline();

  return (
    <div className="chat-container">
      <h2 className="chat-title">💬Response</h2>
      <p className="chat-status">Stage: {stage}</p>

      <div className="chat-reply">
        <h3>🤖 LLM Reply</h3>
        <p>{reply || "No reply yet."}</p>
      </div>
    </div>
  );
}
