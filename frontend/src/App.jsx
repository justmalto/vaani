import React from "react";
import "./App.css";
import VoiceChat from "./components/VoiceChat";
import SpeechTranscriber from "./components/SpeechTranscriber";
import ChatResponse from "./components/ChatResponse";
import { PipelineProvider } from "./pipeline/PipelineWS";
import SpectrogramViewer from "./components/SpectrogramViewer";
import EmotionIndex from "./components/EmotionIndex";

export default function App() {
  return (
    <div className="app-container">
      <div className="left-panel">
        <div className="HeadingFont">VAANI</div>
        <PipelineProvider>
        <div className="Index"><EmotionIndex/></div>
        </PipelineProvider>
        <div className="VC"><VoiceChat/></div>
        <PipelineProvider>
          <div className="SpectoV"><SpectrogramViewer/></div>
        </PipelineProvider>
      </div>
      <PipelineProvider>
        <div className="right-panel">
          <div className="SpeechT"><SpeechTranscriber/></div>
          <div className="ChatR"><ChatResponse/></div>
        </div>
      </PipelineProvider>
    </div>
  );
}
