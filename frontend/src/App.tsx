import { useState } from "react";
import { analyzeAudio, fetchDemoFeedback } from "./api";
import { FeedbackCard } from "./components/FeedbackCard";
import { Recorder } from "./components/Recorder";
import { ToneChart } from "./components/ToneChart";
import type { ToneFeedbackResponse } from "./types";

export default function App() {
  const [targetTone, setTargetTone] = useState(2);
  const [result, setResult] = useState<ToneFeedbackResponse | null>(null);

  async function loadDemo() {
    const nextResult = await fetchDemoFeedback(targetTone);
    setResult(nextResult);
  }

  async function uploadFile(file: File) {
    const nextResult = await analyzeAudio(targetTone, file);
    setResult(nextResult);
  }

  return (
    <main className="app-shell">
      <h1>普通话声调 AI 反馈系统</h1>
      <p>选择目标声调，查看模型预测、F0 曲线和可解释反馈。</p>
      <label>
        目标声调
        <select value={targetTone} onChange={(event) => setTargetTone(Number(event.target.value))}>
          <option value={1}>一声</option>
          <option value={2}>二声</option>
          <option value={3}>三声</option>
          <option value={4}>四声</option>
        </select>
      </label>
      <Recorder onDemo={loadDemo} onFile={uploadFile} />
      {result && <FeedbackCard result={result} />}
      {result && <ToneChart result={result} />}
    </main>
  );
}
