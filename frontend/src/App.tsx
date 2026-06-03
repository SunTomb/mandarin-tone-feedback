import { useState } from "react";
import { analyzeAudio, fetchDemoFeedback } from "./api";
import { FeedbackCard } from "./components/FeedbackCard";
import { Recorder } from "./components/Recorder";
import { ToneChart } from "./components/ToneChart";
import type { ToneFeedbackResponse } from "./types";

export default function App() {
  const [targetTone, setTargetTone] = useState(2);
  const [result, setResult] = useState<ToneFeedbackResponse | null>(null);
  const [resultSource, setResultSource] = useState<"analysis" | "demo">("analysis");
  const [status, setStatus] = useState<string>("等待一次演示、录音或上传后，这里会显示目标声调、F0 曲线诊断和参考预测。");

  async function loadDemo() {
    setStatus("正在生成演示反馈，请稍候…");
    try {
      const nextResult = await fetchDemoFeedback(targetTone);
      setResultSource("demo");
      setResult(nextResult);
      setStatus("演示反馈已更新。");
    } catch {
      setStatus("演示反馈生成失败，请稍后再试。");
    }
  }

  async function uploadFile(file: File) {
    setStatus("正在分析音频，请稍候…");
    setResult(null);
    try {
      const nextResult = await analyzeAudio(targetTone, file);
      setResultSource("analysis");
      setResult(nextResult);
      setStatus("分析完成，结果已更新。");
    } catch {
      setStatus("音频分析失败，请重新录制或上传 WAV 文件。");
    }
  }

  function changeTargetTone(nextTone: number) {
    setTargetTone(nextTone);
    setResult(null);
    setStatus("目标声调已更改，请重新生成演示反馈或重新上传音频。");
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="visual-plane" aria-hidden="true">
          <div>
            <div className="waveform" />
            <div className="visual-label">F0</div>
          </div>
        </div>
        <div className="hero-content">
          <p className="kicker">Mandarin Tone Lab</p>
          <h1>普通话声调 AI 反馈系统</h1>
          <p className="hero-copy">选择目标声调，上传或录制 1–2 秒单字/短词音频，查看 F0 曲线诊断和可解释声学反馈；预测声调仅作为参考。</p>
          <label className="tone-selector">
            目标声调
            <select value={targetTone} onChange={(event) => changeTargetTone(Number(event.target.value))}>
              <option value={1}>一声</option>
              <option value={2}>二声</option>
              <option value={3}>三声</option>
              <option value={4}>四声</option>
            </select>
          </label>
        </div>
      </section>
      <section className="workspace">
        <Recorder onDemo={loadDemo} onFile={uploadFile} />
        <div className="results">
          <div className="status-line" role="status">{status}</div>
          {result ? (
            <>
              <FeedbackCard result={result} source={resultSource} />
              <ToneChart result={result} />
            </>
          ) : null}
        </div>
      </section>
    </main>
  );
}
