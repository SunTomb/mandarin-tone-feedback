import type { ToneFeedbackResponse } from "../types";

type FeedbackSource = "analysis" | "demo";

function confidenceMessage(confidence: number, source: FeedbackSource): string {
  if (source === "demo") {
    return "这是固定演示样例，用于展示界面和反馈格式；请上传或录制音频获得真实 F0 诊断。";
  }
  if (confidence < 0.6) {
    return "参考置信度较低，请优先参考 F0 曲线和具体反馈，不要把预测声调当作最终判断。";
  }
  return "这是基于 F0 轮廓规则的参考判断，请以曲线形状和下方具体反馈为主。";
}

export function FeedbackCard({ result, source = "analysis" }: { result: ToneFeedbackResponse; source?: FeedbackSource }) {
  return (
    <section className="feedback-card">
      <h2>F0 曲线诊断</h2>
      <p className="reference-note">预测仅供参考</p>
      <div className="metric-row">
        <div><span>目标声调</span><strong>{result.target_tone} 声</strong></div>
        <div><span>参考预测</span><strong>{result.predicted_tone} 声</strong></div>
        <div><span>{source === "demo" ? "演示置信度" : "参考置信度"}</span><strong>{Math.round(result.confidence * 100)}%</strong></div>
        <div><span>五度调值</span><strong>{result.five_level_value}</strong></div>
      </div>
      <p className={result.confidence < 0.6 && source !== "demo" ? "confidence-note low" : "confidence-note"}>{confidenceMessage(result.confidence, source)}</p>
      <ul>
        {result.feedback.map((item) => (
          <li key={item.code} data-severity={item.severity}>{item.message}</li>
        ))}
      </ul>
    </section>
  );
}
