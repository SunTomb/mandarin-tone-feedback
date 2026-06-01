import type { ToneFeedbackResponse } from "../types";

export function FeedbackCard({ result }: { result: ToneFeedbackResponse }) {
  return (
    <section className="feedback-card">
      <h2>声调反馈</h2>
      <p>目标：{result.target_tone} 声</p>
      <p>预测：{result.predicted_tone} 声</p>
      <p>置信度：{Math.round(result.confidence * 100)}%</p>
      <p>五度调值：{result.five_level_value}</p>
      <ul>
        {result.feedback.map((item) => (
          <li key={item.code} data-severity={item.severity}>{item.message}</li>
        ))}
      </ul>
    </section>
  );
}
