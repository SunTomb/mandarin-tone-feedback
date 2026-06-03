import { Line, LineChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ToneFeedbackResponse } from "../types";

export function ToneChart({ result }: { result: ToneFeedbackResponse }) {
  const data = result.user_contour.map((point, index) => ({
    time: point.time,
    user: point.value,
    target: result.target_contour[index]?.value
  }));

  return (
    <section className="tone-chart">
      <h2>F0 / 五度曲线</h2>
      <ResponsiveContainer width="100%" height={245}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis domain={[1, 5]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="user" name="你的曲线" stroke="#2563eb" strokeWidth={3} />
          <Line type="monotone" dataKey="target" name="目标曲线" stroke="#dc2626" strokeWidth={3} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
