import { render, screen } from "@testing-library/react";
import { FeedbackCard } from "../components/FeedbackCard";
import type { ToneFeedbackResponse } from "../types";

const response: ToneFeedbackResponse = {
  predicted_tone: 2,
  confidence: 0.86,
  target_tone: 2,
  five_level_value: "234",
  user_contour: [{ time: 0, value: 2 }, { time: 1, value: 4 }],
  target_contour: [{ time: 0, value: 2 }, { time: 1, value: 5 }],
  feedback: [{ code: "insufficient_rise", message: "二声需要明显上升，你的上升幅度不足。", severity: "warning" }]
};

test("renders tone feedback", () => {
  render(<FeedbackCard result={response} />);

  expect(screen.getByText("预测：2 声")).toBeTruthy();
  expect(screen.getByText("五度调值：234")).toBeTruthy();
  expect(screen.getByText("二声需要明显上升，你的上升幅度不足。")).toBeTruthy();
});
