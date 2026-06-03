import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
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

test("renders tone feedback as reference-only contour diagnosis", () => {
  render(<FeedbackCard result={response} />);

  expect(screen.getByText("F0 曲线诊断")).toBeTruthy();
  expect(screen.getByText("预测仅供参考")).toBeTruthy();
  expect(screen.getByText("参考置信度")).toBeTruthy();
  expect(screen.getByText("这是基于 F0 轮廓规则的参考判断，请以曲线形状和下方具体反馈为主。")).toBeTruthy();
  expect(screen.getByText("五度调值")).toBeTruthy();
  expect(screen.getByText("234")).toBeTruthy();
  expect(screen.getByText("二声需要明显上升，你的上升幅度不足。")).toBeTruthy();
});

test("labels demo feedback as a fixed example instead of real detection confidence", () => {
  render(<FeedbackCard result={response} source="demo" />);

  expect(screen.getByText("演示置信度")).toBeTruthy();
  expect(screen.getByText("这是固定演示样例，用于展示界面和反馈格式；请上传或录制音频获得真实 F0 诊断。"));
});

test("warns when confidence is low", () => {
  render(<FeedbackCard result={{ ...response, confidence: 0.42 }} />);

  expect(screen.getByText("参考置信度较低，请优先参考 F0 曲线和具体反馈，不要把预测声调当作最终判断。")).toBeTruthy();
});
