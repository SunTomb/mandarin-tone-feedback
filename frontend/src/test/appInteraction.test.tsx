import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { analyzeAudio, fetchDemoFeedback } from "../api";
import App from "../App";
import type { ToneFeedbackResponse } from "../types";

const response: ToneFeedbackResponse = {
  predicted_tone: 2,
  confidence: 0.5,
  target_tone: 2,
  five_level_value: "135",
  user_contour: [{ time: 0, value: 1 }, { time: 1, value: 5 }],
  target_contour: [{ time: 0, value: 2 }, { time: 1, value: 5 }],
  feedback: [{ code: "good_match", message: "声调轮廓与目标较接近。", severity: "info" }]
};

vi.mock("../api", () => ({
  analyzeAudio: vi.fn(),
  fetchDemoFeedback: vi.fn()
}));

const analyzeAudioMock = vi.mocked(analyzeAudio);
const fetchDemoFeedbackMock = vi.mocked(fetchDemoFeedback);

afterEach(() => {
  cleanup();
});

beforeEach(() => {
  analyzeAudioMock.mockReset();
  fetchDemoFeedbackMock.mockReset();
  analyzeAudioMock.mockImplementation(
    () =>
      new Promise<ToneFeedbackResponse>((resolve) => {
        setTimeout(() => resolve(response), 10);
      })
  );
  fetchDemoFeedbackMock.mockResolvedValue(response);
});

test("shows analyzing state after audio upload", async () => {
  render(<App />);

  const file = new File([new Blob(["audio"])], "tone.wav", { type: "audio/wav" });
  fireEvent.change(screen.getByLabelText("上传音频文件"), { target: { files: [file] } });

  expect(screen.getByText("正在分析音频，请稍候…")).toBeTruthy();
  await waitFor(() => expect(screen.getByText("分析完成，结果已更新。")));
});

test("shows a failure state when audio analysis fails", async () => {
  analyzeAudioMock.mockRejectedValueOnce(new Error("Request failed with status 500"));
  render(<App />);

  const file = new File([new Blob(["audio"])], "recording.webm", { type: "audio/webm" });
  fireEvent.change(screen.getByLabelText("上传音频文件"), { target: { files: [file] } });

  expect(screen.getByText("正在分析音频，请稍候…")).toBeTruthy();
  await waitFor(() => expect(screen.getByText("音频分析失败，请重新录制或上传 WAV 文件。")));
});

test("clears previous analysis when target tone changes", async () => {
  render(<App />);

  fireEvent.click(screen.getByText("生成演示反馈"));
  await waitFor(() => expect(screen.getByText("演示反馈已更新。")));
  expect(screen.getByText("F0 曲线诊断")).toBeTruthy();

  fireEvent.change(screen.getByLabelText("目标声调"), { target: { value: "4" } });

  expect(screen.queryByText("F0 曲线诊断")).toBeNull();
  expect(screen.getByText("目标声调已更改，请重新生成演示反馈或重新上传音频。"));
});

test("demo feedback replaces a failed audio status", async () => {
  analyzeAudioMock.mockRejectedValueOnce(new Error("Request failed with status 500"));
  render(<App />);

  const file = new File([new Blob(["audio"])], "recording.webm", { type: "audio/webm" });
  fireEvent.change(screen.getByLabelText("上传音频文件"), { target: { files: [file] } });
  await waitFor(() => expect(screen.getByText("音频分析失败，请重新录制或上传 WAV 文件。")));

  fireEvent.click(screen.getByText("生成演示反馈"));

  expect(screen.getByText("正在生成演示反馈，请稍候…")).toBeTruthy();
  await waitFor(() => expect(screen.getByText("演示反馈已更新。")));
});
