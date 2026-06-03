import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { Recorder } from "../components/Recorder";

test("renders upload and demo fallback when recording is unavailable", () => {
  const onDemo = vi.fn();
  const onFile = vi.fn();

  render(<Recorder onDemo={onDemo} onFile={onFile} />);

  expect(screen.getByText("录音 / 上传")).toBeTruthy();
  expect(screen.getByText("当前浏览器不支持直接录音，请使用音频文件上传。"));
  fireEvent.click(screen.getByText("生成演示反馈"));
  expect(onDemo).toHaveBeenCalledTimes(1);
  expect(onFile).not.toHaveBeenCalled();
});
