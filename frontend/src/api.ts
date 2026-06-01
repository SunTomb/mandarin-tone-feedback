import type { ToneFeedbackResponse } from "./types";

export async function fetchDemoFeedback(targetTone: number): Promise<ToneFeedbackResponse> {
  const response = await fetch("/api/demo-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_tone: targetTone })
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function analyzeAudio(targetTone: number, file: File): Promise<ToneFeedbackResponse> {
  const body = new FormData();
  body.append("target_tone", String(targetTone));
  body.append("file", file);

  const response = await fetch("/api/analyze", {
    method: "POST",
    body
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}
