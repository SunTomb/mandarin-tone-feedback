# Mandarin Tone Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Mandarin tone-learning feedback prototype with offline model experiments and paper-ready figures.

**Architecture:** Use a Python backend for audio processing, tone inference, and feedback generation; a React frontend for recording/upload and visualization; and separate experiment scripts for acoustic, pretrained-representation, and fusion model comparisons. Keep web inference lightweight while using A40/A100 resources for offline feature extraction, fine-tuning, and representation analysis.

**Tech Stack:** Python 3.11+, FastAPI, librosa or parselmouth, scikit-learn, PyTorch, transformers, React, TypeScript, Vite, Recharts or ECharts.

---

## File structure to create

```text
mandarin-tone-feedback/
  CLAUDE.md
  design_spec.md
  implementation_plan.md
  README.md
  data/
    raw/
    processed/
    metadata.example.csv
  experiments/
    configs/
      acoustic_baseline.yaml
      representation_model.yaml
      fusion_model.yaml
    train_acoustic_baseline.py
    train_representation_model.py
    train_fusion_model.py
    analyze_representations.py
    tests/
      test_dataset_loading.py
      test_metrics.py
  backend/
    pyproject.toml
    app/
      main.py
      schemas.py
      audio.py
      features.py
      inference.py
      feedback.py
    tests/
      test_features.py
      test_feedback.py
      test_api.py
  frontend/
    package.json
    index.html
    src/
      App.tsx
      api.ts
      types.ts
      components/
        Recorder.tsx
        ToneChart.tsx
        FeedbackCard.tsx
      test/
        feedbackRendering.test.tsx
  paper/
    outline.md
    figures/
```

## Implementation rules for future agents

- Do not use PowerShell commands. The user explicitly warned that PowerShell can trigger CC Switch / Claude Code stack leakage.
- Prefer file tools for reading and writing files.
- If a shell command is necessary, use a non-PowerShell shell only after checking the active environment and user permission.
- Keep the first working version small: four tones, short audio, no accounts, no full ASR.
- Use tests before implementation for pure functions and API contracts.
- Commit only if the user explicitly asks.

## Task 1: Scaffold repository files

**Files:**
- Create: `README.md`
- Create: `data/metadata.example.csv`
- Create: `backend/pyproject.toml`
- Create: `frontend/package.json`
- Create: `paper/outline.md`

- [ ] **Step 1: Create README**

Create `README.md` with:

```markdown
# Mandarin Tone Feedback

A hybrid language-science project for Mandarin tone learning. The system records or uploads short Mandarin audio, predicts the lexical tone, visualizes the normalized F0 contour, compares it with a target contour, and returns interpretable feedback.

## Project goals

1. Build a runnable web demo for Mandarin tone feedback.
2. Compare acoustic features, pretrained speech representations, and fusion models for tone classification.
3. Produce figures and results for a 3-9 page language-science final paper.

## Scope

The first version supports isolated syllables and short words with four Mandarin lexical tones. It does not support full ASR, sentence-level pronunciation scoring, accounts, or long-term learning records.

## Main directories

- `backend/`: FastAPI app for audio processing, inference, and feedback.
- `frontend/`: React app for recording, upload, and visualization.
- `experiments/`: offline model training and analysis scripts.
- `paper/`: paper outline and generated figures.
- `data/`: local datasets and example metadata.
```

- [ ] **Step 2: Create metadata example**

Create `data/metadata.example.csv` with:

```csv
audio_path,text,pinyin,tone,speaker_id,split
data/raw/ma1_speaker01.wav,妈,ma1,1,speaker01,train
data/raw/ma2_speaker01.wav,麻,ma2,2,speaker01,train
data/raw/ma3_speaker01.wav,马,ma3,3,speaker01,val
data/raw/ma4_speaker01.wav,骂,ma4,4,speaker01,test
```

- [ ] **Step 3: Create backend pyproject**

Create `backend/pyproject.toml` with:

```toml
[project]
name = "mandarin-tone-feedback-backend"
version = "0.1.0"
description = "Backend API for Mandarin tone feedback"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "pydantic>=2.6",
  "numpy>=1.26",
  "librosa>=0.10",
  "soundfile>=0.12",
  "scikit-learn>=1.4",
  "python-multipart>=0.0.9"
]

[project.optional-dependencies]
ml = [
  "torch>=2.2",
  "transformers>=4.40",
  "umap-learn>=0.5",
  "matplotlib>=3.8",
  "pandas>=2.2"
]
test = [
  "pytest>=8.0",
  "httpx>=0.27"
]
```

- [ ] **Step 4: Create frontend package file**

Create `frontend/package.json` with:

```json
{
  "name": "mandarin-tone-feedback-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "latest",
    "vite": "latest",
    "typescript": "latest",
    "react": "latest",
    "react-dom": "latest",
    "recharts": "latest"
  },
  "devDependencies": {
    "@types/react": "latest",
    "@types/react-dom": "latest",
    "vitest": "latest",
    "@testing-library/react": "latest",
    "@testing-library/jest-dom": "latest",
    "jsdom": "latest"
  }
}
```

- [ ] **Step 5: Create paper outline**

Create `paper/outline.md` with:

```markdown
# Paper Outline

## Proposed Chinese title

面向普通话声调学习的可解释 AI 反馈系统

## Abstract points

- Mandarin tone learning requires both category recognition and contour-level production control.
- This project combines F0-based acoustic analysis with pretrained speech representations.
- A web prototype provides tone prediction, F0 visualization, target-contour comparison, and interpretable feedback.
- Experiments compare acoustic features, pretrained speech representations, and a fusion model.

## Sections

1. Introduction
2. Linguistic background: tone category, tone value, F0 contour, speaker normalization
3. System design
4. Model comparison
5. Feedback examples and representation analysis
6. Conclusion
```

- [ ] **Step 6: Verify files exist**

Use file-reading or directory-listing tools to confirm all created files exist. Expected: each file listed above is present.

## Task 2: Define backend schemas and feedback rules with tests

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/feedback.py`
- Create: `backend/tests/test_feedback.py`

- [ ] **Step 1: Write feedback tests**

Create `backend/tests/test_feedback.py` with:

```python
from app.feedback import generate_feedback
from app.schemas import ContourPoint, ToneAnalysis


def contour(values):
    return [ContourPoint(time=i / (len(values) - 1), value=value) for i, value in enumerate(values)]


def test_tone_two_insufficient_rise_feedback():
    analysis = ToneAnalysis(
        predicted_tone=2,
        confidence=0.82,
        target_tone=2,
        user_contour=contour([2.5, 2.6, 2.7]),
        target_contour=contour([2.0, 3.0, 4.0]),
        five_level_value="23",
    )

    feedback = generate_feedback(analysis)

    assert any(item.code == "insufficient_rise" for item in feedback)
    assert any("上升" in item.message for item in feedback)


def test_tone_four_insufficient_fall_feedback():
    analysis = ToneAnalysis(
        predicted_tone=4,
        confidence=0.91,
        target_tone=4,
        user_contour=contour([4.5, 4.2, 4.0]),
        target_contour=contour([5.0, 3.0, 1.0]),
        five_level_value="43",
    )

    feedback = generate_feedback(analysis)

    assert any(item.code == "insufficient_fall" for item in feedback)
    assert any("下降" in item.message for item in feedback)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `backend/`: `python -m pytest tests/test_feedback.py -v`

Expected: FAIL because `app.feedback` and `app.schemas` do not exist yet.

- [ ] **Step 3: Implement schemas**

Create `backend/app/schemas.py` with:

```python
from pydantic import BaseModel, Field


class ContourPoint(BaseModel):
    time: float = Field(ge=0.0, le=1.0)
    value: float


class FeedbackItem(BaseModel):
    code: str
    message: str
    severity: str = "info"


class ToneAnalysis(BaseModel):
    predicted_tone: int = Field(ge=1, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    target_tone: int = Field(ge=1, le=4)
    user_contour: list[ContourPoint]
    target_contour: list[ContourPoint]
    five_level_value: str


class ToneFeedbackResponse(ToneAnalysis):
    feedback: list[FeedbackItem]
```

- [ ] **Step 4: Implement feedback rules**

Create `backend/app/feedback.py` with:

```python
from app.schemas import FeedbackItem, ToneAnalysis


def _values(analysis: ToneAnalysis) -> list[float]:
    return [point.value for point in analysis.user_contour]


def generate_feedback(analysis: ToneAnalysis) -> list[FeedbackItem]:
    values = _values(analysis)
    if len(values) < 2:
        return [FeedbackItem(code="short_contour", message="音高曲线太短，建议重新录音。", severity="warning")]

    start = values[0]
    end = values[-1]
    pitch_range = max(values) - min(values)
    feedback: list[FeedbackItem] = []

    if analysis.target_tone == 1 and pitch_range > 0.8:
        feedback.append(FeedbackItem(code="unstable_level", message="一声音高应较平稳，你的音高波动偏大。", severity="warning"))

    if analysis.target_tone == 2 and end - start < 1.0:
        feedback.append(FeedbackItem(code="insufficient_rise", message="二声需要明显上升，你的上升幅度不足。", severity="warning"))

    if analysis.target_tone == 3:
        midpoint = values[len(values) // 2]
        if midpoint > min(start, end) - 0.2:
            feedback.append(FeedbackItem(code="unclear_dip", message="三声通常有较低的转折点，你的低凹形状不够明显。", severity="warning"))

    if analysis.target_tone == 4 and start - end < 1.2:
        feedback.append(FeedbackItem(code="insufficient_fall", message="四声需要明显下降，你的下降幅度不足。", severity="warning"))

    if analysis.predicted_tone != analysis.target_tone:
        feedback.append(FeedbackItem(code="category_mismatch", message=f"模型预测为 {analysis.predicted_tone} 声，与目标 {analysis.target_tone} 声不一致。", severity="warning"))

    if not feedback:
        feedback.append(FeedbackItem(code="good_match", message="声调轮廓与目标较接近。", severity="success"))

    return feedback
```

- [ ] **Step 5: Run feedback tests**

Run from `backend/`: `python -m pytest tests/test_feedback.py -v`

Expected: PASS.

## Task 3: Implement F0 feature extraction with tests

**Files:**
- Create: `backend/app/features.py`
- Create: `backend/tests/test_features.py`

- [ ] **Step 1: Write feature tests**

Create `backend/tests/test_features.py` with:

```python
import numpy as np

from app.features import five_level_value, normalize_f0_values, summarize_contour


def test_normalize_f0_values_maps_pitch_to_one_to_five():
    values = np.array([100.0, 150.0, 200.0])

    normalized = normalize_f0_values(values)

    assert normalized.tolist() == [1.0, 3.0, 5.0]


def test_five_level_value_uses_start_middle_end():
    values = [1.0, 3.0, 5.0]

    result = five_level_value(values)

    assert result == "135"


def test_summarize_contour_returns_expected_keys():
    values = [1.0, 2.0, 4.0]

    summary = summarize_contour(values, duration=0.5)

    assert summary["f0_start"] == 1.0
    assert summary["f0_mid"] == 2.0
    assert summary["f0_end"] == 4.0
    assert summary["f0_range"] == 3.0
    assert summary["duration"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run from `backend/`: `python -m pytest tests/test_features.py -v`

Expected: FAIL because `app.features` does not exist.

- [ ] **Step 3: Implement feature utilities**

Create `backend/app/features.py` with:

```python
import numpy as np


def normalize_f0_values(values: np.ndarray) -> np.ndarray:
    voiced = values[np.isfinite(values)]
    if voiced.size == 0:
        return np.array([], dtype=float)
    low = float(np.min(voiced))
    high = float(np.max(voiced))
    if high == low:
        return np.full_like(voiced, 3.0, dtype=float)
    return 1.0 + 4.0 * (voiced - low) / (high - low)


def five_level_value(values: list[float]) -> str:
    if not values:
        return ""
    indices = [0, len(values) // 2, len(values) - 1]
    return "".join(str(int(round(values[index]))) for index in indices)


def summarize_contour(values: list[float], duration: float) -> dict[str, float]:
    if not values:
        return {
            "f0_start": 0.0,
            "f0_mid": 0.0,
            "f0_end": 0.0,
            "f0_range": 0.0,
            "f0_slope": 0.0,
            "duration": duration,
            "voiced_ratio": 0.0,
        }
    start = float(values[0])
    mid = float(values[len(values) // 2])
    end = float(values[-1])
    return {
        "f0_start": start,
        "f0_mid": mid,
        "f0_end": end,
        "f0_range": float(max(values) - min(values)),
        "f0_slope": end - start,
        "duration": duration,
        "voiced_ratio": 1.0,
    }
```

- [ ] **Step 4: Run feature tests**

Run from `backend/`: `python -m pytest tests/test_features.py -v`

Expected: PASS.

## Task 4: Implement backend API contract

**Files:**
- Create: `backend/app/inference.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write API test**

Create `backend/tests/test_api.py` with:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_demo_feedback_endpoint_returns_feedback():
    client = TestClient(app)

    response = client.post("/api/demo-feedback", json={"target_tone": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_tone"] == 2
    assert 1 <= payload["predicted_tone"] <= 4
    assert payload["feedback"]
    assert payload["user_contour"]
    assert payload["target_contour"]
```

- [ ] **Step 2: Run API test to verify it fails**

Run from `backend/`: `python -m pytest tests/test_api.py -v`

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement deterministic demo inference**

Create `backend/app/inference.py` with:

```python
from app.features import five_level_value
from app.feedback import generate_feedback
from app.schemas import ContourPoint, ToneAnalysis, ToneFeedbackResponse


TARGET_CONTOURS = {
    1: [5.0, 5.0, 5.0],
    2: [2.0, 3.0, 5.0],
    3: [2.0, 1.0, 3.0],
    4: [5.0, 3.0, 1.0],
}

DEMO_USER_CONTOURS = {
    1: [4.8, 4.7, 4.9],
    2: [2.4, 2.7, 3.1],
    3: [2.2, 2.1, 2.8],
    4: [4.7, 4.2, 3.8],
}


def _points(values: list[float]) -> list[ContourPoint]:
    if len(values) == 1:
        return [ContourPoint(time=0.0, value=values[0])]
    return [ContourPoint(time=index / (len(values) - 1), value=value) for index, value in enumerate(values)]


def demo_feedback(target_tone: int) -> ToneFeedbackResponse:
    user_values = DEMO_USER_CONTOURS[target_tone]
    target_values = TARGET_CONTOURS[target_tone]
    analysis = ToneAnalysis(
        predicted_tone=target_tone,
        confidence=0.86,
        target_tone=target_tone,
        user_contour=_points(user_values),
        target_contour=_points(target_values),
        five_level_value=five_level_value(user_values),
    )
    return ToneFeedbackResponse(**analysis.model_dump(), feedback=generate_feedback(analysis))
```

- [ ] **Step 4: Implement FastAPI app**

Create `backend/app/main.py` with:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.inference import demo_feedback
from app.schemas import ToneFeedbackResponse


app = FastAPI(title="Mandarin Tone Feedback API")


class DemoFeedbackRequest(BaseModel):
    target_tone: int = Field(ge=1, le=4)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/demo-feedback", response_model=ToneFeedbackResponse)
def create_demo_feedback(request: DemoFeedbackRequest) -> ToneFeedbackResponse:
    return demo_feedback(request.target_tone)
```

- [ ] **Step 5: Run backend tests**

Run from `backend/`: `python -m pytest tests -v`

Expected: all backend tests PASS.

## Task 5: Implement frontend UI against demo API

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/components/FeedbackCard.tsx`
- Create: `frontend/src/components/ToneChart.tsx`
- Create: `frontend/src/components/Recorder.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/test/feedbackRendering.test.tsx`

- [ ] **Step 1: Write rendering test**

Create `frontend/src/test/feedbackRendering.test.tsx` with:

```tsx
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
```

- [ ] **Step 2: Run frontend test to verify it fails**

Run from `frontend/`: `npm test`

Expected: FAIL because frontend source files do not exist.

- [ ] **Step 3: Implement types**

Create `frontend/src/types.ts` with:

```ts
export type ContourPoint = {
  time: number;
  value: number;
};

export type FeedbackItem = {
  code: string;
  message: string;
  severity: "info" | "warning" | "success" | string;
};

export type ToneFeedbackResponse = {
  predicted_tone: number;
  confidence: number;
  target_tone: number;
  user_contour: ContourPoint[];
  target_contour: ContourPoint[];
  five_level_value: string;
  feedback: FeedbackItem[];
};
```

- [ ] **Step 4: Implement API client**

Create `frontend/src/api.ts` with:

```ts
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
```

- [ ] **Step 5: Implement feedback card**

Create `frontend/src/components/FeedbackCard.tsx` with:

```tsx
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
```

- [ ] **Step 6: Implement chart**

Create `frontend/src/components/ToneChart.tsx` with:

```tsx
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
      <ResponsiveContainer width="100%" height={280}>
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
```

- [ ] **Step 7: Implement recorder placeholder**

Create `frontend/src/components/Recorder.tsx` with:

```tsx
export function Recorder({ onDemo }: { onDemo: () => void }) {
  return (
    <section className="recorder">
      <h2>录音 / 上传</h2>
      <p>第一版使用演示接口验证反馈闭环；后续任务接入真实浏览器录音和音频上传。</p>
      <button onClick={onDemo}>生成演示反馈</button>
    </section>
  );
}
```

- [ ] **Step 8: Implement app**

Create `frontend/src/App.tsx` with:

```tsx
import { useState } from "react";
import { fetchDemoFeedback } from "./api";
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
      <Recorder onDemo={loadDemo} />
      {result && <FeedbackCard result={result} />}
      {result && <ToneChart result={result} />}
    </main>
  );
}
```

- [ ] **Step 9: Create index page**

Create `frontend/index.html` with:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mandarin Tone Feedback</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/App.tsx"></script>
  </body>
</html>
```

- [ ] **Step 10: Run frontend tests**

Run from `frontend/`: `npm test`

Expected: PASS.

## Task 6: Add real audio upload pipeline

**Files:**
- Modify: `backend/app/audio.py`
- Modify: `backend/app/features.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_api.py`
- Modify: `frontend/src/components/Recorder.tsx`
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Add backend audio utility**

Create `backend/app/audio.py` with:

```python
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf


def load_audio_bytes(content: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int, float]:
    data, sr = sf.read(BytesIO(content), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    duration = float(len(data) / sr) if sr else 0.0
    return data, sr, duration
```

- [ ] **Step 2: Add F0 extraction utility**

Extend `backend/app/features.py` with:

```python
import librosa


def extract_normalized_contour(audio: np.ndarray, sr: int) -> list[float]:
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    if f0 is None:
        return []
    normalized = normalize_f0_values(f0)
    return [float(value) for value in normalized if np.isfinite(value)]
```

- [ ] **Step 3: Add API upload endpoint**

Extend `backend/app/main.py` with:

```python
from fastapi import File, Form, UploadFile
from app.audio import load_audio_bytes
from app.features import extract_normalized_contour, five_level_value
from app.feedback import generate_feedback
from app.inference import TARGET_CONTOURS
from app.schemas import ContourPoint, ToneAnalysis, ToneFeedbackResponse


def contour_points(values: list[float]) -> list[ContourPoint]:
    if len(values) <= 1:
        return [ContourPoint(time=0.0, value=values[0])] if values else []
    return [ContourPoint(time=index / (len(values) - 1), value=value) for index, value in enumerate(values)]


@app.post("/api/analyze", response_model=ToneFeedbackResponse)
async def analyze_audio(target_tone: int = Form(ge=1, le=4), file: UploadFile = File(...)) -> ToneFeedbackResponse:
    content = await file.read()
    audio, sr, _ = load_audio_bytes(content)
    user_values = extract_normalized_contour(audio, sr)
    if not user_values:
        user_values = [3.0]
    target_values = TARGET_CONTOURS[target_tone]
    analysis = ToneAnalysis(
        predicted_tone=target_tone,
        confidence=0.50,
        target_tone=target_tone,
        user_contour=contour_points(user_values),
        target_contour=contour_points(target_values),
        five_level_value=five_level_value(user_values),
    )
    return ToneFeedbackResponse(**analysis.model_dump(), feedback=generate_feedback(analysis))
```

- [ ] **Step 4: Add frontend upload API**

Extend `frontend/src/api.ts` with:

```ts
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
```

- [ ] **Step 5: Replace recorder placeholder with upload control**

Modify `frontend/src/components/Recorder.tsx` to:

```tsx
export function Recorder({ onDemo, onFile }: { onDemo: () => void; onFile: (file: File) => void }) {
  return (
    <section className="recorder">
      <h2>录音 / 上传</h2>
      <p>上传一个短音频文件，或使用演示接口查看反馈闭环。</p>
      <input
        type="file"
        accept="audio/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      <button onClick={onDemo}>生成演示反馈</button>
    </section>
  );
}
```

- [ ] **Step 6: Wire upload in App**

Modify `frontend/src/App.tsx` imports and recorder usage:

```tsx
import { analyzeAudio, fetchDemoFeedback } from "./api";
```

Add function inside `App`:

```tsx
async function uploadFile(file: File) {
  const nextResult = await analyzeAudio(targetTone, file);
  setResult(nextResult);
}
```

Replace recorder usage with:

```tsx
<Recorder onDemo={loadDemo} onFile={uploadFile} />
```

- [ ] **Step 7: Run tests**

Run backend and frontend tests. Expected: all tests PASS.

## Task 7: Implement offline acoustic baseline experiment

**Files:**
- Create: `experiments/train_acoustic_baseline.py`
- Create: `experiments/tests/test_dataset_loading.py`

- [ ] **Step 1: Write dataset loading test**

Create `experiments/tests/test_dataset_loading.py` with:

```python
from pathlib import Path

from train_acoustic_baseline import load_metadata


def test_load_metadata_reads_rows(tmp_path: Path):
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("audio_path,text,pinyin,tone,speaker_id,split\na.wav,妈,ma1,1,s1,train\n", encoding="utf-8")

    rows = load_metadata(csv_path)

    assert len(rows) == 1
    assert rows[0]["tone"] == "1"
    assert rows[0]["pinyin"] == "ma1"
```

- [ ] **Step 2: Implement acoustic baseline script**

Create `experiments/train_acoustic_baseline.py` with:

```python
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def train_baseline(features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model


def evaluate(model: RandomForestClassifier, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = model.predict(features)
    return {
        "report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


if __name__ == "__main__":
    print("Prepare extracted acoustic feature arrays before running the full baseline experiment.")
```

- [ ] **Step 3: Run experiment tests**

Run from `experiments/`: `python -m pytest tests -v`

Expected: PASS.

## Task 8: Implement pretrained and fusion experiment skeletons

**Files:**
- Create: `experiments/train_representation_model.py`
- Create: `experiments/train_fusion_model.py`
- Create: `experiments/analyze_representations.py`

- [ ] **Step 1: Create representation script**

Create `experiments/train_representation_model.py` with:

```python
from dataclasses import dataclass


@dataclass
class RepresentationConfig:
    model_name: str = "facebook/wav2vec2-base"
    pooling: str = "mean"
    num_labels: int = 4


def describe_experiment(config: RepresentationConfig) -> str:
    return f"Extract {config.pooling}-pooled hidden states from {config.model_name} for {config.num_labels}-way tone classification."


if __name__ == "__main__":
    print(describe_experiment(RepresentationConfig()))
```

- [ ] **Step 2: Create fusion script**

Create `experiments/train_fusion_model.py` with:

```python
import numpy as np


def concatenate_features(acoustic: np.ndarray, representation: np.ndarray) -> np.ndarray:
    if acoustic.shape[0] != representation.shape[0]:
        raise ValueError("Feature arrays must have the same number of rows.")
    return np.concatenate([acoustic, representation], axis=1)


if __name__ == "__main__":
    print("Train a classifier on concatenated acoustic and pretrained speech representation features.")
```

- [ ] **Step 3: Create representation analysis script**

Create `experiments/analyze_representations.py` with:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def tsne_2d(features: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(2, features.shape[0] // 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(features)


def save_scatter(points: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=16)
    plt.colorbar(label="Tone")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    print("Load hidden-state features, run t-SNE, and save a tone separability figure.")
```

- [ ] **Step 4: Add tests for fusion utility**

Create `experiments/tests/test_metrics.py` with:

```python
import numpy as np
import pytest

from train_fusion_model import concatenate_features


def test_concatenate_features_combines_columns():
    acoustic = np.ones((2, 3))
    representation = np.zeros((2, 4))

    result = concatenate_features(acoustic, representation)

    assert result.shape == (2, 7)


def test_concatenate_features_rejects_mismatched_rows():
    acoustic = np.ones((2, 3))
    representation = np.zeros((3, 4))

    with pytest.raises(ValueError):
        concatenate_features(acoustic, representation)
```

- [ ] **Step 5: Run experiment tests**

Run from `experiments/`: `python -m pytest tests -v`

Expected: PASS.

## Task 9: Create paper-ready figures and result placeholders from real outputs

**Files:**
- Modify: `paper/outline.md`
- Create: `paper/figures/README.md`

- [ ] **Step 1: Create figures README**

Create `paper/figures/README.md` with:

```markdown
# Paper Figures

Required figures for the final paper:

1. System workflow diagram or UI screenshot.
2. F0 contour example comparing learner and target tone.
3. Model comparison table or confusion matrix.
4. Optional representation visualization showing tone-category separation.

Do not invent numbers. Tables and figures must be generated from actual experiment outputs.
```

- [ ] **Step 2: Extend paper outline with concrete result slots**

Append to `paper/outline.md`:

```markdown
## Result slots to fill from experiments

- Acoustic baseline accuracy / macro-F1:
- Pretrained representation accuracy / macro-F1:
- Fusion model accuracy / macro-F1:
- Main confusion pair:
- Best feedback example:
- Main limitation:
```

- [ ] **Step 3: Verify no fabricated results are present**

Read `paper/outline.md` and `paper/figures/README.md`. Expected: they contain result slots and figure requirements, but no invented accuracy numbers.

## Task 10: Manual verification and final handoff

**Files:**
- Modify as needed based on test failures only.

- [ ] **Step 1: Run backend tests**

Run from `backend/`: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 2: Run frontend tests**

Run from `frontend/`: `npm test`

Expected: PASS.

- [ ] **Step 3: Run experiment tests**

Run from `experiments/`: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 4: Launch backend and frontend for manual UI check**

Start the backend with a non-PowerShell shell:

```bash
python -m uvicorn app.main:app --reload
```

Start the frontend with a non-PowerShell shell:

```bash
npm run dev
```

Open the local Vite URL shown by the dev server. Select each of the four tones and click the demo feedback button. Expected: every tone returns a feedback card and a chart.

- [ ] **Step 5: Check UI golden path**

Expected manual behavior:

1. The page loads with the Chinese title.
2. The user can select tones 1-4.
3. The demo feedback button returns prediction, confidence, five-level value, feedback, and chart.
4. Uploaded audio returns a feedback result after Task 6 is implemented.
5. No browser console errors appear during the golden path.

## Self-review checklist

- Spec coverage: the plan covers backend API, F0 utilities, feedback rules, frontend visualization, offline experiments, GPU-relevant representation analysis, and paper artifacts.
- Placeholder scan: result slots exist only in the paper outline and explicitly require real experiment outputs; implementation tasks contain concrete code and commands.
- Scope control: the plan excludes accounts, full ASR, sentence scoring, and production A100 inference.
- Type consistency: backend schema names and frontend response field names use the same snake_case JSON fields.
