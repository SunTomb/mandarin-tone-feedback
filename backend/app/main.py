from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.audio import load_audio_bytes
from app.features import extract_normalized_contour, five_level_value
from app.feedback import generate_feedback
from app.inference import TARGET_CONTOURS, demo_feedback, predict_tone_from_contour
from app.schemas import ContourPoint, ToneAnalysis, ToneFeedbackResponse


app = FastAPI(title="Mandarin Tone Feedback API")
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class DemoFeedbackRequest(BaseModel):
    target_tone: int = Field(ge=1, le=4)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/demo-feedback", response_model=ToneFeedbackResponse)
def create_demo_feedback(request: DemoFeedbackRequest) -> ToneFeedbackResponse:
    return demo_feedback(request.target_tone)


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
        user_values = [3.0, 3.0, 3.0]
    target_values = TARGET_CONTOURS[target_tone]
    predicted_tone, confidence = predict_tone_from_contour(user_values)
    analysis = ToneAnalysis(
        predicted_tone=predicted_tone,
        confidence=confidence,
        target_tone=target_tone,
        user_contour=contour_points(user_values),
        target_contour=contour_points(target_values),
        five_level_value=five_level_value(user_values),
    )
    return ToneFeedbackResponse(**analysis.model_dump(), feedback=generate_feedback(analysis))


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
