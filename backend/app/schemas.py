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
