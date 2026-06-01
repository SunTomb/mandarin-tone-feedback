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
