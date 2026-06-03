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


def _edge_means(values: list[float]) -> tuple[float, float, float, float]:
    window = max(1, min(3, len(values) // 4 or 1))
    start = sum(values[:window]) / window
    midpoint = values[len(values) // 2]
    end = sum(values[-window:]) / window
    pitch_range = max(values) - min(values)
    return start, midpoint, end, pitch_range


def predict_tone_from_contour(values: list[float]) -> tuple[int, float]:
    if len(values) < 2:
        return 1, 0.25

    start, midpoint, end, pitch_range = _edge_means(values)
    slope = end - start
    dip_depth = min(start, end) - midpoint
    body = values[:-2] if len(values) >= 5 else values
    body_range = max(body) - min(body)

    if dip_depth >= 0.5 and end >= midpoint + 0.5:
        return 3, min(0.9, 0.45 + dip_depth / 3.0)
    if slope <= -0.9 and body_range <= 0.8:
        return 1, 0.55
    if slope <= -0.9:
        return 4, min(0.9, 0.45 + abs(slope) / 4.0)
    if slope >= 0.8:
        return 2, min(0.9, 0.45 + slope / 4.0)
    if pitch_range <= 1.2:
        return 1, min(0.85, 0.55 + (1.2 - pitch_range) / 4.0)

    if abs(slope) < 0.6:
        return 1, 0.45
    return (2, 0.45) if slope > 0 else (4, 0.45)


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
