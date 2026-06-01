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
