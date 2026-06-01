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
