from app.schemas import FeedbackItem, ToneAnalysis


def _values(analysis: ToneAnalysis) -> list[float]:
    return [point.value for point in analysis.user_contour]


def _edge_means(values: list[float]) -> tuple[float, float, float]:
    window = max(1, min(3, len(values) // 4 or 1))
    start = sum(values[:window]) / window
    midpoint = values[len(values) // 2]
    end = sum(values[-window:]) / window
    return start, midpoint, end


def generate_feedback(analysis: ToneAnalysis) -> list[FeedbackItem]:
    values = _values(analysis)
    if len(values) < 2:
        return [FeedbackItem(code="short_contour", message="有效音高点太少。请靠近麦克风，录 1–2 秒清晰单字或短词，并避免过长静音。", severity="warning")]

    start, midpoint, end = _edge_means(values)
    pitch_range = max(values) - min(values)
    feedback: list[FeedbackItem] = []

    if analysis.target_tone == 1 and pitch_range > 1.4:
        feedback.append(FeedbackItem(code="unstable_level", message="一声音高应较平稳，你的音高波动偏大。", severity="warning"))

    if analysis.target_tone == 2 and end - start < 0.8:
        feedback.append(FeedbackItem(code="insufficient_rise", message="二声需要明显上升，你的上升幅度不足。", severity="warning"))

    if analysis.target_tone == 3:
        edge_floor = min(start, end)
        if midpoint > edge_floor - 0.35:
            feedback.append(FeedbackItem(code="unclear_dip", message="三声通常需要中段明显降低再回升，你的低凹或转折不够清楚。", severity="warning"))

    if analysis.target_tone == 4 and start - end < 0.9:
        feedback.append(FeedbackItem(code="insufficient_fall", message="四声需要从较高位置快速下降，你的下降幅度不足。", severity="warning"))

    if analysis.predicted_tone != analysis.target_tone:
        feedback.append(FeedbackItem(code="category_mismatch", message=f"模型预测为 {analysis.predicted_tone} 声，与目标 {analysis.target_tone} 声不一致。", severity="warning"))

    if not feedback:
        feedback.append(FeedbackItem(code="good_match", message="声调轮廓与目标较接近。", severity="success"))

    return feedback
