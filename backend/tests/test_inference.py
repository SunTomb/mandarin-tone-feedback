from app.inference import predict_tone_from_contour


def test_predicts_tone_one_from_stable_high_contour():
    predicted_tone, confidence = predict_tone_from_contour([4.7, 4.8, 4.6, 4.7])

    assert predicted_tone == 1
    assert 0.0 <= confidence <= 1.0


def test_keeps_tone_one_reference_for_mostly_level_contour_with_tail_fall():
    predicted_tone, confidence = predict_tone_from_contour([4.7, 4.8, 4.6, 4.5, 4.4, 3.7, 3.2])

    assert predicted_tone == 1
    assert confidence < 0.7


def test_predicts_tone_two_from_rising_contour():
    predicted_tone, _ = predict_tone_from_contour([1.8, 2.4, 3.3, 4.6])

    assert predicted_tone == 2


def test_predicts_tone_three_from_dipping_contour():
    predicted_tone, _ = predict_tone_from_contour([3.4, 2.0, 1.3, 2.8])

    assert predicted_tone == 3


def test_predicts_tone_four_from_falling_contour():
    predicted_tone, _ = predict_tone_from_contour([4.8, 3.4, 2.2, 1.2])

    assert predicted_tone == 4
