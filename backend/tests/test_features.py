import numpy as np

from app.features import five_level_value, normalize_f0_values, smooth_contour, resample_contour, summarize_contour


def test_normalize_f0_values_maps_pitch_to_one_to_five():
    values = np.array([100.0, 150.0, 200.0])

    normalized = normalize_f0_values(values)

    assert normalized.tolist() == [1.0, 3.0, 5.0]


def test_normalize_f0_values_keeps_small_pitch_variation_near_level():
    values = np.array([205.0, 203.0, 201.0, 199.0])

    normalized = normalize_f0_values(values)

    assert max(normalized) - min(normalized) < 1.0
    assert normalized[-1] >= 2.5


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




def test_smooth_contour_reduces_trailing_pitch_spike():
    result = smooth_contour([3.0, 3.1, 3.0, 3.1, 5.0])

    assert result[-1] < 4.2


def test_smooth_contour_preserves_tone_three_rise_after_dip():
    result = smooth_contour([3.8, 2.5, 1.4, 2.4, 3.1])

    assert result[-1] > result[2]
