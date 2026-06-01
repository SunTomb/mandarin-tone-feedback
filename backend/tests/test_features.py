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
