import numpy as np

from features_acoustic import summarize_f0_contour


def test_summarize_f0_contour_returns_expected_values():
    contour = np.array([1.0, 2.0, 4.0], dtype=float)

    features = summarize_f0_contour(contour, duration=0.5, voiced_ratio=0.75)

    assert features["f0_start"] == 1.0
    assert features["f0_mid"] == 2.0
    assert features["f0_end"] == 4.0
    assert features["f0_range"] == 3.0
    assert features["f0_slope"] == 3.0
    assert features["duration"] == 0.5
    assert features["voiced_ratio"] == 0.75


def test_summarize_f0_contour_handles_empty_contour():
    contour = np.array([], dtype=float)

    features = summarize_f0_contour(contour, duration=0.0, voiced_ratio=0.0)

    assert features["f0_start"] == 0.0
    assert features["f0_mid"] == 0.0
    assert features["f0_end"] == 0.0
    assert features["f0_range"] == 0.0
    assert features["f0_slope"] == 0.0
