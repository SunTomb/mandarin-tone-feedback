from experiments.prepare_thchs_metadata import choose_tone_label, extract_tone_digits


def test_extract_tone_digits_ignores_neutral_tone():
    assert extract_tone_digits("lv4 shi4 yang2 de5") == [4, 4, 2]


def test_choose_tone_label_returns_majority_when_dominant():
    assert choose_tone_label([4, 4, 2], min_ratio=0.5) == 4


def test_choose_tone_label_rejects_tie():
    assert choose_tone_label([1, 2, 3, 4], min_ratio=0.5) is None
