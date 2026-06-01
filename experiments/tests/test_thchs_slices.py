from pathlib import Path

from thchs_slices import (
    choose_duration_filtered_interval,
    extract_pinyin_tone,
    extract_tonal_pinyin_tokens,
    pinyin_token_slots,
    resolve_trn_path,
    speaker_id_from_stem,
    token_intervals,
)


def test_extract_pinyin_tone_returns_base_and_tone():
    token = extract_pinyin_tone("ma3")

    assert token.base == "ma"
    assert token.tone == 3
    assert token.original == "ma3"


def test_extract_tonal_pinyin_tokens_ignores_neutral_tone():
    tokens = extract_tonal_pinyin_tokens("lv4 shi4 de5 ma3")

    assert [token.original for token in tokens] == ["lv4", "shi4", "ma3"]
    assert [token.tone for token in tokens] == [4, 4, 3]




def test_pinyin_token_slots_preserve_neutral_token_positions():
    slots = pinyin_token_slots("ma1 de5 ma3")

    assert [slot.original if slot else None for slot in slots] == ["ma1", None, "ma3"]
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "train"
    data_dir.mkdir()
    split_dir.mkdir()
    real = data_dir / "A11_0.wav.trn"
    ref = split_dir / "A11_0.wav.trn"
    real.write_text("字\nzi4\nz iy4\n", encoding="utf-8")
    ref.write_text("../data/A11_0.wav.trn\n", encoding="utf-8")

    assert resolve_trn_path(ref) == real.resolve()


def test_token_intervals_cover_duration_in_order():
    intervals = token_intervals(token_count=4, duration=2.0, margin=0.1)

    assert intervals == [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1.9)]


def test_choose_duration_filtered_interval_rejects_too_short():
    assert choose_duration_filtered_interval(0.0, 0.1, min_duration=0.18, max_duration=1.2) is None


def test_choose_duration_filtered_interval_accepts_valid_interval():
    assert choose_duration_filtered_interval(0.0, 0.5, min_duration=0.18, max_duration=1.2) == (0.0, 0.5)


def test_speaker_id_from_stem_uses_prefix_before_underscore():
    assert speaker_id_from_stem("A11_123") == "A11"
