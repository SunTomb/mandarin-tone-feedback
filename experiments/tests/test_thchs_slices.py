from pathlib import Path

from thchs_slices import (
    choose_duration_filtered_interval,
    extract_pinyin_tone,
    extract_tonal_pinyin_tokens,
    filter_metadata_rows,
    is_clean_slice_row,
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


def test_resolve_trn_path_follows_relative_reference(tmp_path: Path):
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




def test_is_clean_slice_row_accepts_single_character_tone_slice():
    row = {"text": "妈", "pinyin": "ma1", "tone": "1", "start_sec": "0.10", "end_sec": "0.62", "split": "train"}

    assert is_clean_slice_row(row, min_duration=0.30, max_duration=0.90)


def test_is_clean_slice_row_rejects_ambiguous_or_invalid_slice():
    assert not is_clean_slice_row({"text": "大块", "pinyin": "da4", "tone": "4", "start_sec": "0.10", "end_sec": "0.62", "split": "train"})
    assert not is_clean_slice_row({"text": "的", "pinyin": "de5", "tone": "5", "start_sec": "0.10", "end_sec": "0.62", "split": "train"})
    assert not is_clean_slice_row({"text": "妈", "pinyin": "ma1", "tone": "1", "start_sec": "0.10", "end_sec": "0.20", "split": "train"}, min_duration=0.30)


def test_filter_metadata_rows_keeps_balanced_quota_per_split_and_tone():
    rows = [
        {"text": "妈", "pinyin": "ma1", "tone": "1", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
        {"text": "麻", "pinyin": "ma2", "tone": "2", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
        {"text": "马", "pinyin": "ma3", "tone": "3", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
        {"text": "骂", "pinyin": "ma4", "tone": "4", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
        {"text": "妈", "pinyin": "ma1", "tone": "1", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
        {"text": "大块", "pinyin": "da4", "tone": "4", "start_sec": "0.0", "end_sec": "0.5", "split": "train"},
    ]

    filtered = filter_metadata_rows(rows, quotas={"train": 1})

    assert [(row["split"], row["tone"], row["text"]) for row in filtered] == [
        ("train", "1", "妈"),
        ("train", "2", "麻"),
        ("train", "3", "马"),
        ("train", "4", "骂"),
    ]
