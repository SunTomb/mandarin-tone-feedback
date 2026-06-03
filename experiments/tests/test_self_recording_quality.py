from pathlib import Path

from self_recording_quality import (
    QualityThresholds,
    assign_loso_split,
    filter_quality_rows,
    quality_decision,
)


def row(**overrides: str) -> dict[str, str]:
    base = {
        "audio_path": "data/self/1-1-1/speaker01_ma1_01.wav",
        "text": "妈",
        "pinyin": "ma1",
        "tone": "1",
        "speaker_id": "speaker01",
        "repetition": "1",
        "split": "train",
        "batch_dir": "1-1-1",
    }
    base.update(overrides)
    return base


def test_quality_decision_rejects_short_or_unvoiced_recordings():
    thresholds = QualityThresholds(min_duration=0.25, max_duration=1.5, min_voiced_ratio=0.45, min_voiced_frames=5)

    short = quality_decision({"duration": 0.2, "voiced_ratio": 0.8, "voiced_frames": 10}, thresholds)
    unvoiced = quality_decision({"duration": 0.6, "voiced_ratio": 0.2, "voiced_frames": 10}, thresholds)
    sparse = quality_decision({"duration": 0.6, "voiced_ratio": 0.8, "voiced_frames": 3}, thresholds)

    assert short.keep is False
    assert short.reason == "duration_too_short"
    assert unvoiced.keep is False
    assert unvoiced.reason == "low_voiced_ratio"
    assert sparse.keep is False
    assert sparse.reason == "too_few_voiced_frames"


def test_filter_quality_rows_drops_bad_rows_and_preserves_split():
    rows = [
        row(audio_path="data/self/1-1-1/good.wav", split="train"),
        row(audio_path="data/self/1-1-1/bad.wav", split="test"),
    ]
    quality_rows = [
        {"audio_path": "data/self/1-1-1/good.wav", "keep": "1", "reason": "ok"},
        {"audio_path": "data/self/1-1-1/bad.wav", "keep": "0", "reason": "low_voiced_ratio"},
    ]

    filtered = filter_quality_rows(rows, quality_rows)

    assert filtered == [rows[0]]
    assert filtered[0]["split"] == "train"


def test_assign_loso_split_uses_one_test_speaker_and_validation_repetition():
    rows = []
    for speaker_id in ["speaker01", "speaker02", "speaker03"]:
        for repetition in ["1", "2", "3"]:
            rows.append(row(speaker_id=speaker_id, repetition=repetition, split="train"))

    split_rows = assign_loso_split(rows, test_speaker_id="speaker02", val_repetition="3")

    assert [r["split"] for r in split_rows if r["speaker_id"] == "speaker02"] == ["test", "test", "test"]
    assert {r["split"] for r in split_rows if r["speaker_id"] != "speaker02" and r["repetition"] == "3"} == {"val"}
    assert {r["split"] for r in split_rows if r["speaker_id"] != "speaker02" and r["repetition"] != "3"} == {"train"}


def test_quality_threshold_defaults_are_suitable_for_isolated_characters():
    thresholds = QualityThresholds()

    assert thresholds.min_duration == 0.25
    assert thresholds.max_duration == 1.8
    assert thresholds.min_voiced_ratio == 0.35
    assert thresholds.min_voiced_frames == 5
