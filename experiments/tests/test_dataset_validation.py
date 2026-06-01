from pathlib import Path

import pytest

from dataset import DatasetValidationError, load_validated_metadata, summarize_metadata


def write_csv(path: Path, rows: str) -> None:
    path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split\n" + rows,
        encoding="utf-8",
    )


def test_load_validated_metadata_accepts_existing_audio(tmp_path: Path):
    audio_train = tmp_path / "ma1.wav"
    audio_test = tmp_path / "ma2.wav"
    audio_train.write_bytes(b"RIFF")
    audio_test.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(
        metadata,
        f"{audio_train.name},妈,ma1,1,speaker01,train\n{audio_test.name},麻,ma2,2,speaker01,test\n",
    )

    rows = load_validated_metadata(metadata)

    assert len(rows) == 2
    assert rows[0]["tone"] == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["audio_path"] == audio_train


def test_load_validated_metadata_rejects_missing_audio(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, "missing.wav,妈,ma1,1,speaker01,train\n")

    with pytest.raises(DatasetValidationError, match="missing audio file"):
        load_validated_metadata(metadata)


def test_load_validated_metadata_rejects_invalid_tone(tmp_path: Path):
    audio = tmp_path / "ma5.wav"
    audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, f"{audio.name},吗,ma5,5,speaker01,train\n")

    with pytest.raises(DatasetValidationError, match="invalid tone"):
        load_validated_metadata(metadata)


def test_summarize_metadata_counts_split_and_tone(tmp_path: Path):
    audio1 = tmp_path / "ma1.wav"
    audio2 = tmp_path / "ma2.wav"
    audio1.write_bytes(b"RIFF")
    audio2.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(
        metadata,
        f"{audio1.name},妈,ma1,1,speaker01,train\n{audio2.name},麻,ma2,2,speaker02,test\n",
    )
    rows = load_validated_metadata(metadata)

    summary = summarize_metadata(rows)

    assert summary["total"] == 2
    assert summary["by_split"] == {"train": 1, "test": 1}
    assert summary["by_tone"] == {1: 1, 2: 1}
    assert summary["by_speaker"] == {"speaker01": 1, "speaker02": 1}
