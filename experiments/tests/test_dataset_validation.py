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


def test_load_validated_metadata_resolves_project_root_relative_data_paths(tmp_path: Path):
    audio_train = tmp_path / "data" / "self" / "1-1-1" / "speaker01_ma1_01.wav"
    audio_test = tmp_path / "data" / "self" / "3-1-1" / "speaker03_ma1_01.wav"
    audio_train.parent.mkdir(parents=True)
    audio_test.parent.mkdir(parents=True)
    audio_train.write_bytes(b"RIFF")
    audio_test.write_bytes(b"RIFF")
    metadata = tmp_path / "data" / "self" / "metadata.csv"
    metadata.write_text(
        "audio_path,text,pinyin,tone,speaker_id,repetition,split,batch_dir\n"
        "data/self/1-1-1/speaker01_ma1_01.wav,妈,ma1,1,speaker01,1,train,1-1-1\n"
        "data/self/3-1-1/speaker03_ma1_01.wav,妈,ma1,1,speaker03,1,test,3-1-1\n",
        encoding="utf-8",
    )

    rows = load_validated_metadata(metadata)

    assert rows[0]["audio_path"] == audio_train
    assert rows[1]["audio_path"] == audio_test


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


def test_load_validated_metadata_preserves_extra_columns(tmp_path: Path):
    train_audio = tmp_path / "ma1.wav"
    test_audio = tmp_path / "ma2.wav"
    train_audio.write_bytes(b"RIFF")
    test_audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split,source_utterance,start_sec,end_sec,source_dataset,label_method\n"
        f"{train_audio.name},妈,ma1,1,speaker01,train,utt1.wav,0.0,0.4,THCHS-30,pinyin_tone_digit_with_energy_slicing\n"
        f"{test_audio.name},麻,ma2,2,speaker01,test,utt2.wav,0.2,0.7,THCHS-30,pinyin_tone_digit_with_energy_slicing\n",
        encoding="utf-8",
    )

    rows = load_validated_metadata(metadata)

    assert rows[0]["source_utterance"] == "utt1.wav"
    assert rows[0]["start_sec"] == "0.0"
    assert rows[0]["label_method"] == "pinyin_tone_digit_with_energy_slicing"
