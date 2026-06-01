from pathlib import Path

from train_acoustic_baseline import load_metadata


def test_load_metadata_reads_validated_rows(tmp_path: Path):
    train_audio = tmp_path / "ma1.wav"
    test_audio = tmp_path / "ma2.wav"
    train_audio.write_bytes(b"RIFF")
    test_audio.write_bytes(b"RIFF")
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split\n"
        f"{train_audio.name},妈,ma1,1,s1,train\n"
        f"{test_audio.name},麻,ma2,2,s1,test\n",
        encoding="utf-8",
    )

    rows = load_metadata(csv_path)

    assert len(rows) == 2
    assert rows[0]["tone"] == 1
    assert rows[0]["pinyin"] == "ma1"
