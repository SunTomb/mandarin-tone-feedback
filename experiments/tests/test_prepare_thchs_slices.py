from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_thchs_slices import SliceConfig, build_dataset, build_slice_rows_for_file


def test_build_slice_rows_for_file_writes_tonal_slices(tmp_path: Path):
    source_wav = tmp_path / "A11_0.wav"
    trn = tmp_path / "A11_0.wav.trn"
    output_root = tmp_path / "slices"
    sr = 16000
    audio = 0.05 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr).astype("float32")
    sf.write(source_wav, audio, sr)
    trn.write_text("妈 麻 马 骂\nma1 ma2 ma3 ma4\nm a1 m a2 m a3 m a4\n", encoding="utf-8")
    config = SliceConfig(min_duration=0.18, max_duration=1.2, margin=0.0, rms_threshold=0.001, min_voiced_ratio=0.0)

    rows = build_slice_rows_for_file(source_wav, trn, output_root, split="train", config=config)

    assert [row["tone"] for row in rows] == ["1", "2", "3", "4"]
    assert all((tmp_path / row["audio_path"]).exists() for row in rows)
    assert rows[0]["source_utterance"] == "A11_0.wav"


def test_build_dataset_does_not_write_slices_beyond_quota(tmp_path: Path):
    root = tmp_path / "thchs"
    train = root / "train"
    train.mkdir(parents=True)
    sr = 16000
    audio = 0.05 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr).astype("float32")
    for file_index in range(1):
        wav = train / f"A11_{file_index}.wav"
        trn = train / f"A11_{file_index}.wav.trn"
        sf.write(wav, audio, sr)
        trn.write_text("妈 妈 妈 妈\nma1 ma1 ma1 ma1\nm a1 m a1 m a1 m a1\n", encoding="utf-8")
    output_root = tmp_path / "slices"

    rows = build_dataset(
        root,
        output_root,
        SliceConfig(min_duration=0.18, max_duration=1.2, margin=0.0, rms_threshold=0.001, min_voiced_ratio=0.0),
        train_quota=1,
        val_quota=0,
        test_quota=0,
    )

    written = list(output_root.glob("**/*.wav"))
    assert len(rows) == 1


def test_build_dataset_cleans_output_root_before_generation(tmp_path: Path):
    root = tmp_path / "thchs"
    train = root / "train"
    train.mkdir(parents=True)
    sr = 16000
    audio = 0.05 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype("float32")
    wav = train / "A11_0.wav"
    trn = train / "A11_0.wav.trn"
    sf.write(wav, audio, sr)
    trn.write_text("妈\nma1\nm a1\n", encoding="utf-8")
    output_root = tmp_path / "slices"
    stale = output_root / "train" / "tone4" / "stale.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    build_dataset(
        root,
        output_root,
        SliceConfig(min_duration=0.18, max_duration=1.2, margin=0.0, rms_threshold=0.001, min_voiced_ratio=0.0),
        train_quota=1,
        val_quota=0,
        test_quota=0,
    )

    assert not stale.exists()


def test_build_slice_rows_for_file_keeps_tonal_timing_slots(tmp_path: Path):
    source_wav = tmp_path / "A11_0.wav"
    trn = tmp_path / "A11_0.wav.trn"
    output_root = tmp_path / "slices"
    sr = 16000
    audio = 0.05 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr).astype("float32")
    sf.write(source_wav, audio, sr)
    trn.write_text("妈 的 马\nma1 de5 ma3\nm a1 d e5 m a3\n", encoding="utf-8")

    rows = build_slice_rows_for_file(
        source_wav,
        trn,
        output_root,
        split="train",
        config=SliceConfig(min_duration=0.18, max_duration=1.2, margin=0.0, rms_threshold=0.001, min_voiced_ratio=0.0),
    )

    assert [row["pinyin"] for row in rows] == ["ma1", "ma3"]
    assert rows[0]["start_sec"] == "0.000"
    assert rows[0]["end_sec"] == "1.000"
    assert rows[1]["start_sec"] == "2.000"
    assert rows[1]["end_sec"] == "3.000"
