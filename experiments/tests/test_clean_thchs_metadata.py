import csv
from pathlib import Path

import numpy as np

from clean_thchs_metadata import clean_metadata_file, clean_metadata_file_with_features


def test_clean_metadata_file_writes_filtered_rows(tmp_path: Path):
    input_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "metadata_clean.csv"
    input_path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split,source_utterance,start_sec,end_sec,source_dataset,label_method\n"
        "a.wav,妈,ma1,1,s1,train,u.wav,0.0,0.5,THCHS,test\n"
        "b.wav,大块,da4,4,s1,train,u.wav,0.0,0.5,THCHS,test\n"
        "c.wav,麻,ma2,2,s1,train,u.wav,0.0,0.5,THCHS,test\n",
        encoding="utf-8",
    )

    rows = clean_metadata_file(input_path, output_path, quotas={"train": 10})

    assert [row["text"] for row in rows] == ["妈", "麻"]
    loaded = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["pinyin"] for row in loaded] == ["ma1", "ma2"]


def test_clean_metadata_file_with_features_filters_low_quality_rows(tmp_path: Path):
    input_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "metadata_clean.csv"
    features_path = tmp_path / "features.npy"
    input_path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split,source_utterance,start_sec,end_sec,source_dataset,label_method\n"
        "a.wav,大块,da4,4,s1,train,u.wav,0.0,0.26,THCHS,test\n"
        "b.wav,文章,wen2,2,s1,train,u.wav,0.0,0.26,THCHS,test\n"
        "c.wav,诗意,shi1,1,s1,train,u.wav,0.0,0.26,THCHS,test\n",
        encoding="utf-8",
    )
    features = np.array([
        [1.0, 2.0, 3.0, 4.0, 2.0, 0.26, 0.8],
        [1.0, 2.0, 3.0, 0.0, 2.0, 0.26, 0.8],
        [1.0, 2.0, 3.0, 4.0, 2.0, 0.26, 0.1],
    ])
    np.save(features_path, features)

    rows = clean_metadata_file_with_features(input_path, features_path, output_path, quotas={"train": 10})



def test_clean_metadata_file_with_features_accepts_rows_json(tmp_path: Path):
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "metadata_clean.csv"
    features_path = tmp_path / "features.npy"
    input_path.write_text(
        '[{"audio_path":"data/processed/thchs30_slices/train/tone4/a.wav","text":"大块","pinyin":"da4","tone":4,"speaker_id":"s1","split":"train","source_utterance":"u.wav","start_sec":"0.0","end_sec":"0.26","source_dataset":"THCHS","label_method":"test"}]',
        encoding="utf-8",
    )
    np.save(features_path, np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 0.26, 0.8]]))

    rows = clean_metadata_file_with_features(input_path, features_path, output_path, quotas={"train": 10})

    assert rows[0]["tone"] == "4"
    loaded = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert loaded[0]["audio_path"] == "processed/thchs30_slices/train/tone4/a.wav"
