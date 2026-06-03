import subprocess
import sys
from pathlib import Path

from self_recording_manifest import (
    REQUIRED_FIELDS,
    assign_speaker_independent_split,
    build_manifest_rows,
    parse_batch_dir_name,
    validate_manifest_rows,
)


def write_batch(batch_dir: Path, rows: list[dict[str, str]]) -> None:
    batch_dir.mkdir(parents=True)
    lines = [",".join(REQUIRED_FIELDS)]
    for row in rows:
        lines.append(",".join(row[field] for field in REQUIRED_FIELDS))
        (batch_dir / row["audio_path"]).write_bytes(b"RIFF")
    (batch_dir / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_batch_dir_name_maps_speaker_tone_and_repetition():
    batch = parse_batch_dir_name("2-3-2")

    assert batch.speaker_id == "speaker02"
    assert batch.tone == "3"
    assert batch.repetition == "2"


def test_build_manifest_rows_resolves_audio_paths_and_keeps_batch_metadata(tmp_path: Path):
    write_batch(
        tmp_path / "2-3-2",
        [
            {
                "audio_path": "speaker02_ma3_02.wav",
                "text": "马",
                "pinyin": "ma3",
                "tone": "3",
                "speaker_id": "speaker02",
                "repetition": "2",
                "split": "train",
            }
        ],
    )

    rows = build_manifest_rows(tmp_path)

    assert rows == [
        {
            "audio_path": "data/self/2-3-2/speaker02_ma3_02.wav",
            "text": "马",
            "pinyin": "ma3",
            "tone": "3",
            "speaker_id": "speaker02",
            "repetition": "2",
            "split": "train",
            "batch_dir": "2-3-2",
        }
    ]


def test_validate_manifest_rows_reports_directory_and_audio_errors(tmp_path: Path):
    rows = [
        {
            "audio_path": "data/self/2-3-2/speaker02_ma2_02.wav",
            "text": "麻",
            "pinyin": "ma2",
            "tone": "2",
            "speaker_id": "speaker02",
            "repetition": "2",
            "split": "train",
            "batch_dir": "2-3-2",
        }
    ]

    errors = validate_manifest_rows(rows, project_root=tmp_path)

    assert "data/self/2-3-2/speaker02_ma2_02.wav is missing" in errors
    assert "data/self/2-3-2/speaker02_ma2_02.wav tone 2 does not match batch 2-3-2 tone 3" in errors


def test_manifest_cli_validates_from_experiments_working_directory(tmp_path: Path):
    project_root = tmp_path
    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir()
    write_batch(
        project_root / "data" / "self" / "1-1-1",
        [
            {
                "audio_path": f"speaker01_item{index}1_01.wav",
                "text": f"字{index}",
                "pinyin": f"item{index}1",
                "tone": "1",
                "speaker_id": "speaker01",
                "repetition": "1",
                "split": "train",
            }
            for index in range(30)
        ],
    )

    script = Path(__file__).parents[1] / "self_recording_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--self-root",
            str(project_root / "data" / "self"),
            "--output",
            str(project_root / "data" / "self" / "metadata.csv"),
        ],
        cwd=experiments_dir,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Wrote 30 rows" in result.stdout
    assert (project_root / "data" / "self" / "metadata.csv").exists()

    rows = []
    for speaker_id in ["speaker01", "speaker02", "speaker03"]:
        for tone in ["1", "2"]:
            for pinyin in [f"ma{tone}", f"ba{tone}"]:
                for repetition in ["1", "2", "3"]:
                    rows.append(
                        {
                            "audio_path": f"data/self/1-{tone}-{repetition}/{speaker_id}_{pinyin}_{int(repetition):02d}.wav",
                            "text": pinyin,
                            "pinyin": pinyin,
                            "tone": tone,
                            "speaker_id": speaker_id,
                            "repetition": repetition,
                            "split": "train",
                            "batch_dir": f"1-{tone}-{repetition}",
                        }
                    )

    split_rows = assign_speaker_independent_split(rows, test_speaker_id="speaker03", val_repetition="3")

    test_rows = [row for row in split_rows if row["split"] == "test"]
    val_rows = [row for row in split_rows if row["split"] == "val"]
    train_rows = [row for row in split_rows if row["split"] == "train"]

    assert {row["speaker_id"] for row in test_rows} == {"speaker03"}
    assert {row["repetition"] for row in val_rows} == {"3"}
    assert {row["speaker_id"] for row in val_rows} == {"speaker01", "speaker02"}
    assert len(test_rows) == 12
    assert len(val_rows) == 8
    assert len(train_rows) == 16
