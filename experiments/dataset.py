import csv
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"audio_path", "text", "pinyin", "tone", "speaker_id", "split"}
VALID_TONES = {1, 2, 3, 4}
VALID_SPLITS = {"train", "val", "test"}


class DatasetValidationError(ValueError):
    pass


def _resolve_audio_path(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return metadata_path.parent / path


def load_validated_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetValidationError(f"metadata file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise DatasetValidationError(f"metadata missing columns: {names}")

        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            audio_path = _resolve_audio_path(path, row["audio_path"])
            if not audio_path.exists():
                raise DatasetValidationError(f"line {line_number}: missing audio file: {audio_path}")

            try:
                tone = int(row["tone"])
            except ValueError as exc:
                raise DatasetValidationError(f"line {line_number}: invalid tone: {row['tone']}") from exc
            if tone not in VALID_TONES:
                raise DatasetValidationError(f"line {line_number}: invalid tone: {tone}")

            split = row["split"]
            if split not in VALID_SPLITS:
                raise DatasetValidationError(f"line {line_number}: invalid split: {split}")

            rows.append(
                {
                    "audio_path": audio_path,
                    "text": row["text"],
                    "pinyin": row["pinyin"],
                    "tone": tone,
                    "speaker_id": row["speaker_id"],
                    "split": split,
                }
            )

    if not rows:
        raise DatasetValidationError("metadata contains no rows")

    splits = {row["split"] for row in rows}
    if "train" not in splits or "test" not in splits:
        raise DatasetValidationError("metadata must contain non-empty train and test splits")

    return rows


def summarize_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_tone": dict(Counter(row["tone"] for row in rows)),
        "by_speaker": dict(Counter(row["speaker_id"] for row in rows)),
    }


def split_indices(rows: list[dict[str, Any]], split: str) -> list[int]:
    return [index for index, row in enumerate(rows) if row["split"] == split]
