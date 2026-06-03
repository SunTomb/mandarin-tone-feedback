import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ["audio_path", "text", "pinyin", "tone", "speaker_id", "repetition", "split"]
OUTPUT_FIELDS = REQUIRED_FIELDS + ["batch_dir"]


@dataclass(frozen=True)
class RecordingBatch:
    name: str
    speaker_id: str
    tone: str
    repetition: str


def parse_batch_dir_name(name: str) -> RecordingBatch:
    parts = name.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid self-recording batch directory: {name}")
    speaker_index, tone, repetition = parts
    return RecordingBatch(
        name=name,
        speaker_id=f"speaker{int(speaker_index):02d}",
        tone=str(int(tone)),
        repetition=str(int(repetition)),
    )


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_fields = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(f"{path} missing required fields: {', '.join(missing_fields)}")
        return [{field: row[field].strip() for field in REQUIRED_FIELDS} for row in reader]


def _project_relative_audio_path(batch_dir: Path, audio_path: str) -> str:
    relative = Path("data") / "self" / batch_dir.name / Path(audio_path).name
    return relative.as_posix()


def build_manifest_rows(self_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metadata_path in sorted(self_root.glob("*/metadata.csv")):
        batch = parse_batch_dir_name(metadata_path.parent.name)
        for row in _read_metadata(metadata_path):
            rows.append(
                {
                    **row,
                    "audio_path": _project_relative_audio_path(metadata_path.parent, row["audio_path"]),
                    "batch_dir": batch.name,
                }
            )
    return rows


def assign_speaker_independent_split(
    rows: list[dict[str, str]],
    *,
    test_speaker_id: str = "speaker03",
    val_repetition: str = "3",
) -> list[dict[str, str]]:
    split_rows: list[dict[str, str]] = []
    for row in rows:
        next_row = dict(row)
        if row["speaker_id"] == test_speaker_id:
            next_row["split"] = "test"
        elif row["repetition"] == val_repetition:
            next_row["split"] = "val"
        else:
            next_row["split"] = "train"
        split_rows.append(next_row)
    return split_rows


def validate_manifest_rows(rows: list[dict[str, str]], *, project_root: Path) -> list[str]:
    errors: list[str] = []
    seen_audio_paths: set[str] = set()
    batch_counts: dict[tuple[str, str, str], int] = {}

    for row in rows:
        missing_fields = [field for field in OUTPUT_FIELDS if field not in row or not row[field]]
        if missing_fields:
            errors.append(f"row missing required fields: {', '.join(missing_fields)}")
            continue

        audio_path = row["audio_path"]
        if audio_path in seen_audio_paths:
            errors.append(f"{audio_path} is duplicated")
        seen_audio_paths.add(audio_path)

        resolved_audio = project_root / audio_path
        if not resolved_audio.exists():
            errors.append(f"{audio_path} is missing")

        try:
            batch = parse_batch_dir_name(row["batch_dir"])
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if row["speaker_id"] != batch.speaker_id:
            errors.append(f"{audio_path} speaker {row['speaker_id']} does not match batch {batch.name} speaker {batch.speaker_id}")
        if row["tone"] != batch.tone:
            errors.append(f"{audio_path} tone {row['tone']} does not match batch {batch.name} tone {batch.tone}")
        if row["repetition"] != batch.repetition:
            errors.append(f"{audio_path} repetition {row['repetition']} does not match batch {batch.name} repetition {batch.repetition}")
        if row["tone"] not in {"1", "2", "3", "4"}:
            errors.append(f"{audio_path} has invalid tone {row['tone']}")
        if row["split"] not in {"train", "val", "test"}:
            errors.append(f"{audio_path} has invalid split {row['split']}")
        if not row["pinyin"].endswith(row["tone"]):
            errors.append(f"{audio_path} pinyin {row['pinyin']} does not end with tone {row['tone']}")

        expected_name = f"{row['speaker_id']}_{row['pinyin']}_{int(row['repetition']):02d}.wav"
        if Path(audio_path).name != expected_name:
            errors.append(f"{audio_path} filename does not match expected {expected_name}")

        batch_key = (row["speaker_id"], row["tone"], row["repetition"])
        batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1

    for (speaker_id, tone, repetition), count in sorted(batch_counts.items()):
        if count != 30:
            errors.append(f"{speaker_id} tone {tone} repetition {repetition} has {count} rows, expected 30")

    return errors


def infer_project_root(self_root: Path) -> Path:
    resolved = self_root.resolve()
    if resolved.name == "self" and resolved.parent.name == "data":
        return resolved.parent.parent
    return Path.cwd()


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unified manifest for self-recorded Mandarin tone data.")
    parser.add_argument("--self-root", type=Path, default=Path("data/self"))
    parser.add_argument("--output", type=Path, default=Path("data/self/metadata.csv"))
    parser.add_argument("--test-speaker", default="speaker03")
    parser.add_argument("--val-repetition", default="3")
    args = parser.parse_args()

    rows = build_manifest_rows(args.self_root)
    rows = assign_speaker_independent_split(rows, test_speaker_id=args.test_speaker, val_repetition=args.val_repetition)
    errors = validate_manifest_rows(rows, project_root=infer_project_root(args.self_root))
    if errors:
        for error in errors:
            print(error)
        return 1
    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
