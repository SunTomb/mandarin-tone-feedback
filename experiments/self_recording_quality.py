import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from self_recording_manifest import OUTPUT_FIELDS, write_manifest

QUALITY_FIELDS = [
    "audio_path",
    "duration",
    "voiced_ratio",
    "voiced_frames",
    "f0_range",
    "rms",
    "keep",
    "reason",
]


@dataclass(frozen=True)
class QualityThresholds:
    min_duration: float = 0.25
    max_duration: float = 1.8
    min_voiced_ratio: float = 0.35
    min_voiced_frames: int = 5
    min_rms: float = 0.005


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    reason: str


def quality_decision(metrics: dict[str, Any], thresholds: QualityThresholds) -> QualityDecision:
    if float(metrics["duration"]) < thresholds.min_duration:
        return QualityDecision(False, "duration_too_short")
    if float(metrics["duration"]) > thresholds.max_duration:
        return QualityDecision(False, "duration_too_long")
    if float(metrics["voiced_ratio"]) < thresholds.min_voiced_ratio:
        return QualityDecision(False, "low_voiced_ratio")
    if int(metrics["voiced_frames"]) < thresholds.min_voiced_frames:
        return QualityDecision(False, "too_few_voiced_frames")
    if float(metrics.get("rms", 1.0)) < thresholds.min_rms:
        return QualityDecision(False, "low_rms")
    return QualityDecision(True, "ok")


def audit_recording(row: dict[str, Any], thresholds: QualityThresholds) -> dict[str, str]:
    import numpy as np

    from features_acoustic import extract_f0_contour, load_audio

    audio, sr, duration = load_audio(row["audio_path"])
    contour, voiced_ratio = extract_f0_contour(audio, sr)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    metrics: dict[str, Any] = {
        "duration": float(duration),
        "voiced_ratio": float(voiced_ratio),
        "voiced_frames": int(contour.size),
        "f0_range": float(np.max(contour) - np.min(contour)) if contour.size else 0.0,
        "rms": rms,
    }
    decision = quality_decision(metrics, thresholds)
    return {
        "audio_path": row["audio_path"].as_posix() if isinstance(row["audio_path"], Path) else str(row["audio_path"]),
        "duration": f"{metrics['duration']:.6f}",
        "voiced_ratio": f"{metrics['voiced_ratio']:.6f}",
        "voiced_frames": str(metrics["voiced_frames"]),
        "f0_range": f"{metrics['f0_range']:.6f}",
        "rms": f"{metrics['rms']:.6f}",
        "keep": "1" if decision.keep else "0",
        "reason": decision.reason,
    }


def audit_recordings(rows: list[dict[str, Any]], thresholds: QualityThresholds) -> list[dict[str, str]]:
    return [audit_recording(row, thresholds) for row in rows]


def filter_quality_rows(rows: list[dict[str, str]], quality_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep_by_path = {row["audio_path"]: row["keep"] == "1" for row in quality_rows}
    return [row for row in rows if keep_by_path.get(str(row["audio_path"]), False)]


def assign_loso_split(rows: list[dict[str, str]], *, test_speaker_id: str, val_repetition: str = "3") -> list[dict[str, str]]:
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


def write_quality_report(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_quality_report(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _manifest_rows_as_strings(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        next_row = {field: str(row[field]) for field in OUTPUT_FIELDS if field in row}
        audio_path = row["audio_path"]
        next_row["audio_path"] = audio_path.as_posix() if isinstance(audio_path, Path) else str(audio_path)
        converted.append(next_row)
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and filter self-recorded Mandarin tone data.")
    parser.add_argument("--metadata", type=Path, default=Path("data/self/metadata.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("outputs/self_quality/quality_report.csv"))
    parser.add_argument("--filtered-metadata", type=Path, default=Path("data/self/metadata_filtered.csv"))
    parser.add_argument("--loso-dir", type=Path, default=Path("data/self/loso"))
    parser.add_argument("--min-duration", type=float, default=QualityThresholds.min_duration)
    parser.add_argument("--max-duration", type=float, default=QualityThresholds.max_duration)
    parser.add_argument("--min-voiced-ratio", type=float, default=QualityThresholds.min_voiced_ratio)
    parser.add_argument("--min-voiced-frames", type=int, default=QualityThresholds.min_voiced_frames)
    parser.add_argument("--min-rms", type=float, default=QualityThresholds.min_rms)
    return parser.parse_args()


def main() -> int:
    from dataset import load_validated_metadata

    args = parse_args()
    thresholds = QualityThresholds(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_voiced_ratio=args.min_voiced_ratio,
        min_voiced_frames=args.min_voiced_frames,
        min_rms=args.min_rms,
    )
    rows = load_validated_metadata(args.metadata)
    manifest_rows = _manifest_rows_as_strings(rows)
    quality_rows = audit_recordings(rows, thresholds)
    write_quality_report(quality_rows, args.quality_report)
    filtered_rows = filter_quality_rows(manifest_rows, quality_rows)
    write_manifest(filtered_rows, args.filtered_metadata)

    speakers = sorted({row["speaker_id"] for row in filtered_rows})
    for speaker_id in speakers:
        write_manifest(assign_loso_split(filtered_rows, test_speaker_id=speaker_id), args.loso_dir / f"metadata_test_{speaker_id}.csv")

    kept = sum(1 for row in quality_rows if row["keep"] == "1")
    print(f"Audited {len(quality_rows)} rows; kept {kept}; dropped {len(quality_rows) - kept}")
    print(f"Wrote {args.quality_report.as_posix()}")
    print(f"Wrote {args.filtered_metadata.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
