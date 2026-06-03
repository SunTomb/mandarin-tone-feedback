from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from thchs_slices import extract_pinyin_tone, filter_metadata_rows


def read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        normalized = [{key: str(value) for key, value in row.items()} for row in rows]
        for row in normalized:
            audio_path = row.get("audio_path", "")
            if audio_path.startswith("data/"):
                row["audio_path"] = audio_path.removeprefix("data/")
        fieldnames = list(normalized[0].keys()) if normalized else []
        return fieldnames, normalized
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_metadata(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_metadata_file(
    input_path: Path,
    output_path: Path,
    quotas: dict[str, int],
    min_duration: float = 0.30,
    max_duration: float = 0.90,
) -> list[dict[str, str]]:
    fieldnames, rows = read_metadata(input_path)
    filtered = filter_metadata_rows(rows, quotas=quotas, min_duration=min_duration, max_duration=max_duration)
    write_metadata(output_path, fieldnames, filtered)
    return filtered


def clean_metadata_file_with_features(
    input_path: Path,
    features_path: Path,
    output_path: Path,
    quotas: dict[str, int],
    min_voiced_ratio: float = 0.50,
    min_f0_range: float = 0.40,
) -> list[dict[str, str]]:
    fieldnames, rows = read_metadata(input_path)
    features = np.load(features_path)
    counts: Counter[tuple[str, str]] = Counter()
    filtered: list[dict[str, str]] = []
    for row, feature in zip(rows, features):
        token = extract_pinyin_tone(row.get("pinyin", ""))
        if token is None or str(token.tone) != str(row.get("tone", "")):
            continue
        if float(feature[3]) < min_f0_range or float(feature[6]) < min_voiced_ratio:
            continue
        split = row.get("split", "")
        tone = row.get("tone", "")
        quota = quotas.get(split)
        if quota is not None and counts[(split, tone)] >= quota:
            continue
        filtered.append(row)
        counts[(split, tone)] += 1
    write_metadata(output_path, fieldnames, filtered)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter THCHS-derived tone-slice metadata to cleaner candidates.")
    parser.add_argument("--input", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata_clean.csv"))
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--mode", choices=["single-char", "features"], default="single-char")
    parser.add_argument("--train-quota", type=int, default=80)
    parser.add_argument("--val-quota", type=int, default=20)
    parser.add_argument("--test-quota", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-duration", type=float, default=0.90)
    parser.add_argument("--min-voiced-ratio", type=float, default=0.50)
    parser.add_argument("--min-f0-range", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quotas = {"train": args.train_quota, "val": args.val_quota, "test": args.test_quota}
    if args.mode == "features":
        if args.features is None:
            raise SystemExit("--features is required when --mode=features")
        rows = clean_metadata_file_with_features(
            args.input,
            args.features,
            args.output,
            quotas,
            min_voiced_ratio=args.min_voiced_ratio,
            min_f0_range=args.min_f0_range,
        )
    else:
        rows = clean_metadata_file(args.input, args.output, quotas, args.min_duration, args.max_duration)
    counts = Counter(f"{row['split']}_tone{row['tone']}" for row in rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output), "by_split_tone": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
