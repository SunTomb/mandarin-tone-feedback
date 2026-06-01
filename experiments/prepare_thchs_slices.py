from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from thchs_slices import (
    choose_duration_filtered_interval,
    pinyin_token_slots,
    read_thchs_trn,
    speaker_id_from_stem,
    token_intervals,
)


@dataclass(frozen=True)
class SliceConfig:
    min_duration: float = 0.18
    max_duration: float = 1.20
    margin: float = 0.05
    rms_threshold: float = 0.001
    min_voiced_ratio: float = 0.2


FIELDNAMES = [
    "audio_path",
    "text",
    "pinyin",
    "tone",
    "speaker_id",
    "split",
    "source_utterance",
    "start_sec",
    "end_sec",
    "source_dataset",
    "label_method",
]


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def voiced_ratio(audio: np.ndarray, sr: int) -> float:
    if audio.size == 0:
        return 0.0
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    if f0 is None or len(f0) == 0:
        return 0.0
    return float(np.isfinite(f0).sum() / len(f0))


def write_slice(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def relative_to_data(path: Path) -> str:
    resolved = path.resolve()
    for parent in [resolved, *resolved.parents]:
        if parent.name == "data":
            return str(resolved.relative_to(parent))
    return str(path)


def portable_source_utterance(path: Path) -> str:
    resolved = path.resolve()
    for parent in [resolved, *resolved.parents]:
        if parent.name in {"train", "dev", "test"}:
            return str(resolved.relative_to(parent))
    return path.name


def build_slice_rows_for_file(
    wav_path: Path,
    trn_path: Path,
    output_root: Path,
    split: str,
    config: SliceConfig,
) -> list[dict[str, str]]:
    text_line, pinyin_line, _phone_line = read_thchs_trn(trn_path)
    token_slots = pinyin_token_slots(pinyin_line)
    if not any(token is not None for token in token_slots):
        return []

    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    duration = float(len(audio) / sr) if sr else 0.0
    intervals = token_intervals(len(token_slots), duration, margin=config.margin)
    characters = [piece for piece in text_line.split() if piece.strip()]
    rows: list[dict[str, str]] = []

    for index, (token, interval) in enumerate(zip(token_slots, intervals)):
        if token is None:
            continue
        filtered = choose_duration_filtered_interval(
            interval[0],
            interval[1],
            min_duration=config.min_duration,
            max_duration=config.max_duration,
        )
        if filtered is None:
            continue
        start_sec, end_sec = filtered
        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(audio), int(end_sec * sr))
        slice_audio = audio[start_sample:end_sample]
        if rms(slice_audio) < config.rms_threshold:
            continue
        if config.min_voiced_ratio > 0 and voiced_ratio(slice_audio, sr) < config.min_voiced_ratio:
            continue

        stem = wav_path.stem
        slice_name = f"{stem}_{index:03d}_{token.original}.wav"
        slice_path = output_root / split / f"tone{token.tone}" / slice_name
        write_slice(slice_path, slice_audio, sr)
        rows.append(
            {
                "audio_path": relative_to_data(slice_path),
                "text": characters[index] if index < len(characters) else token.original,
                "pinyin": token.original,
                "tone": str(token.tone),
                "speaker_id": speaker_id_from_stem(stem),
                "split": split,
                "source_utterance": portable_source_utterance(wav_path),
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "source_dataset": "THCHS-30 / OpenSLR SLR18",
                "label_method": "pinyin_tone_digit_with_energy_slicing",
            }
        )
    return rows


def iter_split_files(root: Path, source_split: str):
    split_dir = root / source_split
    for wav_path in sorted(split_dir.glob("*.wav")):
        trn_path = wav_path.with_suffix(wav_path.suffix + ".trn")
        if trn_path.exists():
            yield wav_path, trn_path


def quota_reached(counts: Counter[tuple[str, str]], split: str, tone: str, quota: int) -> bool:
    return counts[(split, tone)] >= quota


def remove_slice_for_row(row: dict[str, str], output_root: Path) -> None:
    path = Path(row["audio_path"])
    candidates = [path]
    parents = list(output_root.resolve().parents)
    if len(parents) >= 2:
        candidates.append(parents[1] / path)
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()
            return


def build_dataset(
    root: Path,
    output_root: Path,
    config: SliceConfig,
    train_quota: int,
    val_quota: int,
    test_quota: int,
) -> list[dict[str, str]]:
    if output_root.exists():
        shutil.rmtree(output_root)
    split_map = {"train": "train", "dev": "val", "test": "test"}
    quotas = {"train": train_quota, "val": val_quota, "test": test_quota}
    counts: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, str]] = []

    for source_split, output_split in split_map.items():
        for wav_path, trn_path in iter_split_files(root, source_split):
            candidate_rows = build_slice_rows_for_file(wav_path, trn_path, output_root, output_split, config)
            for row in candidate_rows:
                tone = row["tone"]
                if quota_reached(counts, output_split, tone, quotas[output_split]):
                    remove_slice_for_row(row, output_root)
                    continue
                rows.append(row)
                counts[(output_split, tone)] += 1
            if all(counts[(output_split, str(tone))] >= quotas[output_split] for tone in [1, 2, 3, 4]):
                break
    return rows


def write_metadata(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], output: Path, config: SliceConfig) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "total": len(rows),
        "by_split_tone": dict(Counter(f"{row['split']}_tone{row['tone']}" for row in rows)),
        "config": config.__dict__,
        "source": "THCHS-30 / OpenSLR SLR18",
        "source_url": "https://www.openslr.org/18/",
        "label_method": "pinyin_tone_digit_with_energy_slicing",
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output: Path, metadata_path: Path, summary_path: Path) -> None:
    output.write_text(
        "# THCHS-30 Derived Tone Slices\n\n"
        "This directory contains Mandarin tone slices derived from THCHS-30 / OpenSLR SLR18.\n\n"
        "The slice labels come from pinyin tone digits in THCHS transcript files. Boundaries are deterministic approximate intervals refined only by duration/RMS/F0 filters, not manual forced alignment.\n\n"
        f"Metadata: `{metadata_path}`\n\n"
        f"Summary: `{summary_path}`\n\n"
        "Use these data as a reproducible course-project baseline and state the boundary-label limitation in the paper.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare THCHS-derived Mandarin tone slices.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/thchs30/data_thchs30"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/thchs30_slices"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/thchs30_slices/summary.json"))
    parser.add_argument("--train-quota", type=int, default=400)
    parser.add_argument("--val-quota", type=int, default=80)
    parser.add_argument("--test-quota", type=int, default=80)
    parser.add_argument("--min-duration", type=float, default=0.18)
    parser.add_argument("--max-duration", type=float, default=1.20)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--rms-threshold", type=float, default=0.001)
    parser.add_argument("--min-voiced-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SliceConfig(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        margin=args.margin,
        rms_threshold=args.rms_threshold,
        min_voiced_ratio=args.min_voiced_ratio,
    )
    rows = build_dataset(args.root, args.output_root, config, args.train_quota, args.val_quota, args.test_quota)
    write_metadata(rows, args.metadata)
    write_summary(rows, args.summary, config)
    write_readme(args.output_root / "README.md", args.metadata, args.summary)
    print(json.dumps({"rows": len(rows), "summary": str(args.summary), "metadata": str(args.metadata)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
