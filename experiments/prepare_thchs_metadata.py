import argparse
import csv
import re
from collections import Counter
from pathlib import Path


TONE_RE = re.compile(r"[a-zv:]+([1-5])\b", re.IGNORECASE)
SPLITS = ("train", "dev", "test")


def extract_tone_digits(pinyin_line: str) -> list[int]:
    tones = [int(match) for match in TONE_RE.findall(pinyin_line)]
    return [tone for tone in tones if tone in {1, 2, 3, 4}]


def choose_tone_label(tones: list[int], min_ratio: float) -> int | None:
    if not tones:
        return None
    counts = Counter(tones)
    tone, count = counts.most_common(1)[0]
    if count / len(tones) < min_ratio:
        return None
    if list(counts.values()).count(count) > 1:
        return None
    return tone


def resolve_referenced_path(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 1 and lines[0].startswith("../"):
        return (path.parent / lines[0]).resolve()
    return path


def read_trn(path: Path) -> tuple[str, str]:
    resolved = resolve_referenced_path(path)
    lines = resolved.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"transcript has fewer than two lines: {resolved}")
    return lines[0].strip(), lines[1].strip()


def iter_split_wavs(root: Path, split: str):
    split_dir = root / split
    for wav_path in sorted(split_dir.glob("*.wav")):
        trn_path = wav_path.with_suffix(wav_path.suffix + ".trn")
        if trn_path.exists():
            yield wav_path, trn_path


def speaker_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def build_rows(root: Path, min_ratio: float, max_per_tone_split: int | None, output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts: Counter[tuple[str, int]] = Counter()
    split_map = {"train": "train", "dev": "val", "test": "test"}

    for source_split in SPLITS:
        output_split = split_map[source_split]
        for wav_path, trn_path in iter_split_wavs(root, source_split):
            text, pinyin = read_trn(trn_path)
            tones = extract_tone_digits(pinyin)
            tone = choose_tone_label(tones, min_ratio=min_ratio)
            if tone is None:
                continue
            key = (output_split, tone)
            if max_per_tone_split is not None and counts[key] >= max_per_tone_split:
                continue
            counts[key] += 1
            audio_path = wav_path
            try:
                audio_value = str(audio_path.relative_to(output_dir))
            except ValueError:
                audio_value = str(audio_path.resolve())
            rows.append(
                {
                    "audio_path": audio_value,
                    "text": text,
                    "pinyin": pinyin,
                    "tone": str(tone),
                    "speaker_id": speaker_id_from_stem(wav_path.stem),
                    "split": output_split,
                }
            )
    return rows


def write_metadata(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "text", "pinyin", "tone", "speaker_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(output: Path, root: Path, min_ratio: float, row_count: int) -> None:
    output.write_text(
        "# THCHS-30 Derived Tone Metadata\n\n"
        "- Source dataset: THCHS-30 / OpenSLR SLR18\n"
        "- Source page: https://www.openslr.org/18/\n"
        "- License stated on source page: Apache License v2.0; free for academic users\n"
        f"- Local root: `{root}`\n"
        "- Derivation: read the second line of each `.wav.trn` file, extract pinyin tone digits 1-4, ignore neutral tone 5, and assign the utterance to the dominant tone if it meets the configured ratio.\n"
        f"- Dominance ratio: {min_ratio}\n"
        f"- Rows written: {row_count}\n"
        "- Limitation: labels are utterance-level dominant-tone labels derived from transcripts, not manually segmented isolated-syllable tone labels. Use this as a reproducible public-data baseline, and describe the limitation in the paper.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mandarin tone metadata from THCHS-30 transcripts.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/thchs30/data_thchs30"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--provenance", type=Path, default=Path("data/metadata_thchs30_provenance.md"))
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-per-tone-split", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.root, args.min_ratio, args.max_per_tone_split, args.output.parent)
    write_metadata(rows, args.output)
    write_provenance(args.provenance, args.root, args.min_ratio, len(rows))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
