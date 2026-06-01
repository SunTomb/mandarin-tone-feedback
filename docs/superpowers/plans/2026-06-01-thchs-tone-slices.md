# THCHS Tone Slice Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a reproducible syllable/short-word Mandarin tone dataset from THCHS-30 transcripts and audio, then run acoustic, pretrained-representation, and fusion experiments on the derived slice metadata.

**Architecture:** Add a small, testable slice-preparation layer under `experiments/` that resolves THCHS transcript references, extracts pinyin tone labels, creates approximate audio intervals, filters slices by duration/RMS/F0 availability, and writes validated metadata plus provenance. Reuse the existing acoustic, representation, fusion, and evaluation scripts by making them consume the derived `data/metadata.csv`.

**Tech Stack:** Python 3.10/3.11, pathlib/csv/wave, numpy, soundfile, librosa, scikit-learn, torch/transformers for later representation extraction, Bash over SSH for server execution. Do not use PowerShell.

---

## File map

- Create: `experiments/thchs_slices.py` — reusable pure functions for THCHS transcript resolution, pinyin tone extraction, interval generation, audio slicing, and metadata row creation.
- Create: `experiments/tests/test_thchs_slices.py` — unit tests for the pure THCHS slice helpers.
- Modify: `experiments/dataset.py` — accept extended metadata columns while preserving the required core columns.
- Modify: `experiments/tests/test_dataset_validation.py` — verify metadata with extra provenance columns still validates.
- Create: `experiments/prepare_thchs_slices.py` — CLI script that generates slice wav files, metadata, summary JSON, and provenance README.
- Create: `experiments/tests/test_prepare_thchs_slices.py` — smoke tests for CLI helper functions without processing the full corpus.
- Modify: `experiments/run_full_pipeline.sh` — point the full training pipeline at the slice metadata and new output run names.
- Create: `data/processed/thchs30_slices/README.md` on the server — generated provenance for the derived slice dataset.
- Modify: `data/PUBLIC_AUDIO_SOURCES.md` — add the final derived-dataset note after slice generation.

---

## Task 1: Add pure THCHS slice helpers

**Files:**
- Create: `experiments/thchs_slices.py`
- Create: `experiments/tests/test_thchs_slices.py`

- [ ] **Step 1: Write failing tests**

Create `experiments/tests/test_thchs_slices.py` with:

```python
from pathlib import Path

import numpy as np

from thchs_slices import (
    choose_duration_filtered_interval,
    extract_pinyin_tone,
    extract_tonal_pinyin_tokens,
    resolve_trn_path,
    speaker_id_from_stem,
    token_intervals,
)


def test_extract_pinyin_tone_returns_base_and_tone():
    token = extract_pinyin_tone("ma3")

    assert token.base == "ma"
    assert token.tone == 3
    assert token.original == "ma3"


def test_extract_tonal_pinyin_tokens_ignores_neutral_tone():
    tokens = extract_tonal_pinyin_tokens("lv4 shi4 de5 ma3")

    assert [token.original for token in tokens] == ["lv4", "shi4", "ma3"]
    assert [token.tone for token in tokens] == [4, 4, 3]


def test_resolve_trn_path_follows_reference_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "train"
    data_dir.mkdir()
    split_dir.mkdir()
    real = data_dir / "A11_0.wav.trn"
    ref = split_dir / "A11_0.wav.trn"
    real.write_text("字\nzi4\nz iy4\n", encoding="utf-8")
    ref.write_text("../data/A11_0.wav.trn\n", encoding="utf-8")

    assert resolve_trn_path(ref) == real.resolve()


def test_token_intervals_cover_duration_in_order():
    intervals = token_intervals(token_count=4, duration=2.0, margin=0.1)

    assert intervals == [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1.9)]


def test_choose_duration_filtered_interval_rejects_too_short():
    assert choose_duration_filtered_interval(0.0, 0.1, min_duration=0.18, max_duration=1.2) is None


def test_choose_duration_filtered_interval_accepts_valid_interval():
    assert choose_duration_filtered_interval(0.0, 0.5, min_duration=0.18, max_duration=1.2) == (0.0, 0.5)


def test_speaker_id_from_stem_uses_prefix_before_underscore():
    assert speaker_id_from_stem("A11_123") == "A11"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `experiments/`:

```bash
python -m pytest tests/test_thchs_slices.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'thchs_slices'`.

If the current environment lacks `pytest`, run this import check instead and record that pytest is unavailable:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('thchs_slices'))
PY
```

Expected before implementation: prints `None`.

- [ ] **Step 3: Implement pure helper module**

Create `experiments/thchs_slices.py` with:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PINYIN_TONE_RE = re.compile(r"^([a-züv:]+)([1-5])$", re.IGNORECASE)


@dataclass(frozen=True)
class PinyinToneToken:
    original: str
    base: str
    tone: int


def extract_pinyin_tone(token: str) -> PinyinToneToken | None:
    match = PINYIN_TONE_RE.match(token.strip())
    if match is None:
        return None
    tone = int(match.group(2))
    if tone == 5:
        return None
    if tone not in {1, 2, 3, 4}:
        return None
    return PinyinToneToken(original=token.strip(), base=match.group(1), tone=tone)


def extract_tonal_pinyin_tokens(pinyin_line: str) -> list[PinyinToneToken]:
    tokens: list[PinyinToneToken] = []
    for raw_token in pinyin_line.split():
        token = extract_pinyin_tone(raw_token)
        if token is not None:
            tokens.append(token)
    return tokens


def resolve_trn_path(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 1 and lines[0].startswith("../"):
        return (path.parent / lines[0]).resolve()
    return path.resolve()


def read_thchs_trn(path: Path) -> tuple[str, str, str]:
    resolved = resolve_trn_path(path)
    lines = resolved.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"THCHS transcript must contain at least 3 lines: {resolved}")
    return lines[0].strip(), lines[1].strip(), lines[2].strip()


def token_intervals(token_count: int, duration: float, margin: float = 0.0) -> list[tuple[float, float]]:
    if token_count <= 0 or duration <= 0:
        return []
    usable_start = min(margin, duration / 2)
    usable_end = max(usable_start, duration - margin)
    usable_duration = usable_end - usable_start
    step = usable_duration / token_count
    return [(usable_start + index * step, usable_start + (index + 1) * step) for index in range(token_count)]


def choose_duration_filtered_interval(
    start_sec: float,
    end_sec: float,
    min_duration: float,
    max_duration: float,
) -> tuple[float, float] | None:
    duration = end_sec - start_sec
    if duration < min_duration or duration > max_duration:
        return None
    return (start_sec, end_sec)


def speaker_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]
```

- [ ] **Step 4: Run helper tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_thchs_slices.py -v
```

Expected: PASS.

If pytest is unavailable, run:

```bash
python - <<'PY'
from pathlib import Path
from thchs_slices import extract_tonal_pinyin_tokens, token_intervals
print([token.tone for token in extract_tonal_pinyin_tokens('ma1 ma2 ma3 ma4 de5')])
print(token_intervals(4, 2.0, margin=0.1))
PY
```

Expected:

```text
[1, 2, 3, 4]
[(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1.9)]
```

- [ ] **Step 5: Commit if the project becomes a git repository**

```bash
git add experiments/thchs_slices.py experiments/tests/test_thchs_slices.py
git commit -m "feat: add THCHS tone slice helpers"
```

---

## Task 2: Allow extended metadata columns

**Files:**
- Modify: `experiments/dataset.py`
- Modify: `experiments/tests/test_dataset_validation.py`

- [ ] **Step 1: Add a failing test for extended metadata columns**

Append this test to `experiments/tests/test_dataset_validation.py`:

```python

def test_load_validated_metadata_preserves_extra_columns(tmp_path: Path):
    train_audio = tmp_path / "ma1.wav"
    test_audio = tmp_path / "ma2.wav"
    train_audio.write_bytes(b"RIFF")
    test_audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split,source_utterance,start_sec,end_sec,source_dataset,label_method\n"
        f"{train_audio.name},妈,ma1,1,speaker01,train,utt1.wav,0.0,0.4,THCHS-30,pinyin_tone_digit_with_energy_slicing\n"
        f"{test_audio.name},麻,ma2,2,speaker01,test,utt2.wav,0.2,0.7,THCHS-30,pinyin_tone_digit_with_energy_slicing\n",
        encoding="utf-8",
    )

    rows = load_validated_metadata(metadata)

    assert rows[0]["source_utterance"] == "utt1.wav"
    assert rows[0]["start_sec"] == "0.0"
    assert rows[0]["label_method"] == "pinyin_tone_digit_with_energy_slicing"
```

- [ ] **Step 2: Run test to verify current behavior**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_validation.py::test_load_validated_metadata_preserves_extra_columns -v
```

Expected before implementation: FAIL because `load_validated_metadata` drops extra columns.

- [ ] **Step 3: Preserve extra columns in metadata loader**

In `experiments/dataset.py`, replace the `rows.append({...})` block inside `load_validated_metadata` with:

```python
            parsed_row: dict[str, Any] = dict(row)
            parsed_row["audio_path"] = audio_path
            parsed_row["tone"] = tone
            parsed_row["split"] = split
            rows.append(parsed_row)
```

Keep all existing validation before this block.

- [ ] **Step 4: Run metadata validation tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_validation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if the project becomes a git repository**

```bash
git add experiments/dataset.py experiments/tests/test_dataset_validation.py
git commit -m "feat: preserve derived metadata provenance columns"
```

---

## Task 3: Implement THCHS slice generation CLI

**Files:**
- Create: `experiments/prepare_thchs_slices.py`
- Create: `experiments/tests/test_prepare_thchs_slices.py`

- [ ] **Step 1: Write smoke tests for row construction**

Create `experiments/tests/test_prepare_thchs_slices.py` with:

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_thchs_slices import SliceConfig, build_slice_rows_for_file


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
    assert rows[0]["source_utterance"] == str(source_wav)
    assert rows[0]["label_method"] == "pinyin_tone_digit_with_energy_slicing"
```

- [ ] **Step 2: Run test to verify it fails**

Run from `experiments/`:

```bash
python -m pytest tests/test_prepare_thchs_slices.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_thchs_slices'`.

- [ ] **Step 3: Implement slice generation CLI**

Create `experiments/prepare_thchs_slices.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from thchs_slices import (
    choose_duration_filtered_interval,
    extract_tonal_pinyin_tokens,
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


def build_slice_rows_for_file(
    wav_path: Path,
    trn_path: Path,
    output_root: Path,
    split: str,
    config: SliceConfig,
) -> list[dict[str, str]]:
    text_line, pinyin_line, _phone_line = read_thchs_trn(trn_path)
    tokens = extract_tonal_pinyin_tokens(pinyin_line)
    if not tokens:
        return []

    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    duration = float(len(audio) / sr) if sr else 0.0
    intervals = token_intervals(len(tokens), duration, margin=config.margin)
    characters = [piece for piece in text_line.split() if piece.strip()]
    rows: list[dict[str, str]] = []

    for index, (token, interval) in enumerate(zip(tokens, intervals)):
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
                "source_utterance": str(wav_path),
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


def build_dataset(
    root: Path,
    output_root: Path,
    config: SliceConfig,
    train_quota: int,
    val_quota: int,
    test_quota: int,
) -> list[dict[str, str]]:
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
```

- [ ] **Step 4: Run preparation tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_prepare_thchs_slices.py -v
```

Expected: PASS.

If pytest is unavailable, run a smoke script from `experiments/` using temporary files and verify four rows are created.

- [ ] **Step 5: Commit if the project becomes a git repository**

```bash
git add experiments/prepare_thchs_slices.py experiments/tests/test_prepare_thchs_slices.py
git commit -m "feat: prepare THCHS-derived tone slices"
```

---

## Task 4: Generate a pilot THCHS slice dataset on Sui-3-Wu

**Files:**
- Writes on server: `data/processed/thchs30_slices/`
- Writes on server: `data/metadata.csv`
- Writes on server: `data/processed/thchs30_slices/summary.json`

- [ ] **Step 1: Sync slice scripts to server**

Run locally from the project root using Bash:

```bash
tar -czf - experiments/thchs_slices.py experiments/prepare_thchs_slices.py experiments/tests/test_thchs_slices.py experiments/tests/test_prepare_thchs_slices.py | ssh Sui-3-Wu 'tar -xzf - -C /NAS/yesh/mandarin-tone-feedback'
```

Expected: command exits successfully.

- [ ] **Step 2: Run helper smoke check on server**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && cd experiments && python - <<'"'"'PY'"'"'
from thchs_slices import extract_tonal_pinyin_tokens
print([token.tone for token in extract_tonal_pinyin_tokens("ma1 ma2 ma3 ma4 de5")])
PY'
```

Expected:

```text
[1, 2, 3, 4]
```

- [ ] **Step 3: Generate small pilot slices**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && python experiments/prepare_thchs_slices.py --root data/raw/thchs30/data_thchs30 --output-root data/processed/thchs30_slices --metadata data/metadata.csv --summary data/processed/thchs30_slices/summary.json --train-quota 80 --val-quota 20 --test-quota 20 --min-voiced-ratio 0.2'
```

Expected: prints JSON with `rows`, `summary`, and `metadata`.

- [ ] **Step 4: Validate generated metadata**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && python - <<'"'"'PY'"'"'
from pathlib import Path
from dataset import load_validated_metadata, summarize_metadata
rows = load_validated_metadata(Path("data/metadata.csv"))
print(summarize_metadata(rows))
print(Path("data/processed/thchs30_slices/summary.json").read_text(encoding="utf-8"))
PY'
```

Expected: metadata validates and reports non-empty train/test splits. If any tone count is low, record that actual count rather than fabricating samples.

- [ ] **Step 5: Inspect generated slice directory size**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && du -sh data/processed/thchs30_slices && find data/processed/thchs30_slices -type f -name "*.wav" | wc -l'
```

Expected: reports manageable disk usage and a slice count matching metadata rows.

---

## Task 5: Run acoustic baseline on slice metadata

**Files:**
- Reads: `data/metadata.csv`
- Writes: `outputs/thchs_slices_pilot/acoustic/`

- [ ] **Step 1: Run acoustic baseline**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && python experiments/train_acoustic_baseline.py --metadata data/metadata.csv --output-dir outputs/thchs_slices_pilot/acoustic --classifier random_forest'
```

Expected: prints JSON with `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/thchs_slices_pilot/acoustic/acoustic_metrics.json`.

- [ ] **Step 2: Verify acoustic outputs**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f outputs/thchs_slices_pilot/acoustic/acoustic_metrics.json && test -f outputs/thchs_slices_pilot/acoustic/acoustic_confusion_matrix.csv && test -f outputs/thchs_slices_pilot/acoustic/processed/acoustic_features.npy && echo acoustic_ok'
```

Expected: prints `acoustic_ok`.

- [ ] **Step 3: Inspect acoustic metrics without editing paper yet**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && python - <<'"'"'PY'"'"'
import json
from pathlib import Path
metrics = json.loads(Path("outputs/thchs_slices_pilot/acoustic/acoustic_metrics.json").read_text(encoding="utf-8"))
print(metrics["accuracy"], metrics["macro_f1"])
PY'
```

Expected: prints real values from the pilot run. Do not copy them into paper prose until all models finish.

---

## Task 6: Run pretrained representation baseline on slice metadata

**Files:**
- Reads: `data/metadata.csv`
- Writes: `outputs/thchs_slices_pilot/representation/`

- [ ] **Step 1: Check GPU availability**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'nvidia-smi'
```

Expected: at least one GPU is free enough for frozen representation extraction.

- [ ] **Step 2: Run representation extraction and classifier**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && CUDA_VISIBLE_DEVICES=0 python experiments/train_representation_model.py --metadata data/metadata.csv --output-dir outputs/thchs_slices_pilot/representation --model-name facebook/wav2vec2-base --batch-size 8 --device cuda'
```

Expected: prints JSON with `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/thchs_slices_pilot/representation/representation_metrics.json`.

- [ ] **Step 3: Verify representation outputs**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f outputs/thchs_slices_pilot/representation/representation_metrics.json && test -f outputs/thchs_slices_pilot/representation/representation_confusion_matrix.csv && test -f outputs/thchs_slices_pilot/representation/processed/representation_features.npy && echo representation_ok'
```

Expected: prints `representation_ok`.

---

## Task 7: Run fusion model on slice outputs

**Files:**
- Reads: acoustic and representation processed arrays.
- Writes: `outputs/thchs_slices_pilot/fusion/`

- [ ] **Step 1: Run fusion model**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && python experiments/train_fusion_model.py --acoustic-features outputs/thchs_slices_pilot/acoustic/processed/acoustic_features.npy --representation-features outputs/thchs_slices_pilot/representation/processed/representation_features.npy --labels outputs/thchs_slices_pilot/acoustic/processed/tone_labels.npy --splits outputs/thchs_slices_pilot/acoustic/processed/splits.npy --output-dir outputs/thchs_slices_pilot/fusion'
```

Expected: prints JSON with `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/thchs_slices_pilot/fusion/fusion_metrics.json`.

- [ ] **Step 2: Verify fusion outputs**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f outputs/thchs_slices_pilot/fusion/fusion_metrics.json && test -f outputs/thchs_slices_pilot/fusion/fusion_confusion_matrix.csv && echo fusion_ok'
```

Expected: prints `fusion_ok`.

---

## Task 8: Generate representation visualization and model comparison artifact

**Files:**
- Writes: `paper/figures/representation_tsne_thchs_slices_pilot.png`
- Writes: `paper/figures/model_comparison_thchs_slices_pilot.md`

- [ ] **Step 1: Generate t-SNE figure**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && source ./activate.sh && python experiments/analyze_representations.py --features outputs/thchs_slices_pilot/representation/processed/representation_features.npy --labels outputs/thchs_slices_pilot/representation/processed/tone_labels.npy --output paper/figures/representation_tsne_thchs_slices_pilot.png'
```

Expected: writes `paper/figures/representation_tsne_thchs_slices_pilot.png`.

- [ ] **Step 2: Generate comparison markdown from real metrics**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && python - <<'"'"'PY'"'"'
import json
from pathlib import Path
run = "thchs_slices_pilot"
rows = []
for label, name in [("Acoustic", "acoustic"), ("Pretrained", "representation"), ("Fusion", "fusion")]:
    path = Path("outputs") / run / name / f"{name}_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append((label, metrics["accuracy"], metrics["macro_f1"], str(path)))
lines = ["# Model comparison from thchs_slices_pilot", "", "| Model | Accuracy | Macro-F1 | Source |", "| --- | ---: | ---: | --- |"]
for label, accuracy, macro_f1, source in rows:
    lines.append(f"| {label} | {accuracy:.4f} | {macro_f1:.4f} | `{source}` |")
Path("paper/figures/model_comparison_thchs_slices_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY'
```

Expected: writes model comparison markdown using only real metrics files.

- [ ] **Step 3: Sync paper artifacts back to local**

Run locally using Bash:

```bash
scp Sui-3-Wu:/NAS/yesh/mandarin-tone-feedback/paper/figures/model_comparison_thchs_slices_pilot.md paper/figures/
scp Sui-3-Wu:/NAS/yesh/mandarin-tone-feedback/paper/figures/representation_tsne_thchs_slices_pilot.png paper/figures/
scp Sui-3-Wu:/NAS/yesh/mandarin-tone-feedback/data/processed/thchs30_slices/summary.json data/processed_thchs30_slices_summary.json
```

Expected: local paper artifacts are available for paper writing.

---

## Task 9: Update documentation with derived dataset provenance

**Files:**
- Modify: `data/PUBLIC_AUDIO_SOURCES.md`
- Modify: `paper/outline.md`

- [ ] **Step 1: Update public source notes**

Append this section to `data/PUBLIC_AUDIO_SOURCES.md`:

```markdown
## Derived THCHS-30 tone-slice subset

The final experiment subset is derived from THCHS-30 rather than provided directly by THCHS-30. We use the pinyin tone digits in `.wav.trn` transcripts and deterministic approximate time slicing to create short tone-labeled wav clips. The derived metadata includes `source_utterance`, `start_sec`, `end_sec`, `source_dataset`, and `label_method` columns.

This should be described in the paper as a transcript-derived syllable/short-word subset. It should not be described as a manually segmented isolated-syllable corpus.
```

- [ ] **Step 2: Add cautious result wording to paper outline**

Append to `paper/outline.md` after the result slots:

```markdown
## Dataset note for experiment section

The experiment uses a derived subset from THCHS-30 / OpenSLR SLR18. Tone labels are obtained from pinyin tone digits in the transcript, while slice boundaries are generated by a deterministic approximate slicing procedure and filtered by duration, RMS energy, and voiced-frame availability. Therefore, the results should be interpreted as a reproducible public-data baseline for tone-related modeling, not as a learner-production evaluation on manually segmented isolated syllables.
```

- [ ] **Step 3: Sync documentation to server**

Run locally using Bash:

```bash
tar -czf - data/PUBLIC_AUDIO_SOURCES.md paper/outline.md | ssh Sui-3-Wu 'tar -xzf - -C /NAS/yesh/mandarin-tone-feedback'
```

Expected: server docs are updated.

---

## Task 10: Final verification before claiming completion

**Files:**
- No new source edits unless verification fails.

- [ ] **Step 1: Verify server data artifacts**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f data/downloads/data_thchs30.tgz && test -f data/metadata.csv && test -f data/processed/thchs30_slices/summary.json && echo data_artifacts_ok'
```

Expected: prints `data_artifacts_ok`.

- [ ] **Step 2: Verify experiment artifacts**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f outputs/thchs_slices_pilot/acoustic/acoustic_metrics.json && test -f outputs/thchs_slices_pilot/representation/representation_metrics.json && test -f outputs/thchs_slices_pilot/fusion/fusion_metrics.json && echo experiment_artifacts_ok'
```

Expected: prints `experiment_artifacts_ok`.

- [ ] **Step 3: Verify paper artifacts**

Run locally using Bash:

```bash
ssh Sui-3-Wu 'cd /NAS/yesh/mandarin-tone-feedback && test -f paper/figures/model_comparison_thchs_slices_pilot.md && test -f paper/figures/representation_tsne_thchs_slices_pilot.png && echo paper_artifacts_ok'
```

Expected: prints `paper_artifacts_ok`.

- [ ] **Step 4: Report completion with limitations**

Report:

- slice count by tone and split;
- acoustic, pretrained, and fusion metrics source files;
- figure paths;
- explicit limitation that boundaries are approximate and transcript-derived.

Do not claim learner-pronunciation validity or manually segmented labels.

---

## Self-review

- Spec coverage: covers THCHS transcript parsing, tone-digit extraction, slice generation, quality filtering, metadata validation, acoustic/representation/fusion experiments, visualization, and paper provenance.
- Placeholder scan: no fabricated metrics, no unbounded “TODO” steps, and all commands are explicit.
- Type consistency: metadata uses required core columns plus provenance fields; `tone` remains a string in CSV and is parsed to int by `experiments/dataset.py`.
- Scope control: this plan focuses on reproducible data and experiments; it does not expand into ASR, accounts, dashboards, or production GPU inference.
