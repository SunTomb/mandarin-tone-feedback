# Server Sync and Full Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the local Mandarin Tone Feedback project to `/NAS/yesh/mandarin-tone-feedback`, prepare a real metadata-driven audio dataset pipeline, and run acoustic, pretrained-representation, and fusion tone-classification experiments with real outputs for the course paper and presentation.

**Architecture:** Keep the website prototype lightweight while moving offline training to the NAS/GPU server. Add focused experiment modules for metadata validation, feature extraction, model evaluation, and figure generation, then run them on the server under timestamped output directories. Preserve paper integrity by updating paper/PPT materials only from generated metrics and figures.

**Tech Stack:** Bash for synchronization and server commands; Python 3.11, numpy, pandas, librosa, soundfile, scikit-learn, torch, transformers, matplotlib, pytest; FastAPI/React code remains part of the synchronized project but the first training work is offline.

---

## File map

### Existing files to keep and synchronize

- `CLAUDE.md`: project safety, NAS, GPU, and research framing rules.
- `README.md`: project overview.
- `design_spec.md`: language-science framing and system design.
- `implementation_plan.md`: original local implementation sequence.
- `proposal_ppt_outline.md`:开题报告结构.
- `backend/`: API, audio/F0 utilities, feedback rules, backend tests.
- `frontend/`: Vite/React demo UI and tests.
- `experiments/configs/*.yaml`: experiment configuration files.
- `paper/outline.md`: paper skeleton with empty result fields.
- `paper/figures/README.md`: figure requirements.

### New or modified files in this plan

- Create: `experiments/dataset.py` — metadata validation, split extraction, and dataset summary.
- Create: `experiments/features_acoustic.py` — reusable audio loading and acoustic feature extraction for experiments.
- Create: `experiments/evaluation.py` — shared classifier training, metric serialization, and confusion-matrix saving.
- Modify: `experiments/train_acoustic_baseline.py` — run real acoustic baseline from metadata.
- Modify: `experiments/train_representation_model.py` — extract frozen pretrained embeddings and evaluate them.
- Modify: `experiments/train_fusion_model.py` — align and concatenate acoustic and representation features, then evaluate.
- Modify: `experiments/analyze_representations.py` — generate representation visualization from saved embeddings.
- Create: `experiments/tests/test_dataset_validation.py` — tests for metadata validation.
- Create: `experiments/tests/test_evaluation_outputs.py` — tests for metric and confusion-matrix outputs.
- Create: `experiments/tests/test_acoustic_features.py` — tests for feature extraction from synthetic audio.
- Create: `experiments/run_full_pipeline.sh` — server-side orchestration script for the three experiment paths.
- Create: `docs/server_training_runbook.md` — operator runbook for SSH sync, NAS environment, GPU checks, and training commands.

---

## Task 1: Create metadata validation module

**Files:**
- Create: `experiments/dataset.py`
- Test: `experiments/tests/test_dataset_validation.py`

- [ ] **Step 1: Write metadata validation tests**

Create `experiments/tests/test_dataset_validation.py` with:

```python
from pathlib import Path

import pytest

from dataset import DatasetValidationError, load_validated_metadata, summarize_metadata


def write_csv(path: Path, rows: str) -> None:
    path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split\n" + rows,
        encoding="utf-8",
    )


def test_load_validated_metadata_accepts_existing_audio(tmp_path: Path):
    audio = tmp_path / "ma1.wav"
    audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, f"{audio.name},妈,ma1,1,speaker01,train\n")

    rows = load_validated_metadata(metadata)

    assert len(rows) == 1
    assert rows[0]["tone"] == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["audio_path"] == audio


def test_load_validated_metadata_rejects_missing_audio(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, "missing.wav,妈,ma1,1,speaker01,train\n")

    with pytest.raises(DatasetValidationError, match="missing audio file"):
        load_validated_metadata(metadata)


def test_load_validated_metadata_rejects_invalid_tone(tmp_path: Path):
    audio = tmp_path / "ma5.wav"
    audio.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, f"{audio.name},吗,ma5,5,speaker01,train\n")

    with pytest.raises(DatasetValidationError, match="invalid tone"):
        load_validated_metadata(metadata)


def test_summarize_metadata_counts_split_and_tone(tmp_path: Path):
    audio1 = tmp_path / "ma1.wav"
    audio2 = tmp_path / "ma2.wav"
    audio1.write_bytes(b"RIFF")
    audio2.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(
        metadata,
        f"{audio1.name},妈,ma1,1,speaker01,train\n{audio2.name},麻,ma2,2,speaker02,test\n",
    )
    rows = load_validated_metadata(metadata)

    summary = summarize_metadata(rows)

    assert summary["total"] == 2
    assert summary["by_split"] == {"train": 1, "test": 1}
    assert summary["by_tone"] == {1: 1, 2: 1}
    assert summary["by_speaker"] == {"speaker01": 1, "speaker02": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_validation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dataset'`.

- [ ] **Step 3: Implement metadata validation**

Create `experiments/dataset.py` with:

```python
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
```

- [ ] **Step 4: Adjust tests for required train/test split**

Modify `test_load_validated_metadata_accepts_existing_audio` so its metadata has both train and test rows:

```python
def test_load_validated_metadata_accepts_existing_audio(tmp_path: Path):
    audio_train = tmp_path / "ma1.wav"
    audio_test = tmp_path / "ma2.wav"
    audio_train.write_bytes(b"RIFF")
    audio_test.write_bytes(b"RIFF")
    metadata = tmp_path / "metadata.csv"
    write_csv(
        metadata,
        f"{audio_train.name},妈,ma1,1,speaker01,train\n{audio_test.name},麻,ma2,2,speaker01,test\n",
    )

    rows = load_validated_metadata(metadata)

    assert len(rows) == 2
    assert rows[0]["tone"] == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["audio_path"] == audio_train
```

Modify the other tests that call `load_validated_metadata` and expect success so they include at least one train row and one test row.

- [ ] **Step 5: Run metadata tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_validation.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit if this is later placed in a git repository**

The current local directory is not a git repository. If the project is later initialized as git, commit with:

```bash
git add experiments/dataset.py experiments/tests/test_dataset_validation.py
git commit -m "feat: validate tone metadata"
```

---

## Task 2: Create shared evaluation utilities

**Files:**
- Create: `experiments/evaluation.py`
- Test: `experiments/tests/test_evaluation_outputs.py`

- [ ] **Step 1: Write evaluation tests**

Create `experiments/tests/test_evaluation_outputs.py` with:

```python
import json
from pathlib import Path

import numpy as np

from evaluation import evaluate_predictions, save_confusion_matrix, save_metrics


def test_evaluate_predictions_returns_accuracy_and_macro_f1():
    labels = np.array([1, 2, 2, 4])
    predictions = np.array([1, 2, 3, 4])

    metrics = evaluate_predictions(labels, predictions)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "classification_report" in metrics


def test_save_metrics_writes_json(tmp_path: Path):
    metrics = {"accuracy": 1.0, "macro_f1": 1.0, "classification_report": {}}
    output = tmp_path / "metrics.json"

    save_metrics(metrics, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == 1.0


def test_save_confusion_matrix_writes_csv(tmp_path: Path):
    labels = np.array([1, 2, 2, 4])
    predictions = np.array([1, 2, 3, 4])
    output = tmp_path / "confusion_matrix.csv"

    save_confusion_matrix(labels, predictions, output)

    text = output.read_text(encoding="utf-8")
    assert "true_label" in text
    assert "predicted_1" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run from `experiments/`:

```bash
python -m pytest tests/test_evaluation_outputs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Implement evaluation utilities**

Create `experiments/evaluation.py` with:

```python
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


TONE_LABELS = [1, 2, 3, 4]


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=TONE_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=TONE_LABELS)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *[f"predicted_{label}" for label in TONE_LABELS]])
        for label, row in zip(TONE_LABELS, matrix.tolist()):
            writer.writerow([label, *row])
```

- [ ] **Step 4: Run evaluation tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_evaluation_outputs.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if this is later placed in a git repository**

```bash
git add experiments/evaluation.py experiments/tests/test_evaluation_outputs.py
git commit -m "feat: add experiment evaluation outputs"
```

---

## Task 3: Create acoustic feature extraction for experiments

**Files:**
- Create: `experiments/features_acoustic.py`
- Test: `experiments/tests/test_acoustic_features.py`

- [ ] **Step 1: Write acoustic feature tests**

Create `experiments/tests/test_acoustic_features.py` with:

```python
import numpy as np

from features_acoustic import summarize_f0_contour


def test_summarize_f0_contour_returns_expected_values():
    contour = np.array([1.0, 2.0, 4.0], dtype=float)

    features = summarize_f0_contour(contour, duration=0.5, voiced_ratio=0.75)

    assert features["f0_start"] == 1.0
    assert features["f0_mid"] == 2.0
    assert features["f0_end"] == 4.0
    assert features["f0_range"] == 3.0
    assert features["f0_slope"] == 3.0
    assert features["duration"] == 0.5
    assert features["voiced_ratio"] == 0.75


def test_summarize_f0_contour_handles_empty_contour():
    contour = np.array([], dtype=float)

    features = summarize_f0_contour(contour, duration=0.0, voiced_ratio=0.0)

    assert features["f0_start"] == 0.0
    assert features["f0_mid"] == 0.0
    assert features["f0_end"] == 0.0
    assert features["f0_range"] == 0.0
    assert features["f0_slope"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run from `experiments/`:

```bash
python -m pytest tests/test_acoustic_features.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'features_acoustic'`.

- [ ] **Step 3: Implement acoustic feature utilities**

Create `experiments/features_acoustic.py` with:

```python
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


FEATURE_NAMES = [
    "f0_start",
    "f0_mid",
    "f0_end",
    "f0_range",
    "f0_slope",
    "duration",
    "voiced_ratio",
]


def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int, float]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    duration = float(len(audio) / sr) if sr else 0.0
    return audio, sr, duration


def normalize_f0(f0: np.ndarray) -> np.ndarray:
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return np.array([], dtype=float)
    low = float(np.min(voiced))
    high = float(np.max(voiced))
    if high == low:
        return np.full(voiced.shape, 3.0, dtype=float)
    return 1.0 + 4.0 * (voiced - low) / (high - low)


def extract_f0_contour(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    if f0 is None:
        return np.array([], dtype=float), 0.0
    voiced_ratio = float(np.isfinite(f0).sum() / len(f0)) if len(f0) else 0.0
    return normalize_f0(f0), voiced_ratio


def summarize_f0_contour(contour: np.ndarray, duration: float, voiced_ratio: float) -> dict[str, float]:
    if contour.size == 0:
        return {
            "f0_start": 0.0,
            "f0_mid": 0.0,
            "f0_end": 0.0,
            "f0_range": 0.0,
            "f0_slope": 0.0,
            "duration": float(duration),
            "voiced_ratio": float(voiced_ratio),
        }
    start = float(contour[0])
    mid = float(contour[len(contour) // 2])
    end = float(contour[-1])
    return {
        "f0_start": start,
        "f0_mid": mid,
        "f0_end": end,
        "f0_range": float(np.max(contour) - np.min(contour)),
        "f0_slope": end - start,
        "duration": float(duration),
        "voiced_ratio": float(voiced_ratio),
    }


def extract_acoustic_feature_row(path: Path) -> list[float]:
    audio, sr, duration = load_audio(path)
    contour, voiced_ratio = extract_f0_contour(audio, sr)
    summary = summarize_f0_contour(contour, duration, voiced_ratio)
    return [summary[name] for name in FEATURE_NAMES]
```

- [ ] **Step 4: Run acoustic feature tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_acoustic_features.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit if this is later placed in a git repository**

```bash
git add experiments/features_acoustic.py experiments/tests/test_acoustic_features.py
git commit -m "feat: extract acoustic tone features"
```

---

## Task 4: Replace acoustic baseline skeleton with real metadata-driven training

**Files:**
- Modify: `experiments/train_acoustic_baseline.py`
- Test: `experiments/tests/test_dataset_loading.py`

- [ ] **Step 1: Update dataset loading test to use validated metadata**

Replace `experiments/tests/test_dataset_loading.py` with:

```python
from pathlib import Path

from train_acoustic_baseline import load_metadata


def test_load_metadata_reads_validated_rows(tmp_path: Path):
    train_audio = tmp_path / "ma1.wav"
    test_audio = tmp_path / "ma2.wav"
    train_audio.write_bytes(b"RIFF")
    test_audio.write_bytes(b"RIFF")
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "audio_path,text,pinyin,tone,speaker_id,split\n"
        f"{train_audio.name},妈,ma1,1,s1,train\n"
        f"{test_audio.name},麻,ma2,2,s1,test\n",
        encoding="utf-8",
    )

    rows = load_metadata(csv_path)

    assert len(rows) == 2
    assert rows[0]["tone"] == 1
    assert rows[0]["pinyin"] == "ma1"
```

- [ ] **Step 2: Run acoustic baseline tests to verify current mismatch**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_loading.py -v
```

Expected: FAIL because the existing `load_metadata` returns string tone values and does not validate train/test splits.

- [ ] **Step 3: Implement metadata-driven acoustic baseline**

Replace `experiments/train_acoustic_baseline.py` with:

```python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import load_validated_metadata, split_indices, summarize_metadata
from evaluation import evaluate_predictions, save_confusion_matrix, save_metrics
from features_acoustic import FEATURE_NAMES, extract_acoustic_feature_row


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return load_validated_metadata(path)


def build_classifier(name: str, random_state: int):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced")
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
        )
    raise ValueError(f"unsupported classifier: {name}")


def extract_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([extract_acoustic_feature_row(row["audio_path"]) for row in rows], dtype=float)


def run_acoustic_baseline(
    metadata_path: Path,
    output_dir: Path,
    classifier_name: str = "random_forest",
    random_state: int = 42,
) -> dict[str, Any]:
    rows = load_metadata(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_feature_matrix(rows)
    labels = np.array([row["tone"] for row in rows], dtype=int)
    train_idx = split_indices(rows, "train")
    test_idx = split_indices(rows, "test")

    model = build_classifier(classifier_name, random_state)
    model.fit(features[train_idx], labels[train_idx])
    predictions = model.predict(features[test_idx])
    metrics = evaluate_predictions(labels[test_idx], predictions)
    metrics["dataset_summary"] = summarize_metadata(rows)
    metrics["feature_names"] = FEATURE_NAMES
    metrics["classifier"] = classifier_name

    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "acoustic_features.npy", features)
    np.save(processed_dir / "tone_labels.npy", labels)
    (processed_dir / "rows.json").write_text(
        json.dumps(
            [
                {
                    **row,
                    "audio_path": str(row["audio_path"]),
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_metrics(metrics, output_dir / "acoustic_metrics.json")
    save_confusion_matrix(labels[test_idx], predictions, output_dir / "acoustic_confusion_matrix.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an acoustic-feature Mandarin tone classifier.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_acoustic")
    metrics = run_acoustic_baseline(args.metadata, output_dir, args.classifier, args.random_state)
    print(json.dumps({"output_dir": str(output_dir), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run experiment tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_dataset_loading.py tests/test_dataset_validation.py tests/test_acoustic_features.py tests/test_evaluation_outputs.py -v
```

Expected: PASS.

- [ ] **Step 5: Run a smoke command only after real `data/metadata.csv` exists**

Run from project root on the server:

```bash
python experiments/train_acoustic_baseline.py --metadata data/metadata.csv --output-dir outputs/smoke_acoustic --classifier random_forest
```

Expected: prints JSON containing `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/smoke_acoustic/acoustic_metrics.json`.

- [ ] **Step 6: Commit if this is later placed in a git repository**

```bash
git add experiments/train_acoustic_baseline.py experiments/tests/test_dataset_loading.py
git commit -m "feat: train acoustic baseline from metadata"
```

---

## Task 5: Implement pretrained representation extraction and classifier

**Files:**
- Modify: `experiments/train_representation_model.py`

- [ ] **Step 1: Replace representation skeleton with a runnable frozen-encoder pipeline**

Replace `experiments/train_representation_model.py` with:

```python
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor

from dataset import load_validated_metadata, split_indices, summarize_metadata
from evaluation import evaluate_predictions, save_confusion_matrix, save_metrics
from features_acoustic import load_audio


@dataclass
class RepresentationConfig:
    model_name: str = "facebook/wav2vec2-base"
    pooling: str = "mean"
    num_labels: int = 4
    batch_size: int = 4
    random_state: int = 42


def describe_experiment(config: RepresentationConfig) -> str:
    return f"Extract {config.pooling}-pooled hidden states from {config.model_name} for {config.num_labels}-way tone classification."


def _pool_hidden_state(hidden_state: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return hidden_state.mean(dim=1)
    if pooling == "first":
        return hidden_state[:, 0, :]
    raise ValueError(f"unsupported pooling: {pooling}")


def extract_representations(rows: list[dict[str, Any]], config: RepresentationConfig, device: str) -> np.ndarray:
    processor = AutoProcessor.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    for start in range(0, len(rows), config.batch_size):
        batch_rows = rows[start : start + config.batch_size]
        audios = [load_audio(row["audio_path"], target_sr=16000)[0] for row in batch_rows]
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
            pooled = _pool_hidden_state(output.last_hidden_state, config.pooling)
        vectors.append(pooled.cpu().numpy())

    return np.concatenate(vectors, axis=0)


def run_representation_experiment(
    metadata_path: Path,
    output_dir: Path,
    config: RepresentationConfig,
    device: str,
) -> dict[str, Any]:
    rows = load_validated_metadata(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_representations(rows, config, device)
    labels = np.array([row["tone"] for row in rows], dtype=int)
    train_idx = split_indices(rows, "train")
    test_idx = split_indices(rows, "test")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=config.random_state, class_weight="balanced"),
    )
    classifier.fit(features[train_idx], labels[train_idx])
    predictions = classifier.predict(features[test_idx])

    metrics = evaluate_predictions(labels[test_idx], predictions)
    metrics["dataset_summary"] = summarize_metadata(rows)
    metrics["model_name"] = config.model_name
    metrics["pooling"] = config.pooling

    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "representation_features.npy", features)
    np.save(processed_dir / "tone_labels.npy", labels)

    save_metrics(metrics, output_dir / "representation_metrics.json")
    save_confusion_matrix(labels[test_idx], predictions, output_dir / "representation_confusion_matrix.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen speech-representation Mandarin tone classifier.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--pooling", choices=["mean", "first"], default="mean")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RepresentationConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    output_dir = args.output_dir or Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_representation")
    metrics = run_representation_experiment(args.metadata, output_dir, config, args.device)
    print(json.dumps({"output_dir": str(output_dir), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run import smoke test**

Run from `experiments/`:

```bash
python - <<'PY'
from train_representation_model import RepresentationConfig, describe_experiment
print(describe_experiment(RepresentationConfig()))
PY
```

Expected: prints `Extract mean-pooled hidden states from facebook/wav2vec2-base for 4-way tone classification.`

- [ ] **Step 3: Run full tests that do not download pretrained models**

Run from `experiments/`:

```bash
python -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 4: Run representation training only after metadata and GPU check pass**

Run from project root on the server:

```bash
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub
CUDA_VISIBLE_DEVICES=0 python experiments/train_representation_model.py --metadata data/metadata.csv --output-dir outputs/representation_wav2vec2 --model-name facebook/wav2vec2-base --batch-size 4 --device cuda
```

Expected: prints JSON containing `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/representation_wav2vec2/representation_metrics.json` and `outputs/representation_wav2vec2/processed/representation_features.npy`.

- [ ] **Step 5: Commit if this is later placed in a git repository**

```bash
git add experiments/train_representation_model.py
git commit -m "feat: train frozen representation baseline"
```

---

## Task 6: Implement fusion model using saved feature arrays

**Files:**
- Modify: `experiments/train_fusion_model.py`
- Test: `experiments/tests/test_metrics.py`

- [ ] **Step 1: Keep existing fusion utility tests and add evaluation smoke test**

Append to `experiments/tests/test_metrics.py`:

```python
from pathlib import Path

from train_fusion_model import run_fusion_experiment


def test_run_fusion_experiment_writes_metrics(tmp_path: Path):
    acoustic = np.array([[1.0, 2.0], [1.5, 2.5], [4.0, 3.0], [4.5, 3.5]])
    representation = np.array([[0.1, 0.2], [0.2, 0.1], [0.9, 0.8], [0.8, 0.9]])
    labels = np.array([1, 1, 4, 4])
    splits = np.array(["train", "train", "test", "test"])

    acoustic_path = tmp_path / "acoustic.npy"
    representation_path = tmp_path / "representation.npy"
    labels_path = tmp_path / "labels.npy"
    splits_path = tmp_path / "splits.npy"
    output_dir = tmp_path / "outputs"
    np.save(acoustic_path, acoustic)
    np.save(representation_path, representation)
    np.save(labels_path, labels)
    np.save(splits_path, splits)

    metrics = run_fusion_experiment(acoustic_path, representation_path, labels_path, splits_path, output_dir)

    assert "accuracy" in metrics
    assert (output_dir / "fusion_metrics.json").exists()
    assert (output_dir / "fusion_confusion_matrix.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `experiments/`:

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: FAIL because `run_fusion_experiment` is not defined.

- [ ] **Step 3: Implement fusion training**

Replace `experiments/train_fusion_model.py` with:

```python
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_predictions, save_confusion_matrix, save_metrics


def concatenate_features(acoustic: np.ndarray, representation: np.ndarray) -> np.ndarray:
    if acoustic.shape[0] != representation.shape[0]:
        raise ValueError("Feature arrays must have the same number of rows.")
    return np.concatenate([acoustic, representation], axis=1)


def run_fusion_experiment(
    acoustic_features_path: Path,
    representation_features_path: Path,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    acoustic = np.load(acoustic_features_path)
    representation = np.load(representation_features_path)
    labels = np.load(labels_path)
    splits = np.load(splits_path, allow_pickle=True)

    features = concatenate_features(acoustic, representation)
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Fusion training requires non-empty train and test splits.")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
    )
    classifier.fit(features[train_idx], labels[train_idx])
    predictions = classifier.predict(features[test_idx])

    metrics = evaluate_predictions(labels[test_idx], predictions)
    metrics["acoustic_feature_count"] = int(acoustic.shape[1])
    metrics["representation_feature_count"] = int(representation.shape[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, output_dir / "fusion_metrics.json")
    save_confusion_matrix(labels[test_idx], predictions, output_dir / "fusion_confusion_matrix.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fusion classifier from acoustic and pretrained speech features.")
    parser.add_argument("--acoustic-features", type=Path, required=True)
    parser.add_argument("--representation-features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_fusion")
    metrics = run_fusion_experiment(
        args.acoustic_features,
        args.representation_features,
        args.labels,
        args.splits,
        output_dir,
        args.random_state,
    )
    print(json.dumps({"output_dir": str(output_dir), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Modify acoustic and representation scripts to save splits**

In both `experiments/train_acoustic_baseline.py` and `experiments/train_representation_model.py`, after saving labels, add:

```python
np.save(processed_dir / "splits.npy", np.array([row["split"] for row in rows], dtype=object))
```

- [ ] **Step 5: Run fusion tests**

Run from `experiments/`:

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 6: Run fusion training after acoustic and representation outputs exist**

Run from project root on the server:

```bash
python experiments/train_fusion_model.py --acoustic-features outputs/acoustic/processed/acoustic_features.npy --representation-features outputs/representation_wav2vec2/processed/representation_features.npy --labels outputs/acoustic/processed/tone_labels.npy --splits outputs/acoustic/processed/splits.npy --output-dir outputs/fusion_wav2vec2
```

Expected: prints JSON containing `output_dir`, `accuracy`, and `macro_f1`; writes `outputs/fusion_wav2vec2/fusion_metrics.json`.

- [ ] **Step 7: Commit if this is later placed in a git repository**

```bash
git add experiments/train_fusion_model.py experiments/train_acoustic_baseline.py experiments/train_representation_model.py experiments/tests/test_metrics.py
git commit -m "feat: train acoustic representation fusion model"
```

---

## Task 7: Update representation analysis for saved outputs

**Files:**
- Modify: `experiments/analyze_representations.py`

- [ ] **Step 1: Replace analysis script with CLI-compatible version**

Replace `experiments/analyze_representations.py` with:

```python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def tsne_2d(features: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(2, features.shape[0] // 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=42, init="random", learning_rate="auto").fit_transform(features)


def save_scatter(points: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=18)
    plt.colorbar(scatter, label="Tone")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Pretrained speech representation by Mandarin tone")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved speech representations by Mandarin tone.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = np.load(args.features)
    labels = np.load(args.labels)
    points = tsne_2d(features)
    save_scatter(points, labels, args.output)
    print(str(args.output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run import smoke test**

Run from `experiments/`:

```bash
python - <<'PY'
import numpy as np
from analyze_representations import tsne_2d
features = np.random.default_rng(0).normal(size=(8, 4))
points = tsne_2d(features)
print(points.shape)
PY
```

Expected: prints `(8, 2)`.

- [ ] **Step 3: Generate representation figure after representation training exists**

Run from project root on the server:

```bash
python experiments/analyze_representations.py --features outputs/representation_wav2vec2/processed/representation_features.npy --labels outputs/representation_wav2vec2/processed/tone_labels.npy --output paper/figures/representation_tsne.png
```

Expected: writes `paper/figures/representation_tsne.png`.

- [ ] **Step 4: Commit if this is later placed in a git repository**

```bash
git add experiments/analyze_representations.py
git commit -m "feat: visualize tone representations"
```

---

## Task 8: Create server orchestration script

**Files:**
- Create: `experiments/run_full_pipeline.sh`

- [ ] **Step 1: Create server pipeline script**

Create `experiments/run_full_pipeline.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/NAS/yesh/mandarin-tone-feedback}"
METADATA_PATH="${METADATA_PATH:-data/metadata.csv}"
MODEL_NAME="${MODEL_NAME:-facebook/wav2vec2-base}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_ID}}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_ROOT"

export HF_HOME="${HF_HOME:-/NAS/yesh/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/NAS/yesh/hf_cache/hub}"

{
  echo "run_id=$RUN_ID"
  echo "project_root=$PROJECT_ROOT"
  echo "metadata_path=$METADATA_PATH"
  echo "model_name=$MODEL_NAME"
  echo "gpu_id=$GPU_ID"
  hostname
  date
  nvidia-smi || true

  python experiments/train_acoustic_baseline.py \
    --metadata "$METADATA_PATH" \
    --output-dir "$OUTPUT_ROOT/acoustic" \
    --classifier random_forest

  CUDA_VISIBLE_DEVICES="$GPU_ID" python experiments/train_representation_model.py \
    --metadata "$METADATA_PATH" \
    --output-dir "$OUTPUT_ROOT/representation" \
    --model-name "$MODEL_NAME" \
    --batch-size 4 \
    --device cuda

  python experiments/train_fusion_model.py \
    --acoustic-features "$OUTPUT_ROOT/acoustic/processed/acoustic_features.npy" \
    --representation-features "$OUTPUT_ROOT/representation/processed/representation_features.npy" \
    --labels "$OUTPUT_ROOT/acoustic/processed/tone_labels.npy" \
    --splits "$OUTPUT_ROOT/acoustic/processed/splits.npy" \
    --output-dir "$OUTPUT_ROOT/fusion"

  python experiments/analyze_representations.py \
    --features "$OUTPUT_ROOT/representation/processed/representation_features.npy" \
    --labels "$OUTPUT_ROOT/representation/processed/tone_labels.npy" \
    --output "paper/figures/representation_tsne_${RUN_ID}.png"

  echo "outputs saved to $OUTPUT_ROOT"
} 2>&1 | tee "logs/full_pipeline_${RUN_ID}.log"
```

- [ ] **Step 2: Make script executable on server**

Run from project root on the server:

```bash
chmod +x experiments/run_full_pipeline.sh
```

Expected: command exits with status 0.

- [ ] **Step 3: Run the script after metadata and environment pass checks**

Run from project root on the server:

```bash
RUN_ID=first_real_run GPU_ID=0 MODEL_NAME=facebook/wav2vec2-base ./experiments/run_full_pipeline.sh
```

Expected: writes `outputs/first_real_run/acoustic`, `outputs/first_real_run/representation`, `outputs/first_real_run/fusion`, and `logs/full_pipeline_first_real_run.log`.

- [ ] **Step 4: Commit if this is later placed in a git repository**

```bash
git add experiments/run_full_pipeline.sh
git commit -m "chore: add server training pipeline script"
```

---

## Task 9: Create server training runbook

**Files:**
- Create: `docs/server_training_runbook.md`

- [ ] **Step 1: Write runbook**

Create `docs/server_training_runbook.md` with:

```markdown
# Server Training Runbook

## Purpose

This runbook describes how to synchronize the local Mandarin Tone Feedback project to `/NAS/yesh/mandarin-tone-feedback`, prepare the NAS-backed Python environment, validate data, and run the full acoustic + pretrained representation + fusion experiment pipeline.

## Local sync command

Set the SSH target from the user's server information:

```bash
export MTF_REMOTE="user@server"
```

Synchronize necessary files from the local project root:

```bash
rsync -avz --delete \
  --exclude 'node_modules/' \
  --exclude '.venv/' \
  --exclude '.conda/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.vite/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  ./ "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/"
```

## Server environment

```bash
cd /NAS/yesh/mandarin-tone-feedback
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub
conda create --prefix /NAS/yesh/mandarin-tone-feedback/.conda/mtf python=3.11 -y
conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf
pip install -e "backend[test,ml]"
pip install pandas pyyaml
```

## Server checks before training

```bash
hostname
df -h /NAS/yesh
nvidia-smi
python --version
python -m pytest backend/tests -q
cd experiments && python -m pytest tests -q && cd ..
```

## Data requirement

Create `data/metadata.csv` with:

```csv
audio_path,text,pinyin,tone,speaker_id,split
data/raw/ma1_speaker01.wav,妈,ma1,1,speaker01,train
data/raw/ma2_speaker01.wav,麻,ma2,2,speaker01,train
data/raw/ma3_speaker01.wav,马,ma3,3,speaker01,test
data/raw/ma4_speaker01.wav,骂,ma4,4,speaker01,test
```

The real file must reference actual audio files and include non-empty train and test splits.

## Full run

```bash
RUN_ID=first_real_run GPU_ID=0 MODEL_NAME=facebook/wav2vec2-base ./experiments/run_full_pipeline.sh
```

## Outputs to inspect

- `outputs/<RUN_ID>/acoustic/acoustic_metrics.json`
- `outputs/<RUN_ID>/representation/representation_metrics.json`
- `outputs/<RUN_ID>/fusion/fusion_metrics.json`
- `outputs/<RUN_ID>/*/*_confusion_matrix.csv`
- `paper/figures/representation_tsne_<RUN_ID>.png`
- `logs/full_pipeline_<RUN_ID>.log`

## Paper integrity rule

Only copy metrics into `paper/outline.md`, slides, or final prose after verifying that the metric files above were produced by a real run over the test split.
```

- [ ] **Step 2: Commit if this is later placed in a git repository**

```bash
git add docs/server_training_runbook.md
git commit -m "docs: add server training runbook"
```

---

## Task 10: Synchronize local files to server

**Files:**
- No source-code edits.
- Uses Bash only, not PowerShell.

- [ ] **Step 1: Obtain SSH target**

Ask the user for SSH information in this form:

```text
SSH target: user@host
SSH port: 22 unless different
Authentication: existing key, password prompt, or jump host command
```

Expected: user provides a usable SSH target.

- [ ] **Step 2: Run remote directory check**

Run from the local project root using Bash:

```bash
ssh "$MTF_REMOTE" 'mkdir -p /NAS/yesh/mandarin-tone-feedback && test -w /NAS/yesh/mandarin-tone-feedback && echo writable'
```

Expected: prints `writable`.

- [ ] **Step 3: Sync project files**

Run from the local project root using Bash:

```bash
rsync -avz --delete \
  --exclude 'node_modules/' \
  --exclude '.venv/' \
  --exclude '.conda/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.vite/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  ./ "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/"
```

Expected: copies project files and does not copy `node_modules`.

- [ ] **Step 4: Verify server file list**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && test -f CLAUDE.md && test -f experiments/run_full_pipeline.sh && test -f docs/server_training_runbook.md && echo synced'
```

Expected: prints `synced`.

---

## Task 11: Prepare server environment and run tests

**Files:**
- No source-code edits unless tests fail.

- [ ] **Step 1: Check server resources**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'hostname; df -h /NAS/yesh; nvidia-smi || true; which conda || true; which python || true'
```

Expected: prints node name, NAS space, GPU status, and available Python/Conda paths.

- [ ] **Step 2: Create or reuse Conda environment**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && if [ ! -d .conda/mtf ]; then conda create --prefix /NAS/yesh/mandarin-tone-feedback/.conda/mtf python=3.11 -y; fi'
```

Expected: environment exists at `/NAS/yesh/mandarin-tone-feedback/.conda/mtf`.

- [ ] **Step 3: Install dependencies**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf && export HF_HOME=/NAS/yesh/hf_cache && export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub && pip install -e "backend[test,ml]" && pip install pandas pyyaml'
```

Expected: pip exits successfully.

- [ ] **Step 4: Run backend tests**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf && python -m pytest backend/tests -q'
```

Expected: backend tests PASS.

- [ ] **Step 5: Run experiment tests**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback/experiments && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf && python -m pytest tests -q'
```

Expected: experiment tests PASS.

---

## Task 12: Prepare real dataset before training

**Files:**
- Create or update on server: `/NAS/yesh/mandarin-tone-feedback/data/metadata.csv`
- Add audio files under: `/NAS/yesh/mandarin-tone-feedback/data/raw/`

- [ ] **Step 1: Put real audio files under server data directory**

Use the agreed collection or dataset source and place short Mandarin clips under:

```text
/NAS/yesh/mandarin-tone-feedback/data/raw/
```

Expected: audio files are real WAV/FLAC/MP3 clips containing isolated syllables or short words with known tone labels.

- [ ] **Step 2: Create metadata file**

Create `/NAS/yesh/mandarin-tone-feedback/data/metadata.csv` with this exact header:

```csv
audio_path,text,pinyin,tone,speaker_id,split
```

Example row format:

```csv
data/raw/ma1_speaker01.wav,妈,ma1,1,speaker01,train
```

Expected: every row points to an existing audio file and `tone` is 1, 2, 3, or 4.

- [ ] **Step 3: Validate metadata through Python**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback/experiments && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf && python - <<'"'"'PY'"'"'
from pathlib import Path
from dataset import load_validated_metadata, summarize_metadata
rows = load_validated_metadata(Path("../data/metadata.csv"))
print(summarize_metadata(rows))
PY'
```

Expected: prints counts by split, tone, and speaker; no exception is raised.

---

## Task 13: Run full server training pipeline

**Files:**
- Reads: `/NAS/yesh/mandarin-tone-feedback/data/metadata.csv`
- Writes: `/NAS/yesh/mandarin-tone-feedback/outputs/<RUN_ID>/`
- Writes: `/NAS/yesh/mandarin-tone-feedback/logs/full_pipeline_<RUN_ID>.log`

- [ ] **Step 1: Check GPU before long run**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'hostname; nvidia-smi; df -h /NAS/yesh'
```

Expected: at least one GPU has enough free memory for frozen wav2vec2 extraction; NAS has enough free disk space.

- [ ] **Step 2: Start full pipeline in tmux**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && tmux new-session -d -s mtf_train "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate /NAS/yesh/mandarin-tone-feedback/.conda/mtf && RUN_ID=first_real_run GPU_ID=0 MODEL_NAME=facebook/wav2vec2-base ./experiments/run_full_pipeline.sh" && tmux ls'
```

Expected: tmux lists session `mtf_train`.

- [ ] **Step 3: Monitor log**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'tail -n 80 /NAS/yesh/mandarin-tone-feedback/logs/full_pipeline_first_real_run.log'
```

Expected: log shows acoustic, representation, and fusion stages progressing or completed.

- [ ] **Step 4: Verify outputs**

Run using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && test -f outputs/first_real_run/acoustic/acoustic_metrics.json && test -f outputs/first_real_run/representation/representation_metrics.json && test -f outputs/first_real_run/fusion/fusion_metrics.json && test -f paper/figures/representation_tsne_first_real_run.png && echo training_outputs_ready'
```

Expected: prints `training_outputs_ready`.

---

## Task 14: Update paper and PPT artifacts from real outputs

**Files:**
- Modify: `paper/outline.md`
- Create or update: `paper/figures/model_comparison_first_real_run.md`
- Use existing figure files under: `paper/figures/`

- [ ] **Step 1: Inspect metrics files**

Run on server using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && python - <<'"'"'PY'"'"'
import json
from pathlib import Path
for name in ["acoustic", "representation", "fusion"]:
    path = Path("outputs/first_real_run") / name / f"{name}_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(name, metrics["accuracy"], metrics["macro_f1"])
PY'
```

Expected: prints three rows with real accuracy and macro-F1 values.

- [ ] **Step 2: Create model comparison markdown from metrics**

Run on server using Bash:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && python - <<'"'"'PY'"'"'
import json
from pathlib import Path
run = "first_real_run"
rows = []
for label, name in [("Acoustic", "acoustic"), ("Pretrained", "representation"), ("Fusion", "fusion")]:
    path = Path("outputs") / run / name / f"{name}_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows.append((label, metrics["accuracy"], metrics["macro_f1"], str(path)))
lines = ["# Model comparison from first_real_run", "", "| Model | Accuracy | Macro-F1 | Source |", "| --- | ---: | ---: | --- |"]
for label, accuracy, macro_f1, source in rows:
    lines.append(f"| {label} | {accuracy:.4f} | {macro_f1:.4f} | `{source}` |")
Path("paper/figures/model_comparison_first_real_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY'
```

Expected: writes `paper/figures/model_comparison_first_real_run.md` with real metrics and source paths.

- [ ] **Step 3: Update paper outline manually from generated comparison**

Open `paper/figures/model_comparison_first_real_run.md` and copy its real values into the result fields in `paper/outline.md`. Keep wording conservative and include the run ID `first_real_run` near the numbers.

Expected: `paper/outline.md` contains real values from `outputs/first_real_run`, not invented estimates.

- [ ] **Step 4: Sync updated server artifacts back to local if needed**

Run from local project root using Bash:

```bash
rsync -avz "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/paper/" ./paper/
rsync -avz "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/logs/" ./logs/
```

Expected: local `paper/` receives generated figures and comparison files.

---

## Final verification checklist

- [ ] Local plan and spec exist:

```text
docs/superpowers/specs/2026-05-31-server-training-design.md
docs/superpowers/plans/2026-05-31-server-training.md
```

- [ ] Server sync excludes `node_modules/`, `.venv/`, `.conda/`, caches, logs, and outputs.
- [ ] Backend tests pass on server.
- [ ] Experiment tests pass on server.
- [ ] Real `data/metadata.csv` validates with train and test splits.
- [ ] Full pipeline writes acoustic, representation, and fusion metrics.
- [ ] Paper/PPT materials only use metrics from real output files.
- [ ] Any limitations are stated as limitations, not hidden behind fabricated values.

## Self-review

- Spec coverage: this plan covers server sync, NAS environment, metadata validation, acoustic feature extraction, acoustic training, frozen representation training, fusion training, t-SNE visualization, orchestration, runbook, and paper artifact update.
- Red-flag scan: the plan avoids invented results and does not ask an engineer to invent missing details. SSH connection values and the real dataset are explicit execution inputs requested from the user.
- Type consistency: metadata rows use integer `tone`, string `split`, and `Path` audio paths throughout; saved arrays use aligned row order across acoustic, representation, labels, and splits.
- Scope control: no ASR, accounts, dashboards, payment, dialect classification, or online GPU inference is added.
