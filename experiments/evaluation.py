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
