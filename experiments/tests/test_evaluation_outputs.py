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
