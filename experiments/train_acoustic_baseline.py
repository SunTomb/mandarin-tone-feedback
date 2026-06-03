import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
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
    np.save(processed_dir / "splits.npy", np.array([row["split"] for row in rows], dtype=object))
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
    dump(model, output_dir / "acoustic_classifier.joblib")
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
