import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
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
    dump(classifier, output_dir / "fusion_classifier.joblib")
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
