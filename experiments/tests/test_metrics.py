import numpy as np
import pytest
from pathlib import Path

from train_fusion_model import concatenate_features, run_fusion_experiment


def test_concatenate_features_combines_columns():
    acoustic = np.ones((2, 3))
    representation = np.zeros((2, 4))

    result = concatenate_features(acoustic, representation)

    assert result.shape == (2, 7)


def test_concatenate_features_rejects_mismatched_rows():
    acoustic = np.ones((2, 3))
    representation = np.zeros((3, 4))

    with pytest.raises(ValueError):
        concatenate_features(acoustic, representation)


def test_run_fusion_experiment_writes_metrics(tmp_path: Path):
    acoustic = np.array([
        [1.0, 2.0],
        [1.5, 2.5],
        [4.0, 3.0],
        [4.5, 3.5],
        [1.1, 2.1],
        [4.2, 3.2],
    ])
    representation = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.15, 0.25],
        [0.85, 0.75],
    ])
    labels = np.array([1, 1, 4, 4, 1, 4])
    splits = np.array(["train", "train", "train", "train", "test", "test"])

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
    assert (output_dir / "fusion_classifier.joblib").exists()
