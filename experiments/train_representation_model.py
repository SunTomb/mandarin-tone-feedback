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
    np.save(processed_dir / "splits.npy", np.array([row["split"] for row in rows], dtype=object))

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
