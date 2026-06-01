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
