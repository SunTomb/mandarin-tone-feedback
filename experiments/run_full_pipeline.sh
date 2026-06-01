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
