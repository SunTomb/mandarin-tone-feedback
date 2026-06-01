# Server Training Runbook

## Purpose

This runbook describes how to synchronize the local Mandarin Tone Feedback project to `/NAS/yesh/mandarin-tone-feedback`, prepare the NAS-backed Python environment, validate data, and run the full acoustic + pretrained representation + fusion experiment pipeline.

## Local sync command

Set the SSH target from the user's server information:

```bash
export MTF_REMOTE="Sui-3-Wu"
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
