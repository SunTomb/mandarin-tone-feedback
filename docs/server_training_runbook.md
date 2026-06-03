# Server Training Runbook

## Purpose

This runbook records the reproducible workflow for the course project **面向普通话声调学习的可解释 AI 反馈系统**. It covers the self-recorded isolated-character dataset, quality filtering, server-side training, and demo launch checks.

The project is a course final assignment. The goal is not to chase perfect classification scores, but to provide:

1. a complete training pipeline;
2. a runnable web demo;
3. real experimental results that can be reported honestly in the final paper.

## Safety and scope

- Do not use PowerShell in this project.
- Use `/NAS/yesh/mandarin-tone-feedback` as the server project directory.
- Use the user's existing Miniconda installation and project entrypoint:

```bash
source /NAS/yesh/miniconda3/etc/profile.d/conda.sh
cd /NAS/yesh/mandarin-tone-feedback
source ./activate.sh
```

- Use NAS cache for HuggingFace assets:

```bash
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub
```

- Do not reinstall large PyTorch/CUDA packages unless a missing dependency is proven necessary.
- Keep the web prototype lightweight. It must not depend on real-time GPU inference.
- Do not invent experimental results. Only report metrics produced by actual runs.

## Local-to-server sync

Set the SSH target:

```bash
export MTF_REMOTE="Sui-3-Wu"
```

If `rsync` is available, sync only training-relevant files:

```bash
rsync -av \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  experiments/ "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/experiments/"
rsync -av data/self/ "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/data/self/"
rsync -av backend/ "$MTF_REMOTE:/NAS/yesh/mandarin-tone-feedback/backend/"
```

If local `rsync` is unavailable, use tar over SSH from the repository root:

```bash
ROOT=$(git rev-parse --show-toplevel)
tar -C "$ROOT" \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  -cf - experiments backend data/self \
  | ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && tar -xf -'
```

Do not upload local virtual environments, `node_modules`, frontend caches, PPT assets, or presentation-only files unless explicitly needed.

## Server environment check

Run this before training:

```bash
ssh "$MTF_REMOTE" 'cd /NAS/yesh/mandarin-tone-feedback && \
  source /NAS/yesh/miniconda3/etc/profile.d/conda.sh && \
  source ./activate.sh && \
  export HF_HOME=/NAS/yesh/hf_cache && \
  python - <<'"'"'PY'"'"'
import importlib
for name in ["torch", "transformers", "librosa", "sklearn", "soundfile", "joblib", "numpy"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"))
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
PY'
```

Observed Sui-3-Wu environment for the completed runs:

- PyTorch: `2.4.0+cu121`
- transformers: `4.44.0`
- librosa: `0.11.0`
- scikit-learn: `1.7.2`
- CUDA available: yes
- GPU count: 8 × RTX 3090

## Self-recorded dataset structure

The self-recorded dataset lives under `data/self/`.

Folder names encode recording batches:

```text
data/self/{speaker_index}-{tone}-{repetition}/
```

Example:

```text
data/self/2-3-2/
```

means speaker 2, tone 3, second reading repetition. Each batch folder contains 30 isolated-character recordings for that tone and repetition.

Each batch has a `metadata.csv` with:

```csv
audio_path,text,pinyin,tone,speaker_id,repetition,split
```

## Build unified self-recording manifest

From the repository root:

```bash
python experiments/self_recording_manifest.py \
  --self-root data/self \
  --output data/self/metadata.csv \
  --test-speaker speaker03 \
  --val-repetition 3
```

This produces `data/self/metadata.csv` with:

```csv
audio_path,text,pinyin,tone,speaker_id,repetition,split,batch_dir
```

The completed manifest contains 1080 rows:

- train: 480
- val: 240
- test: 360
- speaker01/02/03: 360 each
- tone1/2/3/4: 270 each

## Validate self-recording metadata

```bash
python experiments/validate_self_recordings.py \
  --self-root data/self \
  --project-root .
```

Expected output for the completed dataset:

```text
Validated 1080 rows
```

The validator checks:

- required metadata fields;
- audio file existence;
- folder semantics against `speaker_id`, `tone`, and `repetition`;
- tone values in `{1,2,3,4}`;
- pinyin ending in the tone digit;
- split values in `{train,val,test}`;
- expected filename format;
- 30 rows per speaker/tone/repetition batch.

## Quality audit and filtering

The strict F0 quality filter was too aggressive for this dataset because many recordings include long leading/trailing silence. The course-project version uses a lenient filter to remove only clear failures.

Run on the server:

```bash
python experiments/self_recording_quality.py \
  --metadata data/self/metadata.csv \
  --quality-report outputs/self_quality/quality_report_lenient.csv \
  --filtered-metadata data/self/metadata_filtered_lenient.csv \
  --loso-dir data/self/loso_lenient \
  --min-duration 0.25 \
  --max-duration 3.2 \
  --min-voiced-ratio 0.12 \
  --min-voiced-frames 8 \
  --min-rms 0.005
```

Observed result:

```text
Audited 1080 rows; kept 953; dropped 127
```

Filtered manifest distribution:

- total: 953
- train: 401
- val: 209
- test: 343
- speaker01: 320
- speaker02: 290
- speaker03: 343
- tone1: 259
- tone2: 249
- tone3: 250
- tone4: 195

The script also writes leave-one-speaker-out manifests:

```text
data/self/loso_lenient/metadata_test_speaker01.csv
data/self/loso_lenient/metadata_test_speaker02.csv
data/self/loso_lenient/metadata_test_speaker03.csv
```

## Train acoustic baseline

Raw self dataset:

```bash
python experiments/train_acoustic_baseline.py \
  --metadata data/self/metadata.csv \
  --output-dir outputs/self_acoustic_baseline
```

Filtered self dataset:

```bash
python experiments/train_acoustic_baseline.py \
  --metadata data/self/metadata_filtered_lenient.csv \
  --output-dir outputs/self_filtered_lenient_acoustic
```

Leave-one-speaker-out acoustic analysis:

```bash
for speaker in speaker01 speaker02 speaker03; do
  python experiments/train_acoustic_baseline.py \
    --metadata data/self/loso_lenient/metadata_test_${speaker}.csv \
    --output-dir outputs/self_loso_lenient_acoustic_${speaker}
done
```

## Train frozen wav2vec2 representation baseline

Sui-3-Wu may not reliably reach HuggingFace. Use the NAS HuggingFace cache and offline mode.

Check that `facebook/wav2vec2-base` exists in cache:

```bash
python - <<'PY'
from pathlib import Path
p = Path('/NAS/yesh/hf_cache/hub/models--facebook--wav2vec2-base')
print(p.exists())
for f in sorted(p.rglob('*'))[:20]:
    print(f.relative_to(p))
PY
```

Run representation extraction and classification:

```bash
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python experiments/train_representation_model.py \
  --metadata data/self/metadata_filtered_lenient.csv \
  --output-dir outputs/self_filtered_lenient_representation_wav2vec2 \
  --batch-size 8 \
  --device cuda \
  2>&1 | tee logs/self_filtered_lenient_representation_wav2vec2.log
```

Use `set -o pipefail` whenever piping through `tee`; otherwise a Python failure can be hidden by the pipe exit code.

## Train fusion model

```bash
set -o pipefail
python experiments/train_fusion_model.py \
  --acoustic-features outputs/self_filtered_lenient_acoustic/processed/acoustic_features.npy \
  --representation-features outputs/self_filtered_lenient_representation_wav2vec2/processed/representation_features.npy \
  --labels outputs/self_filtered_lenient_acoustic/processed/tone_labels.npy \
  --splits outputs/self_filtered_lenient_acoustic/processed/splits.npy \
  --output-dir outputs/self_filtered_lenient_fusion_wav2vec2 \
  2>&1 | tee logs/self_filtered_lenient_fusion_wav2vec2.log
```

## Verified experiment results

Main comparison:

| Dataset / model | Accuracy | Macro-F1 |
|---|---:|---:|
| THCHS raw acoustic | 0.2250 | 0.1983 |
| THCHS raw wav2vec2 | 0.3125 | 0.3149 |
| THCHS raw fusion | 0.2625 | 0.2599 |
| Self raw acoustic | 0.3639 | 0.3198 |
| Self raw wav2vec2 | 0.3389 | 0.3211 |
| Self raw fusion | 0.3556 | 0.3373 |
| Self filtered acoustic | 0.3761 | 0.3275 |
| Self filtered wav2vec2 | 0.3644 | 0.3354 |
| Self filtered fusion | 0.3965 | 0.3722 |

Filtered leave-one-speaker-out acoustic baseline:

| Test speaker | Accuracy | Macro-F1 |
|---|---:|---:|
| speaker01 | 0.5156 | 0.5028 |
| speaker02 | 0.3276 | 0.3186 |
| speaker03 | 0.3761 | 0.3275 |

Result files:

```text
paper/self_experiment_results.md
paper/self_experiment_summary.json
outputs/self_quality/self_experiment_summary.json
outputs/self_quality/quality_report_lenient.csv
data/self/metadata_filtered_lenient.csv
```

## Browser demo with tmux on server

The web prototype should remain runnable without real-time GPU inference.

Start API and frontend on the server:

```bash
cd /NAS/yesh/mandarin-tone-feedback
./scripts/run_web_tmux.sh
```

Attach if needed:

```bash
tmux attach -t mtf-demo
```

The session contains:

- `api`: FastAPI on port `8000`.
- `frontend`: Vite on port `5173`, with `/api` proxied to FastAPI.

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected:

```json
{"status":"ok"}
```

From local machine, open a tunnel:

```bash
ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 Sui-3-Wu
```

Open:

```text
http://127.0.0.1:5173
```

Manual browser checklist:

1. Page loads with the Chinese project title.
2. Select each target tone 1-4.
3. Use demo or upload flow.
4. Confirm `/api/analyze` or `/api/demo-feedback` returns 200.
5. Confirm F0 chart and feedback card render.
6. Confirm prediction is presented as a reference, not as the main scoring basis.

## Local demo commands

Backend:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Then open:

```text
http://127.0.0.1:5173
```

## Paper integrity rule

The final paper should use the metrics above only as real measured results. It is acceptable that the scores are not high. The paper narrative should emphasize:

- THCHS slicing is a public-data baseline and limitation analysis;
- the self-recorded dataset is a controlled isolated-character dataset;
- quality filtering improves results but speaker-independent generalization remains difficult;
- the demo therefore focuses on F0 contour diagnosis and interpretable feedback, while the classifier is only a reference signal.
