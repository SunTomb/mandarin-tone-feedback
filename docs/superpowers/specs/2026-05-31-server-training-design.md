# Server Sync and Full Training Design

## Goal

Move the local Mandarin Tone Feedback project to `/NAS/yesh/mandarin-tone-feedback`, verify the server environment, prepare real audio metadata, then run the full three-path experiment pipeline for acoustic, pretrained-representation, and fusion tone classification.

## Scope

This design covers the remaining work after local code drafting:

1. synchronize necessary project files to the lab server;
2. create or reuse a NAS-based Python environment;
3. validate the code and server runtime with smoke tests;
4. prepare a real audio dataset and metadata file;
5. implement and run reproducible acoustic, pretrained, and fusion experiments;
6. generate paper/PPT artifacts only from real experiment outputs.

It does not include full ASR, sentence-level pronunciation scoring, user accounts, classroom dashboards, payment, dialect classification, or real-time A40/A100-dependent web inference.

## Constraints

- Do not use PowerShell for project work.
- Use Bash or dedicated file tools for local and server operations.
- Do not upload local dependency directories such as `node_modules`, `.venv`, `.conda`, caches, or temporary outputs.
- Store the server project under `/NAS/yesh/mandarin-tone-feedback`.
- Store Conda environments under NAS, preferably `/NAS/yesh/mandarin-tone-feedback/.conda/mtf`.
- Store Hugging Face caches under `/NAS/yesh/hf_cache`.
- Before long GPU jobs, check node name, GPU occupancy, NAS disk space, and whether other users are using the target GPU.
- Do not report accuracy, F1, confusion matrices, or figure conclusions until real experiment output files exist.

## Approach

Use a two-stage workflow.

### Stage A: synchronization and server smoke test

Synchronize project source and documentation to the server with SSH-based `rsync` or `scp` after the user provides connection details. The sync includes code, configs, paper scaffolds, and essential assets, while excluding local dependencies and generated caches.

On the server, verify:

- `/NAS/yesh/mandarin-tone-feedback` exists or can be created;
- NAS disk space is sufficient;
- Conda is available;
- the project environment can install backend test dependencies and ML dependencies;
- backend and experiment tests pass;
- the web demo remains lightweight and does not require online GPU inference.

Stage A does not produce final paper results. It only proves that the project can run on the server.

### Stage B: data preparation and full training

Because real audio data is not ready yet, training must be preceded by a dataset preparation step. The dataset must provide a real `data/metadata.csv` with these columns:

```csv
audio_path,text,pinyin,tone,speaker_id,split
```

The training pipeline should validate metadata, extract acoustic features, extract pretrained speech representations, train comparable classifiers, and save all metrics and figures under timestamped output directories.

## Components

### File synchronization

Responsibility: copy only source-controlled and research-relevant files to the server.

Include:

- `CLAUDE.md`
- `README.md`
- `design_spec.md`
- `implementation_plan.md`
- `proposal_ppt_outline.md`
- `backend/`
- `frontend/`
- `experiments/`
- `data/metadata.example.csv`
- `paper/`
- `ppt_assets/prepare.tex`
- `ppt_assets/prepare.pdf`
- `build_proposal_ppt.py`
- selected generated presentation assets if needed for reporting

Exclude:

- `node_modules/`
- `.venv/`
- `.conda/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.vite/`
- large local temporary files
- generated experiment outputs unless explicitly requested

### Server environment

Responsibility: provide a reproducible environment for tests, feature extraction, and model training.

Environment variables:

```bash
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub
```

Preferred Conda prefix:

```bash
/NAS/yesh/mandarin-tone-feedback/.conda/mtf
```

The environment should install:

- backend runtime and test dependencies;
- `numpy`, `pandas`, `scikit-learn`, `librosa`, `soundfile`;
- `torch`, `transformers`, `matplotlib`, and optionally `umap-learn`;
- frontend dependencies only if server-side frontend build verification is needed.

### Dataset validation

Responsibility: fail early before model training if audio paths or labels are invalid.

Validation checks:

- metadata file exists;
- required columns are present;
- every `audio_path` exists relative to the project root or metadata file;
- every `tone` is one of `1`, `2`, `3`, `4`;
- every `split` is one of `train`, `val`, `test`;
- each tone has at least one sample;
- train/test split is non-empty;
- report counts by split, tone, and speaker.

### Acoustic baseline

Responsibility: produce the first interpretable model result.

Inputs:

- `data/metadata.csv`;
- audio files referenced by metadata.

Features:

- normalized F0 start;
- normalized F0 middle;
- normalized F0 end;
- F0 range;
- F0 slope;
- duration;
- voiced-frame ratio.

Outputs:

- `data/processed/acoustic_features.npy`;
- `data/processed/tone_labels.npy`;
- `outputs/<run_id>/acoustic_metrics.json`;
- `outputs/<run_id>/acoustic_confusion_matrix.csv`.

### Pretrained representation baseline

Responsibility: test whether frozen speech representations improve tone classification.

Recommended first model:

- `facebook/wav2vec2-base` or a HuBERT equivalent if available in the server cache.

Inputs:

- audio clips and labels from metadata.

Process:

- load audio at 16 kHz;
- pass batches through the frozen speech encoder;
- mean-pool hidden states;
- save embeddings;
- train a lightweight classifier on embeddings.

Outputs:

- `data/processed/representation_features.npy`;
- `outputs/<run_id>/representation_metrics.json`;
- `outputs/<run_id>/representation_confusion_matrix.csv`.

### Fusion model

Responsibility: compare explicit F0 features with pretrained features combined.

Inputs:

- `data/processed/acoustic_features.npy`;
- `data/processed/representation_features.npy`;
- labels and split assignments.

Process:

- concatenate feature matrices by row;
- verify row alignment with metadata IDs;
- train a lightweight classifier;
- evaluate using the same metrics as the other paths.

Outputs:

- `outputs/<run_id>/fusion_metrics.json`;
- `outputs/<run_id>/fusion_confusion_matrix.csv`;
- a combined model comparison table.

### Figure and paper artifact generation

Responsibility: convert real outputs into paper/PPT-ready artifacts.

Artifacts:

- F0 target-versus-user example plot;
- model comparison table;
- confusion matrix figure;
- optional t-SNE/UMAP representation visualization.

Rules:

- Generated figures must include a source pointer to the output run directory.
- Paper prose may contain result slots before training, but final numeric claims must reference real output files.

## Data flow

1. Local project files are synchronized to `/NAS/yesh/mandarin-tone-feedback`.
2. Server environment is activated with NAS-backed caches.
3. Tests verify backend, feature utilities, API contract, and experiment helpers.
4. Real metadata and audio are placed under `data/`.
5. Metadata validation produces a dataset summary.
6. Acoustic extraction generates explicit F0 features.
7. Pretrained extraction generates frozen speech embeddings.
8. Acoustic, representation, and fusion classifiers are trained and evaluated.
9. Metrics and figures are saved under `outputs/<run_id>/` and `paper/figures/`.
10. Paper and PPT materials are updated only from real outputs.

## Error handling and stopping rules

- If SSH connection fails, stop and request corrected connection details.
- If `/NAS/yesh` is not writable, stop and request server-side permission correction.
- If Conda is unavailable, inspect existing Python environments before creating alternatives.
- If metadata validation fails, do not train; report the exact invalid rows and missing files.
- If GPU is occupied, do not preempt other users; either wait, choose another GPU, or ask the user.
- If pretrained model download fails, prefer using an already cached model or ask before changing model family.
- If tests fail, fix the code before starting long training jobs.

## Testing strategy

Before server training:

- backend tests for feedback, F0 utilities, and API response shape;
- experiment tests for metadata loading, split handling, feature concatenation, and metric serialization;
- a smoke run on a tiny dataset or fixture that exercises the full pipeline without claiming research results.

During server training:

- log command, node, GPU, environment variables, and git/project state equivalent;
- save metrics as machine-readable JSON/CSV;
- save all logs under `logs/`.

After training:

- verify output files exist;
- verify metrics are computed from test split, not train split only;
- verify figures match output files;
- update paper result slots with real values only.

## Open execution inputs

The user still needs to provide:

- SSH host, username, and port if non-default;
- the server login method if key-based SSH is not already configured;
- the real audio dataset or a decision on how to collect it;
- whether pretrained extraction should start with wav2vec2-base or HuBERT if both are available.

## Self-review

- Spec coverage: covers synchronization, NAS environment, data preparation, acoustic baseline, pretrained baseline, fusion model, output management, and paper/PPT artifact generation.
- Placeholder scan: no numeric results or fabricated conclusions are included.
- Scope check: focused on one cohesive remaining-work workflow; no unrelated product features are included.
- Ambiguity check: training is explicitly blocked until real metadata/audio pass validation.
