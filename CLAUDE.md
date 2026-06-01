# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Absolute safety instruction

Do not use PowerShell commands in this project. The user is connected through CC Switch and explicitly warned that PowerShell tool use may trigger Claude Code internal stack leakage.

Use dedicated file tools for file reads/writes/searches whenever possible. If a terminal command is unavoidable, use Bash or another non-PowerShell shell and explain why before running it.

## Project goal

Build a hybrid research-and-demo project titled:

**面向普通话声调学习的可解释 AI 反馈系统**

The final result should support a 3-9 page language-science paper and a small runnable web prototype. The system records or uploads short Mandarin audio, predicts the lexical tone, visualizes the normalized F0 contour, compares it with a target contour, and returns interpretable feedback.

## Read first

Before implementing anything, read these files in this folder:

1. `README.md` — project goal, scope, and main directories.
2. `design_spec.md` — research framing, scope, system design, and paper contribution.
3. `implementation_plan.md` — task-by-task implementation handoff.
4. `CLAUDE.md` — this instruction file.

Also read the course context files in the parent directory when paper framing matters:

1. `../论文要求.md`
2. `../语言科学课件-1总结.md`
3. `../语言科学课件-2总结.md`

## Common commands

Run commands from the repository root unless noted.

Backend setup and API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "backend[test,ml]"
uvicorn backend.app.main:app --reload
```

Backend tests:

```bash
pytest backend/tests -q
pytest backend/tests/test_feedback.py -q
pytest backend/tests/test_api.py::test_feedback_endpoint -q
```

Frontend setup and development:

```bash
cd frontend
npm install
npm run dev
npm run build
npm test
```

Offline experiments:

```bash
python experiments/train_acoustic_baseline.py --config experiments/configs/acoustic_baseline.yaml
python experiments/train_representation_model.py --config experiments/configs/representation_model.yaml
python experiments/train_fusion_model.py --config experiments/configs/fusion_model.yaml
python experiments/analyze_representations.py --config experiments/configs/representation_model.yaml
pytest experiments/tests -q
```

Proposal/presentation assets:

```bash
python build_proposal_ppt.py
```

## High-level architecture

This repo is split into three cooperating layers:

- `backend/`: FastAPI service for audio upload, feature extraction, tone inference, and feedback generation. `backend/app/audio.py` handles audio/F0 extraction, `features.py` computes normalized acoustic summaries, `inference.py` owns prediction paths, `feedback.py` turns model outputs into interpretable learner feedback, and `main.py` exposes API endpoints.
- `frontend/`: React + Vite interface for recording/uploading audio, selecting a target tone, rendering F0 contours, and displaying feedback. `src/api.ts` mirrors backend response types from `src/types.ts`; UI components are split into recorder, tone chart, and feedback card responsibilities.
- `experiments/`: Offline research scripts for acoustic baselines, pretrained speech representation models, fusion models, and representation analysis. These scripts should produce real metrics and paper figures rather than hard-coded results.
- `paper/` and `ppt_assets/`: paper outline, figures, and presentation materials. Only use figures generated from real experiment outputs in final paper prose.

The intended data flow is: user audio or dataset sample -> audio loading/F0 extraction -> feature or representation extraction -> tone prediction -> target-contour comparison -> interpretable feedback -> frontend chart/card rendering.

## Scope boundaries

Build the first version around controlled Mandarin tone feedback:

- isolated syllables or short words;
- four Mandarin lexical tones;
- F0 contour extraction;
- target-contour comparison;
- interpretable feedback;
- offline model comparison.

Do not add these before the paper demo works:

- full automatic speech recognition;
- sentence-level pronunciation scoring;
- user accounts;
- payment;
- classroom dashboards;
- long-term practice history;
- dialect classification;
- production A100 online inference.

## Research framing

Keep the paper grounded in language science, not just engineering.

Every major feature should connect to at least one course concept:

- acoustic phonetics: F0 contour, duration, pitch curve;
- phonology: tone category versus tone value;
- Mandarin tone research: 阴平、阳平、上声、去声 and 五度调值;
- language acquisition: learner production feedback and tone-category learning;
- computation: wav2vec2/HuBERT/Whisper representations, model comparison, representation visualization.

## Lab GPU cluster and NAS notes

The user's lab environment has shared GPU servers and a NAS workspace. This project already has a corresponding server-side folder:

```bash
/NAS/yesh/mandarin-tone-feedback
```

Known cluster resources from the Self-RAG project experience:

- `Sui-3-Wu`: 8 × NVIDIA RTX 3090.
- `Tang-1-Wu`, `Tang-2-Wu`, `Tang-3-Wu`: A40 GPU nodes. Self-RAG work used A40 48GB cards; `Tang-2-Wu` was observed as an 8 × A40 server.
- Use A40/A100-class resources for offline research work such as pretrained speech encoder feature extraction, fine-tuning/adapters, repeated experiments, hidden-state extraction, and UMAP/t-SNE visualization.
- Do not make the web prototype depend on real-time A40/A100 inference. Keep the demo runnable through a lightweight inference path when possible.

Conda/NAS practices learned from the previous project and updated for this project:

```bash
# Use the user's own Miniconda installation under /NAS/yesh.
# Do not use other users' environments or install another Miniconda inside this project.
source /NAS/yesh/miniconda3/etc/profile.d/conda.sh

# Prefer a project-level activate.sh entrypoint for reproducibility.
cd /NAS/yesh/mandarin-tone-feedback
source ./activate.sh

# Put model/dataset caches on NAS, not in the home directory.
export HF_HOME=/NAS/yesh/hf_cache
```

Before installing packages, inspect existing environments under `/NAS/yesh` and avoid duplicate installs when compatible packages already exist. The current server-side `activate.sh` may point to an existing compatible environment such as `/NAS/yesh/NLP/.conda/selfrag`; this is acceptable because it is inside the user's own `/NAS/yesh` workspace. Do not duplicate large packages such as PyTorch/CUDA unless a missing dependency is proven necessary and lightweight alternatives are unavailable.

Keep `/NAS/yesh/mandarin-tone-feedback` focused on server-side code, data, configs, logs, outputs, and training artifacts. Do not upload presentation-only or local artifact files such as finished PPTX decks, school logos, `ppt_assets/`, Visio automation scripts, `node_modules/`, local virtual environments, caches, or root-level package-lock files unless explicitly requested.

For long-running experiments, use tmux/screen and write logs under the project directory, for example:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python experiments/train_acoustic_baseline.py \
  --config experiments/configs/acoustic_baseline.yaml \
  2>&1 | tee logs/acoustic_baseline_$(date +%Y%m%d_%H%M%S).log
```

## Development priorities

1. Create a working demo loop first: target tone -> demo prediction -> feedback -> chart.
2. Add real audio upload and F0 extraction.
3. Add acoustic baseline experiments.
4. Add pretrained representation and fusion experiments.
5. Generate paper-ready figures only from real outputs.

## Testing expectations

Backend:

- Test feedback rules.
- Test F0 normalization and summary utilities.
- Test API response shape.

Frontend:

- Test feedback rendering.
- Manually verify the golden path in a browser after UI changes.

Experiments:

- Test metadata loading.
- Test feature concatenation and small utilities.

## Paper integrity

Do not invent experimental results. Accuracy, F1, confusion matrices, and figures must come from actual experiment outputs.

It is acceptable for early paper files to contain clearly marked result slots, but final prose must replace them with real results or explicitly state that a component is a prototype limitation.

## Communication style

The user prefers project directions that combine AI research, practical web engineering, GPU compute, and deployable infrastructure. When making trade-offs, prefer options that produce both a credible paper and a demonstrable web system.
