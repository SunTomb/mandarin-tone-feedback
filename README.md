# Mandarin Tone Feedback

A hybrid language-science project for Mandarin tone learning. The system records or uploads short Mandarin audio, predicts the lexical tone, visualizes the normalized F0 contour, compares it with a target contour, and returns interpretable feedback.

## Project goals

1. Build a runnable web demo for Mandarin tone feedback.
2. Compare acoustic features, pretrained speech representations, and fusion models for tone classification.
3. Produce figures and results for a 3-9 page language-science final paper.

## Scope

The first version supports isolated syllables and short words with four Mandarin lexical tones. It does not support full ASR, sentence-level pronunciation scoring, accounts, or long-term learning records.

## Main directories

- `backend/`: FastAPI app for audio processing, inference, and feedback.
- `frontend/`: React app for recording, upload, and visualization.
- `experiments/`: offline model training and analysis scripts.
- `paper/`: paper outline and generated figures.
- `data/`: local datasets and example metadata.
