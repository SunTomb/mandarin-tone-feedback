# Mandarin Tone Feedback Design Spec

## Project positioning

**Working title:** 面向普通话声调学习的可解释 AI 反馈系统

**English project name:** Mandarin Tone Feedback

**Paper type:** hybrid language-science project. The final paper should combine a small empirical study, a model comparison, and a deployable web prototype.

**Length target:** 3-9 pages in the required NCMMSC-style format.

## Core research question

Can pretrained speech representations, combined with interpretable acoustic features, provide useful and explainable feedback for Mandarin tone learning?

The project should not be framed as a generic speech-recognition app. It should be framed as a language-science system that connects tone categories, pitch contours, speaker normalization, model representations, and learner-facing feedback.

## Course alignment

This topic maps directly onto the course content:

1. **Acoustic phonetics**
   - Speech is a measurable acoustic signal.
   - Mandarin tones can be described through F0 contour, duration, and relative pitch movement.
   - The system visualizes the learner's pitch contour and compares it with a target contour.

2. **Phonology and tone**
   - The system distinguishes tone category from tone value.
   - The paper should explicitly discuss 阴平, 阳平, 上声, 去声 as tone categories and 五度调值 as an approximate tone-value representation.

3. **Language acquisition**
   - The system targets tone category learning and production feedback.
   - Feedback should explain what is wrong, not only whether the predicted label is correct.

4. **Language and computation**
   - The experiment compares traditional acoustic features with pretrained speech representations.
   - Representation visualization should connect model behavior to tone-category separability.

## Intended users

Primary users are Mandarin learners or students practicing isolated syllables and short disyllabic words. The first version should not support open-ended speech recognition or full-sentence pronunciation scoring.

## Scope

### In scope

- A small web app that supports recording or uploading short audio.
- A fixed practice list of Mandarin syllables or short words.
- Reference tone estimation for four Mandarin lexical tones.
- F0 extraction and speaker-normalized pitch visualization.
- Comparison between learner contour and target contour.
- Error-type feedback such as:
  - level tone not stable enough;
  - rising tone has insufficient rise;
  - dipping tone has unclear low turning point;
  - falling tone has insufficient fall;
  - pitch starts too high or too low relative to the target.
- Offline experiments comparing:
  - acoustic-feature baseline;
  - pretrained speech representation baseline;
  - acoustic + pretrained fusion model.
- Optional representation analysis using UMAP or t-SNE.

### Out of scope

- Full automatic speech recognition.
- Full sentence-level pronunciation scoring.
- Dialect classification.
- Large-scale learner modeling.
- Account system, payment, classroom management, or long-term learning plans.
- Real-time A100 inference in production.

## System concept

The user selects a target item, records or uploads audio, and receives a feedback card:

1. target tone, reference prediction, and confidence;
2. normalized F0 curve;
3. target tone curve;
4. approximate five-level tone value;
5. concrete feedback explaining the main deviation.

The prototype should prioritize clarity and reproducibility over product completeness.

## Model design

### Data unit

The first implementation should use short audio clips containing one syllable or one disyllabic word. Each clip should have:

- an audio path;
- a target pinyin or display text;
- a tone label in `{1, 2, 3, 4}`;
- optional speaker ID;
- optional split label in `{train, val, test}`.

### Feature paths

The experiment should compare three model families.

#### 1. Acoustic baseline

Extract compact interpretable features:

- normalized F0 mean;
- normalized F0 start, middle, end;
- F0 slope from start to end;
- F0 range;
- duration;
- voiced-frame ratio.

Train a small classifier such as logistic regression, random forest, or MLP. This baseline is important because it connects directly to acoustic phonetics.

#### 2. Pretrained speech representation baseline

Use a pretrained speech encoder such as wav2vec2, HuBERT, or Whisper encoder. Extract utterance-level representations by pooling hidden states, then train a classification head for four tone labels.

The recommended first model is wav2vec2 or HuBERT because the course specifically discusses wav2vec2-style speech representations.

#### 3. Fusion model

Concatenate acoustic features with pretrained encoder representations before classification. The hypothesis is that pretrained representations capture broad speech information while F0 features make tone-specific pitch movement explicit.

### GPU usage

A40/A100 resources should be used for:

- batch feature extraction from pretrained speech encoders;
- fine-tuning the speech encoder or adapter/classification head;
- running repeated model comparisons;
- extracting hidden states for UMAP/t-SNE representation analysis.

Deployment should use a lightweight inference path. The web app can serve a small trained classifier or cached encoder-based model rather than requiring an A100 online.

## Feedback design

Feedback must be explainable. The output should avoid vague messages such as “pronunciation is wrong.” Each feedback item should connect to an observable pitch property.

Example rules:

- Tone 1 target: stable high-level contour. If F0 range is large, report “pitch is not level enough.”
- Tone 2 target: rising contour. If end pitch is not sufficiently higher than start pitch, report “rise is insufficient.”
- Tone 3 target: low/dipping contour. If there is no low turning point, report “low-dipping shape is unclear.”
- Tone 4 target: falling contour. If end pitch is not sufficiently lower than start pitch, report “fall is insufficient.”

These rule-based explanations can be combined with the model prediction. The paper should present them as interpretable feedback derived from acoustic analysis, not as model-generated free text.

## Web architecture

Use a simple client-server architecture:

- frontend: recording/upload UI, target selection, result visualization;
- backend API: audio upload, feature extraction, model inference, feedback generation;
- experiment scripts: training, evaluation, figure generation;
- shared schema: typed result objects used by both backend and frontend.

The frontend should visualize the F0 contour with a standard charting library. The backend should return normalized contour points, target contour points, prediction, confidence, and feedback items.

## Recommended repository structure

```text
mandarin-tone-feedback/
  CLAUDE.md
  design_spec.md
  implementation_plan.md
  README.md
  data/
    raw/
    processed/
    metadata.example.csv
  experiments/
    configs/
    train_acoustic_baseline.py
    train_representation_model.py
    train_fusion_model.py
    analyze_representations.py
  backend/
    pyproject.toml
    app/
      main.py
      schemas.py
      audio.py
      features.py
      inference.py
      feedback.py
    tests/
      test_features.py
      test_feedback.py
      test_api.py
  frontend/
    package.json
    src/
      App.tsx
      api.ts
      components/
        Recorder.tsx
        ToneChart.tsx
        FeedbackCard.tsx
      test/
        toneFeedback.test.tsx
  paper/
    outline.md
    figures/
```

The current request only creates planning documents. Future implementation agents should create source files according to this structure.

## Paper structure

A 3-9 page paper can use this structure:

1. **Introduction**
   - Mandarin tones are difficult for learners because they require category-level and contour-level control.
   - Existing speech models can classify tones, but learners need interpretable feedback.
   - This work builds a prototype that combines model prediction and acoustic visualization.

2. **Linguistic background**
   - Tone category versus tone value.
   - Mandarin four tones and five-level notation.
   - F0 contour and speaker normalization.

3. **System design**
   - Web recording/upload flow.
   - F0 extraction and visualization.
   - Tone prediction and feedback generation.

4. **Model comparison**
   - Acoustic baseline.
   - Pretrained representation model.
   - Fusion model.
   - Accuracy/F1 and confusion matrix.

5. **Representation and feedback analysis**
   - UMAP/t-SNE visualization if available.
   - Example feedback cases.

6. **Conclusion**
   - The project shows how language-science concepts can guide practical AI pronunciation feedback.
   - Limitations include dataset size, controlled target list, and lack of large learner evaluation.

## Success criteria

The project is successful if it produces:

1. a runnable demo where a user records or uploads a short Mandarin item and receives tone feedback;
2. at least one trained tone classifier with an evaluation table;
3. F0 contour visualization with target comparison;
4. a paper-ready explanation of how the system relates to acoustic phonetics, phonology, tone learning, and pretrained speech models;
5. at least two paper-ready figures:
   - a system workflow diagram or UI screenshot;
   - a model comparison table, confusion matrix, or representation visualization.

## Risks and controls

### Risk: data collection takes too long

Control: begin with a small controlled dataset and add public datasets only if accessible. A manually recorded pilot set is enough for the website demo, while public or larger collected data can support model comparison.

### Risk: model fine-tuning dominates the timeline

Control: implement the acoustic baseline and frozen-representation classifier first. Full fine-tuning is an enhancement, not a dependency for the web demo.

### Risk: feedback becomes too product-like and weakly linguistic

Control: every feedback message must map to an acoustic or phonological property: F0 level, rise, fall, turning point, duration, or target tone category.

### Risk: website scope expands

Control: do not add accounts, dashboards, payment, full ASR, or curriculum features before the paper demo is complete.

## Final decision

Use the hybrid direction: a Mandarin tone-learning feedback prototype whose research contribution is the combination of interpretable F0-based feedback with pretrained speech representation experiments.
