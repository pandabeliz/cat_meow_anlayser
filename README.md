# 🐱 Cat Distress Detection

A machine learning pipeline for detecting distress in cat vocalisations using acoustic feature extraction and a tuned LightGBM classifier. Built as a data science portfolio project.

**Live demo:** [kitty-comm-analyzer.lovable.app](https://kitty-comm-analyzer.lovable.app/analyze)  
**Model on Hugging Face:** [belpekkan/cat_distress_detection](https://huggingface.co/belpekkan/cat_distress_detection)

---

## Overview

The model classifies cat meow recordings as **distress** (isolation context) or **normal** (brushing or food-anticipation context). It returns a predicted label and a confidence probability for every recording.

This project covers the full data science pipeline — from raw audio preprocessing through feature engineering, model selection, evaluation, and deployment as a live web application.

---

## Dataset

[CatMeows](https://zenodo.org/records/4008297) — Ntalampiras et al. (2019).  
*Automatic Classification of Cat Vocalizations Emitted in Different Contexts.* Animals 9(8):543.

- 440 recordings from **21 cats** (10 Maine Coon, 11 European Shorthair)
- Recorded via Bluetooth collar microphone at **8 kHz**
- Three contexts: brushing (127), isolation (221), food anticipation (92)
- Binary label: isolation → **distress (1)**, brushing + food → **normal (0)**

---

## Repository Structure

```
├── Modelling_v2.ipynb          # Full pipeline: preprocessing → features → modelling
├── features_extracted.csv      # Extracted feature matrix (434 recordings × 81 features)
├── cat_distress_model_tuned.joblib  # Saved model artefacts (model + scaler + config)
├── loco_results.csv            # Leave-one-cat-out evaluation results per cat
└── hf_upload/
    ├── app.py                  # Gradio inference app (deployed to HF Spaces)
    ├── requirements.txt        # Space dependencies
    ├── README.md               # Hugging Face model card
    ├── config.json             # Inference configuration
    └── feature_cols.json       # Ordered feature list for inference
```

---

## Pipeline

### 1. Preprocessing
Handled by collaborator Erin prior to feature extraction:
- Native 8 kHz sample rate preserved — hardware Nyquist ceiling is already 4 kHz, upsampling would add no information
- DC offset removal
- High-pass filter: 100 Hz, 5th-order Butterworth, zero-phase
- Padded to 2.5 s with trailing zeros
- **No amplitude normalisation** — consistent collar placement means raw amplitude carries real signal

### 2. Feature Extraction (81 features)

| Group | Count | Description |
|---|---|---|
| MFCCs | 52 | 13 coefficients × (mean, std) + delta × (mean, std) — matches Ntalampiras et al. |
| Spectral | 20 | Centroid, bandwidth, rolloff, flatness, contrast (4 bands), ZCR |
| Spectral entropy | 2 | Mean + std — captures tonal vs. noisy structure, relevant for stress |
| Temporal | 4 | RMS mean/std, onset rate, temporal centroid |
| Pitch (F0) | 3 | Mean, std, voiced ratio via `pyin` |

### 3. Modelling

- **Algorithm:** LightGBM classifier
- **Hyperparameter tuning:** Optuna, 100 trials, maximising CV AUC
- **Class imbalance:** `class_weight='balanced'`
- **Threshold optimisation:** F2 score (recall weighted 2× over precision) — appropriate for welfare monitoring where a missed distress event is worse than a false alarm

### 4. Evaluation — Leave-One-Cat-Out (LOCO)

Standard random train/test splitting is **incorrect** for this dataset. With multiple recordings per cat, a random split causes the model to learn individual vocal signatures rather than distress patterns — 19 of 21 cats leaked in the naive split, producing an inflated AUC of 0.476 per-cat on inspection (some cats 1.000, others 0.286).

The correct approach is **leave-one-cat-out cross-validation**: train on 20 cats, evaluate on the held-out cat, repeat for all 21. This tests whether the model generalises to cats it has never heard.

| Metric | Value |
|---|---|
| Mean LOCO AUC | 0.780 ± 0.248 |
| Cats with AUC ≥ 0.80 | 11 / 16 evaluable |
| Decision threshold | 0.135 |

5 cats (BRI01, CLE01, IND01, JJX01, LEO01) could not be evaluated — they only appear in one context in the dataset so the hold-out set contains only one class. NIG01 was excluded from the mean (only 1 distress recording when held out).

The high standard deviation (±0.248) reflects the small population size — 21 individuals is a genuine dataset constraint, not a modelling failure.

---

## Key Methodological Finding

An initial naive `train_test_split` (ignoring cat identity) produced:
- Recall: **88.6%**, F1: **0.847** — looked strong
- On inspection: **19/21 cats leaked** between train and val
- Per-cat AUC ranged from 0.286 to 1.000 — the model was identifying individual cats, not detecting distress
- Overall leaked AUC: **0.476** (worse than random)

Switching to group-aware evaluation (LOCO by `cat_id`) gave honest results. This finding is documented as part of the project narrative.

---

## Deployment

The model is deployed as a **Gradio app on Hugging Face Spaces**, used purely as an inference API. The frontend is a React web application built with Lovable that calls the Space endpoint and renders results in its own interface.

```
User uploads .wav or records audio
        ↓
Lovable React frontend
        ↓  POST audio as base64
HuggingFace Space (Gradio — inference only, not displayed)
        ↓  returns label + confidence + interpretation
Lovable displays result with colour-coded confidence
        (green = normal, orange = uncertain, red = distress)
```

---

## Limitations

- **Small population:** 21 cats from 2 breeds limits generalisation across the broader cat population
- **Recording setup:** Collar microphone at 8 kHz — may not generalise to other microphone types or distances
- **Contextual labelling:** Isolation is used as a proxy for distress; individual variation in how cats express distress is not accounted for
- **5 cats unevaluable:** Single-context recordings prevent LOCO evaluation for these individuals
- **No augmentation:** Dataset size did not support augmentation without risk of train/val leakage

---

## References

- Ntalampiras et al. (2019). *Automatic Classification of Cat Vocalizations Emitted in Different Contexts.* Animals 9(8):543. https://doi.org/10.3390/ani9080543
- CatMeows dataset: https://zenodo.org/records/4008297

---

## Collaborators

**Erin Wall** — Preprocessing pipeline, quality flagging methodology, and technical notes. Erin's detailed documentation of the 8 kHz hardware constraint, DC offset removal, high-pass filtering, and feature recommendations directly shaped this pipeline.

**[Anna N. Osiecka](https://www.linkedin.com/in/anna-n-osiecka-b49758bb/?locale=en)** — Domain expertise and project guidance throughout the development of this project.

---

## Acknowledgements

This project was developed as part of the **[Electric Sheep Futurekind Fellowship](https://electric-sheep-org.squarespace.com/futurekind)** — a 12-week AI for impact programme immersing fellows in the intersection of artificial intelligence and animal protection. The fellowship provided mentorship, community, and the project framework that made this work possible.

> *"Be part of a global community of impact-focused fellows making AI work for our planet and all its sentient life."*
> — Electric Sheep Futurekind
