# Multimodal Emotion Recognition Using Text, Speech, and Facial Features with Robust Feature Fusion

A research-oriented benchmark and implementation framework for conversational emotion recognition across textual, acoustic, and visual modalities, integrating reliability-aware feature fusion.

---

## 1. Project Overview & Research Objectives

Conversational Emotion Recognition (CER) requires capturing emotional transitions and subtle affective cues across multiple parallel communication channels. This repository provides an end-to-end framework for multimodal emotion classification that:
- **Models Conversational Context**: Tracks multi-party dialogue dynamics and emotional shifts across consecutive speaker turns.
- **Extracts Complementary Unimodal Representations**:
  - **Text**: Linguistic embeddings capturing dialogue semantics and contextual sentiment.
  - **Speech**: Acoustic features (MFCCs, spectral descriptors, pitch, and voice energy) reflecting prosody and vocal affect.
  - **Facial Video**: Spatial-temporal representations of facial action units and expressions across video frames.
- **Applies Robust Feature Fusion**: Mitigates unimodal uncertainty and sensory noise by evaluating cross-modal consistency before computing joint prediction distributions.
- **Preserves Dataset Integrity**: Implements deterministic, reproducible data preparation and strict split boundaries without data leakage.

---

## 2. System Architecture

The framework consists of three specialized unimodal processing branches unified by a reliability-aware multimodal fusion module:

```text
 ┌────────────────┐       ┌───────────────────────────────┐
 │ Dialogue Text  │ ───>  │ Linguistic Context Encoder    │ ──┐
 └────────────────┘       └───────────────────────────────┘   │
                                                              │
 ┌────────────────┐       ┌───────────────────────────────┐   │     ┌────────────────────────┐       ┌──────────────────┐
 │ Speech Audio   │ ───>  │ Acoustic Feature Extractor    │ ──┼──>  │ Reliability Estimation │ ───>  │ Emotion Class    │
 └────────────────┘       └───────────────────────────────┘   │     │ & Cross-Modal Fusion   │       │ (7 Categories)   │
                                                              │     └────────────────────────┘       └──────────────────┘
 ┌────────────────┐       ┌───────────────────────────────┐   │
 │ Facial Video   │ ───>  │ Spatial-Temporal Visual Model │ ──┘
 └────────────────┘       └───────────────────────────────┘
```

### Unimodal Branches
1. **Text Branch**: Encodes conversational history and current utterance tokens to capture contextual dialogue semantics.
2. **Speech Audio Branch**: Extracts acoustic prosodic features (MFCCs, energy envelopes, fundamental frequency) robust to background acoustic variability.
3. **Facial Feature Branch**: Processes face-aligned bounding boxes and visual frames to identify temporal micro-expressions and emotion indicators.
4. **Robust Multimodal Fusion**: Estimates unimodal reliability and cross-modal agreement, dynamically adjusting modality weights to maintain high performance even when a modality is degraded or noisy.

---

## 3. Supported Benchmarks & Dataset Infrastructure

This framework evaluates dynamic reliability and cross-modal consistency across two complementary benchmark datasets:

### Dataset Comparison

| Feature Dimension | Benchmark 1: [MELD](datasets/MELD/) | Benchmark 2: [CMU-MOSEI](datasets/CMU-MOSEI/) |
| :--- | :--- | :--- |
| **Discourse Context** | Multi-party conversational dialogue (*Friends* sitcom) | In-the-wild monologue video presentations (YouTube) |
| **Primary Challenge** | Conversational context tracking & speaker turn dynamics | Sensory noise robustness & unconstrained web variability |
| **Segment Count** | 13,708 utterances (1,433 dialogues) | 23,453 monologue segments (3,294 videos) |
| **Speaker Count** | Fixed cast of recurring sitcom characters | >1,000 distinct real-world speakers |
| **Emotion Schema** | 7 Categorical classes (`neutral`, `joy`, `sadness`, `anger`, `surprise`, `disgust`, `fear`) | 6 Continuous intensities in `[0.0, 3.0]` (`happy`, `sad`, `anger`, `surprise`, `disgust`, `fear`) |
| **Sentiment Schema** | 3 Discrete classes (`positive`, `negative`, `neutral`) | Continuous sentiment score in `[-3.0, +3.0]` |
| **Modality Alignment** | Utterance-level video (.mp4) and audio (.wav) | Temporal segment timestamps `[start_time, end_time]` (.mp4, .wav, .csd) |
| **Documentation** | [MELD Dataset Card](datasets/MELD/dataset_card.md) | [CMU-MOSEI Dataset Card](datasets/CMU-MOSEI/dataset_card.md) |

---

## 4. Repository Structure

```text
.
├── README.md                          # Main project documentation & benchmark overview
├── .gitignore                         # Strict repository exclusions (media, features, models)
└── datasets/
    ├── MELD/                          # Benchmark 1: Conversational Emotion Recognition
    │   ├── README.md                  # MELD module guide & setup instructions
    │   ├── dataset_card.md            # MELD specifications & anomaly documentation
    │   ├── download_dataset.ps1       # Annotation download & safe extraction utility
    │   ├── prepare_dataset.py         # Deterministic preprocessing & manifest generator
    │   ├── annotations/               # Local annotation CSV directory (.gitkeep tracked)
    │   ├── manifests/                 # Reproducible split manifests (tracked in git)
    │   │   ├── train_manifest.csv
    │   │   ├── dev_manifest.csv
    │   │   └── test_manifest.csv
    │   └── reports/                   # Preprocessing validation & exclusion reports
    │       ├── preprocessing_report.json
    │       ├── preprocessing_report.md
    │       └── exclusions.csv
    └── CMU-MOSEI/                     # Benchmark 2: In-the-Wild Multimodal Monologues
        ├── README.md                  # CMU-MOSEI module guide & SDK setup reference
        ├── dataset_card.md            # CMU-MOSEI specifications & continuous schema
        ├── download_dataset.ps1       # Environment check, SDK helper & archive unpacker
        ├── prepare_dataset.py         # Deterministic preprocessing & split isolation validator
        ├── annotations/               # Local annotations directory (.gitkeep tracked)
        │   └── .gitkeep
        ├── manifests/                 # Generated split manifests directory (.gitkeep tracked)
        │   └── .gitkeep
        ├── reports/                   # Preprocessing audit reports directory (.gitkeep tracked)
        │   └── .gitkeep
        └── tests/                     # Automated regression test suite
            ├── __init__.py
            ├── fixtures/              # Non-restricted synthetic test fixtures
            │   ├── synthetic_mosei_annotations.csv
            │   └── synthetic_splits.json
            └── test_prepare_dataset.py # Comprehensive unit tests
```

---

## 5. Quickstart & Data Preparation

### 1. Clone the Repository
```bash
git clone https://github.com/alokgupta7b-creator/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. MELD Dataset Preparation
Fetch official MELD annotations and generate manifests:
```powershell
# Fetch official annotations
.\datasets\MELD\download_dataset.ps1

# Generate manifests and audit reports
python datasets/MELD/prepare_dataset.py --meld-raw-dir "datasets/MELD/raw"
```

### 3. CMU-MOSEI Dataset Preparation
Verify environment, inspect SDK instructions, and execute the preparation pipeline:
```powershell
# Check environment and display SDK download instructions
.\datasets\CMU-MOSEI\download_dataset.ps1 -CheckEnvironment

# Run metadata validation and generate split manifests
python datasets/CMU-MOSEI/prepare_dataset.py `
    --annotations-file "datasets/CMU-MOSEI/annotations/mosei_annotations.csv" `
    --output-dir "datasets/CMU-MOSEI" `
    --skip-media-verify
```

### 4. Run Automated Test Suites
Execute the regression test suite covering schema validation, split isolation, and deterministic manifests:
```bash
python -m unittest discover -s datasets/CMU-MOSEI/tests -v
```

---

## 6. Research Notice & Citations

All raw media files, large precomputed feature matrices (`.npy`, `.h5`, `.csd`), and third-party transcripts are excluded from version control in compliance with academic data governance policies and copyright considerations.

### Academic Citations

```bibtex
@inproceedings{poria-etal-2019-meld,
    title = "{MELD}: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations",
    author = "Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Gautam, Gautam and Chhaya, Niyati and Cambria, Erik",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1050",
    doi = "10.18653/v1/P19-1050",
    pages = "527--536"
}

@inproceedings{zadeh2018multimodal,
    title = "Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Approaches",
    author = "Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1208",
    doi = "10.18653/v1/P18-1208",
    pages = "2236--2246"
}
```
---

## Text Emotion Recognition Module

This module predicts emotions from text using TF-IDF feature extraction and Logistic Regression, trained on the MELD dataset.

### Features
- Text preprocessing and cleaning.
- TF-IDF feature extraction using unigrams and bigrams.
- Logistic Regression with class balancing.
- Prediction of seven emotions: anger, disgust, fear, joy, neutral, sadness, and surprise.
- Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.

### Dataset Setup

Download the MELD dataset and locate these files:
- `train_sent_emo.csv`
- `dev_sent_emo.csv`
- `test_sent_emo.csv`

Place the `meld_dataset` folder in the parent directory of the repository, matching the path configured in `text_emotion/train.py`. The dataset is not included in Git version control.

### Installation

```bash
python -m pip install pandas scikit-learn joblib
```

### Train the Model

```bash
python -m text_emotion.train
```

### Predict an Emotion

```bash
python -m text_emotion.predict
```

Enter a sentence when prompted. Type `exit` to stop.

### Evaluate the Model

```bash
python -m text_emotion.evaluate
```

### Results

The model achieved **41.38% test accuracy** on 2,610 samples from the official MELD test split. The weighted F1-score was **0.43**. Performance varies across emotion classes, with less frequent emotions being more difficult to classify.

These results provide a baseline for future improvements.

**Note:** The trained model (`text_emotion_model.joblib`) is generated locally and is not included in Git version control.