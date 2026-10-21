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

## 3. Dataset Infrastructure: MELD

The primary benchmark supported by this project is the **Multimodal EmotionLines Dataset (MELD)**:
- **13,708 utterances** across **1,433 dialogues** from multi-party conversations.
- **7 Emotion Categories**: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.
- **3 Sentiment Categories**: `positive`, `negative`, `neutral`.
- **Predefined Partitions**:
  - **Train**: 9,989 utterances (9,988 multimodal-usable)
  - **Development**: 1,109 utterances (1,108 multimodal-usable)
  - **Test**: 2,610 utterances (2,610 multimodal-usable)

Detailed dataset documentation, class distribution analyses, and known data limitation records are documented in the [MELD Dataset Card](datasets/MELD/dataset_card.md).

---

## 4. Repository Structure

```text
.
├── README.md                      # Main project documentation
├── .gitignore                     # Repository exclusions (media, models, caches)
└── datasets/
    └── MELD/
        ├── README.md              # MELD module instructions & schema reference
        ├── dataset_card.md        # Dataset specifications & anomaly documentation
        ├── download_dataset.ps1   # PowerShell download, verification & extraction utility
        ├── prepare_dataset.py     # Deterministic preprocessing & manifest generator
        ├── annotations/           # Local annotations folder (.gitkeep tracked)
        ├── manifests/             # Reproducible split manifests (tracked in git)
        │   ├── train_manifest.csv
        │   ├── dev_manifest.csv
        │   └── test_manifest.csv
        └── reports/               # Preprocessing validation & audit reports
            ├── preprocessing_report.json
            ├── preprocessing_report.md
            └── exclusions.csv
```

---

## 5. Quickstart & Data Preparation

### 1. Clone the Repository
```bash
git clone https://github.com/alokgupta7b-creator/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. Obtain Annotations & Raw Data
Run the included setup script to fetch the official MELD annotations:
```powershell
.\datasets\MELD\download_dataset.ps1
```

To extract local `.tar.gz` archives if present in `datasets/MELD/raw`:
```powershell
.\datasets\MELD\download_dataset.ps1 -ExtractArchives
```

### 3. Generate Manifests & Validate Integrity
Run the reproducible preprocessing pipeline:
```bash
python datasets/MELD/prepare_dataset.py --meld-raw-dir "datasets/MELD/raw"
```

To generate local manifests containing dialogue transcripts for text-only experiments:
```bash
python datasets/MELD/prepare_dataset.py --meld-raw-dir "datasets/MELD/raw" --include-transcript-text
```

---

## 6. Research Notice & Citations

The media clips in MELD are derived from the television show *Friends* (Warner Bros. Entertainment) and are utilized under fair-use for non-commercial academic research. In accordance with data governance policies, raw audio/video files, feature dumps, and third-party annotation files are not hosted in this Git repository.

### Academic Citation
```bibtex
@inproceedings{poria-etal-2019-meld,
    title = "{MELD}: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations",
    author = "Poria, Soujanya  and
      Hazarika, Devamanyu  and
      Majumder, Navonil  and
      Gautam, Gautam  and
      Chhaya, Niyati  and
      Cambria, Erik",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1050",
    doi = "10.18653/v1/P19-1050",
    pages = "527--536"
}
```
