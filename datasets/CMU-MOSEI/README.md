# CMU-MOSEI: Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity Module

## Multimodal Emotion Recognition Using Text, Speech, and Facial Features with Robust Feature Fusion

This directory contains the dataset preparation pipeline, schema validation, alignment manifests, and validation audit reports for the **CMU-MOSEI** dataset.

---

## 1. What is CMU-MOSEI and Why Was It Selected?

### Dataset Overview
**CMU-MOSEI (Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity)** is the largest public benchmark for multimodal language analysis in the wild. Created by the CMU MultiComp Lab, it contains **23,453 annotated monologue segments** extracted from **3,294 YouTube videos** spoken by more than **1,000 distinct speakers** across **250 distinct topics**.

### How CMU-MOSEI Complements MELD

| Research Dimension | Dataset 1: MELD | Dataset 2: CMU-MOSEI |
| :--- | :--- | :--- |
| **Acoustic Environment** | Studio environment with laugh tracks, overlapping dialogue, room acoustics | In-the-wild YouTube audio with diverse microphones, background music, ambient noise |
| **Visual Environment** | Multi-speaker sitcom scenes, varied camera cuts, staged facial expressions | Single-speaker webcam monologues, dynamic head poses, lighting variations, micro-expressions |
| **Discourse Structure** | Multi-party conversational dialogue with rapid emotional transitions across turns | Continuous self-contained monologues focused on opinions, reviews, and debate topics |
| **Annotation Space** | 7 Categorical emotion classes + 3 discrete sentiments | 6 Continuous emotion intensities in `[0.0, 3.0]` + continuous sentiment in `[-3.0, +3.0]` |
| **Speaker Diversity** | Fixed cast of recurring actors (*Friends*) | Over 1,000 distinct real-world speakers across diverse demographics |

### Strategic Research Rationale
The primary research objective of this repository is to develop a **dynamic reliability and consistency estimation mechanism** that weighs text, speech, and facial features prior to fusion. Evaluating this mechanism across both MELD and CMU-MOSEI ensures that:
1. **Reliability Weighting Generalization**: The reliability estimator is tested against both television studio noise (laugh tracks, scene cuts) and unconstrained web noise (webcam compression, microphone clipping, ambient noise).
2. **Cross-Domain Robustness**: The framework is benchmarked on both multi-speaker conversational dialogue (MELD) and in-the-wild monologue presentations (CMU-MOSEI).
3. **Continuous and Categorical Targets**: The system supports both categorical classification and continuous affective regression.

---

## 2. Official Sources & Citations

- **Official Dataset Page**: [https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
- **Official SDK (CMU-MultimodalSDK)**: [https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- **ACL Anthology Research Paper**: [https://aclanthology.org/P18-1208/](https://aclanthology.org/P18-1208/)

### Academic Citation
```bibtex
@inproceedings{zadeh2018multimodal,
  title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Approaches},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2236--2246},
  year={2018},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/P18-1208}
}
```

---

## 3. Data Governance & Redistribution Policy

> [!IMPORTANT]
> **Data Governance Policy**:
> - Raw YouTube video clips, large computational sequence arrays (`.csd`, `.h5`), and copyright-restricted transcripts are **not tracked or committed to this Git repository**.
> - All media directories (`raw/`, `features/`, `data/`) and raw annotation CSVs are strictly ignored via `.gitignore`.
> - Researchers obtain features directly through the CMU-MultimodalSDK or authorized CMU portals into local workspace directories.
> - This repository tracks the reproducible pipeline code, documentation, synthetic test fixtures, and validation audit reports.

---

## 4. Directory Structure

```text
datasets/CMU-MOSEI/
├── README.md                 # This module guide
├── dataset_card.md           # Official specifications, label space, and anomaly records
├── download_dataset.ps1      # PowerShell verification, SDK helper & safe extraction utility
├── prepare_dataset.py        # Deterministic preprocessing, validation & manifest generator
├── annotations/              # Local annotations directory (preserved via .gitkeep)
│   └── .gitkeep
├── manifests/                # Generated split manifests directory (preserved via .gitkeep)
│   └── .gitkeep              # Manifests are generated upon local pipeline execution
├── reports/                  # Preprocessing validation & audit reports (preserved via .gitkeep)
│   └── .gitkeep              # Reports are generated upon local pipeline execution
└── tests/                    # Automated regression & integrity test suite
    ├── __init__.py
    ├── fixtures/             # Non-restricted synthetic test fixtures
    │   ├── synthetic_mosei_annotations.csv
    │   └── synthetic_splits.json
    └── test_prepare_dataset.py # Comprehensive unit test suite
```

> [!NOTE]
> In accordance with data governance policies, full-dataset manifests and reports are generated locally by running `prepare_dataset.py` after acquiring official annotations or SDK data. Public repository tracking retains code, tests, documentation, and directory placeholders (`.gitkeep`).

---

## 5. Manifest Schema Reference

Each generated manifest CSV contains the following standardized columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | string | Unique identifier formatted as `{video_id}_{clip_id}` |
| `split` | string | Partition assignment (`train`, `dev`, `test`) |
| `video_id` | string | Source YouTube video identifier (ensures speaker isolation) |
| `clip_id` | string | Utterance/segment index within the source video |
| `start_time` | float | Segment start timestamp in seconds |
| `end_time` | float | Segment end timestamp in seconds |
| `sentiment` | float | Continuous ground-truth sentiment score in `[-3.0, +3.0]` |
| `sentiment_discrete` | string | Configurable mapped categorical sentiment (`positive`, `negative`, `neutral`) |
| `happy` | float | Ground-truth happiness intensity score in `[0.0, 3.0]` |
| `sad` | float | Ground-truth sadness intensity score in `[0.0, 3.0]` |
| `anger` | float | Ground-truth anger intensity score in `[0.0, 3.0]` |
| `surprise` | float | Ground-truth surprise intensity score in `[0.0, 3.0]` |
| `disgust` | float | Ground-truth disgust intensity score in `[0.0, 3.0]` |
| `fear` | float | Ground-truth fear intensity score in `[0.0, 3.0]` |
| `dominant_emotion` | string | Name of the highest-intensity emotion dimension above threshold, or `none` |
| `text_path` | string | Relative path to text transcript or linguistic feature array |
| `audio_path` | string | Relative path to audio WAV or acoustic feature array |
| `visual_path` | string | Relative path to visual MP4 or facial feature array |
| `has_text` | boolean | Availability flag for the textual modality |
| `has_audio` | boolean | Availability flag for the acoustic modality |
| `has_visual` | boolean | Availability flag for the visual modality |
| `is_fully_usable` | boolean | Set to `True` only when schema and all required modalities are valid |
| `exclusion_reason` | string | Explicit exclusion reason if unusable (empty if valid) |

---

## 6. Pipeline Usage & Execution

### 1. Environment & Setup Verification
Validate the local environment and display SDK download instructions:
```powershell
.\datasets\CMU-MOSEI\download_dataset.ps1 -CheckEnvironment
```

### 2. Standard Metadata Validation (Fast, Non-Media)
Validate annotations, verify speaker isolation, and generate manifests:
```bash
python datasets/CMU-MOSEI/prepare_dataset.py \
    --annotations-file "datasets/CMU-MOSEI/annotations/mosei_annotations.csv" \
    --output-dir "datasets/CMU-MOSEI" \
    --skip-media-verify
```

### 3. Full Multimodal Media & Feature Verification
Verify local raw media files or pre-extracted feature arrays:
```bash
python datasets/CMU-MOSEI/prepare_dataset.py \
    --annotations-file "datasets/CMU-MOSEI/annotations/mosei_annotations.csv" \
    --audio-dir "datasets/CMU-MOSEI/features/audio" \
    --visual-dir "datasets/CMU-MOSEI/features/visual" \
    --text-dir "datasets/CMU-MOSEI/features/text" \
    --output-dir "datasets/CMU-MOSEI"
```

### 4. Configurable Threshold Tuning
Configure explicit decision thresholds for sentiment and emotion presence without mutating raw ground-truth values:
```bash
python datasets/CMU-MOSEI/prepare_dataset.py \
    --annotations-file "datasets/CMU-MOSEI/annotations/mosei_annotations.csv" \
    --sentiment-threshold 0.1 \
    --emotion-threshold 0.5 \
    --output-dir "datasets/CMU-MOSEI"
```

---

## 7. Dynamic Reliability Estimation & Multimodal Fusion Integration

The planned multimodal fusion architecture operates as follows:
1. **Unimodal Feature Extraction**:
   - **Text**: Contextual token representations from Transformer backbones.
   - **Speech**: Acoustic descriptors capturing pitch, energy, and prosodic stability.
   - **Facial**: Spatial-temporal representations of Facial Action Units and gaze vectors.
2. **Reliability Estimation**:
   - An uncertainty estimation module computes confidence indicators for each modality (e.g., acoustic signal-to-noise ratio, facial tracking confidence, lexical ambiguity).
3. **Cross-Modal Consistency**:
   - Evaluates pairwise directional agreement between modality predictions before final aggregation.
4. **Dynamic Weighted Fusion**:
   - Unimodal representations are dynamically scaled by their reliability and consistency weights, minimizing the impact of corrupted or absent modalities.

---

## 8. Automated Testing

Run the automated test suite using Python's standard `unittest` framework:
```bash
python -m unittest discover -s datasets/CMU-MOSEI/tests -v
```

All test cases execute against self-contained synthetic fixtures, verifying:
- Strict speaker-independent split isolation and cross-split leakage prevention.
- Validation of continuous sentiment and emotion bounds.
- Deterministic manifest generation and reproducible output ordering.
- Detection of duplicate identifiers and missing modality files.
- Accurate exclusion categorization and markdown report path sanitization.

### Verification Status & Storage Requirements
- **Verification Basis**: All pipeline validations and automated tests execute against synthetic fixtures. End-to-end processing of the full official CMU-MOSEI dataset requires obtaining the official distribution locally.
- **Estimated Local Storage Requirements**:
  - Raw video clips (.mp4): ~35 GB (estimated).
  - Aligned computational sequence features (.csd via SDK): ~8–12 GB (estimated).
  - Annotation metadata: ~15 MB (estimated).
