# MELD: Multimodal EmotionLines Dataset Module

## Multimodal Emotion Recognition Using Text, Speech, and Facial Features with Robust Feature Fusion

This directory contains the dataset preparation module, reproducible preprocessing pipeline, official annotations, alignment manifests, and validation reports for the **Multimodal EmotionLines Dataset (MELD)**.

---

## 1. What is MELD and Why Was It Selected?

### Dataset Overview
**MELD (Multimodal EmotionLines Dataset)** is a conversational benchmark developed by NTU, Singapore and Universidad Nacional de Educación a Distancia (UNED). It extends the text-only *EmotionLines* dataset by pairing each conversational turn with its corresponding synchronized video and audio clips extracted from the television series *Friends*.

### Key Characteristics
- **Dialogues & Utterances**: 1,433 dialogues and 13,708 utterances across multiple interacting speakers.
- **Three Core Modalities**:
  - **Text**: Written dialogue transcripts with speaker identification.
  - **Speech / Audio**: Utterance-level speech acoustics (sampled at 16 kHz or 48 kHz).
  - **Facial / Video**: Frontal and semi-profile facial expressions and body language in video clips (.mp4 container).
- **Dual Label Space**:
  - **Emotion (7 classes)**: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.
  - **Sentiment (3 classes)**: `positive`, `negative`, `neutral`.

### Why MELD Was Selected for This Project
1. **Multi-Party Conversational Context**: Unlike dyadic (2-person) datasets such as IEMOCAP, MELD features realistic multi-speaker dialogues with frequent emotional transitions, interruptions, and contextual inter-dependencies.
2. **True Modality Alignment**: Every utterance in the official annotation files contains synchronized start and end timestamps matching segmented media clips.
3. **Multimodal Branch Alignment**:
   - **Text Branch**: Extracts linguistic tokens and conversational history to capture semantic dialogue context.
   - **Facial Feature Branch**: Extracts face bounding boxes, facial action units/landmarks, and spatial-temporal representations from visual frames.
   - **Speech Feature Branch**: Extracts acoustic features (MFCCs, spectral descriptors, pitch, and energy) aligned to conversational turns.
   - **Robust Feature Fusion**: Estimates unimodal reliability and cross-modal consistency before combining representations.
   - **System Integration**: Provides standardized inference interfaces and evaluation pipelines across all modalities.

---

## 2. Official Sources & Citations

- **Official Website**: [https://affective-meld.github.io/](https://affective-meld.github.io/)
- **Upstream Code & Data Repository**: [https://github.com/declare-lab/MELD](https://github.com/declare-lab/MELD)
- **Hugging Face Hub**: [https://huggingface.co/datasets/declare-lab/MELD](https://huggingface.co/datasets/declare-lab/MELD)
- **ACL Anthology Research Paper**: [https://aclanthology.org/P19-1050/](https://aclanthology.org/P19-1050/)

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

---

## 3. Usage & Redistribution Restrictions

> [!IMPORTANT]
> **Data Governance & Repository Policy**:
> - The video and audio recordings in MELD are derived from the television show *Friends* (Warner Bros. Entertainment / Bright/Kauffman/Crane Productions).
> - **Redistribution Terms**: Because explicit redistribution permission for this public repository was not sufficiently established, raw MP4 clips, WAV extractions, nested archives, model checkpoints, and third-party annotation CSV files are **not tracked in Git**.
> - **How to Obtain Annotations**: Users and researchers can automatically download the official source annotation CSV files into `datasets/MELD/annotations/` using the included utility `download_dataset.ps1`, or obtain them directly from the official upstream Declare Lab repository (`https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/`).
> - The directory `datasets/MELD/annotations/` is preserved via `.gitkeep`, and `.gitignore` prevents local annotation CSVs from being accidentally committed.

---

## 4. Expected Directory Structure

```text
datasets/MELD/
├── README.md                 # This guide
├── dataset_card.md           # Dataset documentation, emotion classes & anomalies
├── download_dataset.ps1      # Automated download and verification script for Windows
├── prepare_dataset.py        # Deterministic preprocessing & manifest generator
├── annotations/              # Local MELD CSV files (downloaded via download_dataset.ps1, unversioned)
│   ├── .gitkeep              # Tracks directory structure in git
│   ├── train_sent_emo.csv    # 9,989 train utterances (local download)
│   ├── dev_sent_emo.csv      # 1,109 dev utterances (local download)
│   └── test_sent_emo.csv     # 2,610 test utterances (local download)
├── manifests/                # Deterministic manifests with relative paths (tracked in git)
│   ├── train_manifest.csv    # Train split manifest with modality usability flags
│   ├── dev_manifest.csv      # Development split manifest
│   └── test_manifest.csv     # Test split manifest
├── reports/                  # Validation and audit reports (tracked in git)
│   ├── preprocessing_report.json # Complete machine-readable statistics
│   ├── preprocessing_report.md   # Human-readable markdown audit summary
│   └── exclusions.csv        # Explicit record of all excluded/corrupted samples
└── raw/                      # Local unpacked video clips (excluded from git)
    ├── train_splits/
    ├── dev_splits_complete/
    └── output_repeated_splits_test/
```

---

## 5. Quickstart & Setup Guide (Windows)

### Step 1: Clone Repository
```powershell
git clone https://github.com/alokgupta7b-creator/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### Step 2: Download or Place the Raw MELD Dataset
Run the included PowerShell setup script:
```powershell
.\datasets\MELD\download_dataset.ps1
```
If you already possess the official tarball archives (`train.tar.gz`, `dev.tar.gz`, `test.tar.gz`):
1. Extract them into `datasets\MELD\raw\` (or your local project storage).
2. Ensure the resulting split folders are named:
   - `train_splits` (Train video clips)
   - `dev_splits_complete` (Development video clips)
   - `output_repeated_splits_test` (Test video clips)

### Step 3: Run the Preprocessing Pipeline
Execute `prepare_dataset.py` with Python 3.10+:
```powershell
# Standard run (produces clean manifests without copyrighted transcript text)
py datasets/MELD/prepare_dataset.py --meld-raw-dir "datasets/MELD/raw"

# Or pointing to custom storage
py datasets/MELD/prepare_dataset.py `
    --meld-raw-dir "path/to/MELD.Raw" `
    --audio-dir "path/to/audio"

# To include dialogue transcripts for text-based emotion recognition experiments:
py datasets/MELD/prepare_dataset.py `
    --meld-raw-dir "datasets/MELD/raw" `
    --include-transcript-text
```

The pipeline:
- Validates all 13,708 annotations and label values against official bounds.
- Checks video container integrity (`moov` atom scan) without external tools.
- Generates reproducible split manifests in `datasets/MELD/manifests/`.
- Documents all exclusions in `datasets/MELD/reports/exclusions.csv`.

---

## 6. Manifest Schema Reference

Each manifest CSV file (`train_manifest.csv`, `dev_manifest.csv`, `test_manifest.csv`) contains:

| Column | Type | In Git? | Description |
| :--- | :--- | :--- | :--- |
| `sample_id` | string | Yes | Globally unique identifier (e.g. `train_dia0_utt0`) |
| `split` | string | Yes | `train`, `dev`, or `test` |
| `dialogue_id` | int | Yes | Dialogue index within the split |
| `utterance_id` | int | Yes | Sequential utterance index within the dialogue |
| `emotion` | string | Yes | One of the 7 emotion categories |
| `sentiment` | string | Yes | `positive`, `negative`, or `neutral` |
| `video_path` | string | Yes | Relative path to video clip (e.g. `train_splits/dia0_utt0.mp4`) |
| `audio_path` | string | Yes | Relative path to extracted audio (e.g. `audio/train/dia0_utt0.wav`) |
| `has_text` | bool | Yes | `True` if transcript text is present in official annotations |
| `has_video` | bool | Yes | `True` if video exists and is container-valid |
| `has_audio` | bool | Yes | `True` if extracted WAV audio exists |
| `video_status` | string | Yes | `ok`, `corrupted_moov_atom_missing`, or `missing_file` |
| `audio_status` | string | Yes | `ok`, `missing_audio`, or `not_provided` |
| `is_fully_usable` | bool | Yes | `True` only when all required modalities are intact |
| `exclusion_reason` | string | Yes | Explains why a sample is excluded from multimodal training |
| `speaker` | string | Local | Speaker character name (added via `--include-transcript-text`) |
| `text` | string | Local | Utterance dialogue text (added via `--include-transcript-text`) |
| `season` / `episode` | string | Local | Season and episode ID (added via `--include-transcript-text`) |
| `start_time` / `end_time` | string | Local | Subtitle timestamp bounds (added via `--include-transcript-text`) |

---

## 7. Known Dataset Anomalies

1. **Train Split Corruption (`dia125_utt3.mp4`)**:
   - The original MP4 container lacks the index `moov` atom.
   - It cannot be decoded by standard media decoders or FFmpeg.
   - It is flagged with `is_fully_usable=False` and `exclusion_reason=corrupted_video_moov_atom_missing`.
2. **Development Split Missing Video (`dia110_utt7.mp4`)**:
   - Missing from the official development archive.
   - Although a recording with the same dialogue/utterance number exists in the test split folder, **it is NOT copied into development** to preserve strict split isolation and avoid data contamination.
   - It is flagged with `is_fully_usable=False` and `exclusion_reason=missing_video_recording`.
