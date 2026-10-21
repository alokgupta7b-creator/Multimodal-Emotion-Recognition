# Dataset Card for Multimodal EmotionLines Dataset (MELD)

## 1. Dataset Summary

The **Multimodal EmotionLines Dataset (MELD)** is a multimodal, multi-party conversational dataset created by enriching the *EmotionLines* dataset with audiovisual recordings. It contains **13,708 utterances** across **1,433 dialogues** from the television series *Friends*, involving conversational exchanges among recurring primary and guest characters.

Each conversational turn is aligned with:
1. **Text**: Subtitle and dialogue transcripts.
2. **Audio**: Acoustic speech waveforms corresponding to the utterance.
3. **Video**: Head and torso visual recordings showing facial expressions and contextual interactions.

### Intended Research Use
This dataset is intended for non-commercial academic research on:
- Conversational Emotion Recognition (CER)
- Multimodal feature extraction (Text, Speech, Facial)
- Cross-modal consistency and uncertainty estimation
- Late, early, and hybrid fusion architectures for affective computing

---

## 2. Modalities & Label Space

### Available Modalities
- **Text**: UTF-8 textual strings of spoken dialogue, tokenized by sentence/turn.
- **Speech (Audio)**: Segmented audio containing vocal tone, pitch, acoustic prosody, and cadence.
- **Facial (Video)**: MP4 video segments (typically 24/25 fps) showing the speaker's facial expressions, mouth movements, and upper-body gestures.

### Emotion Labels (7 Classes)
The dataset is annotated with discrete categorical emotions based on Ekman's basic emotions plus neutral:
1. **Anger**: Hostile, irritated, or confrontational speech and expressions.
2. **Disgust**: Revulsion or profound disapproval.
3. **Fear**: Apprehension, anxiety, or panic.
4. **Joy**: Happiness, amusement, triumph, or satisfaction.
5. **Neutral**: Non-affective informational or matter-of-fact dialogue turns.
6. **Sadness**: Sorrow, dejection, or disappointment.
7. **Surprise**: Shock, astonishment, or unexpected revelation.

### Sentiment Labels (3 Classes)
1. **Positive** (primarily associated with Joy and pleasant Surprise)
2. **Negative** (Anger, Disgust, Fear, Sadness)
3. **Neutral** (Neutral dialogue turns)

---

## 3. Split Conventions & Class Distribution

The dataset provides official, isolated partitions by complete dialogues to ensure that conversational context from an episode does not leak into the test set.

| Split | Dialogues | Utterances | Avg. Length | Primary Use |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,039 | 9,989 | 8.03 words | Model training & representation learning |
| **Development (Dev)** | 114 | 1,109 | 7.99 words | Hyperparameter tuning, early stopping, unimodal validation |
| **Test** | 280 | 2,610 | 8.28 words | Final held-out evaluation & fusion benchmark |
| **Total** | **1,433** | **13,708** | **—** | |

### Class Imbalance Breakdown

| Emotion | Train | Dev | Test | Total | Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neutral** | 4,710 | 470 | 1,256 | 6,436 | 46.95% |
| **Joy** | 1,743 | 163 | 402 | 2,308 | 16.84% |
| **Surprise** | 1,205 | 150 | 281 | 1,636 | 11.93% |
| **Anger** | 1,109 | 153 | 345 | 1,607 | 11.72% |
| **Sadness** | 683 | 111 | 208 | 1,002 | 7.31% |
| **Disgust** | 271 | 22 | 68 | 361 | 2.63% |
| **Fear** | 268 | 40 | 50 | 358 | 2.61% |
| **Total** | **9,989** | **1,109** | **2,610** | **13,708** | **100.0%** |

> [!NOTE]
> **Severe Class Imbalance**: Neutral dominates nearly 47% of the dataset, whereas Disgust and Fear each constitute less than 3%. Models evaluated on MELD must report **Weighted F1-score** and **Macro F1-score** rather than unweighted accuracy to reflect true performance across minority emotion categories.

---

## 4. Known Data Limitations & Corruptions

1. **Train Split: `dia125_utt3.mp4`**:
   - **Status**: Corrupted in the official MELD raw distribution.
   - **Root Cause**: The MP4 container lacks a valid `moov` atom (movie header box), making it unreadable by standard media parsers (FFprobe reports `moov atom not found`).
   - **Policy**: Do not overwrite or mutate the original raw file. In the manifest, it is recorded with `video_status=corrupted_moov_atom_missing` and excluded from multimodal model training (`is_fully_usable=False`).
2. **Development Split: `dia110_utt7.mp4`**:
   - **Status**: Missing from the official `dev_splits_complete` archive.
   - **Policy**: Although an utterance with ID `dia110_utt7.mp4` exists in the test split folder, **it must never be copied into the development partition**. Cross-split transfer violates split isolation and constitutes data leakage. It is recorded as `video_status=missing_file` and `is_fully_usable=False`.
3. **Dialogue Context Dependence**:
   - Utterances are often elliptical and heavily rely on previous speaker turns to infer emotion. Unimodal models (especially audio-only) may struggle on short phrases like *"Yeah"* or *"No way"* without conversational history.
4. **Studio Noise & Laugh Tracks**:
   - Because MELD is recorded from a television sitcom, background audience laughter, music cues, and overlapping speech are present in the audio track, requiring noise-robust acoustic feature extraction.

---

## 5. Provenance & Licensing Considerations

- **Original Dataset Authors**: Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Gautam, Niyati Chhaya, Erik Cambria (NTU Singapore, UNED Madrid, Adobe Research).
- **Primary Source**: Television series *Friends*, copyrighted by Warner Bros. Television.
- **Redistribution Status & Governance**:
  - The video and audio recordings are provided under fair-use for academic research purposes.
  - Because redistribution permission was not sufficiently established for public GitHub hosting, neither the raw audiovisual recordings nor the third-party annotation CSV files are tracked in this repository.
  - Users and researchers must download annotations and media directly from official sources (`https://affective-meld.github.io/` or `https://github.com/declare-lab/MELD`) into local workspace directories.
  - Commercial exploitation or re-hosting of raw media archives on public Git repositories without permission is strictly prohibited.
