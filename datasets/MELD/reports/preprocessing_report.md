# MELD Dataset Preprocessing & Validation Report

- **Generated At**: `2026-10-21T19:01:19.393671`
- **Raw Media Source**: `.../MELD.Raw`
- **Audio Source**: `.../audio`

## 1. Split Summary & Baseline Alignment

| Split | Annotations | Usable Multimodal | Missing Media | Corrupted Media | Missing Audio | Excluded |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **train** | 9989 | 9988 | 0 | 1 | 1 | 1 |
| **dev** | 1109 | 1108 | 1 | 0 | 1 | 1 |
| **test** | 2610 | 2610 | 0 | 0 | 0 | 0 |

## 2. Verification of Known Dataset Anomalies

The pipeline explicitly verified the historical MELD dataset issues:

1. **Train Split Corruption (`dia125_utt3.mp4`)**:
   - **Issue**: Container corrupted (`moov atom not found`).
   - **Resolution**: Kept in official annotation sequence, flagged with `is_fully_usable=False` and `exclusion_reason=corrupted_video_moov_atom_missing`. Original corrupted file preserved without alteration.

2. **Development Split Missing Media (`dia110_utt7.mp4`)**:
   - **Issue**: File absent from official development split media (`dev_splits_complete`). Although a file named `dia110_utt7.mp4` exists in the test split folder, **it was not copied into dev** to strictly preserve split isolation and prevent cross-split leakage.
   - **Resolution**: Flagged with `is_fully_usable=False` and `exclusion_reason=missing_video_recording`.

## 3. Emotion Label Distribution

| Emotion | Train | Dev | Test | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Anger** | 1109 | 153 | 345 | 1607 |
| **Disgust** | 271 | 22 | 68 | 361 |
| **Fear** | 268 | 40 | 50 | 358 |
| **Joy** | 1743 | 163 | 402 | 2308 |
| **Neutral** | 4710 | 470 | 1256 | 6436 |
| **Sadness** | 683 | 111 | 208 | 1002 |
| **Surprise** | 1205 | 150 | 281 | 1636 |

## 4. Exclusion Details

| Sample ID | Split | Dia/Utt | Video Status | Audio Status | Exclusion Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_dia125_utt3` | train | dia125_utt3 | `corrupted_moov_atom_missing` | `missing_audio` | **corrupted_video_moov_atom_missing** |
| `dev_dia110_utt7` | dev | dia110_utt7 | `missing_file` | `missing_audio` | **missing_video_recording** |
