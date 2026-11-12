# Final Model Evaluation Report

**Date:** 2026-11-12  
**Project:** Multimodal Emotion Recognition System  
**Class Order:** `anger, disgust, fear, joy, neutral, sadness, surprise`  

> **Evaluation Split Boundary:** Held-out MELD `test` (`N = 2,610`) metrics are **reused strictly from the completed frozen Phase 16 benchmark** (`models/phase16_final_evaluation/final_results.json`, zero further test-split tuning or re-evaluation performed). All newer P0/P1/P2 speech (`Wav2Vec2_v2`) and facial (`ResNet18_v2`) models were trained strictly on MELD `train` and evaluated strictly on the dialogue-grouped disjoint MELD `dev_val` (`N = 548` / `544` detected videos) and full MELD `dev` (`N = 1,108`) splits.

---

## 1. Existing Frozen Held-Out MELD `test` Results (Phase 16 Baseline, `N = 2,610`)

Evaluated on RTX 4050 CUDA (`81.68 s` total for 2,610 samples; median per-sample latencies: DistilBERT `6.71 ms`, Wav2Vec2 v1 `16.76 ms`, Candidate 4 fusion `0.07 ms`):

| System (Frozen V1 Phase 16) | Accuracy | Macro-F1 | Weighted-F1 | Brier | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Text Baseline** | 63.10% | 41.55% | 60.77% | 0.5240 | 0.0953 |
| **Wav2Vec2 Speech v1** *(matched valid `N=2,606`)* | 50.08% | 24.34% | 45.05% | 0.6579 | 0.0309 |
| **Naive Equal-Weight Fusion (T+A)** | 63.56% | 37.61% | 59.64% | 0.5221 | 0.0819 |
| **Candidate 4 Adaptive Fusion (T+A)** | **63.37%** | **41.93%** | **61.13%** | **0.5199** | **0.0687** |

- Statistical significance vs DistilBERT (`10,000` paired bootstrap resamples): Candidate 4 `Delta Macro-F1 = +0.0038` (`p = 0.0230`, 95% CI `[+0.0005, +0.0072]`), `Delta Weighted-F1 = +0.0036` (`p = 0.0256`), `Delta Anger F1 = +0.0190` (`p = 0.0146`).
- **Phase 16 Candidate 4 Test Confusion Matrix** (`rows = true`, `cols = pred` in canonical order):
  `[[141, 1, 4, 44, 89, 7, 59], [18, 10, 2, 0, 27, 5, 6], [7, 0, 3, 2, 23, 6, 9], [36, 0, 1, 202, 110, 8, 45], [46, 3, 7, 50, 1075, 27, 48], [23, 1, 4, 11, 107, 43, 19], [28, 1, 1, 19, 49, 3, 180]]`
- *(Note: Facial models and V2 speech were intentionally not run on MELD `test` to avoid expensive re-evaluation on `2,610` raw test videos and keep held-out `test` untouched after remediation.)*

---

## 2. Disjoint MELD `dev_val` Results (Post-P0+P1+P2 Selected Models, `N = 548`)

### 2.1 Overall & Calibration Metrics (`dev_val`: Text/Speech/Fusion `N = 548`, Facial `N = 544` face-detected)

| Modality / Configuration | Acc (%) | Macro-F1 (%) | Weighted-F1 (%) | ECE | NLL | Brier | Mean Conf | `<45%` Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text Only (DistilBERT)** | 58.58 | 37.23 | 54.50 | 0.0951 | 1.2588 | — | — | — |
| Speech V1 P0 (`raw`) | 46.35 | 24.50 | 41.07 | 0.0638 | 1.4995 | 0.6946 | 46.2% | 51.6% |
| Speech V1 + P1 (`L2 cal`) | 47.26 | 24.58 | 40.91 | 0.0602 | 1.4834 | 0.6895 | 48.7% | 49.5% |
| Speech V2 Uncalibrated *(ablation)* | 42.15 | **27.93** | 41.06 | 0.0883 | 1.5761 | 0.7205 | 50.6% | 40.5% |
| **Speech V2 + L2 Cal (`Selected`)** | **47.26** | **26.26** | **41.98** | **0.0537** | **1.4510** | **0.6791** | 49.7% | **40.1%** |
| Facial V1 P0 (`exp5_ce_weighted`) | 37.68 | 17.96 | 32.17 | 0.0934 | 1.6891 | 0.7838 | 29.6% | 97.8% |
| **Facial V2 (`Selected`, `raw_all_detections_mean_prob`)** | 34.38 | **21.52** | **33.31** | **0.0473** | 1.7242 | 0.7904 | **38.6%** | **76.7%** |
| Fusion `T + A` (`V2 L2`) | 58.76 | 37.63 | 54.95 | 0.0909 | — | — | — | — |
| Fusion `T + V` (`V2 Face`) | 58.94 | 37.41 | 54.76 | 0.0639 | — | — | — | — |
| **Fusion `T + A + V` (`Active Default`)** | **59.12** | **37.82** | **55.21** | **0.0646** | **1.2154** | — | — | — |

On **full MELD `dev` (`N = 1,108`)**: Text-only scored `58.48% Acc / 38.44% MF1 / 55.07% WF1 / 0.1076 ECE`, `T+A` scored `59.03% Acc / 39.05% MF1`, `T+V` scored `58.57% Acc / 38.43% MF1`, and **`T+A+V` reached `59.12% Acc / 39.04% MF1 / 55.79% WF1 / 0.0546 ECE`**.

### 2.2 Per-Class Precision, Recall & F1 on Disjoint `dev_val`

| Emotion | Text Only (`P / R / F1`) | Speech V2 L2 (`P / R / F1`) | Facial V2 (`P / R / F1`) | Fusion T+A+V (`P / R / F1`) |
| :--- | :---: | :---: | :---: | :---: |
| **Anger** (`75` / `74` face) | `49.1 / 36.0 / 41.5` | `46.4 / 52.0 / 49.1` | `27.8 / 20.3 / 23.4` | `50.0 / 41.3 / 45.3` |
| **Disgust** (`11`) | `0.0 / 0.0 / 0.0` | `0.0 / 0.0 / 0.0` | `0.0 / 0.0 / 0.0` | `0.0 / 0.0 / 0.0` |
| **Fear** (`20`) | `66.7 / 10.0 / 17.4` | `0.0 / 0.0 / 0.0` | `10.0 / 5.0 / 6.7` | `66.7 / 10.0 / 17.4` |
| **Joy** (`80` / `79` face) | `56.9 / 51.3 / 53.9` | `33.3 / 15.0 / 20.7` | `21.7 / 35.4 / 26.9` | `56.2 / 51.3 / 53.6` |
| **Neutral** (`228` / `227` face) | `65.4 / 83.8 / 73.5` | `50.7 / 78.9 / 61.7` | `47.2 / 52.0 / 49.5` | `66.3 / 83.8 / 74.0` |
| **Sadness** (`58`) | `50.0 / 10.3 / 17.1` | `34.6 / 15.5 / 21.4` | `41.7 / 17.2 / 24.4` | `50.0 / 10.3 / 17.1` |
| **Surprise** (`76` / `75` face) | `47.8 / 71.1 / 57.1` | `40.4 / 25.0 / 30.9` | `19.5 / 20.0 / 19.7` | `48.6 / 69.7 / 57.3` |

### 2.3 `dev_val` Confusion Matrices (`rows = true`, `cols = pred`: `[anger, disgust, fear, joy, neutral, sadness, surprise]`)
- **Speech V2 + L2 Calibration (`N=548`):** `[[39, 0, 0, 3, 29, 0, 4], [3, 0, 0, 0, 6, 0, 2], [5, 0, 0, 0, 12, 0, 3], [10, 0, 0, 12, 49, 5, 4], [16, 0, 0, 12, 180, 9, 11], [2, 0, 0, 4, 39, 9, 4], [9, 0, 0, 5, 40, 3, 19]]`
- **Facial V2 (`ResNet18_v2`, `N=544`):** `[[15, 0, 5, 14, 28, 5, 7], [1, 0, 0, 4, 4, 1, 1], [1, 0, 1, 6, 11, 1, 0], [5, 0, 0, 28, 34, 1, 11], [17, 0, 3, 51, 118, 2, 36], [7, 0, 1, 6, 27, 10, 7], [8, 0, 0, 20, 28, 4, 15]]`
- **Multimodal `T+A+V` (`N=548`):** `[[31, 0, 0, 7, 27, 1, 9], [3, 0, 0, 0, 6, 1, 1], [4, 0, 2, 3, 5, 0, 6], [8, 0, 1, 41, 17, 0, 13], [6, 0, 0, 10, 191, 3, 18], [4, 1, 0, 6, 32, 6, 9], [6, 0, 0, 6, 10, 1, 53]]`

---

## 3. Inference Latencies Measured on Target Hardware (RTX 4050 6GB CUDA)
- **Live FastAPI endpoint latencies** (`dia0_utt0` real sample, warmed CUDA): Text-only `36.2 ms` (`/predict/text`); Speech-only `29.4 ms` (`/predict/audio` warmed, `162.5 ms` first clip); Text+Speech `44.6 ms`; Facial video `~601–986 ms` (`/predict/video`, 8-frame MTCNN + ResNet18); Three-modality (`T+A+V`) `699.8–710.4 ms`.
- **Offline `dev_val` throughput:** `Wav2Vec2_v2` `18.33 ms/clip`.

