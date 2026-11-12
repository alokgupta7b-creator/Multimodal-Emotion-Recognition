# Phase 16 — Final Scientific Evaluation on Held-Out Test Split

## Executive Summary

Phase 16 executed the final, definitive benchmark of the Multimodal Emotion Recognition system on the previously untouched MELD test split ($N = 2,610$). All model checkpoints (DistilBERT, Wav2Vec2, and Facial CNN) and Candidate 4 fusion parameters ($\tau_a = 0.20, \lambda_{\max} = 0.40, \kappa_a = 1.50, \lambda_v = 0.00$, $M_a = p_a[\hat{y}_a] - p_a[\text{neutral}]$) were strictly frozen prior to evaluation. No model retraining, fine-tuning, threshold tuning, or hyperparameter adjustments were performed.

### Key Empirical Findings

1. **Macro F1 Improvement (Statistically Significant):** Candidate 4 achieved a Macro F1 of **0.4193**, surpassing the standalone DistilBERT text baseline of **0.4155** ($\Delta\text{Macro F1} = \mathbf{+0.0038}$, 95% paired bootstrap CI: `[+0.0005, +0.0072]`, $p = \mathbf{0.0230}$). Because the 95% CI strictly excludes zero, the Macro F1 gain is **statistically significant at $\alpha = 0.05$**.
2. **Reversal of Equal-Weight Fusion Collapse:** Naive equal-weight late fusion suffered a catastrophic Macro F1 collapse on the test split ($0.4155 \to \mathbf{0.3761}$, $\Delta = -0.0390$, 95% CI: `[-0.0624, -0.0174]`, $p = \mathbf{0.0002}$), completely wiping out Fear ($F_1 = 0.0000$) and halving Disgust ($F_1: 0.2381 \to 0.1096$). Candidate 4's class-conditional safeguard completely eliminated this collapse, preserving rare-class performance while unlocking acoustic synergy.
3. **Targeted Acoustic Synergy on Anger (Statistically Significant):** Speech prosody provided significant true positive gains on Anger ($F_1: 0.4187 \to \mathbf{0.4379}$, $\Delta = \mathbf{+0.0192}$, 95% CI: `[+0.0033, +0.0363]`, $p = \mathbf{0.0146}$), adding +11 net true positive Anger detections.
4. **Accuracy Gain (Descriptive, Not Statistically Significant):** Candidate 4 achieved an overall accuracy of **63.37%** (1,654 / 2,610) vs **63.10%** (1,647 / 2,610) for DistilBERT ($\Delta\text{Accuracy} = \mathbf{+0.26\%}$, +7 net true positives, 95% CI: `[-0.038%, +0.613%]`, $p = \mathbf{0.1324}$). While positive, this difference crosses zero and is therefore characterized as a descriptive improvement rather than a statistically significant one.
5. **Superior Probability Calibration:** Candidate 4 achieved the lowest multiclass Brier score (**0.5199** vs 0.5240 for text and 0.5221 for equal fusion) and reduced the 10-bin Expected Calibration Error (ECE) from **0.0953** to **0.0687** (a **27.9% relative calibration error reduction**).
6. **Real-Time Latency on GPU:** Evaluated on an NVIDIA GeForce RTX 4050 Laptop GPU, complete end-to-end multimodal inference across all 2,610 test samples averaged **25.57 ms per sample** (DistilBERT: 6.71 ms median, Wav2Vec2: 16.76 ms median, Fusion: 0.07 ms median).

---

## 1. Frozen Preflight Verification & Integrity Manifest

Prior to evaluating the test split, all component hashes and manifest counts were verified:

| Component | File Path | Size (Bytes) | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :---: |
| **DistilBERT Checkpoint** | `models/text_emotion/distilbert/best/model.safetensors` | 267,847,948 | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | Verified Frozen |
| **Wav2Vec2 Checkpoint** | `models/speech_emotion/wav2vec2/best/model.safetensors` | 378,307,452 | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` | Verified Frozen |
| **Facial CNN Checkpoint** | `models/facial_emotion/facial_cnn_best.pt` | 380,128 | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | Verified Frozen |
| **Candidate 4 Fusion** | `api/fusion.py` | 7,949 | `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d` | Verified Frozen |
| **Canonical Label Map** | `speech_emotion/model/label_mapping.py` | 4,501 | `f8c09b87c4124c4ce92583a60ac29fad5c83a1463d488c7f67b5fdba65fdad2a` | Verified Frozen |

### Test Split Cohort Audit
- **Annotation Source:** `datasets/MELD/annotations/test_sent_emo.csv` ($N = 2,610$ rows)
- **Audio Manifest:** `datasets/MELD_raw_acquisition/prepared/reports/test/audio_manifest.csv` ($N = 2,610$ rows)
- **Alignment:** 100% 1-to-1 match across `(Dialogue_ID, Utterance_ID)`.
- **Audio Availability on Disk:** 2,606 valid audio files ($99.85\%$), 2 silent audio files ($0.08\%$), 2 corrupt/unreadable files ($0.08\%$), 0 missing files.

### Test Split Ground Truth Distribution

```
Neutral   : 1,256 (48.12%)  ████████████████████████
Joy       :   402 (15.40%)  ████████
Anger     :   345 (13.22%)  ███████
Surprise  :   281 (10.77%)  █████
Sadness   :   208 ( 7.97%)  ████
Disgust   :    68 ( 2.61%)  █
Fear      :    50 ( 1.92%)  █
Total     : 2,610 (100.0%)
```

---

## 2. Comprehensive Test Split Metrics Comparison

The four evaluation strategies were evaluated on exactly the same test split cohort ($N = 2,610$):

| Evaluation Strategy | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Brier Score | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Text Baseline** | 63.10% | 0.4155 | 0.6077 | 0.4977 | 0.3998 | 0.5240 | 0.0953 |
| **Wav2Vec2 Speech Baseline** *(Matched Valid Cohort)* | 50.08% | 0.2434 | 0.4505 | 0.3444 | 0.3477 | 0.6579 | 0.0309 |
| **Naive Equal-Weight Fusion** | 63.56% | 0.3761 | 0.5964 | 0.5192 | 0.3641 | 0.5221 | 0.0819 |
| **Candidate 4 Adaptive Fusion** | **63.37%** | **0.4193** | **0.6113** | **0.5002** | **0.4031** | **0.5199** | **0.0687** |

---

## 3. Per-Class F1 Score Breakdown

The per-class F1 metric table demonstrates how Candidate 4 safeguards minority classes while capturing speech prosody:

| Emotion Class | Support | DistilBERT Text | Equal Fusion | Candidate 4 Fused | $\Delta(C_4 - \text{Text})$ | 95% Paired Bootstrap CI | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anger** | 345 | 0.4187 | 0.4633 | **0.4379** | **+0.0192** | `[+0.0033, +0.0363]` | **Yes ($p = 0.0146$)** |
| **Disgust** | 68 | 0.2381 | 0.1096 | **0.2381** | **0.0000** | `[ 0.0000,  0.0000]` | Safeguarded (Unchanged) |
| **Fear** | 50 | 0.0833 | 0.0000 | **0.0833** | **0.0000** | `[ 0.0000,  0.0000]` | Safeguarded (Unchanged) |
| **Joy** | 402 | 0.5534 | 0.5227 | **0.5534** | **0.0000** | `[ 0.0000,  0.0000]` | Safeguarded (Unchanged) |
| **Neutral** | 1,256 | 0.7850 | 0.7816 | **0.7858** | **+0.0008** | `[-0.0014, +0.0030]` | No ($p = 0.4646$) |
| **Sadness** | 208 | 0.2801 | 0.2038 | **0.2801** | **0.0000** | `[ 0.0000,  0.0000]` | Safeguarded (Unchanged) |
| **Surprise** | 281 | 0.5498 | 0.5521 | **0.5564** | **+0.0066** | `[-0.0045, +0.0168]` | No ($p = 0.2254$) |

> [!IMPORTANT]
> **Minority Class Preservation:** In equal-weight fusion, Fear collapsed from 0.0833 to **0.0000** (zero true positives), and Disgust collapsed from 0.2381 to **0.1096**. Candidate 4's text-anchored protected class lockout (`fear`, `disgust`, `sadness`, `joy`) completely preserved text performance without any auxiliary dilution.

---

## 4. Paired Bootstrap Statistical Significance (10,000 Iterations)

Statistical significance was evaluated using 10,000 paired bootstrap resamples (`seed = 42`) across the 2,610 test samples:

### Candidate 4 vs DistilBERT Text Baseline

```
Δ Macro F1    : +0.0038  (95% CI: [+0.0005, +0.0072], p = 0.0230)  --> STATISTICALLY SIGNIFICANT
Δ Weighted F1 : +0.0036  (95% CI: [+0.0004, +0.0071], p = 0.0256)  --> STATISTICALLY SIGNIFICANT
Δ Accuracy    : +0.0026  (95% CI: [-0.0004, +0.0061], p = 0.1324)  --> Descriptive gain (crosses zero)
Δ Anger F1    : +0.0192  (95% CI: [+0.0033, +0.0363], p = 0.0146)  --> STATISTICALLY SIGNIFICANT
Δ Surprise F1 : +0.0066  (95% CI: [-0.0045, +0.0168], p = 0.2254)  --> Descriptive gain
```

### Naive Equal-Weight Late Fusion vs DistilBERT Text Baseline

```
Δ Macro F1    : -0.0390  (95% CI: [-0.0624, -0.0174], p = 0.0002)  --> STATISTICALLY SIGNIFICANT DEGRADATION
Δ Disgust F1  : -0.1269  (95% CI: [-0.2341, -0.0342], p = 0.0046)  --> Severe collapse
Δ Sadness F1  : -0.0763  (95% CI: [-0.1277, -0.0288], p = 0.0014)  --> Severe collapse
Δ Joy F1      : -0.0310  (95% CI: [-0.0564, -0.0066], p = 0.0104)  --> Severe collapse
```

---

## 5. Failure & Availability Accounting

Out of 2,610 test utterances:

- **Effective Multimodal Cohort:** 2,606 samples (**99.85%** coverage).
- **Silent Audio Exclusions:** 2 samples (waveform RMS below $-60\text{ dB}$).
- **Corrupt Audio Exclusions:** 2 samples (failed decode/read).
- **Missing Audio Files:** 0 samples.
- **Handling:** All 4 invalid audio samples triggered the zero-synthetic fallback, where text was 100% authoritative ($w_{\text{text}} = 1.0, w_{\text{audio}} = 0.0$) with explicit exclusion logging.

### Candidate 4 Dynamic Rule Execution Breakdown ($N = 2,610$)

| Rule Identifier | Trigger Condition | Samples Triggered | Share (%) |
| :--- | :--- | :---: | :---: |
| `text_anchor_default` | Speech margin $M_a \le 0.20$ or text neutral | 1,720 | 65.90% |
| `acoustic_synergy_anger` | Audio predicts anger with $M_a > 0.20$ | 328 | 12.57% |
| `acoustic_synergy_surprise` | Audio predicts surprise with $M_a > 0.20$ | 97 | 3.72% |
| `protected_class_lockout_joy` | DistilBERT predicts joy | 328 | 12.57% |
| `protected_class_lockout_sadness` | DistilBERT predicts sadness | 99 | 3.79% |
| `protected_class_lockout_fear` | DistilBERT predicts fear | 22 | 0.84% |
| `protected_class_lockout_disgust` | DistilBERT predicts disgust | 16 | 0.61% |

---

## 6. Confusion Matrix Analysis & Visual Artifacts

The four confusion matrices are rendered and saved in [`models/phase16_final_evaluation/`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation):

1. [`confusion_matrix_text.png`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation/confusion_matrix_text.png) (335 KB) — DistilBERT Text Baseline Confusion Matrix.
2. [`confusion_matrix_candidate4.png`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation/confusion_matrix_candidate4.png) (337 KB) — Candidate 4 Adaptive Multimodal Fusion Confusion Matrix.
3. [`confusion_matrix_audio.png`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation/confusion_matrix_audio.png) (322 KB) — Wav2Vec2 Standalone Speech Confusion Matrix.
4. [`confusion_matrix_equal_fusion.png`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation/confusion_matrix_equal_fusion.png) (314 KB) — Naive Equal-Weight Late Fusion Confusion Matrix.

### True Positive Shifts (Candidate 4 vs DistilBERT):
- **Anger:** Correct predictions increased from **130** to **141** (+11 true positives).
- **Neutral:** Correct predictions changed from **1,077** to **1,075** (-2 false conversions).
- **Surprise:** Correct predictions changed from **182** to **180** (-2 true positives, but precision improved from 47.77% to 49.18%, boosting F1).
- **Fear, Disgust, Joy, Sadness:** True positive counts remained identical (3, 10, 202, 43 respectively), demonstrating the safeguard.

---

## 7. Inference Latency & Efficiency on GPU

Timing statistics measured across all 2,610 test samples on CUDA:

| Pipeline Stage | Mean | Median | p90 | p95 | p99 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Text Inference** | 7.06 ms | 6.71 ms | 9.37 ms | 10.19 ms | 12.71 ms |
| **Wav2Vec2 Audio Inference** | 18.42 ms | 16.76 ms | 27.62 ms | 32.22 ms | 43.28 ms |
| **Candidate 4 Fusion Math** | 0.086 ms | 0.074 ms | 0.111 ms | 0.128 ms | 0.225 ms |
| **Combined Multimodal Pipeline** | **25.57 ms** | **23.55 ms** | **37.10 ms** | **42.54 ms** | **56.22 ms** |

Total wall-clock runtime for all 2,610 test samples was **81.68 seconds** (~32 samples per second).

---

## 8. Honest Limitations & Scientific Conclusions

1. **Macro F1 vs Accuracy:** Candidate 4 achieved a **statistically significant Macro F1 improvement** ($p = 0.0230$), but its overall accuracy gain (+0.26%, $p = 0.1324$) did not reach significance at $\alpha = 0.05$. In dialogue emotion recognition where conversational neutral accounts for 48% of the data, Macro F1 is the clinically and scientifically relevant metric because accuracy is heavily dominated by conversational neutral.
2. **Acoustic Dominance on Anger:** The primary driver of multimodal synergy is vocal intensity in Anger. When speakers exhibit sharp acoustic activation, Wav2Vec2 reliably corrects text errors where sarcastic or ambiguous utterances are misclassified as Neutral or Joy.
3. **Facial Video Remains Unneeded:** Consistent with Phase 12B and Phase 13 findings, setting $\lambda_v = 0.00$ avoided ~2.2 seconds of video processing per sample while delivering superior accuracy and Macro F1 compared to tri-modal fusion.
4. **Reproducibility:** All per-sample predictions are persisted in [`models/phase16_final_evaluation/test_predictions.jsonl`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/phase16_final_evaluation/test_predictions.jsonl), allowing researchers to verify every decision.

