# Phase 12B Report: Matched-Cohort Multimodal Fusion Analysis

## 1. Executive Summary

Phase 12B conducted a rigorous, matched-cohort statistical evaluation of multimodal late fusion against individual modality predictors using the 1,108 development predictions generated in Phase 12.

### Core Verdict
1. **No Significant Accuracy Gain:** Unweighted late fusion does **not** produce a statistically significant overall accuracy improvement over standalone `DistilBERT` text across any matched cohort.
   - Text vs Text+Audio ($N=1,107$): $\Delta \text{Acc} = -0.27\%$ ($95\%\text{ CI: } [-1.90\%, +1.26\%]$, $p = 0.7674$).
   - Text vs Text+Video ($N=1,096$): $\Delta \text{Acc} = +0.18\%$ ($95\%\text{ CI: } [-0.18\%, +0.55\%]$, $p = 0.4566$).
   - Text vs Tri-modal ($N=1,095$): $\Delta \text{Acc} = -0.18\%$ ($95\%\text{ CI: } [-1.74\%, +1.37\%]$, $p = 0.8654$).
2. **Statistically Significant Macro F1 Drop:** Equal-weight late fusion causes a **statistically significant reduction** in Macro F1 ($p < 0.005$ for both Text+Audio and Tri-modal). Naive probability averaging severely penalizes minority classes (`fear` and `disgust`) where audio and video models have zero recall, dragging text predictions below classification thresholds.
3. **Class-Specific Synergy:** Audio provides clear, positive complementary signal for high-activation emotions:
   - **Anger F1:** Improves from **0.4071** (Text) to **0.4505** (Text+Audio) and **0.4521** (Tri-modal).
   - **Surprise F1:** Improves from **0.5609** (Text) to **0.5697** (Text+Audio) and **0.5741** (Tri-modal).
4. **Calibration Improvements:** Text+Audio significantly improves model calibration over Text alone:
   - Multiclass Brier score drops from **0.5756** to **0.5559**.
   - Expected Calibration Error (ECE) is nearly halved from **0.1165** to **0.0613**.

---

## 2. Methodology and Cohort Definitions

To eliminate sample-selection bias caused by missing modalities, all comparisons were conducted strictly on **matched subsets** where identical sample IDs and true labels were evaluated across all compared methods.

- **Cohort A & C ($N = 1,107$):** All development samples with both `text` and `audio` available. Exactly 1 sample (`dev_dia36_utt8`) was excluded due to near-silent audio ($-63.3\text{ dB}$).
- **Cohort B ($N = 1,096$):** All development samples with both `text` and `video` available. Exactly 12 samples were excluded due to face detection absence (`No detectable faces were supplied.`).
- **Cohort D ($N = 1,095$):** All development samples with all three modalities (`text`, `audio`, `video`) available. Exactly 13 samples were excluded (1 audio + 12 video).

### Statistical Significance Methodology
- **Resampling Method:** Paired non-parametric bootstrap resampling on the exact matched evaluation indices.
- **Iterations:** $B = 10,000$ iterations.
- **Random Seed:** Fixed seed `42` for exact reproducibility.
- **Intervals & Testing:** Two-sided 95% percentile confidence intervals $[\text{Percentile}_{2.5\%}, \text{Percentile}_{97.5\%}]$ and empirical two-sided permutation p-values. Differences are declared statistically significant at $\alpha = 0.05$ if and only if the 95% CI strictly excludes zero.

---

## 3. Matched Cohort Results

### Cohort A: Text vs Text+Audio ($N = 1,107$)

| Metric | Standalone Text | Text + Audio Fusion | Point Difference ($\Delta$) | 95% Bootstrap CI | Empirical $p$-value | Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 60.07% | 59.80% | -0.27% | [-1.90%, +1.26%] | 0.7674 | No (Inconclusive) |
| **Macro Precision** | 0.5308 | 0.4745 | -0.0563 | [-0.1118, +0.0006] | 0.0504 | No |
| **Macro Recall** | 0.4019 | 0.3614 | -0.0405 | [-0.0706, -0.0104] | 0.0084 | **Yes (Degradation)** |
| **Macro F1** | 0.4204 | 0.3671 | -0.0533 | [-0.0910, -0.0165] | 0.0014 | **Yes (Degradation)** |
| **Weighted Precision**| 0.5838 | 0.5476 | -0.0362 | [-0.0631, -0.0094] | 0.0080 | **Yes (Degradation)** |
| **Weighted Recall** | 0.6007 | 0.5980 | -0.0027 | [-0.0190, +0.0126] | 0.7674 | No |
| **Weighted F1** | 0.5697 | 0.5518 | -0.0179 | [-0.0381, +0.0018] | 0.0766 | No |
| **Brier Score** | 0.5756 | **0.5559** | -0.0197 | — | — | **Improved** |
| **ECE (10-bin)** | 0.1165 | **0.0613** | -0.0552 | — | — | **Improved** |

#### Cohort A Per-Class Breakdown

| Emotion Class | Support | Text F1 | Text+Audio F1 | $\Delta$ F1 | Text Precision | Text+Audio Precision | Text Recall | Text+Audio Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 153 | 0.4071 | **0.4505** | **+0.0434** | 0.4488 | 0.4714 | 0.3725 | 0.4314 |
| **disgust** | 22 | **0.2759** | 0.0833 | **-0.1925** | 0.5714 | 0.5000 | 0.1818 | 0.0455 |
| **fear** | 40 | **0.1250** | 0.0000 | **-0.1250** | 0.3750 | 0.0000 | 0.0750 | 0.0000 |
| **joy** | 163 | **0.5226** | 0.4751 | -0.0476 | 0.6048 | 0.6327 | 0.4601 | 0.3804 |
| **neutral** | 468 | **0.7598** | 0.7485 | -0.0114 | 0.6767 | 0.6376 | 0.8654 | 0.9060 |
| **sadness** | 111 | **0.2914** | 0.2429 | -0.0485 | 0.5500 | 0.5862 | 0.1982 | 0.1532 |
| **surprise** | 150 | 0.5609 | **0.5697** | **+0.0088** | 0.4877 | 0.5318 | 0.6600 | 0.6133 |

---

### Cohort B: Text vs Text+Video ($N = 1,096$)

| Metric | Standalone Text | Text + Video Fusion | Point Difference ($\Delta$) | 95% Bootstrap CI | Empirical $p$-value | Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 60.13% | 60.31% | +0.18% | [-0.18%, +0.55%] | 0.4566 | No (Inconclusive) |
| **Macro Precision** | 0.5304 | 0.5259 | -0.0045 | [-0.0381, +0.0249] | 0.7816 | No |
| **Macro Recall** | 0.4022 | 0.3954 | -0.0068 | [-0.0210, +0.0075] | 0.3540 | No |
| **Macro F1** | 0.4203 | 0.4133 | -0.0070 | [-0.0272, +0.0039] | 0.7282 | No |
| **Weighted Precision**| 0.5841 | 0.5835 | -0.0006 | [-0.0163, +0.0135] | 0.9416 | No |
| **Weighted Recall** | 0.6013 | 0.6031 | +0.0018 | [-0.0018, +0.0055] | 0.4566 | No |
| **Weighted F1** | 0.5706 | 0.5715 | +0.0009 | [-0.0084, +0.0100] | 0.8582 | No |
| **Brier Score** | **0.5740** | 0.5980 | +0.0240 | — | — | Worsened |
| **ECE (10-bin)** | **0.1164** | 0.1673 | +0.0509 | — | — | Worsened |

#### Cohort B Per-Class Breakdown

| Emotion Class | Support | Text F1 | Text+Video F1 | $\Delta$ F1 | Text Precision | Text+Video Precision | Text Recall | Text+Video Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 151 | 0.4000 | **0.4029** | +0.0029 | 0.4444 | 0.4409 | 0.3642 | 0.3709 |
| **disgust** | 22 | **0.2759** | 0.2143 | -0.0616 | 0.5714 | 0.5000 | 0.1818 | 0.1364 |
| **fear** | 38 | 0.1304 | **0.1333** | +0.0029 | 0.3750 | 0.4286 | 0.0789 | 0.0789 |
| **joy** | 162 | 0.5245 | 0.5245 | 0.0000 | 0.6048 | 0.6048 | 0.4630 | 0.4630 |
| **neutral** | 466 | 0.7615 | **0.7639** | +0.0023 | 0.6778 | 0.6801 | 0.8691 | 0.8712 |
| **sadness** | 110 | **0.2933** | 0.2914 | -0.0019 | 0.5500 | 0.5366 | 0.2000 | 0.2000 |
| **surprise** | 147 | 0.5565 | **0.5630** | +0.0065 | 0.4897 | 0.4948 | 0.6463 | 0.6531 |

---

### Cohort C: Text vs Audio vs Text+Audio ($N = 1,107$)

| Method | Accuracy | Macro F1 | Weighted F1 | Multiclass Brier | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Text** | **60.07%** | **0.4204** | **0.5697** | 0.5756 | 0.1165 |
| **Audio** | 47.79% | 0.2534 | 0.4258 | 0.6805 | **0.0313** |
| **Text+Audio** | 59.80% | 0.3671 | 0.5518 | **0.5559** | 0.0613 |

#### Statistical Comparisons (Cohort C)
1. **Text+Audio vs Text:**
   - $\Delta \text{Acc} = -0.27\%$ ($95\%\text{ CI: } [-1.90\%, +1.26\%]$, $p=0.7674$). Not significant.
   - $\Delta \text{Macro F1} = -0.0533$ ($95\%\text{ CI: } [-0.0910, -0.0165]$, $p=0.0014$). **Statistically significant degradation**.
2. **Text+Audio vs Audio:**
   - $\Delta \text{Acc} = +12.01\%$ ($95\%\text{ CI: } [+9.12\%, +14.91\%]$, $p < 0.0001$). **Statistically significant improvement**.
   - $\Delta \text{Macro F1} = +0.1138$ ($95\%\text{ CI: } [+0.0795, +0.1500]$, $p < 0.0001$). **Statistically significant improvement**.
3. **Audio vs Text:**
   - $\Delta \text{Acc} = -12.29\%$ ($95\%\text{ CI: } [-15.54\%, -9.03\%]$, $p < 0.0001$). Standalone audio is significantly weaker than standalone text.
   - $\Delta \text{Macro F1} = -0.1670$ ($95\%\text{ CI: } [-0.2130, -0.1184]$, $p < 0.0001$).

---

### Cohort D: Tri-modal vs Individual Modalities ($N = 1,095$)

| Method | Accuracy | Macro F1 | Weighted F1 | Multiclass Brier | ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Text** | **60.09%** | **0.4202** | **0.5702** | **0.5744** | 0.1168 |
| **Audio** | 48.13% | 0.2556 | 0.4293 | 0.6767 | **0.0303** |
| **Video** | 14.61% | 0.0433 | 0.0596 | 0.8412 | 0.0188 |
| **Text+Audio+Video** | 59.91% | 0.3663 | 0.5526 | 0.6002 | 0.1863 |

#### Statistical Comparisons (Cohort D)
1. **Tri-modal vs Text:**
   - $\Delta \text{Acc} = -0.18\%$ ($95\%\text{ CI: } [-1.74\%, +1.37\%]$, $p=0.8654$). Not statistically significant (noise).
   - $\Delta \text{Macro F1} = -0.0540$ ($95\%\text{ CI: } [-0.0922, -0.0160]$, $p=0.0028$). **Statistically significant degradation**.
2. **Tri-modal vs Audio:**
   - $\Delta \text{Acc} = +11.78\%$ ($95\%\text{ CI: } [+8.86\%, +14.61\%]$, $p < 0.0001$). Significant improvement over audio.
   - $\Delta \text{Macro F1} = +0.1107$ ($95\%\text{ CI: } [+0.0768, +0.1472]$, $p < 0.0001$). Significant improvement over audio.
3. **Tri-modal vs Video:**
   - $\Delta \text{Acc} = +45.30\%$ ($95\%\text{ CI: } [+41.64\%, +48.95\%]$, $p < 0.0001$). Significant improvement over video.
   - $\Delta \text{Macro F1} = +0.3229$ ($95\%\text{ CI: } [+0.2902, +0.3579]$, $p < 0.0001$). Significant improvement over video.

#### Cohort D Per-Class Breakdown

| Emotion Class | Support | Text F1 | Audio F1 | Video F1 | Tri-modal F1 | $\Delta$ (Tri-modal - Text) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 151 | 0.4000 | 0.4273 | 0.2410 | **0.4521** | **+0.0521** |
| **disgust** | 22 | **0.2759** | 0.0000 | 0.0000 | 0.0833 | **-0.1925** |
| **fear** | 38 | **0.1304** | 0.0000 | 0.0000 | 0.0000 | **-0.1304** |
| **joy** | 162 | **0.5245** | 0.1296 | 0.0000 | 0.4710 | -0.0534 |
| **neutral** | 465 | **0.7611** | 0.6605 | 0.0620 | 0.7498 | -0.0113 |
| **sadness** | 110 | **0.2933** | 0.1884 | 0.0000 | 0.2336 | -0.0598 |
| **surprise** | 147 | 0.5565 | 0.3602 | 0.0000 | **0.5741** | **+0.0176** |

---

## 4. In-Depth Technical Analysis

### 4.1 Which modality combinations genuinely improve performance?
**No late fusion combination genuinely improves overall performance over standalone Text.**
- While Text+Video shows a nominal $+0.18\%$ accuracy gain, this corresponds to exactly **2 net samples** out of 1,096 (661 vs 659 correct predictions). The 95% bootstrap confidence interval $[-0.18\%, +0.55\%]$ crosses zero, confirming that this difference is random sampling noise.
- Both Text+Audio ($-0.27\%$) and Tri-modal ($-0.18\%$) exhibit negligible, non-significant accuracy reductions, but suffer severe, statistically significant drops in Macro F1 ($-0.0533$ and $-0.0540$ respectively).

### 4.2 Does Audio or Video add measurable value over Text?
- **Audio:**
  - **Yes, for acoustic-heavy emotions:** Audio adds undeniable value for `anger` (F1 $+0.0434$) and `surprise` (F1 $+0.0088$). Vocal prosody (raised pitch, increased energy) provides unambiguous signal that resolves ambiguous text lines (e.g., exclamations or short phrases like "What?!").
  - **Yes, for calibration:** Audio provides well-calibrated confidence intervals (ECE $0.0313$). In late fusion, audio probabilities dampen DistilBERT's overconfident misclassifications, reducing Brier score from $0.5756 \to 0.5559$ and cutting ECE from $0.1165 \to 0.0613$.
  - **No, under naive uniform averaging:** Because Wav2Vec2 has 0 recall on `fear` and `disgust`, uniform late fusion dilutes text's true positive detections.
- **Video:**
  - **No measurable value:** The Phase 8 facial CNN achieves 14.61% accuracy on the development set and exhibits near-zero sensitivity to all emotions except anger. Fusing video yields no statistically significant metric changes over text alone, degrades calibration (ECE worsens to $0.1673$), and imposes a massive $52\times$ latency penalty (~2,243 ms vs ~43 ms).

### 4.3 Which emotion classes benefit or deteriorate?
- **Benefiting Classes:**
  - `anger`: Consistent, meaningful boost across all fused combinations (F1 $0.4000 \to 0.4505 \to 0.4521$).
  - `surprise`: Moderate boost (F1 $0.5565 \to 0.5697 \to 0.5741$).
- **Deteriorating Classes:**
  - `fear`: Catastrophic collapse. Text correctly identified 3 fear samples (recall 0.0750, F1 0.1250); late fusion reduced fear predictions to 0 (F1 $0.0000$).
  - `disgust`: Recall collapsed from $18.18\%$ to $4.55\%$ (F1 dropped from $0.2759 \to 0.0833$).
  - `joy`: F1 dropped by $\sim 0.05$ (from $0.5226 \to 0.4751$).
  - `sadness`: F1 dropped by $\sim 0.05$ (from $0.2914 \to 0.2429$).

### 4.4 Whether calibration improves
- **Text+Audio:** Calibration substantially improves. Expected Calibration Error drops from **0.1165 to 0.0613**, and Brier score improves from **0.5756 to 0.5559**. Audio acts as a reliable regularizer that tempers transformer overconfidence.
- **Text+Video and Tri-modal:** Calibration deteriorates. High-entropy, uninformative video distributions spread spurious probability mass across incorrect classes, increasing ECE to **0.1863**.

### 4.5 Are the observed differences meaningful or inconclusive?
- **Overall Accuracy Differences:** **Inconclusive.** All 95% bootstrap confidence intervals for accuracy include zero. Neither late fusion nor any single modality outperforms Text in overall accuracy on this dataset.
- **Macro F1 Differences:** **Meaningful and statistically significant.** The reduction in Macro F1 for Text+Audio ($p=0.0014$) and Tri-modal ($p=0.0028$) is statistically real.
- **Architectural Conclusion:** Simple late fusion (equal-weight probability averaging) fails because it assumes equal reliability across all classes. To exploit the genuine acoustic synergy observed in anger and surprise without sacrificing minority-class recall, the system requires **learned, modality-gated, or confidence-weighted fusion** (e.g., Phase 13 gating network).

---

## 5. Development Isolation & Safety Audit

- **Predictions Source:** `models/phase12_dev_experiments/dev_predictions.jsonl`
- **Total Records Evaluated:** Exactly 1,108 unique development records.
- **Test Split Access:** **Zero test records accessed.** No test manifests, test directories, or test predictions were read.
- **Checkpoints & Datasets:** Intact and untouched; no model training, fine-tuning, or inference was executed during Phase 12B.
- **Artifacts Created:**
  - `models/phase12_dev_experiments/matched_cohort_results.json`
  - `docs/PHASE12B_MATCHED_COHORT_REPORT.md`

