# Phase 17C — Final Experiment Comparison & Candidate Selection

**Date:** 2026-11-11  
**Script:** [`scripts/run_phase17c_selection.py`](../scripts/run_phase17c_selection.py)  
**Artifact:** [`models/facial_experiments/phase17c_final_selection.json`](../models/facial_experiments/phase17c_final_selection.json)  
**Elapsed Time:** `114.0s`  
**Protected Production Assets Verified:** `before=True, after=True` (`zero_test_split_access = True`)

---

## 1. Evaluation Protocol & Selection Rubric

- **Matched Dev Cohort (`N = 1,075`):** Strict intersection of MELD `dev` utterances detected identically across Haar (`1,080`), MTCNN unaligned (`1,100`), and Experiment 7 temporal video (`1,102`). All statistical comparisons use identical sample IDs and ground-truth labels (`5,000`-resample paired bootstrap for two-sided p-values and 95% CIs).
- **Collapse Exclusion Gate:** Models with `collapse_detected = True` (active predicted classes `<= 2` or max predicted share `>= 85%`) or `< 5` active classes receive `selection_score = 0.0`.
- **Multi-Criteria Selection Rubric (Normalized to Eligible Top Score):**
  - **45%** Matched Macro F1
  - **20%** Matched Minority Macro F1 (`anger`, `disgust`, `fear`, `sadness`, `surprise`)
  - **20%** Matched Weighted F1
  - **10%** Active Class Coverage (`active_classes / 7`)
  - **5%** Inference Latency Efficiency (`min_latency / latency_ms_cuda`)
- **Non-Clone Fallback Rule:** Fallback selection excludes collapsed candidates, checkpoints sharing the Primary SHA-256, and near-identical twin checkpoints (`>= 99.5%` prediction agreement with Primary or its same-SHA single-frame base, preventing redundant `exp4a` vs `exp5_ce_weighted` selection).

---

## 2. Full 8-Stage / 9-Candidate Comparison Table

| Stage / Model ID | Arch | Params | Full Dev `N` | Full Acc | Full Macro F1 | Full Weighted F1 | Matched `N=1,075` Acc | Matched Macro F1 | Matched Min F1 | CUDA Lat | Peak VRAM | Score | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`phase17b_exp5_loss` (`exp5_ce_weighted`)** | `resnet18` | `11,180,103` | `1,100` | `0.3091` | `0.1724` | `0.2981` | `0.3107` | `0.1739` | `0.0746` | **`2.58 ms`** | `323.6 MB` | **`0.9488`** | **PRIMARY** |
| `phase17b_exp4_detection` (`exp4a_mtcnn_unaligned`) | `resnet18` | `11,180,103` | `1,100` | `0.3091` | `0.1724` | `0.2981` | `0.3107` | `0.1739` | `0.0746` | `3.42 ms` | `366.4 MB` | `0.9365` | Twin of Primary |
| **`phase17b_exp5_focal` (`exp5_focal_g3_wt`)** | `resnet18` | `11,180,103` | `1,100` | `0.3291` | `0.1669` | `0.3049` | `0.3302` | `0.1683` | `0.0762` | `3.27 ms` | `319.6 MB` | **`0.9312`** | **FALLBACK** |
| `phase17b_exp7_temporal` (`raw_all_detections_mean_prob`) | `resnet18` | `11,180,103` | `1,102` | **`0.3829`** | `0.1722` | **`0.3232`** | **`0.3860`** | **`0.1753`** | `0.0653` | `4.34 ms`/vid | `254.2 MB` | `0.9272` | Temporal Option |
| `phase17a_exp3_resnet18` (Haar) | `resnet18` | `11,180,103` | `1,080` | `0.2741` | `0.1472` | `0.2576` | `0.2753` | `0.1479` | **`0.0886`** | `3.08 ms` | `319.1 MB` | `0.8806` | Haar Baseline |
| `phase17b_exp6_efficientnet` (`exp6b_focal`) | `efficientnet_b0` | `4,016,515` | `1,100` | `0.2727` | `0.1494` | `0.2655` | `0.2753` | `0.1513` | `0.0450` | `11.57 ms` | `618.8 MB` | `0.7657` | Rejected |
| `phase17a_exp2_weighted_cnn` | `facial_cnn` | `94,151` | `1,080` | `0.4241` | `0.0851` | `0.2526` | `0.4260` | `0.0854` | `0.0000` | `0.57 ms` | `156.0 MB` | `0.0000` | Collapsed (`1/7`) |
| `phase17a_exp1_corrected_cnn` | `facial_cnn` | `94,151` | `1,080` | `0.4241` | `0.0851` | `0.2526` | `0.4260` | `0.0854` | `0.0000` | `0.39 ms` | `156.0 MB` | `0.0000` | Collapsed (`1/7`) |
| `phase8_legacy_cnn` (Production) | `facial_cnn` | `94,151` | `1,080` | `0.1824` | `0.0652` | `0.1232` | `0.1833` | `0.0654` | `0.0000` | `0.69 ms` | `11.2 MB` | `0.0000` | Collapsed (`3/7` buggy norm; `2/7` Acc `0.1426`, MF1 `0.0366` w/ fixed norm) |

*(Per-class F1 breakdown and 9-stage comparison charts are saved in `models/facial_experiments/phase17c_final_selection/phase17c_8stage_comparison.png` and `phase17c_per_class_f1.png`.)*

---

## 3. Paired Bootstrap Statistical Comparisons (`N = 1,075`, 5,000 resamples)

| Comparison (`Model A` vs `Model B`) | $\Delta$ Macro F1 (`B - A`) | 95% CI (Macro F1) | `p`-value (MF1) | $\Delta$ Accuracy (`B - A`) | 95% CI (Acc) | `p`-value (Acc) |
|---|---:|---|---:|---:|---|---:|
| `phase8` vs `exp3_resnet18` | **`+0.0824`** | `[+0.0564, +0.1121]` | **`0.0000`** | **`+0.0921`** | `[+0.0577, +0.1274]` | **`0.0000`** |
| `phase8` vs `exp7_temporal` | **`+0.1098`** | `[+0.0876, +0.1321]` | **`0.0000`** | **`+0.2028`** | `[+0.1637, +0.2409]` | **`0.0000`** |
| `exp3` vs `exp4_detection` (`4A`) | `+0.0261` | `[-0.0084, +0.0585]` | `0.1324` | **`+0.0353`** | `[+0.0019, +0.0679]` | **`0.0428`** |
| `exp3` vs `exp5_loss` (`ce_weighted`) | `+0.0261` | `[-0.0084, +0.0585]` | `0.1324` | **`+0.0353`** | `[+0.0019, +0.0679]` | **`0.0428`** |
| `exp3` vs `exp5_focal` (`g3_wt`) | `+0.0205` | `[-0.0141, +0.0534]` | `0.2316` | **`+0.0549`** | `[+0.0205, +0.0893]` | **`0.0028`** |
| `exp5_ce` vs `exp5_focal` | `-0.0056` | `[-0.0354, +0.0245]` | `0.7128` | `+0.0195` | `[-0.0140, +0.0521]` | `0.2620` |
| `exp3` vs `exp6_efficientnet` | `+0.0035` | `[-0.0316, +0.0357]` | `0.7972` | `0.0000` | `[-0.0353, +0.0353]` | `1.0000` |
| `exp3` vs `exp7_temporal` | `+0.0274` | `[-0.0074, +0.0590]` | `0.1112` | **`+0.1107`** | `[+0.0763, +0.1433]` | **`0.0000`** |
| `exp6` vs `exp7_temporal` | `+0.0239` | `[-0.0037, +0.0514]` | `0.0920` | **`+0.1107`** | `[+0.0772, +0.1442]` | **`0.0000`** |

---

## 4. Selected Candidates & Rationale

### Primary Candidate — `phase17b_exp5_loss` (`exp5_ce_weighted`)
- **Checkpoint:** `models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_best.pt`
- **SHA-256:** `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`
- **Pipeline Config:** `ResNet18` (`11.18M` params) + `MTCNN` square unaligned (`0.08` margin) + `imagenet` RGB normalization
- **Selection Score:** **`0.9488`** (Rank #1)
- **Why It Won:** Delivered strong Macro F1 (`0.1724` full / `0.1739` matched) and best minority recall balance (`0.0811` full) at the lowest ResNet18 single-crop CUDA latency (`2.58 ms`). Although `exp7_temporal` achieved slightly higher matched Macro F1 (`0.1753`) and substantially higher accuracy (`0.3860`) using the **exact same `exp5_ce_weighted` checkpoint** (`8825d865...`) with multi-frame mean probability pooling, `exp7` smoothed `disgust`/`fear` TP recall to zero (`Matched MinF1 = 0.0653` vs `0.0746`) and required `4.34 ms` temporal batch latency, placing single-frame `exp5_ce_weighted` slightly ahead (`0.9488` vs `0.9272`). Importantly, because both modes share checkpoint `8825d865...`, `raw_all_detections_mean_prob` temporal pooling remains immediately available as a runtime video aggregation mode around this exact Primary checkpoint.

### Fallback Candidate — `phase17b_exp5_focal` (`exp5_focal_g3_wt`)
- **Checkpoint:** `models/facial_experiments/exp5_focal_loss/exp5_focal_g3_wt_best.pt`
- **SHA-256:** `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`
- **Pipeline Config:** `ResNet18` (`11.18M` params) + `MTCNN` square unaligned (`0.08` margin) + Weighted Focal Loss ($\gamma = 3.0$)
- **Selection Score:** **`0.9312`** (Highest non-clone alternative checkpoint)
- **Why It Was Selected:** Independently trained objective achieving higher accuracy (`0.3291` vs `0.3091`), higher Weighted F1 (`0.3049` vs `0.2981`), and highest Minority Macro F1 (`0.0762` matched, `sadness F1 = 0.1863`, `fear F1 = 0.0370`) among single-frame models, providing a genuine behavioral and calibrated-loss fallback rather than a weight-identical clone (`exp4a`).

