# Phase 17B — Facial Emotion Recognition Experiment Report (Experiments 4–7)

**Date:** 2026-11-11  
**Status:** Completed and Validated (Isolated Experimental Artifacts Only — Not Deployed to Production)  
**Machine-Readable Summary:** `models/facial_experiments/phase17b_results.json`

---

## 1. Environment, Safety & Baseline Reference

### Hardware & Software Configuration
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **OS / Runtime:** Windows 11 (`Windows-11-10.0.26300-SP0`), Python `3.13.9`
- **Libraries:** PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, OpenCV `4.13.0`, `facenet-pytorch` (MTCNN bundled offline), `scikit-learn` `1.7.2`, NumPy `2.3.5`
- **Random Seed:** `42` (`seed_everything(42)`)
- **Canonical 7-Class Order (0–6):** `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

### Strict Isolation & Safety Compliance
- **Zero Test-Split Access:** The held-out MELD `test` split (used once in Phase 16) was completely blocked via path-guard assertions (`assert_no_test_path`). All feature extraction, training, hyperparameter comparisons, and temporal pooling decisions used MELD `train` and `dev` only.
- **Protected Production Assets Untouched:** Live production predictor [`facial_emotion/model.py`](../facial_emotion/model.py), [`facial_emotion/detector.py`](../facial_emotion/detector.py), [`facial_emotion/remediation.py`](../facial_emotion/remediation.py), production CNN checkpoint (`models/facial_emotion/facial_cnn_best.pt`, SHA-256 `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760`), Phase 17A checkpoints (`exp1`, `exp2`, `exp3`), `text_emotion/`, `speech_emotion/`, and [`api/fusion.py`](../api/fusion.py) (`ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d`) were SHA-256 verified before and after every experiment (`before=True, after=True`).
- **Strict Preprocessing Separation:** Experimental ResNet18/EfficientNet-B0 models use `imagenet` preprocessing (`BGR -> RGB`, `96x96`, scale `1/255.0`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`), isolated from legacy CNN `[-1, 1]` normalization.

### Phase 17A Reference Baseline
- **Phase 17A Exp 3 ResNet18 (Haar `dev N=1,080`):** Accuracy `0.2741` (`27.41%`), Macro F1 `0.1472`, Weighted F1 `0.2576`, `7/7` active predicted classes, `11,180,103` parameters, SHA-256 `106b524e837417373dd904518da3ad389287c843d93c7f92596c5b1daf2b1446`.

---

## 2. Experiment 4 — Improved Face Detection & Landmark Alignment

**Directory:** `models/facial_experiments/exp4_face_detection/`  
**Script:** [`scripts/run_exp4_face_detection.py`](../scripts/run_exp4_face_detection.py)  
**Total Elapsed Time:** `2,718.1s`

### 2.1 Multi-Detector Benchmark (`834` frames / `140` stratified MELD `dev` videos)

| Detector | Frames Detected | Detection Rate | Videos w/ Face | Total Boxes | Latency (ms/frame) | Peak GPU VRAM | Peak CPU Traced |
|---|---:|---:|---:|---:|---:|---:|---:|
| `haar_native_720p` | `738 / 834` | `88.49%` | `139 / 140` | `1,481` | `84.82 ms` | `0.0 MB` | `1.70 MB` |
| `haar_downscaled_360p` | `651 / 834` | `78.06%` | `131 / 140` | `998` | `28.39 ms` | `0.0 MB` | `1.48 MB` |
| `mtcnn_native_720p` | `799 / 834` | `95.80%` | `140 / 140` | `1,553` | `269.22 ms` | `141.6 MB` | `6.76 MB` |
| `mtcnn_downscaled_360p` | `794 / 834` | `95.20%` | `140 / 140` | `1,399` | `164.83 ms` | `114.4 MB` | `3.27 MB` |

- **Missed & Unmatched Detections:** Haar missed `69` frames where MTCNN confidence was `>= 0.95` (vs `9` frames where Haar hit and MTCNN missed). `510 / 1,481` (`34.44%`) of native Haar boxes had `IoU < 0.15` against all MTCNN boxes on the same frame (candidate background false-positive rate).

### 2.2 Stratified Detection Recall (`haar_native_720p` vs `mtcnn_downscaled_360p`)

| Condition | Sample Count | Haar Native Hit Rate | MTCNN Downscaled Hit Rate |
|---|---:|---:|---:|
| Small faces (`< 64 px`) | `79` | `39.24%` (`31`) | **`84.81%`** (`67`) |
| Medium faces (`64–128 px`) | `321` | `63.86%` (`205`) | **`97.82%`** (`314`) |
| Large faces (`>= 128 px`) | `399` | `86.22%` (`344`) | **`98.99%`** (`395`) |
| Dark luma (`< 75`) | `683` | `88.43%` (`604`) | **`94.88%`** (`648`) |
| Normal luma (`75–155`) | `146` | `91.78%` (`134`) | **`100.00%`** (`146`) |
| Bright luma (`>= 155`) | `5` | `0.00%` (`0`) | `0.00%` (`0`) |
| Frontal pose (landmarks) | `356` | `90.73%` (`323`) | **`97.75%`** (`348`) |
| Angled / tilted pose | `443` | `58.01%` (`257`) | **`96.61%`** (`428`) |

### 2.3 Crop Quality & Landmark Alignment Geometry (`n = 590` matched frames)
- **Sharpness (Laplacian variance):** Haar native `148.03`, MTCNN square unaligned `248.44`, MTCNN 5-pt aligned `428.87`.
- **Brightness / Contrast:** Haar `71.66 / 32.07`, MTCNN unaligned `66.37 / 34.26`, MTCNN aligned `66.04 / 34.92`.
- **Alignment Precision:** Eye-y standard deviation dropped from `2.18 px` to `3.29e-6 px` across `mean_abs_roll_corrected_deg = 6.05°` (geometrically horizontal eye normalization verified).

### 2.4 Full Split Cache Extraction (`train` / `dev` strictly disjoint)
- **Train Cache (`cache/train_mtcnn_crops.npz`):** `9,937 / 9,988` videos detected (`99.49%` coverage, `51` excluded no-face clips, `74` native fallback hits, `1,808.9s`).
- **Dev Cache (`cache/dev_mtcnn_crops.npz`):** `1,100 / 1,108` videos detected (`99.28%` coverage, `8` excluded, `11` native fallback hits, `254.4s`).
- **Matched Haar ∩ MTCNN Dev Cohort:** `1,075` shared utterance IDs (`train ∩ dev overlap = 0`).

### 2.5 ResNet18 Classification Results

| Model / Protocol | Cohort `N` | Accuracy | Macro F1 | Weighted F1 | Min Class Recall | Active Classes | Train Time | Peak VRAM | CUDA / CPU Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp 3 ResNet18 (Haar ref) | `1,080` (`1,075`) | `0.2741` (`0.2753`) | `0.1472` (`0.1479`) | `0.2576` (`0.2593`) | `0.1196` | `7/7` | — | `319.1 MB` | `5.82 ms` / — |
| Exp 3 Zero-Shot on MTCNN Unaligned | `1,100` (`1,075`) | `0.2282` (`0.2270`) | `0.1371` (`0.1369`) | `0.2249` (`0.2233`) | `0.1879` | `7/7` | `0s` | — | — |
| Exp 3 Zero-Shot on MTCNN Aligned | `1,100` (`1,075`) | `0.2264` (`0.2270`) | `0.1223` (`0.1224`) | `0.2210` (`0.2218`) | `0.1598` | `7/7` | `0s` | — | — |
| **`Exp 4A`: ResNet18 + MTCNN Unaligned (Ep 5)** | `1,100` (`1,075`) | **`0.3091`** (`0.3107`) | **`0.1724`** (**`0.1739`**) | **`0.2979`** (`0.2997`) | `0.0811` | `7/7` | `63.3s` | `366.4 MB` | `3.42 ms` / `18.04 ms` |
| `Exp 4B`: ResNet18 + MTCNN Aligned (Ep 7) | `1,100` (`1,075`) | `0.3264` | `0.1586` (`0.1606`) | `0.2951` (`0.2977`) | `0.0265` | `7/7` | `62.3s` | `366.1 MB` | `3.27 ms` / `18.34 ms` |

- **Experiment 4 Checkpoint Hashes & Validation Gate:**
  - Selected Winner: `exp4a_resnet18_mtcnn_unaligned_best.pt` (SHA-256 `78aa0ebb53c7ea270284e3c3fe772a1363378c81147599201aced0b21a99f40c`).
  - Alternative Evaluated: `exp4b_resnet18_mtcnn_aligned_best.pt` (SHA-256 `783c4f26bbd796be1199db8d7a8b971f77f68c045d6330daf6d3e237db6c0aba`).
  - **Finding:** Zero-shot transfer of a Haar-trained ResNet18 onto tighter MTCNN/aligned crops hurt Macro F1 (`0.1472 -> 0.1371 / 0.1223`), proving retraining on the matched crop distribution is necessary. Re-trained **`Exp 4A` (MTCNN unaligned square margin 0.08)** improved Dev Macro F1 `+0.0252` (`+0.0261` matched) over Exp 3 while preserving peripheral context (`chin/jaw/head`) that aggressive 5-pt tightening in `Exp 4B` cropped away (`0.1724` vs `0.1586`).

---

## 3. Experiment 5 — Focal Loss & Class Imbalance Study

**Directory:** `models/facial_experiments/exp5_focal_loss/`  
**Script:** [`scripts/run_exp5_focal_loss.py`](../scripts/run_exp5_focal_loss.py)  
**Total Elapsed Time:** `343.4s`

All six runs used identical Experiment 4 `exp4a_mtcnn_unaligned` splits (`train=9,937`, `dev=1,100`), Seed `42`, and `ResNet18`.  
**Smoothed Square-Root Class Weights ($\alpha_c$):** `anger=0.8173`, `disgust=1.6518`, `fear=1.6611`, `joy=0.6518`, `neutral=0.3969`, `sadness=1.0386`, `surprise=0.7826`.

| Run ID | Loss Config | Best Ep | Acc | Macro F1 | Weighted F1 | Min Rec | Min F1 | Active | Time | VRAM | CUDA / CPU Lat | SHA-256 (prefix) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `exp5_ce_unweighted` | CE unweighted ($\gamma=0$) | `8` | `0.3573` | `0.1461` | `0.2948` | `0.0152` | `0.0247` | `7/7` | `54.2s` | `323.6 MB` | `2.50` / `16.99 ms` | `4c41757fc37a...` |
| **`exp5_ce_weighted`** *(Best CE)* | CE sqrt weights ($\gamma=0$) | `5` | `0.3091` | **`0.1724`** | `0.2979` | **`0.0811`** | `0.0730` | `7/7` | `53.9s` | `323.6 MB` | `2.58` / `17.00 ms` | **`8825d865c703...`** |
| `exp5_focal_g1_nowt` | Focal ($\gamma=1.0$, nowt) | `6` | `0.3527` | `0.1466` | `0.2945` | `0.0061` | `0.0107` | `6/7` | `55.7s` | `321.6 MB` | `2.93` / `15.73 ms` | `ba0327d122e8...` |
| `exp5_focal_g2_nowt` | Focal ($\gamma=2.0$, nowt) | `3` | `0.3664` | `0.1272` | `0.2759` | `0.0242` | `0.0395` | `5/7` | `46.9s` | `319.6 MB` | `2.64` / `17.51 ms` | `944f63fd6f7a...` |
| `exp5_focal_g2_wt` | Focal ($\gamma=2.0$, sqrt $\alpha$) | `6` | `0.2773` | `0.1572` | `0.2702` | `0.0364` | `0.0442` | `7/7` | `56.3s` | `319.6 MB` | `2.52` / `16.56 ms` | `023f531c7a4b...` |
| **`exp5_focal_g3_wt`** *(Best Focal)* | Focal ($\gamma=3.0$, sqrt $\alpha$) | `4` | **`0.3291`** | `0.1669` | **`0.3049`** | `0.0659` | **`0.0744`** | `7/7` | `55.8s` | `319.6 MB` | `3.27` / `19.97 ms` | **`4830ac8409e7...`** |

### Per-Class Dev F1 Comparison (`N = 1,100`)

| Run ID | `anger` | `disgust` | `fear` | `joy` | `neutral` | `sadness` | `surprise` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp5_ce_unweighted` | `0.1235` | `0.0000` | `0.0000` | `0.1362` | `0.5440` | `0.0741` | `0.1447` |
| **`exp5_ce_weighted`** | **`0.1543`** | `0.0000` | `0.0357` | `0.1961` | `0.4914` | `0.1832` | **`0.1459`** |
| `exp5_focal_g1_nowt` | `0.1429` | `0.0000` | `0.0000` | `0.2037` | `0.5334` | `0.0320` | `0.1140` |
| `exp5_focal_g2_nowt` | `0.0244` | `0.0000` | `0.0000` | `0.1404` | `0.5461` | `0.1185` | `0.0610` |
| `exp5_focal_g2_wt` | `0.1517` | **`0.0400`** | `0.0000` | **`0.2266`** | `0.4368` | `0.0927` | `0.1526` |
| **`exp5_focal_g3_wt`** | `0.1304` | `0.0000` | **`0.0370`** | `0.1944` | `0.5346` | **`0.1863`** | `0.0858` |

- **Validation Gate Decision:** Without class weighting, Focal Loss collapsed toward majority `neutral` (`exp5_focal_g2_nowt`: only `5/7` classes active, `disgust=0`, `fear=0`). Combining Focal $\gamma=3.0$ with smoothed-sqrt $\alpha$ (`exp5_focal_g3_wt`, SHA-256 `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`) achieved the highest single-frame Weighted F1 (`0.3049`) and MinF1 (`0.0744`), while **`exp5_ce_weighted`** (SHA-256 `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`, deterministic match to `exp4a`) retained highest Macro F1 (`0.1724`) and MinRecall (`0.0811`). Both checkpoints were preserved.

---

## 4. Experiment 6 — EfficientNet-B0 Architecture Comparison

**Directory:** `models/facial_experiments/exp6_efficientnet/`  
**Script:** [`scripts/run_exp6_efficientnet.py`](../scripts/run_exp6_efficientnet.py)  
**Total Elapsed Time:** `207.0s`

Evaluated ImageNet-pretrained `EfficientNet-B0` (`4,016,515` params) against retained `ResNet18` (`11,180,103` params) under identical `exp4a_mtcnn_unaligned` splits and both top loss objectives from Experiment 5:

| Model / Variant | Params | Best Ep | Acc | Macro F1 | Weighted F1 | MinRec | Time | Peak VRAM | CUDA / CPU Lat | SHA-256 (prefix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ResNet18` (`exp5_ce_weighted`)** | `11,180,103` | `5` | **`0.3091`** | **`0.1724`** | **`0.2979`** | **`0.0811`** | **`53.9s`** | **`323.6 MB`** | **`2.58` / `17.00 ms`** | `8825d865c703...` |
| `Exp 6A`: EfficientNet-B0 + CE sqrt | `4,016,515` | `3` | `0.3009` | `0.1481` | `0.2802` | `0.0212` | `81.8s` | `620.2 MB` | `12.21` / `17.34 ms` | `f3f5d21e49fb...` |
| `Exp 6B`: EfficientNet-B0 + Focal ($\gamma=3$) | `4,016,515` | `3` | `0.2718` | `0.1497` | `0.2649` | `0.0417` | `111.9s` | `618.8 MB` | `11.57` / `22.26 ms` | `4b373bb7d961...` |

- **Per-Class F1 (`Exp 6A` / `Exp 6B`):** `anger` `0.1181 / 0.1348`, `disgust` `0.0000 / 0.0000`, `fear` `0.0000 / 0.0299`, `joy` `0.2198 / 0.2185`, `neutral` `0.4836 / 0.4389`, `sadness` `0.0741 / 0.1023`, `surprise` `0.1408 / 0.1237`.
- **Validation Gate Decision:** **Retained `ResNet18` (`exp5_ce_weighted`)**. Although EfficientNet-B0 has `2.78x` fewer parameters, at `96x96` resolution its depthwise-separable SE blocks used `~1.9x` higher training VRAM (`620 MB` vs `324 MB`), ran `4.5x` slower on CUDA (`11.57–12.21 ms` vs `2.58 ms`), and scored `-0.0227` lower Dev Macro F1 (`0.1497` vs `0.1724`).

---

## 5. Experiment 7 — Face Tracking & Temporal Pooling

**Directory:** `models/facial_experiments/exp7_temporal/`  
**Script:** [`scripts/run_exp7_temporal.py`](../scripts/run_exp7_temporal.py)  
**Total Elapsed Time:** `1,722.3s`

### 5.1 Multi-Frame Detection & Coherent Tracking Across All `1,108` MELD `dev` Videos
- **Backbone:** `exp5_ce_weighted` (`ResNet18`, SHA-256 `8825d865...`, `square_unaligned_margin_0.08`)
- **Video Coverage:** `1,102 / 1,108` dev videos with valid detections/tracks (`99.46%`)
- **Frame & Crop Accounting:** `8,777` frames sampled and decoded (`7.92/video`), `14,877` raw valid detections, `7,290` primary-track crops (`6.62/detected video`), `2,206` false-positive boxes filtered, `724` videos contained multi-face frames
- **Track Coherence:** `mean 3.19 tracks/detected video`, `mean adjacent IoU = 0.5399` (`median = 0.5633`)
- **Runtime & Memory:** Detector + IoU tracking averaged `1,535.94 ms/video` (`126.7 MB` peak GPU VRAM, `419.7 MB` CPU RSS); batched ResNet18 temporal forward pass averaged `4.34 ms/video` (`254.2 MB` peak VRAM)
- **Tracking Center Defect Fix Verified:** Previous buggy resized-crop coordinates collapsed distinct speakers `(255, 190)` vs `(905, 180)` to `(48.0, 48.0)` (`distance = 0.0`). Normalized original-frame coordinates preserved separation (`0.4426`).

### 5.2 Temporal Pooling Comparison (`N = 1,102` detected dev videos)

| Pooling Method | Accuracy | Macro F1 | Weighted F1 | MinRec | Active | Haar-Matched `N=1,077` MF1 | Key Per-Class F1s (`anger` / `disgust` / `fear` / `joy` / `neutral` / `sad` / `surp`) |
|---|---:|---:|---:|---:|---:|---:|---|
| `single_frame_middle` | `0.3004` | `0.1715` | `0.2858` | **`0.1018`** | `7/7` | `0.1740` | `0.1281 / 0.0488 / 0.0392 / 0.2293 / 0.4722 / 0.1891 / 0.0940` |
| **`raw_all_detections_mean_prob`** | **`0.3829`** | **`0.1722`** | **`0.3232`** | `0.0571` | `7/7` | **`0.1748`** | `0.1328 / 0.0000 / 0.0000 / 0.2179 / 0.5666 / 0.1891 / 0.0990` |
| `track_arith_mean_prob` | `0.3466` | `0.1589` | `0.3019` | `0.0631` | `7/7` | `0.1594` | `0.1040 / 0.0000 / 0.0000 / 0.1944 / 0.5381 / 0.1918 / 0.0837` |
| `track_quality_conf_weighted` | `0.3330` | `0.1549` | `0.2951` | `0.0601` | `7/7` | `0.1556` | `0.1016 / 0.0000 / 0.0000 / 0.1924 / 0.5253 / 0.1739 / 0.0913` |
| `track_topk3_prob` | `0.3113` | `0.1544` | `0.2869` | `0.0721` | `7/7` | `0.1559` | `0.1135 / 0.0000 / 0.0000 / 0.1993 / 0.5005 / 0.1897 / 0.0776` |
| `track_topk3_logit` | `0.3240` | `0.1525` | `0.2909` | `0.0631` | `7/7` | `0.1540` | `0.1176 / 0.0000 / 0.0000 / 0.1837 / 0.5191 / 0.1765 / 0.0708` |
| `track_mean_logit` | `0.3548` | `0.1506` | `0.2976` | `0.0511` | `7/7` | `0.1517` | `0.1008 / 0.0000 / 0.0000 / 0.1697 / 0.5483 / 0.1692 / 0.0664` |

### 5.3 Live Diagnostic Video Audit (`WIN_20261111_03_07_13_Pro.mp4`)
- **Accounting:** `244 / 244` frames decoded, `82` sampled, `67` valid detections, `3` FP boxes filtered, `2` tracks (`56` primary-track crops, `mean_adjacent_iou = 0.8940`).
- **Predictions:** `mean_prob = surprise (0.2301)` (`neutral=0.2165`, `anger=0.1788`, `sadness=0.1747`); `topk3 = neutral (0.3668)` / `surprise (0.2888)`; `mean_logit = surprise`.

### 5.4 Validation Gate & Temporal Tradeoff Analysis
- **`raw_all_detections_mean_prob`** passed Experiment 7 (`+8.25%` Accuracy `0.3829 vs 0.3004`, `+0.0374` Weighted F1 `0.3232 vs 0.2858`, and `0.1722 vs 0.1715` Macro F1 on `N=1,102`).
- **Key Scientific Finding:** In multi-speaker MELD clips (`724` videos with multi-face frames), averaging probabilities across all valid detections avoids single-track speaker-selection errors (without active-speaker audio diarization, `track_*` methods locked onto non-speaking listeners across camera cuts, lowering Macro F1 to `0.1506–0.1589`). However, temporal smoothing across frames also damped rare momentary `disgust`/`fear` peaks into `neutral`, reducing `disgust F1` and `fear F1` to `0.0000` (despite `disgust` and `fear` remaining predicted classes `7/7`), which directly informed Phase 17C multi-criteria ranking.

