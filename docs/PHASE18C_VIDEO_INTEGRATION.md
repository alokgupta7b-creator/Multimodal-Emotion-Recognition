# Phase 18C — Video Inference Integration Report

**Date:** 2026-11-11  
**Script:** [`scripts/validate_phase18c_video.py`](../scripts/validate_phase18c_video.py)  
**Artifact:** [`models/facial_emotion/resnet18/phase18c_video_validation.json`](../models/facial_emotion/resnet18/phase18c_video_validation.json)  
**Gate Decision:** **PASS** (End-to-end MTCNN + original-frame IoU tracking + ResNet18 verified across MELD dev cohort and all stress scenarios; zero held-out test access)

---

## 1. Integrated Video Pipeline Summary

- **Detection & Crop Extraction ([`facial_emotion/mtcnn_detector.py`](../facial_emotion/mtcnn_detector.py)):**
  - `MtcnnFaceDetector` uses `MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709)`.
  - Evaluates at `640px` downscaled width first with native-resolution fallback; scales detections back to full original frame coordinates.
  - Filters false positives (`prob < 0.90`, `min(w, h) < 24`, aspect ratio outside `[0.40, 1.80]`), and extracts square-unaligned `96x96` crops with `margin=0.08`.
- **Sampling & Tracking:**
  - `sample_video_frames()` performs uniform seeking across `n_sample_frames = 8` frames when `CAP_PROP_FRAME_COUNT > 0`, with automatic sequential decode fallback for browser WebM / streamable files lacking header frame counts.
  - `track_mtcnn_detections()` tracks faces across sampled frames in original full-resolution `(x, y, w, h)` coordinates using IoU (`min_iou=0.08`) + frame-diagonal normalized center distance (`max_center_dist=0.28`, `max_frame_gap=3`), avoiding the legacy resized-crop center collapse bug, and selects the primary face track by integrated salience (`len * sqrt(mean_area) * mean_prob`).
- **Honest Frame Accounting & Errors:**
  - Reports exact `total_video_frames`, `frames_decoded`, `frames_sampled`, `frames_with_detected_faces`, `multi_face_frames`, `total_face_detections`, `false_positives_filtered`, `unique_tracks`, `primary_track_crops`, and `mean_adjacent_iou`.
  - Raises explicit `ValueError` (HTTP `422` in FastAPI) when zero valid faces remain — never fabricates a `neutral` distribution.

---

## 2. Development Benchmark Comparison (`zero_test_split_access = True`)

### 2.1 Full MELD `dev` Pooling Comparison (`1,108` videos / `1,102` detected, `14,877` crops, `6.78 s` ResNet18 forward, `616.8 MB` peak VRAM)

| Pooling Method | Detected `N` | Accuracy | Macro F1 | Weighted F1 | Active Classes | Notes |
|---|---:|---:|---:|---:|---:|---|
| **`raw_all_detections_mean_prob`** *(API default)* | `1,102` | **`0.3829`** | **`0.1722`** | **`0.3232`** | `5 / 7` | Highest overall Accuracy, Macro F1, Weighted F1; smooths `disgust`/`fear` F1 to `0.0000` |
| `track_arith_mean_prob` *(primary track)* | `1,102` | `0.3466` | `0.1589` | `0.3019` | `5 / 7` | Best tracked multi-frame Macro & Weighted F1 |
| `track_mean_logit` | `1,102` | `0.3548` | `0.1506` | `0.2976` | `5 / 7` | Higher Neutral precision (`0.5483`), slightly lower Macro F1 |
| `track_quality_conf_weighted` | `1,102` | `0.3321` | `0.1547` | `0.2943` | `5 / 7` | Confidence + Laplacian sharpness weighted |
| `track_topk3_prob` | `1,102` | `0.3122` | `0.1547` | `0.2875` | `5 / 7` | Retains slightly stronger `sadness`/`joy` sensitivity |
| `single_frame_middle` | `1,102` | `0.3004` | `0.1715` | `0.2860` | **`7 / 7`** | Only mode retaining non-zero `disgust` (`0.0488`) & `fear` (`0.0392`) F1 |

**Engineering Tradeoff Preserved:** Both `raw_all_detections_mean_prob` (default via `EMOTION_API_FACIAL_POOLING`) and primary-track pooling (`track_arith_mean_prob`, plus all 6 modes) remain selectable without duplicating inference logic; raw vs tracked crop counts are never conflated.

### 2.2 End-to-End Decoded MELD `dev` Balanced Probe (`N = 70`: 10 videos per canonical class)
- **Detected Videos:** `69 / 70` (`1` valid no-face clip rejected safely)
- **End-to-End Latency (decode + MTCNN + track + ResNet18):** `mean = 1,213.20 ms`, `p95 = 1,722.59 ms`
- **Balanced Subset Scores:** `raw_all_detections_mean_prob` (`Acc=0.1739`, `Macro F1=0.1336`, `Weighted F1=0.1356`) vs `track_arith_mean_prob` (`Acc=0.1884`, `Macro F1=0.1354`, `Weighted F1=0.1374`).

---

## 3. Stress Scenario Suite Results (`ResNetFacialEmotionPredictor.predict_video`)

| Stress Scenario | Source / Condition | Frames (Total / Dec / Samp) | Detections / FPs / Tracks / Primary | Result / Prob Sum / Finite | Latency (`ms`) |
|---|---|---|---|---|---:|
| Normal Dialogue | `dev` dialog sample | `67 / 8 / 8` | `21 det / 0 FP / 3 tracks / 8 crops` (`IoU=0.550`) | `neutral (0.3301)`, `sum=1.0`, finite | `1,357.53` |
| Live Webcam Capture | `WIN_20261111_03_07_13_Pro.mp4` | `244 / 8 / 8` | `6 det / 0 FP / 1 track / 6 crops` (`IoU=0.598`) | `surprise (0.2172)`, `sum=1.0`, finite | `1,115.48` |
| No-Face (MELD Dev) | `dev_dia5_utt9` | `2 / 2 / 2` | `0 det / 2 FP filtered` | Rejected safely (`ValueError`) | — |
| No-Face (Synthetic) | Black `160x120` `.mp4` | `10 / 8 / 8` | `0 det / 0 FP` | Rejected safely (`ValueError`) | — |
| Multi-Face | `dev_dia0_utt0` | `67 / 8 / 8` | `21 det (`8` multi-face frames) / 3 tracks / 8 crops` | `neutral (0.3301)`, `sum=1.0`, finite | `1,291.35` |
| Ultra-Short | `dev_dia99_utt3` | `2 / 2 / 2` | `2 det / 1 track / 2 crops` (`IoU=0.960`) | `neutral (0.6457)`, `sum=1.0`, finite | `407.51` |
| Long | `dev_dia74_utt19` | `684 / 8 / 8` | `9 det / 2 tracks / 7 crops` | `neutral (0.2614)`, `sum=1.0`, finite | `1,099.73` |
| Low-Light | `dev_dia10_utt3` | `146 / 8 / 8` | `24 det / 3 tracks / 8 crops` (`IoU=0.655`) | `neutral (0.2771)`, `sum=1.0`, finite | `1,064.58` |
| Synthetic Dark + Blur | `0.35x` luma + Gaussian blur `sigma=1.5` | `16 / 8 / 8` | `22 det / 3 FP filtered / 3 tracks / 8 crops` | `neutral (0.3060)`, `sum=1.0`, finite | `909.66` |
| WebM Live Fallback | `libvpx` `.webm` encode | `29 / 8 / 8` | `24 det / 3 tracks / 8 crops` (`IoU=0.706`) | `neutral (0.3515)`, `sum=1.0`, finite | `1,178.71` |
| Corrupt Video | Non-MP4 bytes (`corrupt.mp4`) | — | — | Rejected safely (`ValueError`) | — |

**Phase 18C Conclusion:** All video integration and stress gates pass. Advancing automatically to **Phase 18D — Multimodal Integration**.

