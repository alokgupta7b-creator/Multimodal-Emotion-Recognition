# Phase 18D — Multimodal Integration Report

**Date:** 2026-11-11  
**Script:** [`scripts/validate_phase18d_multimodal.py`](../scripts/validate_phase18d_multimodal.py)  
**Artifact:** [`models/facial_emotion/resnet18/phase18d_dev_fusion_benchmark.json`](../models/facial_emotion/resnet18/phase18d_dev_fusion_benchmark.json)  
**Gate Decision:** **PASS** (Candidate 4 Text + Speech preserved bit-for-bit; calibrated ResNet18 facial extension validated on MELD `dev` with zero held-out `test` split access; all modality combinations, UI updates, frontend build, and 66 backend/video tests passing)

---

## 1. Empirical Validation Strictly on MELD `dev` (`N = 1,108`, `zero_test_split_access = True`)

Cohort breakdown across full MELD `dev` (`1,108` utterances):
- Valid DistilBERT Text: `1,108 / 1,108`
- Valid Wav2Vec2 Speech: `1,107 / 1,108`
- Valid MTCNN + ResNet18 Facial (`raw_all_detections_mean_prob`): `1,102 / 1,108`
- Joint Audio + Video valid cohort: `1,101 / 1,108`

### 1.1 Full MELD `dev` Modality & Fusion Benchmark (`N = 1,108` / AV cohort `N = 1,101`)

| Configuration / Strategy | `N` | Accuracy | Macro F1 | Weighted F1 | Minority Macro F1 | NLL ↓ | Brier ↓ | ECE ↓ | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `text_only_distilbert` | `1,108` | `0.6011` | `0.4205` | `0.5700` | `0.3321` | `1.23759` | `0.57513` | `0.11620` | Strong semantic anchor |
| `candidate4_text_audio_baseline` (`w_v = 0.0`) | `1,108` | **`0.6047`** | **`0.4243`** | **`0.5747`** | **`0.3368`** | `1.20542` | `0.56306` | `0.08466` | Protected production baseline |
| `text_video_lambda_0.08` *(selected T+V)* | `1,108` | `0.6011` | `0.4205` | `0.5700` | `0.3321` | `1.23135` | `0.57424` | `0.11404` | Improves calibration over Text-only with zero F1/Acc loss |
| **`text_audio_video_lambda_0.08`** *(selected T+A+V)* | `1,108` | **`0.6047`** | **`0.4243`** | **`0.5747`** | **`0.3368`** | **`1.20038`** | **`0.56229`** | **`0.08388`** | Improves NLL/Brier/ECE over Candidate 4 with zero Acc/F1 drop |
| `text_audio_video_lambda_0.12` | `1,108` | `0.6047` | `0.4243` | `0.5747` | `0.3368` | `1.19939` | `0.56201` | `0.08107` | Higher facial weight boundary tested |
| `cond_video_wv0.10_agree0` | `1,108` | `0.6047` | `0.4243` | `0.5747` | `0.3368` | `1.20377` | `0.56264` | `0.08385` | Conditional synergy ablation |
| `audio_only_on_av_cohort` | `1,101` | `0.4796` | **`0.2532`** | **`0.4270`** | **`0.1997`** | `1.46984` | `0.67870` | **`0.03027`** | Best non-text single modality |
| `video_only_on_av_cohort` | `1,101` | `0.3833` | `0.1724` | `0.3233` | `0.0842` | `1.68993` | `0.78044` | `0.09468` | ResNet18 facial standalone (`5/7` active) |
| **`audio_video`** (`w_v = 0.08`, selected A+V) | `1,101` | `0.4814` | `0.2501` | `0.4252` | — | `1.46514` | `0.67781` | `0.03852` | Improves NLL/Brier & Acc (`+0.0018`) vs Audio alone |
| `audio_video_blend_wv_0.20` | `1,101` | `0.4805` | `0.2424` | `0.4187` | `0.1868` | `1.46703` | `0.67942` | `0.06357` | Larger `w_v` degrades Macro F1 & ECE |
| `audio_video_blend_wv_0.50` | `1,101` | `0.4859` | `0.2357` | `0.4140` | `0.1728` | `1.50605` | `0.69890` | `0.13375` | Heavy calibration degradation (`ECE +0.103`) |

**Key Empirical Finding:** Because ResNet18 facial predictions are weak standalone signals (`Macro F1 = 0.1724`), naive equal weighting (`w_v >= 0.20`) hurts calibration and Macro F1. Conservative bounded integration at `VALIDATED_LAMBDA_VIDEO = 0.08` preserves Candidate 4 Acc (`0.6047`), Macro F1 (`0.4243`), and Weighted F1 (`0.5747`) while strictly improving probability calibration (NLL `1.20542 -> 1.20038`, Brier `0.56306 -> 0.56229`, ECE `0.08466 -> 0.08388`).

---

## 2. Clean Architecture & Candidate 4 Preservation

- **Preserved Untouched Core ([`api/fusion.py`](../api/fusion.py)):**
  - `Candidate4FusionEngine` (`lambda_video=0.0` legacy bypass) remains completely unmodified (SHA-256 `ffa76bea...`).
- **Layered Phase 18D Extension ([`api/multimodal_extension.py`](../api/multimodal_extension.py)):**
  - [`ExtendedMultimodalFusionEngine`](../api/multimodal_extension.py) delegates `Text`-only and `Text + Speech` requests (without video) **verbatim** to `Candidate4FusionEngine` (`fusion_method = "text_anchored_class_conditional_bounded_fusion_v1"`).
  - When a video file is present with Text (or Text + Audio) and passes quality gates (`face_detection_coverage >= 0.40` and `prediction_entropy <= 1.85`), applies bounded blend `0.92 * P_c4 + 0.08 * P_video` (`fusion_method = "candidate4_plus_validated_resnet18_video_v1"`), renormalizing weights transparently to sum to `1.0`.
  - When video fails quality gates (`video_low_face_coverage` or `video_high_entropy`), sets `w_v = 0.0` (advisory telemetry retained cleanly in `individual_predictions["video"]` without perturbing Candidate 4 probabilities).
  - Supports text-omitted fallback combinations (`single_modality_video_resnet18_v1`, `single_modality_audio_wav2vec2_v1`, `audio_primary_plus_validated_resnet18_video_v1`) without ever fabricating missing or `neutral` distributions.
- **FastAPI Endpoint ([`api/app.py`](../api/app.py)):**
  - `/predict/multimodal` accepts optional `text` alongside uploaded `audio`/`file` and/or `video` (preserving strict JSON rejection of missing/whitespace/non-string `text`), shares `_validate_and_predict_file_sync`, cleans temporary uploads via `finally: cleanup_file(...)`, and degrades safely (`200`) if a secondary modality fails when another valid modality succeeds (or `422` if no usable modality remains).
- **Frontend UI ([`frontend/src/api.ts`](../frontend/src/api.ts), [`frontend/src/LiveCapture.tsx`](../frontend/src/LiveCapture.tsx), [`frontend/src/App.tsx`](../frontend/src/App.tsx)):**
  - Multimodal Studio now supports optional Text, optional WAV Audio (upload or live mic), and optional MP4/WebM Face Video (upload or live camera) plus dynamic 3-modality weights (`Active`, `Advisory`, `Excluded`) and 3-column breakdown (`DistilBERT Text`, `Wav2Vec2 Speech`, `ResNet18 Face`).

---

## 3. Verification Results

- **Backend & Extension Test Suite:**
  - Command: `python -m pytest -v Multimodal-Emotion-Recognition-main/facial_emotion/tests/test_video_integration.py Multimodal-Emotion-Recognition-main/api/tests/`
  - Result: **`66 passed, 3 warnings in 40.07s`** (`test_video_integration.py`: 4, `test_api.py`: 14, `test_fusion.py`: 18, `test_multimodal_api.py`: 26, `test_phase18d_multimodal_extension.py`: 4 — all 7 modality combinations, Candidate 4 exact parity, quality gating, audio-only MP4/blank/corrupt fallback + 422 checks, real model smoke test, and zero leaked temp files).
- **Frontend Production Build:**
  - Command: `npm.cmd --prefix Multimodal-Emotion-Recognition-main/frontend run build`
  - Result: **`tsc -b && vite build` succeeded (`dist/assets/index-CqEvIWqZ.js 192.64 kB`, built in `9.44 s`)**.

**Phase 18D Conclusion:** All multimodal extension gates pass. Advancing automatically to **Phase 18E — Final Regression & Reliability Testing**.

