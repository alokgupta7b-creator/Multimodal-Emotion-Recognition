# Phase 18B — Facial Model Integration Report

**Date:** 2026-11-11  
**Gate Decision:** **PASS** (`41 passed` across facial & API regression tests; CPU/CUDA parity `4.77e-7 < 1e-4`; all protected assets untouched)

---

## 1. Architecture & Isolation Summary

- **Independent Production Predictor ([`facial_emotion/resnet_predictor.py`](../facial_emotion/resnet_predictor.py)):**
  - Implemented [`ResNetFacialEmotionPredictor`](../facial_emotion/resnet_predictor.py) independently from the legacy CNN [`FacialEmotionPredictor`](../facial_emotion/model.py) so legacy code and `facial_cnn_best.pt` remain byte-for-byte untouched.
  - Stages the Phase 17 approved models into [`models/facial_emotion/resnet18/`](../models/facial_emotion/resnet18/manifest.json):
    - **Primary (`exp5_ce_weighted_best.pt` -> `facial-resnet18-ce-v1`):** `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`
    - **Fallback (`exp5_focal_g3_wt_best.pt` -> `facial-resnet18-focal-v1`):** `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`
    - Both copied verbatim from `models/facial_experiments/exp5_focal_loss/` (leaving all experimental checkpoints intact).
- **Pre-Phase 18 Backups:** Backed up `23` pre-Phase 18 source files (`api/*.py`, `facial_emotion/*.py`, `frontend/src/*.ts*`) and `facial_cnn_best.pt` into `scratch/pre_phase18_backup/` before modifying production config or registry.
- **Registry & Configurable Rollback ([`api/config.py`](../api/config.py), [`api/model_registry.py`](../api/model_registry.py), [`api/schemas.py`](../api/schemas.py)):**
  - Default `EMOTION_API_FACIAL_BACKEND=resnet18` loads `ResNetFacialEmotionPredictor` with startup SHA-256 verification and automatic verified fallback to `exp5_focal_g3_wt_best.pt` if the primary artifact fails checksum/metadata verification (failing loudly if neither valid checkpoint passes).
  - Rollback `EMOTION_API_FACIAL_BACKEND=legacy_cnn` preserves `FacialEmotionPredictor` (`models/facial_emotion/facial_cnn_best.pt`, `facial_cnn` 94k CNN).
  - [`ModelStatus`](../api/schemas.py) and `ModelRegistry.status()` report additive fields (`architecture`, `checkpoint_sha256`, `active_role`, `preprocessing`, `pooling_method`) while keeping `loaded`, `model_identifier`, `device`, and `error` compatible.

---

## 2. Verification Checks & Observed Metrics

### 2.1 Preprocessing, Checksum, & Canonical Schema
- **Normalization (`imagenet_rgb_96x96`):** `preprocess_resnet_face_crops()` converts uint8 BGR crops to RGB `[0.0, 1.0]` (`1/255.0`) normalized with ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, resizing non-`96x96` inputs via `INTER_CUBIC` (upscale) or `INTER_AREA` (downscale).
- **7-Class Order Verified:** `('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise')`. Outputs checked finite, `>= 0.0`, and normalized (`sum == 1.0`).
- **Missing / Empty / Tampered Rejection:** Empty crop sequences (`predict_faces([])`) and blank no-face frames (`predict_image(blank)`) raise `ValueError` without fabricating `neutral`. Tampered primary checksum without fallback raises `ValueError("SHA-256 mismatch ...")`; tampered primary with verified fallback switches to `active_role = "fallback"` (`facial-resnet18-focal-v1`).

### 2.2 CPU vs CUDA Inference Parity & Latency (`strict_fp32=True`)
Measured on 16 uint8 crops (`96x96x3`) and single-crop `infer_crop_logits()` calls (RTX 4050 vs CPU):
- **Max Absolute Probability Divergence:** **`4.77e-7`** (`4.768e-07 < 1e-4` strict FP32 threshold; `_strict_fp32_guard` disables TF32 during ResNet18 forward pass and restores original backend flags afterward)
- **Top-1 Argmax Match:** **`16 / 16`** identical predictions
- **Single-Crop End-to-End `infer_crop_logits` Call Average (including preprocessing + host-device transfer):** `16.88 ms` (CPU) / `16.55 ms` (CUDA), `610.26 MB` peak CUDA memory with both MTCNN and ResNet18 resident on GPU.

### 2.3 Test Suite & Protected Checksums
- **Command:** `pytest -v facial_emotion/tests/ api/tests/test_api.py`
- **Result:** **`41 passed, 3 warnings in 40.00s`** (`test_facial.py`: 13, `test_remediation.py`: 6, `test_resnet_integration.py`: 8, `test_api.py`: 14).
- **Protected Assets Untouched (SHA-256 Re-verified):**
  - `facial_emotion/model.py`: `08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b`
  - `facial_emotion/detector.py`: `24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d`
  - `facial_emotion/remediation.py`: `3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07`
  - `api/fusion.py`: `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d`
  - `models/facial_emotion/facial_cnn_best.pt`: `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760`
  - `models/text_emotion/distilbert/best/model.safetensors`: `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474`
  - `models/speech_emotion/wav2vec2/best/model.safetensors`: `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e`

**Phase 18B Conclusion:** All validation gates pass. Advancing automatically to **Phase 18C — Video Inference Integration**.

