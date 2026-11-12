# Phase 18E — Final Regression & Reliability Report

- **Date:** 2026-11-11
- **Deployment Status:** `deployed_to_production = False` (local integration complete; held for explicit user review before any public deployment)
- **Canonical 7-Class Order (Preserved End-to-End):** `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

---

## 1. Complete Python Test Suite (`pytest`)

Executed across all six repository package test directories:

```powershell
python -m pytest -v -o pythonpath=Multimodal-Emotion-Recognition-main \
  Multimodal-Emotion-Recognition-main/api/tests \
  Multimodal-Emotion-Recognition-main/facial_emotion/tests \
  Multimodal-Emotion-Recognition-main/text_emotion/tests \
  Multimodal-Emotion-Recognition-main/speech_emotion/tests \
  Multimodal-Emotion-Recognition-main/preprocessing/tests \
  Multimodal-Emotion-Recognition-main/datasets/CMU-MOSEI/tests
```

*(Note: `scripts/test_gradient_flow.py` is a top-level training script rather than a `pytest` suite and was excluded by targeting the six package test directories directly.)*

- **Result:** **`287 passed, 6 warnings in 38.94s`** (0 failed, 0 skipped) across 25 test modules:
  - [`api/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/tests) (**62 passed**): `test_api.py` (14), `test_fusion.py` (20), `test_multimodal_api.py` (24), `test_phase18d_multimodal_extension.py` (4)
  - [`facial_emotion/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/tests) (**31 passed**): `test_facial.py` (13), `test_remediation.py` (6), `test_resnet_integration.py` (8), `test_video_integration.py` (4)
  - [`text_emotion/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/text_emotion/tests) (**6 passed**): `test_distilbert.py` (3), `test_stabilization.py` (3)
  - [`speech_emotion/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/tests) (**86 passed**): `test_audio_loader.py` (13), `test_feature_extraction.py` (16), `test_label_mapping.py` (8), `test_metrics.py` (4), `test_model.py` (16), `test_mosei_training.py` (7), `test_preprocess_audio.py` (6), `test_training_data.py` (7), `test_training_pipeline.py` (5), `test_wav2vec2.py` (4)
  - [`preprocessing/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/preprocessing/tests) (**90 passed**): `test_alignment.py` (27), `test_csd_reader.py` (11), `test_preprocessing.py` (52)
  - [`datasets/CMU-MOSEI/tests/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/datasets/CMU-MOSEI/tests) (**12 passed**): `test_prepare_dataset.py` (12)

---

## 2. Backend Regression, Integrity & Reliability Suite

Executed [`scripts/validate_phase18e_regression.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/validate_phase18e_regression.py) -> [`models/facial_emotion/resnet18/phase18e_regression_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase18e_regression_validation.json) (exit code `0`, `total_script_seconds = 38.77`).

### 2.1 Split Isolation & Protected Artifact Integrity

- **Zero Test Split Audit (`zero_test_split_audit`):** `checked_files = 10`, `violations = []`, `passed = True`
- **Protected & Staged SHA-256 Checksums (`protected_hashes`, 11/11 `matched = True`):**
  - `facial_emotion/model.py`: `08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b`
  - `facial_emotion/detector.py`: `24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d`
  - `facial_emotion/remediation.py`: `3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07`
  - `api/fusion.py`: `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d`
  - `models/facial_emotion/facial_cnn_best.pt`: `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760`
  - `models/text_emotion/distilbert/best/model.safetensors`: `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474`
  - `models/speech_emotion/wav2vec2/best/model.safetensors`: `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e`
  - `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt`: `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`
  - `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt`: `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`
  - `models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_best.pt`: `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`
  - `models/facial_experiments/exp5_focal_loss/exp5_focal_g3_wt_best.pt`: `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`

### 2.2 Parity, Fallback, Rollback & Fusion Behavior

- **CPU vs. CUDA Parity (`cpu_cuda_parity`):**
  - `cpu_single_crop_ms = 17.9`
  - `cuda_single_crop_ms = 15.01`
  - `max_abs_prob_diff = 3.5762786865234375e-07` (`< 1e-4`)
  - `argmax_matches = 16/16`
- **Fallback Verification (`fallback_verification`):** Triggered cleanly when primary checkpoint path was nonexistent; loaded fallback model `facial-resnet18-focal-v1` (SHA-256 prefix `4830ac84...`).
- **Legacy Rollback Check (`legacy_rollback_check`):** `backend = "legacy_cnn"`, `class = "FacialEmotionPredictor"`, `passed = True`.
- **Candidate 4 Parity & Multimodal Combinations:** Candidate 4 exact parity verified (`max diff < 1e-6`) with all multimodal combinations verified on MELD `dev` (`dia0_utt0.wav` and `dia0_utt0.mp4`).

### 2.3 Latency, Rejections, Concurrency & Resource Footprint

- **Negative Rejection Matrix (`negative_rejection_matrix`):** HTTP status codes `422 / 422 / 422 / 415 / 422 / 422 / 413`
- **Endpoint Latencies (`endpoint_latencies_ms`):**
  - Text: mean `15.90 ms`, p95 `20.60 ms`
  - Audio: mean `50.55 ms`, p95 `67.60 ms`
  - Video: mean `877.48 ms`, p95 `941.52 ms`
  - Multimodal T+A+V: mean `920.22 ms`, p95 `1011.09 ms`
- **Concurrency (`concurrency_test`):** `6` requests across `3` workers, `all_200 = True`, `wall_time_ms = 5417.07`
- **Temporary File Cleanup (`temp_file_cleanup`):** `leaked_files = []`
- **Process & GPU Resources (`resources`):** `process_rss_mb = 1760.37`, `cuda_peak_allocated_mb = 787.69`, `cuda_reserved_mb = 834.0`

---

## 3. Frontend Production Build & Headless Browser E2E Verification

- **Frontend Build:** `npm.cmd --prefix Multimodal-Emotion-Recognition-main/frontend run build` exited `0`, producing `dist/assets/index-CqEvIWqZ.js` (`192.64 kB`, built in `9.44s`). No frontend files were edited after this build.
- **Browser E2E Suite:** Executed [`scripts/run_phase18e_browser_suite.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/run_phase18e_browser_suite.py) + [`scripts/e2e_phase18e_browser.cjs`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/e2e_phase18e_browser.cjs) against live `uvicorn api.app:app --port 8000` and `vite preview --host localhost --port 5173` -> [`models/facial_emotion/resnet18/phase18e_browser_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase18e_browser_validation.json) (exit code `0`):
  - `passed = true`
  - `consoleErrors = []`, `pageErrors = []`, `serverErrors = []`, `failedRequests = []`
  - **Desktop (`1920x1080`):** `scrollWidth = 1920`, `clientWidth = 1920`, `hasOverflow = false`
  - **Mobile (`390x844`):** `scrollWidth = 390`, `clientWidth = 390`, `hasOverflow = false`
  - **Text-only UI check (`textOnlyCheck`):** `bodyHasCandidate4 = true`, `videoExcluded = true`
  - **Text + Video UI check (`textVideoCheck`):** `bodyHasExtendedId = true`, `hasResnetBreakdown = true`
  - **Text + Audio + Video UI check (`tavCheck`):** `hasDistilBert = true`, `hasWav2Vec2 = true`, `hasResNet18 = true`, `activeWeightCount = 3`, `hasNotActive = false`
  - **Backend status display (`backendStatusContainsResNetNote`):** `true`
  - Visual review of desktop (`desktop_multimodal_tav.png`) and mobile (`mobile_multimodal_390x844.png`) screenshots confirmed clean dark UI layout, three-modality breakdown rendering, and zero horizontal overflow.

---

## 4. Remaining Limitations

1. **No Public Deployment:** `deployed_to_production = False`; integration remains in the local workspace awaiting explicit review and approval.
2. **Held-Out Test Set Untouched:** Per Phase 16 and Phase 17 data-isolation rules, no Phase 18 tuning or validation accessed the held-out MELD `test` split.
3. **Standalone Facial Capacity:** Standalone `exp5_ce_weighted` ResNet18 dev performance (Macro F1 `0.1739`, Weighted F1 `0.2998`, Accuracy `0.3107`) remains weak relative to text and speech; multimodal fusion mitigates this via quality-gated conservative visual weighting while keeping Candidate 4 unchanged for Text-only and Text+Audio requests.
4. **Physical Media Hardware:** Headless Chrome E2E verification exercised uploaded media and live FastAPI endpoints; physical webcam/microphone hardware permissions are required at runtime for browser live capture (previously verified on physical devices in Phase 17B-0.1).

