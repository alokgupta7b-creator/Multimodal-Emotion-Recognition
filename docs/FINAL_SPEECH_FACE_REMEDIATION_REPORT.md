# Final Speech & Facial Emotion Recognition Remediation Report (`P0 + P1 + P2`)

**Date:** 2026-11-12  
**Project:** Multimodal Emotion Recognition System (Academic Project)  
**Workflow Completed:** Stage 1 (Baseline & Protection) -> Stage 2 (P0 Inference, Pooling & UX) -> Stage 3 (P1 Calibration) -> Stage 4 (P2 V2 Retraining) -> Stage 5 (API Integration, Rollback & Final Regression)  
**Detailed Stage Reports:**
- [`docs/P0_INFERENCE_REMEDIATION_REPORT.md`](P0_INFERENCE_REMEDIATION_REPORT.md)
- [`docs/P1_PROBABILITY_CALIBRATION_REPORT.md`](P1_PROBABILITY_CALIBRATION_REPORT.md)
- [`docs/P2_V2_MODEL_RETRAINING_REPORT.md`](P2_V2_MODEL_RETRAINING_REPORT.md)

---

## 1. Protected Baselines & Experimental Integrity

All protected v1 model checkpoints, frozen fusion rules, canonical class ordering (`anger, disgust, fear, joy, neutral, sadness, surprise`), and held-out MELD `test` artifacts were hashed before remediation and verified unchanged at completion:

| Protected Asset | Path | Verified SHA-256 (Pre & Post) | Status |
| :--- | :--- | :--- | :--- |
| DistilBERT Text | `models/text_emotion/distilbert/best/model.safetensors` | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | Unchanged |
| Wav2Vec2 Speech v1 | `models/speech_emotion/wav2vec2/best/model.safetensors` | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` | Unchanged |
| ResNet18 Primary v1 | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | Unchanged |
| ResNet18 Fallback v1 | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | Unchanged |
| Facial Legacy CNN | `models/facial_emotion/facial_cnn_best.pt` | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | Unchanged |
| Candidate 4 Fusion | `api/fusion.py` (`tau_a=0.20, lambda_max=0.40, kappa_a=1.50, lambda_v=0.00`) | `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d` | Unchanged |
| Held-out MELD Test CSV | `datasets/MELD/annotations/test_sent_emo.csv` | `2d3fdb6586bdcdf172a7197fae9244116b1103efd951db2317e6a67c7042d28e` | Unchanged & Excluded |
| Phase 16 Test Predictions | `models/phase16_final_evaluation/test_predictions.jsonl` | `e913c90a9fc16281846af2b8c62e5fda5bb6d60c153139a92ef7a214fd34926d` | Unchanged |

All calibration fitting, hyperparameter/epoch selection, and V2 evaluation used strictly MELD `train` plus the disjoint dialogue-grouped MELD `dev` split (`dev_cal` = `560` utterances / 57 dialogues; `dev_val` = `548` utterances / 57 dialogues in [`models/remediation_splits/meld_dev_cal_val_split.json`](../models/remediation_splits/meld_dev_cal_val_split.json)).

---

## 2. End-to-End Remediation Summary Across Stages

### P0 — Inference, Temporal Pooling & UX Remediation
- **Speech (`speech_emotion/wav2vec2_model.py`, `api/app.py`):**
  - Eliminated batch padding distortion by enforcing `return_attention_mask=True` in both `_collator` and `Wav2Vec2Predictor.predict_file`.
  - Unified training (`8.0 s`) and inference preprocessing via `trim_leading_trailing_silence()` (scale-invariant `95th percentile - 32.0 dB` with `200 ms` safety margins) and `preprocess_speech_waveform()`, halving padded-silence L1 probability drift across 100 `dev` clips (`0.1575 -> 0.0773`) and improving top-1 agreement (`92.0% -> 96.0%`) while preserving `100.0%` agreement under `-18 dB` quiet scaling.
  - Reduced CUDA mean inference latency by `37.3%` (`27.32 ms -> 17.12 ms`) and surfaced acoustic telemetry (`audio_duration_s`, `processed_duration_s`, `trimmed_silence_s`, `audio_rms_db`, `low_volume_warning`).
- **Facial (`facial_emotion/resnet_predictor.py`):**
  - Benchmarked all 9 temporal pooling strategies on `dev`, `dev_cal`, `dev_val`, single-track vs multi-track subsets, and drop-1 frame perturbations. Proved `track_topk3_prob` artificially sharpened confidence while losing `-7.07 pp` accuracy (`38.29% -> 31.22%`); retained `raw_all_detections_mean_prob` as default and added `raw_mean_logit` to `SUPPORTED_POOLING_METHODS`.
- **Frontend (`frontend/src/`):**
  - Formatted bar percentages to 1 decimal place (`formatProbabilityPercent`) to match confidence badges, added live and post-capture RMS dB microphone guidance (`LiveCapture.tsx`), separated signal-validation rejections (`classifyInputError`, `"Excluded (invalid input)"`) from low-confidence (`<45%`) banners, guarded against double-click duplicate requests (`inFlightRef`), retained `"Low-confidence estimate (<45%)"`, and kept `"Candidate"` absent from all visible UI copy.

### P1 — Post-Hoc Probability Calibration
- Created [`calibration_utils.py`](../calibration_utils.py) and fitted leakage-free `dev_cal` logit calibrations evaluated on `dev_val`:
  - **Speech V1 + P1 (`models/speech_emotion/wav2vec2/calibration_p1.json`):** `temp_plus_l2_bias` (`T=0.9767, reg=0.05`) improved `dev_val` Accuracy (`46.35% -> 47.26%`), Macro-F1 (`24.50% -> 24.58%`), ECE (`0.0638 -> 0.0602`), and NLL (`1.4995 -> 1.4834`).
  - **Facial V1:** Rejected post-hoc prior/L2 bias calibration after observing complete collapse of all `544 / 544` `dev_val` facial predictions into `Neutral` (`MF1 17.96% -> 8.41%`).

### P2 — V2 Model Retraining & Disjoint `dev_val` Comparison
- **Facial `ResNet18_v2` (`facial_emotion/train_resnet18_v2.py` -> `models/facial_emotion/resnet18_v2/resnet18_v2_best.pt`, `sha256: 1822ba9dc13a7406c13ad4bd48a7c8be7d309ead286cbe40352525bf85c1211a`, `facial-resnet18-v2-la_ce_tau040`):**
  - Expanded clean MELD `train` crops from `9,714` to `20,761` via multi-frame temporal extraction on non-neutral utterances + webcam-realistic augmentations + `LogitAdjustedCrossEntropyLoss(tau=0.40)`.
  - On untouched `dev_val` (`N=544`), uncalibrated `ResNet18_v2` boosted **Macro-F1 `+3.56 pp` (`17.96% -> 21.52%`)**, **Weighted-F1 `+1.14 pp` (`32.17% -> 33.31%`)**, cut **ECE nearly in half (`0.0934 -> 0.0473`)**, increased mean confidence (`29.6% -> 38.6%`), and reduced `<45%` warnings (`97.8% -> 76.7%`).
- **Speech `Wav2Vec2_v2` (`speech_emotion/train_wav2vec2_v2.py` -> `models/speech_emotion/wav2vec2_v2/best/`, `sha256: fbbb902b5d77138178f07c8dde847c0f853aed6c5fde04aa73b4a260ca477478`, `speech-wav2vec2-v2`):**
  - Trained with unified P0 preprocessing, `return_attention_mask=True`, acoustic augmentations (`-6 dB` to `+4 dB` gain jitter, additive noise, SpecAugment with `4800`-sample minimum floor), length-bucketed batching, and `LogitAdjustedCELoss(tau=0.45)`, paired with `dev_cal`-fitted `temp_plus_l2_bias` (`T=1.4159, reg=0.005` in `wav2vec2_v2/best/calibration.json`).
  - On untouched `dev_val` (`N=548`), `Wav2Vec2_v2 + L2 Calibration` beat both `V1 Raw` and `V1 + P1` across **every primary metric**: **Acc `47.26%`**, **Macro-F1 `26.26%`** (vs `24.50%` V1 / `24.58%` V1+P1), **Weighted-F1 `41.98%`** (vs `41.07%` / `40.91%`), **ECE `0.0537`** (vs `0.0638` / `0.0602`), **NLL `1.4510`** (vs `1.4995` / `1.4834`), **Brier `0.6791`** (vs `0.6946` / `0.6895`), and **`<45%` `40.1%`** (down `-11.5 pp` vs `51.6%`).
- **Multimodal Impact (Frozen Candidate 4 + Phase 18 Rules):**
  - On disjoint `dev_val` (`N=548`): **T+A+V** improved from Text-only (`58.58% Acc / 37.23% MF1 / 54.50% WF1`) and `V1+P1` (`58.76% Acc / 37.44% MF1 / 54.75% WF1`) to **`59.12% Acc / 37.82% MF1 / 55.21% WF1 / 0.0646 ECE / 1.2154 NLL`**.
  - On full MELD `dev` (`N=1,108`): **T+A+V** reached **`59.12% Acc / 39.04% MF1 / 55.79% WF1 / 0.0546 ECE`**.

---

## 3. Production API Wiring & Instant Rollback Controls

Updated [`api/config.py`](../api/config.py), [`api/model_registry.py`](../api/model_registry.py), [`api/schemas.py`](../api/schemas.py), [`facial_emotion/resnet_predictor.py`](../facial_emotion/resnet_predictor.py), [`speech_emotion/wav2vec2_model.py`](../speech_emotion/wav2vec2_model.py), and [`.env.staging.example`](../.env.staging.example):

- **Default Serving Stack:** `Wav2Vec2_v2` (`speech-wav2vec2-v2`, `calibrated=True`) + `ResNet18_v2` (`facial-resnet18-v2-la_ce_tau040`, `calibrated=False`, SHA-256 verified against `PRIMARY_RESNET_V2_SHA256`) + `DistilBertTextPredictor`.
- **Zero-Code Environment Variable Rollback:**
  - Roll speech back to `V1 + P1`: set `EMOTION_API_AUDIO_MODEL_DIR=models/speech_emotion/wav2vec2/best` (automatically resolves `calibration_p1.json`).
  - Disable speech calibration (`Wav2Vec2_v2 Uncal` / `V1 Raw`): set `EMOTION_API_USE_SPEECH_CALIBRATION=0`.
  - Roll facial back to `V1`: set `EMOTION_API_VIDEO_MODEL_PATH=models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` (automatically matches `PRIMARY_RESNET_SHA256`) or set `EMOTION_API_FACIAL_BACKEND=legacy_cnn`.

---

## 4. Final Post-Edit Verification Results

1. **Full Python Test Suite:** `python -m pytest -q` -> **`314 passed, 0 failed, 6 warnings in 116.11s`** (includes P0/P1/P2 unit tests, V2 default `/models/status` & inference checks, `test_v1_plus_p1_env_rollback_configuration_and_registry`, and Phase 18–20 suites).
2. **Frontend Unit Tests & Production Build:**
   - `node --test frontend/tests/api.test.mjs` -> **`11 passed, 0 failed`**
   - `npm.cmd --prefix frontend run build` (`tsc -b && vite build`) -> **succeeded** (`dist/assets/index-DduZs2ec.js`, `208.08 kB`).
   - Source/dist verification confirmed `"Low-confidence estimate (<45%)"` intact and zero occurrences of `"Candidate"` in rendered UI files.
3. **Live Server CUDA End-to-End Verification (`http://127.0.0.1:8000` & `http://localhost:5173/`):**
   - `/health` & `/ready`: `200 OK`, canonical 7 classes.
   - `/models/status`: `text` (`DistilBertTextPredictor`, `cuda`, `calibrated=False`), `audio` (`speech-wav2vec2-v2`, `cuda`, `calibrated=True`), `video` (`facial-resnet18-v2-la_ce_tau040`, `cuda`, `calibrated=False`).
   - `/predict/text`: `200 OK` (`36.2 ms`, `"surprise" 94.0%`).
   - `/predict/audio` (`dev` `dia0_utt0.wav`): `200 OK` (`162.5 ms` cold / `29.4 ms` warmed, repeated identical call `repeat_max_abs_diff = 0.0`); silent WAV -> `422`; invalid MIME -> `415`.
   - `/predict/video` (`dev` `dia0_utt0.mp4`): `200 OK` (`1265.1 ms`, `21` faces across `8` sampled frames, `100%` coverage, `entropy = 1.7830 < 1.85`); blank video -> `422`.
   - `/predict/multimodal` (3-modality): `200 OK` (`699.8 ms`, `modality_weights = {text: 0.92, audio: 0.0, video: 0.08}`, `rule_applied = "text_anchor_default+validated_video_calibrated"`).
   - `http://localhost:5173/`: `200 OK`. *(Interactive Chrome DevTools browser-subagent check did not run because Chrome remote debugging / `DevToolsActivePort` was not enabled on the host.)*

