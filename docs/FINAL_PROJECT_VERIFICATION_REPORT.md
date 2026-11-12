# Final Project Verification & Academic Summary Report

**Project:** Multimodal Emotion Recognition System (Final-Year CSE/AIML Academic Project)  
**Date:** 2026-11-12  
**Overall Readiness Status:** **READY WITH KNOWN LIMITATIONS** *(ready for local college academic demonstration; hardware/dataset and unrun manual browser checks documented in Section 9)*

---

## 1. Project Architecture
An end-to-end three-modality affective computing web application recognizing 7 emotions (`anger, disgust, fear, joy, neutral, sadness, surprise`) using late text-anchored multimodal fusion:
- **Frontend Layer:** React + Vite + TypeScript + Tailwind CSS UI (`frontend/src/`), supporting Multimodal Studio (`Text`, `WAV/Mic Audio`, `MP4/WebM/Webcam Video`) plus single-modality tabs, live microphone/camera capture, signal quality context, and model health/device status.
- **API Layer:** FastAPI + Uvicorn (`api/`), managing thread-safe singleton model loading, strict upload/duration/MIME validation, sliding-window rate limiting, temporary-file cleanup, Prometheus `/metrics`, and timeout handling.
- **ML Inference Layer:** GPU/CPU DistilBERT (`text_emotion/`), Wav2Vec2 (`speech_emotion/`), MTCNN + ResNet18 (`facial_emotion/`), logit calibration utilities (`calibration_utils.py`), and Candidate 4 + Phase 18 bounded fusion (`api/fusion.py`, `api/multimodal_extension.py`).

## 2. Technologies and Libraries
- **ML / Deep Learning:** PyTorch, Transformers (`distilbert-base-uncased`, `facebook/wav2vec2-base`), TorchVision (`ResNet18`), `facenet-pytorch` (`MTCNN`), OpenCV (`cv2`), SoundFile, NumPy, scikit-learn, SciPy.
- **Backend:** Python 3.12 (Anaconda), FastAPI, Pydantic, Uvicorn, Starlette, `httpx` (tests).
- **Frontend:** React 18, TypeScript, Vite 6, Tailwind CSS, Lucide icons, Recharts.
- **Hardware Used:** AMD Ryzen 7 7445HS, 16 GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), Windows 11.

## 3. Machine Learning and Deep Learning Models
- **Text (`DistilBERT`):** Fine-tuned 6-layer distilled BERT sequence classifier (`267.8 MB`).
- **Speech (`Wav2Vec2_v2`):** Self-supervised speech encoder fine-tuned with gain/noise/SpecAugment + `LogitAdjustedCELoss(tau=0.45)` + post-hoc `temp_plus_l2_bias` (`T=1.4159, reg=0.005`).
- **Facial (`ResNet18_v2 + MTCNN`):** ImageNet-pretrained ResNet18 fine-tuned on `20,761` clean MELD multi-frame crops (`96x96` unaligned square crops) with webcam-realistic augmentations and `LogitAdjustedCrossEntropyLoss(tau=0.40)`.

## 4. Dataset and Preprocessing
- **Dataset:** MELD (*Multimodal EmotionLines Dataset*, `train`, `dev`, held-out `test`), mapped strictly to the canonical 7-class taxonomy. For P1 calibration and P2 model selection, `dev` was split cleanly by dialogue into `dev_cal` (`560` utterances) and `dev_val` (`548` utterances).
- **Speech Preprocessing:** 16 kHz mono resampling, finite waveform checks, gain-invariant leading/trailing silence trimming (`95th percentile - 32.0 dB`), `8.0 s` maximum duration cap, `-60.0 dB` RMS silent-audio rejection (`HTTP 422`), and `return_attention_mask=True`.
- **Facial Preprocessing:** Sequential video decode (`cap.grab()/retrieve()`, `8` uniform frames), MTCNN detection (`min_prob=0.90`, `min_face_size=24`, aspect `[0.40, 1.80]`), `96x96` square RGB ImageNet normalization, strict IEEE FP32 guard on CUDA, and `raw_all_detections_mean_prob` temporal aggregation.

## 5. Training Methodology & Candidate 4 Fusion
- **P2 Retraining Discipline:** Both V2 models trained on MELD `train` with epoch/variant selection on `dev_cal` and single-shot verification on disjoint `dev_val`. Original v1 checkpoints and held-out `test` data remained completely untouched.
- **Candidate 4 Fusion (`api/fusion.py`):** Frozen parameters `tau_a = 0.20`, `lambda_max = 0.40`, `kappa_a = 1.50`, `lambda_v = 0.00` keep Text authoritative while allowing acoustic prosody to resolve Anger and Surprise without degrading minority classes (`Fear`, `Disgust`, `Sadness`, `Joy`). Phase 18 (`api/multimodal_extension.py`) adds bounded facial blending (`w_v = 0.08`) only when `coverage >= 0.40` and `entropy <= 1.85`.

## 6. Summary Experimental Results & Final Selection
- Detailed evaluation metrics, confusion matrices, per-class F1 scores, and calibration statistics are recorded in [`FINAL_MODEL_EVALUATION_REPORT.md`](FINAL_MODEL_EVALUATION_REPORT.md) and [`FINAL_MODEL_INTEGRATION_REPORT.md`](FINAL_MODEL_INTEGRATION_REPORT.md).
- Selected active pipeline: `DistilBERT` + `Wav2Vec2_v2 + L2 Cal` + `ResNet18_v2`, reaching **59.12% Acc / 37.82% Macro-F1** on disjoint `dev_val` (`N=548`) and **59.12% Acc / 39.04% Macro-F1** on full `dev` (`N=1,108`), while preserving frozen **63.37% Acc / 41.93% Macro-F1** Phase 16 `T+A` performance on `test` via env-var rollback.

## 7. Backend & Frontend Implementation
- **Backend Endpoints Verified Live (`http://127.0.0.1:8000`):** `/health`, `/ready`, `/models/status` (exposing `model_identifier`, `device`, `checkpoint_sha256`, `calibrated`), `/predict/text`, `/predict/audio`, `/predict/video`, `/predict/multimodal`, and `/metrics`. Enforces MIME/size/duration/silence/no-face validation, sliding-window rate limits, timeouts, and zero-leak temp-file cleanup.
- **Frontend Verified (`http://localhost:5173/`):** Displays real FastAPI predictions, `1`-decimal confidence badges and probability bars, `"Low-confidence estimate (<45%)"` banner, acoustic/facial quality panels, double-submit locking, validated input rejection guidance, webcam/mic capture (`LiveCapture.tsx`), and removes internal `"Candidate"` wording from user-facing text.

## 8. Testing & Verification Results
- **Python Backend & ML Suite (`python -m pytest -q`):** **314 passed, 0 failed, 6 warnings in 116.11s** (covering model predictors, P0/P1/P2 unit tests, V2 defaults + V1+P1 env rollback tests, API security/staging tests, and multimodal fusion contracts).
- **Frontend Unit Tests (`node --test frontend/tests/api.test.mjs`):** **11 passed, 0 failed**.
- **Frontend Production Build (`npm.cmd --prefix frontend run build`):** Succeeded (`tsc -b && vite build`, `dist/assets/index-DduZs2ec.js`, `208.08 kB`).
- **Live Server CUDA E2E Verification (`http://127.0.0.1:8000` & `http://localhost:5173/`):** Verified `/health` (`200`), `/ready` (`200`), `/models/status` (`text`, `audio`, `video` on `cuda`), all 7 multimodal routes (`200 OK`), repeated identical audio determinism (`repeat_max_abs_diff = 0.0`), silent WAV (`422`), invalid MIME (`415`), blank MP4 (`422`), and `http://localhost:5173/` (`200`).

## 9. Known Limitations & Blocked Items
1. **Facial Single-Modality Accuracy Ceiling (`34.38%` Acc / `21.52%` MF1 on `dev_val`):** Multi-speaker sitcom shots, low `96x96` crop resolution, ImageNet initialization, static per-frame averaging, and rare MELD `disgust`/`fear` visual cues limit standalone face accuracy (`disgust` F1 = `0.0%`). Phase 18 entropy/coverage gating protects multimodal predictions from unreliable faces.
2. **Speech Minority Rarity (`disgust` / `fear`):** Both models score `0.0%` standalone F1 on ultra-rare `disgust` (`11` `dev_val` samples) and `fear` (`20` samples) speech segments due to extreme class skew in MELD audio.
3. **Interactive Headless Chrome Subagent Check (`BLOCKED` in this environment):** Automated Chrome DevTools browser-subagent inspection could not attach (`DevToolsActivePort` missing / remote debugging inactive on Chrome); live HTTP serving of the built Vite app and API JSON contracts were verified directly instead, and physical mic/webcam live capture requires user interaction in a browser.
4. **Test-Split Facial/V2 Evaluation:** Held-out MELD `test` (`2,610` samples) was not re-run with V2 checkpoints or raw MP4 video decoding in order to preserve test-split discipline and avoid unnecessary multi-minute GPU/video runs; Phase 16 `T` / `A` / `Candidate 4` test results are cited separately.

## 10. Future Work (Proposed Facial V3 Roadmap — Not Implemented)
1. **Expression-pretrained facial backbone** (e.g., AffectNet/RAF-DB HSEmotion or VGGFace2 encoder instead of generic ImageNet ResNet18).
2. **Higher input resolution (`160x160`–`224x224`) with 5-landmark affine face alignment** using MTCNN eye/nose/mouth coordinates.
3. **Active-speaker / speaking-track selection** via temporal mouth-region motion to avoid averaging silent background listeners.
4. **Temporal attention / sequence modeling** (BiGRU or lightweight Transformer over frame embeddings) to separate articulation from emotion.
5. **External facial expression pretraining** (RAF-DB / FERPlus) followed by cross-modal noise-filtered MELD adaptation.
