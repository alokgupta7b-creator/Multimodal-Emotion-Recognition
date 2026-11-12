# Multimodal Emotion Recognition System (Text, Speech, and Facial Video)

An end-to-end multimodal conversational emotion recognition application that predicts **7 canonical emotions** (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) across **text**, **speech audio (`.wav`)**, and **facial video (`.mp4` / `.webm`)** using fine-tuned deep learning models, post-hoc probability calibration, quality-gated multimodal fusion, a FastAPI backend, and a React + TypeScript web studio.

---

## 1. Tech Stack

| Layer | Technologies & Versions (Verified in Workspace) | Role |
| :--- | :--- | :--- |
| **Deep Learning & Models** | `torch==2.6.0`, `transformers==5.2.0`, `safetensors==0.6.2`, `facenet-pytorch==2.5.3` | DistilBERT (text), Wav2Vec2 (speech), MTCNN face detection + ResNet18 (facial) |
| **Audio & Vision Processing** | `librosa==0.11.0`, `soundfile==0.13.1`, `opencv-python==4.11.0.86`, `Pillow` | 16 kHz mono WAV loading, silence trimming, sequential video frame decoding, 96×96 RGB crops |
| **Scientific & ML Utilities** | `numpy==2.3.5`, `scipy==1.16.3`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `joblib==1.5.2`, `h5py==3.15.1` | Calibration optimization, manifest processing, evaluation metrics, classical baselines |
| **Backend API & Serving** | `fastapi==0.115.0`, `uvicorn==0.32.0`, `python-multipart==0.0.12` | REST inference endpoints, size/MIME/duration validation, rate limiting, Prometheus `/metrics` |
| **Frontend Web Studio** | `react==18.3.1`, `typescript==5.7.2`, `vite==6.0.5`, `tailwindcss==3.4.17`, `lucide-react==0.468.0` | Multi-tab UI, file uploads, in-browser mic (16 kHz PCM WAV) & webcam (`MediaRecorder`) capture |
| **Testing & Verification** | `pytest==8.3.4`, Node.js built-in test runner (`node --test`), `puppeteer-core==25.17.0` | 314 Python unit/integration/security tests, frontend API/retry tests, headless E2E scripts |
| **Staging & Monitoring** | `Docker`, `docker-compose.staging.yml`, `Nginx`, `Caddy`, `Prometheus`, `Grafana` | Containerized staging templates, reverse-proxy TLS config, and telemetry dashboards |

---

## 2. How It Is Implemented

The system separates training/benchmarking pipelines from the real-time serving stack (`api/` + `frontend/`), enforcing a unified 7-class MELD label order across every module: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.

### A. Unimodal Models & Calibration (`text_emotion/`, `speech_emotion/`, `facial_emotion/`)

1. **Text Emotion Recognition (`DistilBERT`)**
   - **Module:** [`text_emotion/distilbert_model.py`](text_emotion/distilbert_model.py) (`DistilBertTextPredictor`)
   - **Checkpoint:** [`models/text_emotion/distilbert/best/`](models/text_emotion/distilbert/best/config.json) (`DistilBertForSequenceClassification`, 6 transformer layers, `dim=768`, 12 attention heads)
   - **Implementation:** Normalizes input utterances via `preprocess_text()` before tokenization (`max_length=128`) and softmax classification across 7 classes. *(An earlier Phase 2 TF-IDF + Logistic Regression baseline is retained in `text_emotion/` for benchmark comparison.)*

2. **Speech Emotion Recognition (`Wav2Vec2 V2 + Calibration`)**
   - **Module:** [`speech_emotion/wav2vec2_model.py`](speech_emotion/wav2vec2_model.py) (`Wav2Vec2Predictor`) & [`speech_emotion/audio_loader.py`](speech_emotion/audio_loader.py)
   - **Checkpoint:** [`models/speech_emotion/wav2vec2_v2/best/`](models/speech_emotion/wav2vec2_v2/best/config.json) (`Wav2Vec2ForSequenceClassification` over `wav2vec2-base`, `hidden_size=768`, `classifier_proj_size=256`, trained with `LogitAdjustedCELoss`)
   - **Calibration:** [`models/speech_emotion/wav2vec2_v2/best/calibration.json`](models/speech_emotion/wav2vec2_v2/best/calibration.json) applies `temp_plus_l2_bias` (`T = 1.1956`, `regularization = 0.05`) fitted on unpadded `batch_size=1` `dev_cal` logits to match live runtime inference 1:1.
   - **Implementation:** Loads 16 kHz mono PCM WAV audio (`<= 60.0 s`, RMS `>= -60.0 dB`), trims leading/trailing silence (`trim_leading_trailing_silence()`), bounds processed speech to `8.0 s` (`preprocess_speech_waveform()`), enforces `return_attention_mask=True`, and surfaces acoustic telemetry (`audio_duration_s`, `processed_duration_s`, `trimmed_silence_s`, `audio_rms_db`, `low_volume_warning`).

3. **Facial Emotion Recognition (`MTCNN + ResNet18 V2 + Calibration`)**
   - **Modules:** [`facial_emotion/mtcnn_detector.py`](facial_emotion/mtcnn_detector.py) & [`facial_emotion/resnet_predictor.py`](facial_emotion/resnet_predictor.py) (`ResNetFacialEmotionPredictor`)
   - **Checkpoint:** [`models/facial_emotion/resnet18_v2/resnet18_v2_best.pt`](models/facial_emotion/resnet18_v2/calibration.json) (`ResNet18` trained on multi-frame clean MELD crops with webcam augmentations and `LogitAdjustedCrossEntropyLoss(tau=0.40)`)
   - **Calibration:** [`models/facial_emotion/resnet18_v2/calibration.json`](models/facial_emotion/resnet18_v2/calibration.json) applies `crop_prior_shift_a025_temp` (`alpha = 0.25`, `T = 1.25`) to correct Neutral under-representation from multi-frame training crops.
   - **Implementation:** Samples 8 frames uniformly (using single-pass sequential `cap.grab()`/`cap.retrieve()` when `<= 240` frames), detects faces via `MTCNN` (`640px` fast pass with native fallback and false-positive aspect/probability filtering), extracts square `margin=0.08` crops resized to `96x96` ImageNet RGB, enforces IEEE FP32 guard (`allow_tf32=False`), verifies SHA-256 checksums on startup with automatic fallback to `exp5_focal_g3_wt_best.pt` (or optional `legacy_cnn` rollback), and aggregates crop predictions (`raw_all_detections_mean_prob`) while reporting `num_faces`, `num_sampled_frames`, `face_detection_coverage`, and `prediction_entropy`.

### B. Quality-Gated Multimodal Fusion (`api/fusion.py`, `api/multimodal_extension.py`)

Instead of training an unconstrained black-box fusion network that could let noisy acoustic/visual frames overwrite clear linguistic evidence, fusion uses a deterministic, auditable two-stage engine ([`ExtendedMultimodalFusionEngine`](api/multimodal_extension.py)):

1. **Text-Anchored Protected Class Safeguard:** When Text predicts subtle/minority classes (`fear`, `disgust`, `sadness`, `joy`), Text authority is preserved (`w_text = 1.0` before optional video check).
2. **Acoustic Synergy Injection:** When Text is non-protected (e.g., `neutral`) and Speech strongly predicts `anger` or `surprise` with margin $M_a = p_a[\text{pred}] - p_a[\text{neutral}] > 0.20$, Speech receives bounded weight $\lambda_a = \min(0.40, 1.50 \cdot M_a)$.
3. **Quality-Gated Visual Blend:** When Video passes quality checks (`face_detection_coverage >= 0.40` and `prediction_entropy <= 1.85`), calibrated facial probabilities receive bounded weight `w_v = 0.08` (rescaling Text/Audio weights by `0.92`); if Video fails quality thresholds, it is kept as advisory telemetry with `w_v = 0.0`.
4. **No-Text Fallbacks:** Supports `Audio` only, `Video` only, and `Audio + Video` seamlessly without fabricating placeholder probability vectors.

### C. Backend Hardening & Frontend Studio (`api/`, `frontend/`)

- **FastAPI Backend (`api/app.py`, `api/model_registry.py`, `api/validation.py`):**
  - Thread-safe lazy/eager `ModelRegistry` with per-modality `RLock`s and `/health`, `/ready`, `/models/status`, `/metrics`.
  - Streaming upload size enforcement (`50 MB` / `52,428,800` bytes, `HTTP 413`), text cap (`20,000` chars), `RIFF/WAVE` & `MP4/WebM` magic-byte verification (`HTTP 415`), silent/no-face guards (`HTTP 422`), sliding-window rate limiting (`120` requests / `60 s`, `HTTP 429`), request timeouts (`60 s`, `HTTP 504`), security headers (`nosniff`, `DENY`, `no-store`), and `finally`-block deletion of randomized OS temp files.
- **React + Vite Studio (`frontend/src/App.tsx`, `LiveCapture.tsx`, `api.ts`, `media.ts`):**
  - Four interactive modes: **Multimodal**, **Text**, **Speech Audio**, and **Facial Video**.
  - Client-side `50 MB` upload check, duplicate-request lock (`inFlightRef`), 60 s `AbortController` timeouts with Retry recovery, live microphone RMS dB guidance with browser-side 16 kHz PCM WAV encoding, camera device selector with explicit local-processing consent notice, 1-decimal probability bars (`formatProbabilityPercent`), and `"Low-confidence estimate (<45%)"` advisory notices.

---

## 3. How It Works (End-to-End Request Flow)

```text
User Input (React Studio: Text / WAV Upload or Live Mic / MP4-WebM Upload or Live Cam)
   │  [Client 50 MB size check + Consent gate + 60s AbortController timeout]
   ▼
FastAPI Gateway (api/app.py & api/validation.py)
   │  [Rate limit -> Streaming <= 50 MB check -> RIFF/WAVE or MP4/WebM magic-byte check -> Temp file]
   ▼
ModelRegistry (api/model_registry.py, per-modality RLock)
   ├── Text   ──► preprocess_text() -> DistilBERT ─────────────────────────► P_text  (7 probs)
   ├── Audio  ──► 16 kHz, -60 dB gate -> silence trim + 8s + mask -> Wav2Vec2 + Cal ► P_audio (7 probs)
   └── Video  ──► 8 frames -> MTCNN -> 96x96 RGB -> ResNet18 + Cal ───────► P_video + coverage/entropy
   │
   ▼
ExtendedMultimodalFusionEngine (api/fusion.py + api/multimodal_extension.py)
   │  [Protected class lockout / Acoustic synergy (<=0.40) / Video quality gate (w_v=0.08)]
   ▼
JSON Response + Immediate Temp Cleanup (`finally` unlink)
   └── React Studio renders top emotion, 1-decimal 7-class bars, modality weights & <45% advisory
```

---

## 4. Quickstart — Run Locally

### 1. Install Python Dependencies
Tested on `Python 3.10–3.13` (includes GPU PyTorch or CPU fallback, plus `facenet-pytorch` and `Pillow` for MTCNN):
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install facenet-pytorch==2.5.3 Pillow
```

### 2. Start the FastAPI Backend
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:EMOTION_API_DEVICE="auto"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
*(Optional: copy [`.env.example`](.env.example) to `.env` to customize ports, device, or limits.)*

### 3. Build & Start the React Frontend
```powershell
cd frontend
npm.cmd install
npm.cmd run build
npm.cmd run dev -- --host localhost --port 5173
```
Open **`http://localhost:5173`** in your browser. *(Browser microphone/camera permissions require `localhost` or `HTTPS`.)*

### Key API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service liveness and model readiness status |
| `GET` | `/ready` | Staging/container readiness probe (`200` / `503`) |
| `GET` | `/models/status` | Active device, identifier, calibration status, and SHA-256 metadata per modality |
| `GET` | `/metrics` | Prometheus-compatible request, latency, and modality counters (`api/telemetry.py`) |
| `POST` | `/predict/text` | DistilBERT prediction from JSON `{"text": "..."}` (max 20,000 chars) |
| `POST` | `/predict/audio` | Wav2Vec2 V2 calibrated prediction from `RIFF/WAVE` `.wav` upload (max 50 MB, `<= 60 s`) |
| `POST` | `/predict/video` | MTCNN + ResNet18 V2 calibrated prediction from `.mp4` / `.webm` upload (max 50 MB, `<= 60 s`) |
| `POST` | `/predict/multimodal` | Unified fusion across any combination of `text`, `file` (audio), and `video` |

### Zero-Code Environment Rollback Controls
- **Disable calibration:** `EMOTION_API_USE_SPEECH_CALIBRATION=0` or `EMOTION_API_USE_FACIAL_CALIBRATION=0`
- **Roll back Speech to V1 + P1:** `EMOTION_API_AUDIO_MODEL_DIR=models/speech_emotion/wav2vec2/best`
- **Roll back Facial to V1 or Legacy CNN:** `EMOTION_API_VIDEO_MODEL_PATH=models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` or `EMOTION_API_FACIAL_BACKEND=legacy_cnn`
- **Disable single-pass video decode:** `EMOTION_API_OPTIMIZED_VIDEO_PIPELINE=0`

---

## 5. Verified Evaluation & Test Summary

All post-remediation calibration and integration improvements were evaluated on MELD `dev` (`dev_cal` / `dev_val` dialogue-disjoint folds) while keeping the held-out MELD `test` split (`N = 2,610`) untouched (`docs/ACCURACY_RECOVERY_REPORT.md`):

| Modality / Fusion | Evaluation Split (`N`) | Accuracy | Macro-F1 | Weighted-F1 | ECE-15 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Text (`DistilBERT`)** | MELD `test` (`2,610`, Phase 16) / `full_dev` (`1,108`) | `63.10%` / `60.11%` | `41.55%` / `42.05%` | `60.77%` / `57.00%` | `0.1174` (`dev`) |
| **Speech (`Wav2Vec2_v2` + `bs=1 reg=0.05` Cal)** | MELD `full_dev` (`1,107`) | `49.86%` | `30.73%` | `47.07%` | `0.0310` |
| **Facial (`ResNet18_v2` + Prior/Temp Cal)** | MELD `full_dev` (`1,102`) | `38.66%` | `18.06%` | `33.29%` | `0.0531` |
| **Multimodal (`Text + Audio + Video`)** | MELD `full_dev` (`1,108`) | **`60.74%`** | **`42.55%`** | **`57.63%`** | **`0.0655`** |

### Run Automated Verification
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m pytest -q --tb=short
```
Latest verified suite run: **`314 passed, 0 failed, 0 skipped`** (plus frontend API/retry tests via `node --test frontend/tests/api.test.mjs` and TypeScript/Vite build via `npm.cmd --prefix frontend run build`).

---

## 6. Repository Structure

```text
.
├── api/                        # FastAPI app, config, validation, model registry, fusion & telemetry
├── text_emotion/               # DistilBERT predictor, preprocessing & TF-IDF baseline scripts
├── speech_emotion/             # Wav2Vec2 V1/V2 training, audio loader, silence trimming & prediction
├── facial_emotion/             # MTCNN detector, ResNet18 V1/V2, legacy CNN & video temporal pooling
├── calibration_utils.py        # Temperature, L2 bias & prior-shift post-hoc calibration utilities
├── frontend/                   # React + Vite + TypeScript Emotion Studio & LiveCapture UI
├── models/                     # Saved checkpoints, SHA-256 manifests & calibration JSONs
├── datasets/
│   ├── MELD/                   # Conversational Emotion Recognition (13,708 utterances, 7 emotions)
│   └── CMU-MOSEI/              # Monologue multimodal benchmark infrastructure & tests
├── deploy/                     # Nginx, Caddy & Prometheus/Grafana staging configs
├── scripts/                    # Dataset prep, validation benchmarks & E2E verification scripts
├── docs/                       # Architecture diagrams, phase reports & investigation/recovery logs
├── requirements.txt            # Core Python runtime dependencies
├── requirements-dev.txt        # Development & pytest dependencies
├── Dockerfile.backend          # Non-root staging backend container
└── docker-compose.staging.yml  # Full multi-service staging composition
```

---

## 7. Datasets & Academic Citations

All raw media files and large feature caches are excluded from Git version control. Prepare local benchmark data using the dataset guides in [`datasets/MELD/`](datasets/MELD/) and [`datasets/CMU-MOSEI/`](datasets/CMU-MOSEI/):

| Dataset | Discourse Context | Segments | Emotion Schema | Reference |
| :--- | :--- | :---: | :--- | :--- |
| **MELD** | Multi-party dialogue (*Friends*) | 13,708 utterances (1,433 dialogues) | 7 categorical (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) | [MELD Dataset Card](datasets/MELD/dataset_card.md) |
| **CMU-MOSEI** | In-the-wild monologues (YouTube) | 23,453 segments (3,294 videos) | 6 continuous intensities + continuous sentiment `[-3, +3]` | [CMU-MOSEI Dataset Card](datasets/CMU-MOSEI/dataset_card.md) |

```bibtex
@inproceedings{poria-etal-2019-meld,
    title = "{MELD}: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations",
    author = "Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Gautam, Gautam and Chhaya, Niyati and Cambria, Erik",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1050",
    doi = "10.18653/v1/P19-1050",
    pages = "527--536"
}

@inproceedings{zadeh2018multimodal,
    title = "Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Approaches",
    author = "Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1208",
    doi = "10.18653/v1/P18-1208",
    pages = "2236--2246"
}
```
