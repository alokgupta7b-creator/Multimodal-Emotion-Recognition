# Phase 17B-0.1 — Final Integration Verification Report

**Date:** November 11, 2026  
**Scope:** End-to-end verification of live speech/video capture, Windows OpenCV/FFmpeg backend fixes, frontend build/layout parity, regression suite, checkpoint integrity, CPU/CUDA parity, and Phase 17B-0 isolation status.  
**Policy Compliance:** Held-out `test` split untouched; production and experimental model checkpoints unmodified; Candidate 4 fusion preserved; Experiments 4–7 not started; no public deployment.

---

## 1. Executive Summary (PASS / FAIL Matrix)

| Verification Area | Status | Summary of Observed Evidence |
| :--- | :---: | :--- |
| **1. Backend Service & Suites** | **PASS** | Restarted `uvicorn` on `127.0.0.1:8000` (`cuda`, `/health` ready); API suite **`58/58` passed** (`20.86s`); full Python suite **`271/271` passed** (`59.64s`). |
| **1a. Live Uploads, Errors & Cleanup** | **PASS** | Verified live MP4 (`200`), VP9 WebM (`200`), audio-only MP4 fast rejection (`422` in `6.08 ms`), chunked audio (`200`), chunked multimodal text+audio (`200`), oversized `>50 MB` rejection (`413`), and `0` leaked temp files. |
| **2. Physical-Device Browser Testing** | **PASS** | Real `AMD Audio Device` mic -> 16 kHz mono PCM WAV -> playback -> Speech (`200`) & Candidate 4 (`200`); real `HP Wide Vision HD Camera (0408:5482)` -> H.264 MP4 & VP9 WebM playback -> Video (`200`). *(See UX caveat §6.1 on OS virtual camera precedence.)* |
| **3. Checkpoints, Regression & CPU/CUDA** | **PASS** | All 3 production + 3 Phase 17A experimental SHA-256 digests match baseline byte-for-byte. Text, Speech, Candidate 4, and Facial models match across CPU and RTX 4050 CUDA. *(See §4.3 on `facial_emotion/remediation.py` isolation status.)* |
| **4. Frontend Build, Layout & Console** | **PASS** | `npm.cmd run build` (`tsc -b && vite build`) exited `0` in `5.08s`; zero overflow on Desktop (`1440x960`) and Mobile (`375x812`); `0` page errors, `0` console errors, `0` failed network requests. |

---

## 2. Backend Verification

### 2.1 Live Server State
- **Command:** `python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --log-level info` with `EMOTION_API_DEVICE=auto`, `EMOTION_API_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`.
- **`GET /health`:** `{"status": "ok", "ready": true, "classes": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]}`
- **`GET /models/status`:**
  - `text`: `DistilBertTextPredictor`, `loaded: true`, `device: "cuda"`, `error: null`
  - `audio`: `Wav2Vec2Predictor`, `loaded: true`, `device: "cuda"`, `error: null`
  - `video`: `FacialEmotionPredictor`, `loaded: true`, `device: "cuda"`, `error: null`

### 2.2 Automated Python Test Suites

| Scope | Command | Exact Result | Duration |
| :--- | :--- | :--- | :---: |
| **API Test Suite** | `pytest api/tests/ -v` | **`58 passed`, `0 failed`, `0 skipped`, `3 warnings`** | `20.86s` |
| **Complete Python Suite** | `pytest --cache-clear -v` | **`271 passed`, `0 failed`, `0 skipped`, `6 warnings`** | `59.64s` |

Breakdown of the **271** executed Python tests across modules:
- `api/tests/test_api.py`: **14 passed** (including WebM upload & `test_audio_only_mp4_is_rejected_without_hanging`)
- `api/tests/test_fusion.py`: **20 passed**
- `api/tests/test_multimodal_api.py`: **24 passed**
- `datasets/CMU-MOSEI/tests/test_prepare_dataset.py`: **12 passed**
- `facial_emotion/tests/test_facial.py`: **13 passed**
- `facial_emotion/tests/test_remediation.py`: **6 passed**
- `preprocessing/tests/`: **90 passed** (`test_alignment.py`: 27, `test_csd_reader.py`: 11, `test_preprocessing.py`: 52)
- `speech_emotion/tests/`: **86 passed** (`test_audio_loader.py`: 13, `test_feature_extraction.py`: 16, `test_label_mapping.py`: 8, `test_metrics.py`: 4, `test_model.py`: 16, `test_mosei_training.py`: 7, `test_preprocess_audio.py`: 6, `test_training_data.py`: 7, `test_training_pipeline.py`: 5, `test_wav2vec2.py`: 4)
- `text_emotion/tests/`: **6 passed** (`test_distilbert.py`: 3, `test_stabilization.py`: 3)

### 2.3 Live Restarted HTTP Upload, Streaming & Cleanup Verification
Executed directly against `http://127.0.0.1:8000` (clean OS temp-directory diff before/after):

| Check | Endpoint | Observed Response | Latency |
| :--- | :--- | :--- | :---: |
| **MP4 Video Upload** (`WIN_20261111_03_07_13_Pro.mp4`) | `POST /predict/video` | **HTTP 200** — `neutral` (`0.1518`), `num_faces: 52`, `num_sampled_frames: 17`, `coverage: 1.0`, `entropy: 1.9436` | `4,429.88 ms` |
| **WebM VP9 Video Upload** (`diag_short_vp9.webm`) | `POST /predict/video` | **HTTP 200** — `anger` (`0.1649`), `num_faces: 4`, `num_sampled_frames: 1`, `coverage: 1.0`, `entropy: 1.9388` | `692.02 ms` |
| **Audio-Only MP4 Rejection** (`audio_only_aac.mp4`) | `POST /predict/video` | **HTTP 422** — `{"detail": "Video could not be decoded."}` (no `CAP_MSMF` hang) | `6.08 ms` |
| **Chunked Transfer Upload (Audio)** (`dev/dia0_utt0.wav`) | `POST /predict/audio` | **HTTP 200** — `surprise` | `179.06 ms` |
| **Chunked Transfer Upload (Multimodal Text + Audio)** | `POST /predict/multimodal` | **HTTP 200** — `surprise`, `modalities_available: {text: true, audio: true, video: false}` | `204.35 ms` |
| **Oversized Request Header** (`Content-Length: 57671680`) | `POST /predict/audio` | **HTTP 413** — `{"detail": "Request payload exceeds maximum allowed size (52428800 bytes)."}` | — |
| **Temporary File Cleanup** | OS `%TEMP%` | **`0` newly retained `.wav`/`.mp4`/`.webm` temp files**; server log shows clean `200`/`422`/`413` responses with zero exceptions. | — |

---

## 3. Physical-Device Browser Verification (No Fake Media Flags)

Headless Chrome was launched without `--use-fake-device-for-media-stream` against the live frontend (`http://localhost:5173`) and restarted backend (`http://127.0.0.1:8000`).

### 3.1 Enumerated Windows Hardware Devices
- **Audio Inputs:**
  - `Default - Microphone Array (AMD Audio Device)`
  - `Communications - Microphone Array (AMD Audio Device)`
  - `Microphone Array (AMD Audio Device)`
- **Video Inputs:**
  - `OPPO A78 5G (Windows Virtual Camera)` *(OS index 0)*
  - `HP Wide Vision HD Camera (0408:5482)` *(integrated physical laptop camera, index 1)*
- **Audio Outputs:** `Default / Communications / Speaker (Realtek(R) Audio)`

### 3.2 Physical Microphone & Candidate 4 Results (UI End-to-End)
- **Live Microphone Recording & 16 kHz Mono PCM WAV Conversion:**
  - Header verified in browser memory: `RIFF` / `WAVE`, `audioFormat = 1` (PCM), `channels = 1`, `sampleRate = 16000`, `bitsPerSample = 16`, `totalBytes = 49,964`, `dataBytes = 49,920` (`1.56 s`).
  - Live signal levels: `rms = 0.010036`, `peak = 0.275909`.
  - HTML5 `<audio>` playback check: `error = null`.
- **Live Speech Inference (`/predict/audio`):** **HTTP 200** — predicted `neutral`.
- **Candidate 4 Text + Live Microphone (`/predict/multimodal`):** **HTTP 200** — predicted `anger`, `modalities_available: {text: true, audio: true, video: false}`, `exclusion_reasons: {audio: null, video: "outside_production_path"}`, `rule_applied: "text_anchor_default"`, `modality_weights: {text: 1, audio: 0, video: 0}`.

### 3.3 Physical Camera Preview, Playback & Video Inference Results
Because `OPPO A78 5G (Windows Virtual Camera)` sits at OS index 0 ahead of `HP Wide Vision HD Camera (0408:5482)`, both devices were probed explicitly:

| Camera Device & Codec | Capture Telemetry | HTML5 `<video>` Playback | `/predict/video` Result |
| :--- | :--- | :--- | :--- |
| **UI Default (`facingMode: "user"`) -> `OPPO A78 5G (Windows Virtual Camera)`** *(Run 1 — idle virtual cam)* | `1280x720`, `meanLuma: 1.44`, `avc1` MP4 `49,070 B` (`ftypisom`) | `loaded: true`, `1280x720`, `1.814s`, `error: null` | **HTTP 422** (`"No detectable faces were supplied."`) |
| **Default Camera `OPPO A78 5G`** *(Run 2 — container/codec probe)* | `1280x720`, `meanLuma: 1.43`:<br>- `avc1` (`29,497 B`, audio-only track)<br>- `vp9,opus` (`302,791 B`, `1a45dfa3...`)<br>- `vp8,opus` (`549,299 B`, `1a45dfa3...`) | - `avc1`: `0x0`, `error: null`<br>- `vp9`: `1280x720`, `error: null`<br>- `vp8`: `1280x720`, `error: null` | - `avc1`: **HTTP 422** (`"Video could not be decoded."`)<br>- `vp9`: **HTTP 200** (`neutral`, `1` face)<br>- `vp8`: **HTTP 200** (`anger`, `12` faces) |
| **Integrated Physical USB Webcam `HP Wide Vision HD Camera (0408:5482)`** (`avc1` MP4) | `1280x720` @ 30fps, `meanLuma: 116.64`, `684,808 B` (`ftypisom`) | `loaded: true`, `1280x720`, `duration: 1.794s`, **`error: null`** | **HTTP 200** — `anger` (`0.1576`), `num_faces: 12`, `num_sampled_frames: 4`, `entropy: 1.9415`, `1,787.17 ms` |
| **Integrated Physical USB Webcam `HP Wide Vision HD Camera (0408:5482)`** (`vp9,opus` WebM) | `1280x720` @ 30fps, `meanLuma: 116.64–132.38`, `540,469 B` (`1a45dfa3...`) | `loaded: true`, `1280x720`, **`error: null`** *(WebM live metadata `duration: null`)* | **HTTP 200** — `anger` (`0.1586`), `num_faces: 18`, `num_sampled_frames: 6`, `entropy: 1.9411`, `1,257.81 ms` |

---

## 4. Regression, Checkpoint Integrity & CPU/CUDA Verification

### 4.1 Checkpoint SHA-256 & Size Verification

| Role / Artifact Path | Size (Bytes) | Observed SHA-256 | Baseline Match |
| :--- | :---: | :--- | :---: |
| Production Text (`models/text_emotion/distilbert/best/model.safetensors`) | `267,847,948` | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | **EXACT MATCH** |
| Production Speech (`models/speech_emotion/wav2vec2/best/model.safetensors`) | `378,307,452` | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` | **EXACT MATCH** |
| Production Facial (`models/facial_emotion/facial_cnn_best.pt`) | `380,128` | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | **EXACT MATCH** |
| Archive Phase 9 Facial (`models/facial_emotion_phase9/facial_cnn_best.pt`) | `380,128` | `ca52669714b53f6b7206eceab34d6fc2b7d1f4091494705393d6444aa2d365a8` | Recorded |
| Archive Phase 9 v2 Facial (`models/facial_emotion_phase9_v2/facial_cnn_best.pt`) | `380,128` | `578c7fb09aefbee792bf45298d6be26f15a1b507faabedb123b2b6fe1a6ddad0` | Recorded |
| Phase 17A Exp 1 (`models/facial_experiments/exp1_corrected_cnn/exp1_corrected_cnn_best.pt`) | `380,096` | `a2be816a67bb094026bdf43fb31884ee11812882de5f2c8108f26262b0f89613` | **EXACT MATCH** |
| Phase 17A Exp 2 (`models/facial_experiments/exp2_weighted_cnn/exp2_weighted_cnn_best.pt`) | `380,212` | `422a5934c5f1337733eeb9c5274f4bb52f19d1ad81b7d1656b0eb2d25cc8aa8a` | **EXACT MATCH** |
| Phase 17A Exp 3 (`models/facial_experiments/exp3_resnet18/exp3_resnet18_best.pt`) | `44,796,776` | `106b524e837417373dd904518da3ad389287c843d93c7f92596c5b1daf2b1446` | **EXACT MATCH** |

### 4.2 CPU vs. CUDA Parity Verification
Verified sequentially on PyTorch `2.6.0+cu124` (`CPU` vs. `NVIDIA GeForce RTX 4050 Laptop GPU`):

| Component & Input | CPU Prediction (Top Prob) | CUDA Prediction (Top Prob) | Max `|p_cpu - p_cuda|` | Status |
| :--- | :--- | :--- | :---: | :---: |
| **DistilBERT Text** (`5` deterministic utterances) | `joy` (`0.9362`), `anger` (`0.8111`), `surprise` (`0.8720`), `sadness` (`0.7294`), `neutral` (`0.9724`) | Identical 5/5 predicted labels | `2.607e-7` | **PASS** |
| **Wav2Vec2 Speech** (`dev/dia0_utt0.wav`) | `surprise` (`0.29763`), `387.58 ms` | `surprise` (`0.29775`), `443.43 ms` | `1.228e-4` | **PASS** |
| **Candidate 4 Fusion** (`protected_class_lockout_joy` + `dev/dia0_utt0.wav`) | `joy` (`0.9362`), weights `{text: 1.0, audio: 0.0, video: 0.0}` | `joy` (`0.9362`), identical weights | `9.155e-9` | **PASS** |
| **Candidate 4 Fusion** (`neutral` + `dev/dia0_utt0.wav`, `margin_audio=0.1552 <= 0.20`) | `neutral` (`0.9724`), rule `text_anchor_default` | `neutral` (`0.9724`), rule `text_anchor_default` | `3.476e-8` | **PASS** |
| **Candidate 4 Fusion** (`neutral` + synthetic `anger=0.85`, `margin_audio=0.75 > 0.20`) | `neutral` (`0.6234`), `anger` boosted to `0.3426`, `acoustic_synergy_anger`, `weights={text: 0.6, audio: 0.4, video: 0.0}` | — (pure NumPy) | — | **PASS** |
| **Production Facial CNN** (`facial_emotion/model.py`, `WIN_20261111_03_07_13_Pro.mp4`) | `neutral` (`0.151758`), entropy `1.9436`, `17` reported frames | `neutral` (`0.151758`), entropy `1.9436`, `17` reported frames | `2.486e-7` | **PASS** |
| **Remediated Facial CNN** (`facial_emotion/remediation.py`, `LEGACY_CNN`, `downscale=2.0`) | `anger` (`0.724976`), entropy `0.9528`, `frame_accounting={244, 244, 82, 52, 53, 52}` | `anger` (`0.724981`), entropy `0.9528`, identical frame accounting | `6.931e-6` | **PASS** |

### 4.3 Facial Normalization & Frame-Accounting Isolation Status
- **`facial_emotion/remediation.py` (`RemediatedFacialEmotionPredictor` + `remediated_preprocess_face_crops`):** Verified cleanly (`6/6` tests in `facial_emotion/tests/test_remediation.py` plus CPU/CUDA checks above). On synthetic BGR `(0, 128, 255)`, `NormalizationType.LEGACY_CNN` outputs RGB channels `[+1.0, +0.00392, -1.0]` within `[-1.0, 1.0]`, and `VideoFrameAccounting` on `WIN_20261111_03_07_13_Pro.mp4` accurately records `total_video_frames=244`, `frames_decoded=244`, `frames_sampled=82`, `frames_with_detected_faces=52`, `total_face_detections=53`, `unique_tracked_faces=52`.
- **Production `facial_emotion/model.py` (`FacialEmotionPredictor`):** Per Phase 17B-0 isolation constraints ("Keep changes isolated until the audit passes"), `facial_emotion/model.py` has **not** yet been cut over to `remediation.py`. Production `/predict/video` therefore still exhibits the legacy `[0.0, 1.0]` normalization shift (`normalize_input=False`) and `round(len(faces)/sample_every)` sampled-frame calculation (`17` vs `82`) until final cutover approval.

---

## 5. Frontend Build, Responsive Layout & Runtime Telemetry

- **Production TypeScript + Vite Build (`npm.cmd run build`):**
  - `tsc -b && vite build` exited **`0`** in **`5.08 s`** (`1580 modules transformed`).
  - Emitted assets: `dist/index.html` (`0.76 kB` / gzip `0.45 kB`), `dist/assets/index-Bh7wuhXQ.css` (`17.80 kB` / gzip `4.40 kB`), `dist/assets/index-Bb9e3sb1.js` (`188.92 kB` / gzip `58.05 kB`).
- **Layout Responsiveness:**
  - Desktop (`1440x960`): `scrollWidth = 1440`, `innerWidth = 1440`, `overflow = false`.
  - Mobile (`375x812`): `scrollWidth = 375`, `innerWidth = 375`, `overflow = false`.
- **Runtime Error Audit (Physical-Device UI Run):**
  - `pageErrors`: `[]` (`0` errors)
  - `unexpectedConsoleErrors`: `[]` (`0` errors; SVG favicon added in `frontend/index.html` eliminated prior `/favicon.ico` 404)
  - `unexpectedNetFailures`: `[]` (`0` failed requests)

---

## 6. Remaining Limitations & Open Engineering Items

1. **No Camera / Microphone Device Selector Dropdown in `LiveCapture.tsx`:** Default `getUserMedia({ video: { facingMode: "user", ... } })` relies on OS camera ordering. On systems with virtual camera drivers installed (e.g., `OPPO A78 5G (Windows Virtual Camera)` at index 0), default live UI capture may bind to an idle virtual camera (`422`) rather than the physical webcam (`HP Wide Vision HD Camera`). Adding a `navigator.mediaDevices.enumerateDevices()` selector is recommended for a follow-up UI enhancement.
2. **Production `facial_emotion/model.py` Pending Phase 17B-0 Cutover:** While `facial_emotion/detector.py` (`open_video_capture` / `CAP_FFMPEG` Windows deadlock fix) and `api/app.py` (non-blocking threadpool execution) are live, the normalization and frame-accounting fix remains isolated in `facial_emotion/remediation.py` per Phase 17B-0 rules until audit sign-off.
3. **Browser Live-Recorded WebM Metadata Duration:** Chromium's `MediaRecorder` emits EBML WebM streams with `video.duration === Infinity`/`null` until seek/remux; OpenCV (`CAP_FFMPEG`) decodes the frames normally for inference.
4. **Acoustic Content Dependency for Speech/Multimodal Synergy:** Physical microphone recording captures real ambient/spoken room acoustics (`rms: 0.010036`); triggering Candidate 4 Rule 2 (`acoustic_synergy_anger`/`surprise`) requires vocal prosody exceeding `tau_audio = 0.20` over neutral.

