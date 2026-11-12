# Comprehensive Speech & Facial Emotion Recognition Accuracy and Pipeline Investigation

**Date:** 2026-11-12  
**Status:** Investigation Complete — Awaiting Explicit Approval Before Implementing Fixes  
**Scope:** Investigation-only audit of Speech (`Wav2Vec2`), Facial (`ResNet18 + MTCNN`), Text (`DistilBERT`), Multimodal Fusion, FastAPI Backend, and React Frontend (`zero_test_split_access = True`; no models, weights, thresholds, or source code were modified).

---

## 1. Executive Summary

### Why Text Appears Accurate While Speech and Facial Produce Uncertain Predictions

1. **Modal & Task Prior Advantage (`DistilBERT` vs `Wav2Vec2` vs `ResNet18`):**
   - On MELD `dev` (`N = 1,108`), [`DistilBertTextPredictor`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/text_emotion/distilbert_model.py) achieves **60.11% accuracy / 0.4205 Macro F1** with sharp probability distributions (mean confidence **71.73%**, only **14.62%** of clips `< 45%`), because lexical dialogue semantics strongly distinguish MELD emotions.
   - By contrast, acoustic prosody alone ([`Wav2Vec2Predictor`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/wav2vec2_model.py)) reaches **47.79% accuracy / 0.2534 Macro F1** on MELD `dev` (`N = 1,107`) with diffuse probabilities (mean confidence **46.33%**, median **44.30%**, **51.49%** `< 45%`), while static-frame sitcom facial recognition ([`ResNetFacialEmotionPredictor`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/resnet_predictor.py)) reaches **38.29% accuracy / 0.1722 Macro F1** (`N = 1,102`) with near-uniform posteriors (mean confidence **29.82%**, median **28.70%**, **98.19%** `< 45%`, mean entropy **1.7336 / 1.9459**).
   - **Crucially:** Text's confidence masks unimodal uncertainty during multimodal fusion ([`ExtendedMultimodalFusionEngine`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/multimodal_extension.py) keeps Text at `w >= 0.90` / `0.92` when Text is present), whereas standalone **Speech** and **Face** tabs expose those flat unimodal distributions directly.

2. **Confirmed Speech (`Wav2Vec2`) Defects:**
   - **Minority-class collapse from unweighted cross-entropy on imbalanced MELD (`4 epochs`, `lr=1e-5`):** `disgust` (`F1 = 0.0`, `0` dev predictions) and `fear` (`F1 = 0.0`, `0` dev predictions) collapsed completely; `neutral` receives `55.8%` of all predictions.
   - **Training-vs-inference preprocessing & duration mismatches:**
     - Training sets `max_duration_seconds = 8.0` and `silence_threshold_db = -200.0`, whereas runtime inference allows up to `60.0 s` without truncating to `8.0 s` and enforces `silence_threshold_db = -60.0 dB` (causing quiet clips `<-60 dB` to fail with HTTP `422`).
     - [`preprocessor_config.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/speech_emotion/wav2vec2/best/preprocessor_config.json) sets `"return_attention_mask": false` while training uses `padding=True` (measured on `N = 40` dev WAVs: `mean max_prob_diff = 0.0436`, `max = 0.2143`, **`4/40` argmax flips** between padded batch evaluation and unpadded runtime inference).
     - `"do_normalize": true` makes `Wav2Vec2` amplitude-invariant above `-60 dB` but amplifies background room hiss/silence tails when trailing silence lowers global utterance variance.

3. **Confirmed Facial (`ResNet18 + MTCNN`) Defects:**
   - **MTCNN works reliably (`100%` face detection coverage in user test), so the bottleneck is ResNet18 classification + temporal pooling:**
   - **Smoothed class-weight prior distortion in `exp5_ce_weighted`:** Weighting `neutral = 0.3969` vs `sadness = 1.0386`, `disgust = 1.6518`, `fear = 1.6611` intentionally suppressed `neutral` priors (`mean neutral prob 0.5154 -> 0.2780`) and flattened dev posteriors (`mean entropy 1.0574 -> 1.6143`) to chase Macro F1 on MELD static frames.
   - **Temporal softmax averaging (`raw_all_detections_mean_prob`) inflates entropy via Jensen's inequality:** Averaging post-softmax probabilities across all faces/frames without track or sharpness filtering blends non-speaker/blurry/neutral frames with expressive peaks, pushing dev mean entropy from `1.6364` (`single_frame_middle`) to `1.7336`, collapsing `disgust` & `fear` F1 to `0.0`, and pushing **98.19%** of dev predictions under `45%`.
   - **Static 1-frame `96x96` sitcom crop training with `"augment": false`:** Trained on a single MTCNN frame per TV video (`[0.50, 0.25, 0.75, 0.12, 0.88]`) with only horizontal flips—no head-tilt, scale, contrast, or eyeglass-glare augmentation. Controlled perturbation of a MELD `dev` crop under low contrast (`ent=1.8373`) and spectacles-glare proxy (`conf=19.56%`, `ent=1.8635 > 1.85`) reproduces the user's flat **19.5% / `entropy=1.883`** webcam result.

4. **Frontend Notices, Repeated Requests & Backend Errors Explained:**
   - **`"Low-confidence estimate (<45%)"`:** Informational UI banner triggered purely when `confidence < 0.45` (`51.49%` of MELD `dev` speech and `98.19%` of facial predictions qualify).
   - **UI probability sum (`99%`) & `30.9%` vs `31%`:** Header uses `.toFixed(1)%`; bars use `.toFixed(0)%`.
   - **Warnings / `422`s during manual testing:** Live telemetry (`emotion_api_requests_total`, `task-4565.log`) recorded **zero `429`**, **zero `503`**, and **zero `504`** responses. There is **no** automatic frontend retry loop; the **only** backend rejections in the user's session were **2 HTTP `422 Unprocessable Content`** responses on `/predict/audio` in `<= 50 ms`, matching `load_audio` `media_validation` (`silent_audio < -60.0 dB` or duration `< 0.010 s`).

### Explicit Investigation Limitations

- **User's Manual Media Files Not Present in Workspace:** A full non-dataset media search across the workspace returned **0 files**. Neither the user's ~6s 16 kHz mono WAV recording (`Neutral 30.9%`, `814.5 ms`) nor the user's ~8s webcam video (`Surprise 19.5%`, `2280 ms`, `num_faces=8`, `entropy=1.883`) exists on disk in the repository, so direct waveform RMS/SNR inspection of that WAV and frame-pixel inspection of that video could not be performed. Zero measurements have been fabricated for those two files.
- **Held-Out Test Split Untouched:** Per academic evaluation hygiene and project rules, MELD `test` data (`test_sent_emo.csv`, `output_repeated_splits_test`, `phase16_final_evaluation/test_predictions.jsonl`) was **never accessed or inspected**.

---

## 2. Architecture, Checkpoints, Label Contracts & Runtime Status Audit

Live verification against the running backend (`http://127.0.0.1:8000/models/status`) and disk SHA-256 verification confirmed that **all three primary trained models are active on `cuda`**, no fallback or legacy checkpoint is active, and zero mock predictors exist.

| Modality | Runtime Identifier | Device | Artifact File & Size | Verified SHA-256 | Active Role / Guard |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Text** | `DistilBertTextPredictor` | `cuda` | `models/text_emotion/distilbert/best/model.safetensors` (`267,847,948` B) | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | Primary (`max_length=128`) |
| **Speech** | `Wav2Vec2Predictor` | `cuda` | `models/speech_emotion/wav2vec2/best/model.safetensors` (`378,307,452` B) | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` | Primary (`16 kHz` mono) |
| **Facial (Primary)** | `facial-resnet18-ce-v1` | `cuda` | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` (`44,797,666` B) | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | Active (`imagenet_rgb_96x96`, `raw_all_detections_mean_prob`) |
| **Facial (Fallback — Unused)** | `facial-resnet18-focal-v1` | Standby | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` (`44,797,666` B) | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | Fallback only on primary load failure |
| **Facial (Legacy — Unused)** | `FacialEmotionPredictor` | Standby | `models/facial_emotion/facial_cnn_best.pt` (`380,128` B) | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | Rollback via `EMOTION_API_FACIAL_BACKEND=legacy_cnn` |

**Contract & Integrity Checks Verified:**
1. **Canonical 7-Class Order (`Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise`):** Matches verbatim across `models/text_emotion/distilbert/best/config.json`, `models/speech_emotion/wav2vec2/best/config.json`, `exp5_ce_weighted_best.pt` checkpoint metadata, [`api/schemas.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/schemas.py), and [`frontend/src/App.tsx`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/App.tsx).
2. **Numerical Determinism & FP32 Guard:** [`_strict_fp32_guard`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/resnet_predictor.py) enforces `allow_tf32=False` on CUDA with documented strict FP32 parity (`max_abs_diff = 3.58e-7` / `4.77e-7 < 1e-4`), avoiding TF32 rounding drift. Five consecutive HTTP requests per modality returned `max_prob_diff = 0.00e+00`.
3. **Why Historical Reference Baselines Differ Slightly from Unimodal `dev` (`N=1,108`):** Historical reference benchmarks in project prompts (`Text 63.10%`, `Candidate 4 63.37%`) came from the MELD `test` Cohort (`2,610` rows), whereas all checks here measure MELD `dev` (`1,108` rows, where `Text = 60.11%` and `Candidate 4 = 60.47%`). Facial matched dev cohort (`31.07%` / `0.1739`) came from single-frame crops (`N = 1,095`), whereas the integrated multi-frame video predictor (`raw_all_detections_mean_prob`) achieves `38.29%` Accuracy / `0.1722` Macro F1 on `N = 1,102` detected `dev` videos.

---

## 3. Deep Investigation: Speech Emotion Recognition (`Wav2Vec2`)

### 3.1 Code & Training Configuration Audit ([`speech_emotion/wav2vec2_model.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/wav2vec2_model.py) & [`speech_emotion/audio_loader.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/audio_loader.py))

1. **Unweighted Cross-Entropy on Heavily Imbalanced MELD (`42.3%` Neutral):**
   - `Wav2Vec2TrainingConfig` trains `facebook/wav2vec2-base` for `epochs = 4`, `lr = 1e-5`, `batch_size = 2` (`gradient_accumulation_steps = 8`) using Hugging Face's default unweighted `CrossEntropyLoss` (`loss = model(**batch).loss / config.gradient_accumulation_steps`, `wav2vec2_model.py:173`).
   - On MELD `dev` (`true counts`: `neutral=469`, `joy=163`, `anger=153`, `surprise=150`, `sadness=111`, `fear=40`, `disgust=22`), minority emotions suffer total recall collapse:
     - `disgust`: **0 predictions**, **F1 = 0.0000** (`22` true clips)
     - `fear`: **0 predictions**, **F1 = 0.0000** (`40` true clips)
     - `sadness`: `29` predictions, **F1 = 0.2000** (`111` true clips)
     - `joy`: `58` predictions, **F1 = 0.1176** (`163` true clips)
     - `surprise`: `121` predictions, **F1 = 0.3616** (`150` true clips)
     - `anger`: `281` predictions, **F1 = 0.4332** (`153` true clips)
     - `neutral`: `618` predictions (`55.8%`), **F1 = 0.6611** (`469` true clips)
   - Because MELD audio includes overlapping sitcom dialogue, studio laughter, background music, and short turns (`median = 2.453 s`, `min = 0.064 s`, `max = 28.544 s`), prosody alone is inherently less discriminative than text (`Speech dev Acc = 47.79%`, `mean confidence = 0.4633`, `median = 0.4430`, **`51.49%` below 0.45**, `mean entropy = 1.5013`).

2. **Confirmed Training-vs-Inference Preprocessing & Validation Mismatches:**

| Parameter | Training / Batch Eval ([`wav2vec2_model.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/wav2vec2_model.py)) | Runtime Inference (`Wav2Vec2Predictor.predict_file`) | Observed Effect |
| :--- | :--- | :--- | :--- |
| **`silence_threshold_db`** | `-200.0 dB` (`L99`) — never rejects quiet audio | Defaults to **`-60.0 dB`** via `load_audio` (`L267` -> `audio_loader.py:240-247`) | Raises `ValueError: silent_audio` -> **HTTP `422`** (`dev_dia36_utt8` at `-63.3 dB` fails at runtime; `0.02x` scaled clips fail at `-61.4` to `-62.2 dB`). |
| **`max_duration_seconds`** | Explicitly truncates `waveform[:128000]` (**`8.0 s`**, `L38, 103-104`) | Does **not** truncate at `8.0 s`; allows up to **`60.0 s`** (`load_audio` default) | Longer clips or trailing silence dilute mean-pooled hidden states across frames (`4.34%` of MELD `dev` clips exceed `8 s`; padding `dev_dia104_utt15` to `10s` shifted argmax `surprise 0.301` -> `anger 0.343`). |
| **Batch Padding vs `attention_mask`** | `_collator` runs `padding=True` (`L110`) with `"return_attention_mask": false` ([`preprocessor_config.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/speech_emotion/wav2vec2/best/preprocessor_config.json)) | Single-sample unpadded inference (`L270`) | Measured on `N=40` MELD `dev` clips: **`mean max_prob_diff = 0.04361`**, **`max = 0.21427`**, **`4/40` (10%) argmax flips**! Offline batch-eval metrics were computed with trailing-zero attention leakage that differs from live inference. |
| **`do_normalize`** | `true` (per-clip zero-mean, unit-variance) | `true` (per-clip zero-mean, unit-variance) | Makes predictions insensitive to linear gain (`0.1x` `-20 dB` changes conf by `<= 0.002`), **but** trailing/leading silence shrinks global variance and amplifies room noise/hiss into active frames. |
| **Metadata Provenance** | `config = None`, `dev_history = []` in `training_metadata.json` | — | Training hyperparameters and epoch curve were not persisted to disk during Phase 7 training. |

### 3.2 Reproducing the User's Manual Speech Test Pattern on Controlled MELD `dev` Clips

The user's uploaded ~6s 16 kHz mono WAV produced: `Neutral (30.9%)` with `[Anger 10%, Disgust 3%, Fear 3%, Joy 20%, Neutral 31%, Sadness 6%, Surprise 26%]` (`814.5 ms`). While that specific file could not be inspected (absent from workspace), controlled runtime tests on clean MELD `dev` clips and acoustic perturbations reproduce the exact `~30% Neutral/Surprise/Joy` ambiguity:

- **Clean MELD `dev` Clips (`Wav2Vec2Predictor`):**
  - `dev_dia0_utt0` (`2.795 s`, `-28.05 dB`, true=`sadness`) -> `surprise (29.8%)`, `entropy=1.714`
  - `dev_dia103_utt0` (`2.048 s`, `-27.43 dB`, true=`neutral`) -> `neutral (33.8%)`, `entropy=1.652`
  - `dev_dia103_utt2` (`2.176 s`, `-28.11 dB`, true=`joy`) -> **`neutral (31.3%)`**, `entropy=1.719`
  - `dev_dia104_utt10` (`2.261 s`, `-29.44 dB`, true=`surprise`) -> `surprise (30.5%)`, `entropy=1.722`
  - `dev_dia104_utt15` (`2.965 s`, `-25.45 dB`, true=`anger`) -> `surprise (30.1%)`, `entropy=1.676`
- **Controlled Acoustic Perturbations (`task-4783`):**

| Perturbation (applied to `dev_dia104_utt15` / `dev_dia103_utt2` / `dev_dia0_utt0`) | Observed Prediction Behavior | Diagnostic Explanation |
| :--- | :--- | :--- |
| **`quiet_0.1x (-20 dB)`** (`-45.5` to `-48.1 dB`) | Conf shift `<= 0.002`, zero class flips | `do_normalize: true` cancels linear attenuation while RMS `>= -60 dB`. |
| **`very_quiet_0.02x`** (`-59.4` to `-62.2 dB`) | `-59.4 dB` succeeds; `-61.4` to `-62.2 dB` raises **`ValueError: silent_audio`** | Hard cliff at `-60.0 dB` RMS in `load_audio`. |
| **`clipped_4x`** (`-13.4` to `-16.1 dB`) | Minor (`<= 0.003` conf diff) | Feature normalization absorbs modest clipping. |
| **`faster_1.25x` / `slower_0.8x`** | `dev_dia103_utt2` shifts `0.314 -> 0.461` (`1.25x`) / `0.259` (`0.8x`); `dev_dia0_utt0` flips `surprise -> neutral` (`0.8x`) | Speaking rate shifts prosodic temporal statistics in CNN/transformer layers. |
| **`short_0.4s`** | Large calibration instability (`0.338 -> 0.734` on `dia103_utt0`, `0.314 -> 0.665` on `dia103_utt2`, flips `anger` on `dia104_utt15`) | Very short segments lack contextual prosody and match MELD's noisy fragment bias. |
| **`padded_to_6s_trailing_silence` & `padded_to_10s`** | `dev_dia104_utt15` flips `surprise (0.301) -> anger (0.395 at 6s; 0.343 at 10s)`; `dia103_utt2` conf jumps `0.314 -> 0.374` (`ent 1.718 -> 1.547`) | Trailing silence changes global RMS variance (`do_normalize=true`) AND alters sequence mean-pooling beyond the `8s` training envelope. |
| **`noise_snr_10dB`** | `dia103_utt2` jumps `0.314 -> 0.615`; `dia103_utt0` jumps `0.338 -> 0.609`; `dia0_utt0` flips `surprise -> joy (0.310)` | Additive broadband noise produces spurious high-confidence shifts. |
| **`silent_zeros_2s` / `too_short_5ms`** | Fails cleanly with **`ValueError` (`silent_audio` / `duration_too_short`) -> HTTP `422`** | Exact cause of `-60 dB` / `<10 ms` upload rejections on `/predict/audio`. |

**Why the user's ~6s recording saw `814.5 ms` inference:** Warm `Wav2Vec2` GPU forward takes `~26–63 ms` on a `2.8s` clip and `~206 ms` cold; end-to-end `814.5 ms` reflects a `6s` (`96,000` sample) multipart upload, disk spooling, validation, WAV read, and first-hit GPU warmup.

---

## 4. Deep Investigation: Facial Emotion Recognition (`ResNet18 + MTCNN`)

*(Full frame-by-frame crop traces, perturbation sensitivity, and temporal pooling benchmarks are detailed in [`docs/FACIAL_LIVE_TEST_INVESTIGATION.md`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/FACIAL_LIVE_TEST_INVESTIGATION.md).)*

### Key Findings Summary

1. **MTCNN Face Detection is Not the Bottleneck for Centered Webcam Video:**
   - User report: `num_faces = 8`, `num_sampled_frames = 8`, `100%` coverage -> [`MTCNN`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/mtcnn_detector.py) detected a valid `prob >= 0.90` face in every sampled frame.
   - Instead, latency (`2,280 ms`) is dominated by CPU MTCNN detection + video decode (~88–95% of pipeline time, whereas ResNet18 CUDA forward takes only `18–48 ms`).

2. **Three Compounding Causes of Flat ~19% / `entropy = 1.883` Predictions:**
   - **Cause F1 — Class-Weighted Cross-Entropy Prior Shift (`exp5_ce_weighted_config.json`):** Smoothed-sqrt weights (`neutral = 0.3969` vs `sadness = 1.0386`, `disgust = 1.6518`, `fear = 1.6611`) suppress `neutral` and elevate rare emotions (`exp5_ce_weighted` dev `mean neutral = 0.2780`, `mean entropy = 1.6143` vs unweighted `0.5154`, `1.0574`), directly creating ~`15–19%` spread across `neutral`, `anger`, `sadness`, `surprise`, and `disgust`.
   - **Cause F2 — Post-Softmax Pooling (`raw_all_detections_mean_prob`) Entropy Inflation:** Averaging probabilities across all crops inflates entropy by **Jensen's inequality**, pushing MELD `dev` mean entropy from `1.6364` (`single_frame_middle`) to **`1.7336`**, dropping `disgust` & `fear` F1 to **`0.0`**, and making **`98.19%`** of `dev` videos fall `< 45%` confidence.
   - **Cause F3 — Static `96x96` MELD Sitcom Training Without Head-Tilt/Glare Augmentation (`"augment": false`):** Trained on 1 frame per clip (`run_exp4_face_detection.py`) without speaker-expression alignment, pitch/roll rotation, contrast, or eyeglass glare augmentation. On a MELD `dev` crop (`dev_dia0_utt0 f#37`), controlled perturbations shifted `baseline (0.3820, ent=1.6050)` -> `subtle_low_contrast (0.2789, ent=1.8373)` -> **`spectacles_glare_proxy` (`disgust 0.1956`, `joy 0.1845`, `neutral 0.1781`, `ent = 1.8635 > 1.85`)**.

3. **Safety Gate Already Active in Multimodal Fusion:**
   - Because `MAX_VIDEO_ENTROPY = 1.85` ([`api/multimodal_extension.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/multimodal_extension.py)), an uncertain live video (`entropy = 1.883 > 1.85`) is automatically gated to **`w_v = 0.00`** (`video_high_entropy`) in Multimodal mode, preventing flat facial probabilities from degrading Text+Speech predictions.

---

## 5. Repeated Requests, Warnings, Retries, Rate Limits & Concurrency Audit

### 5.1 Live Backend Telemetry & User Session Breakdown (`task-4565.log` + `/metrics` + `task-4800`)

Before running our controlled probe burst, the running uvicorn backend (`task-4565`, started `2026-11-11 23:28:56`) recorded exact telemetry from the user's manual testing session:

- **Total User Session Requests:** `/predict/text` (`1` × `200`, `1.50 s` cold), `/predict/audio` (`11` × `200`, **`2` × `422`**, sum `20.25 s`), `/predict/video` (`6` × `200`, sum `14.64 s`), `/predict/multimodal` (`3` × `200`, sum `5.25 s`), `/models/status` (`20` × `200`), `/health` (`1`), `/ready` (`1`).
- **Error Counts Observed:**
  - `emotion_api_request_errors_total{route="/predict/audio"} = 2`
  - `emotion_api_upload_rejections_total{reason="media_validation"} = 2` (both in the `le="0.05"` second latency bucket!)
  - **Zero** HTTP `429` (rate limit), **zero** HTTP `503` (unavailable), **zero** HTTP `504` (timeout), **zero** HTTP `413` (`>50 MB`), and **`emotion_api_cleanup_failures_total = 0`**. (One pre-existing `%TEMP%\emotion_1r2c_slk.mp4` file carried `CreationTime 2026-11-11 10:23:48`, >13 hours prior to server startup.)

### 5.2 Exact Mechanism of Every Warning / Repeated-Request Behavior

1. **`"Low-confidence estimate (<45%)"` Banner ([`frontend/src/App.tsx:116-135`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/App.tsx#L116-L135)):**
   - Not an error or retry warning—purely a visual calibration threshold (`confidence < 0.45`). It triggers on **51.49%** of MELD `dev` speech clips and **98.19%** of facial clips, vs **14.62%** of text clips.
   - Rounding display difference: header uses `(prediction.confidence * 100).toFixed(1)%` (`30.9%`, `19.5%`), while bar rows use `(value * 100).toFixed(0)%` (`31%`, `19%`), explaining why integer percentages sum to `99%` (`10+3+3+20+31+6+26`).

2. **Why `/predict/audio` Returned `422` Twice in User Session vs `/predict/multimodal` `200`:**
   - `Wav2Vec2Predictor.predict_file` uses `load_audio(..., silence_threshold_db=-60.0, min_duration=0.01)` ([`audio_loader.py:240-247`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/speech_emotion/audio_loader.py#L240-L247)).
   - Directly verified via live HTTP probes (`task-4800`): sending a silent WAV or `<10 ms` WAV to `/predict/audio` returns **HTTP `422`** in `6.7–24.0 ms` (`Invalid audio: silent_audio ...` / `duration_too_short`), matching the user's `<= 0.05 s` latency bucket `422` errors. Sending that same silent WAV alongside text to `/predict/multimodal` returns **HTTP `200`** with `exclusion_reasons.audio` (`api/app.py:437-438`).

3. **Frontend Retry / Request Scope Behavior ([`frontend/src/api.ts`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/api.ts), [`frontend/src/retry.ts`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/retry.ts)):**
   - **No automatic retry loop exists.** Failed/timed-out requests (`60,000 ms` `AbortController`) show an error banner with a **manual** `Retry` button (`retryAction`), while `createRequestScope()` aborts superseded in-flight requests if the user re-submits quickly.

4. **Backend Rate Limiter, Timeouts, Determinism & Concurrency ([`api/app.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/app.py), [`api/model_registry.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/model_registry.py)):**
   - **Rate limit:** [`SlidingWindowRateLimiter`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/app.py) allows `120 requests / 60s` per client IP (`#121` returns `HTTP 429` with `Retry-After: 60` and `"Rate limit exceeded. Wait briefly before retrying."`, verified in `task-4800`). `/health`, `/ready`, `/metrics` are exempt; `/models/status` & `/predict/*` share the window. Normal repeated clicks do **not** hit `429` unless `>120` requests happen inside `60s`.
   - **Repeated identical requests over HTTP (`task-4800`):** `5x Text` (`94.03% Surprise`, `1506ms -> 39.8-55.7ms`), `5x Audio` (`29.77% Surprise`, `206.1ms -> 25.5-62.5ms`), and `3x Video` (`33.01% Neutral`, `1131.9ms -> 755.9-890.7ms`) all returned **HTTP 200** with `max_prob_diff = 0.0` (`Cache-Control: no-store`).
   - **Concurrent 4-request multimodal burst:** `4/4 HTTP 200` (`0.5585 Surprise`, weights `text=0.92, audio=0.0, video=0.08`) completed cleanly in `3,295.8 ms` wall time (`873.1 ms`, `1,697.9 ms`, `2,515.5 ms`, `3,289.3 ms`), demonstrating orderly serialization via per-modality `RLock` (`self._locks`) with zero deadlocks or CUDA errors.

---

## 6. MELD `dev` Cross-Modality Evaluation Summary (`N = 1,108`, `zero_test_split_access = True`)

*(Computed directly from `models/phase12_dev_experiments/dev_predictions.jsonl` and `models/facial_emotion/resnet18/phase18d_dev_fusion_benchmark.json`.)*

| Modality Combination | `N` Evaluated | Accuracy | Macro F1 | Weighted F1 | Mean Confidence | Median Conf. | Fraction `< 45%` | Mean Entropy | Key Operational Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Text only (`DistilBERT`)** | `1,108` | **60.11%** | 0.4205 | 0.5700 | **0.7173** | 0.7679 | **14.62%** | **0.8832** | Dominant calibrated semantic signal |
| **Speech only (`Wav2Vec2`)** | `1,107` | 47.79% | 0.2534 | 0.4258 | 0.4633 | 0.4430 | **51.49%** | 1.5013 | `disgust/fear F1=0.0`; half `<45%` |
| **Face only (`ResNet18`)** | `1,102` | 38.29% | 0.1722 | 0.3232 | **0.2982** | **0.2870** | **98.19%** | **1.7336** | Nearly all clips `<45%`; `90th pct=0.3639` |
| **Text + Speech (`Candidate 4`)** | `1,108` | **60.47%** | **0.4243** | **0.5747** | 0.6862 | — | 17.06% | — | Audio adds slight boost when text-anchored |
| **Text + Face** | `1,108` | 60.20% | 0.4210 | 0.5707 | 0.6804 | — | 16.25% | — | Face bounded cleanly by `w_v <= 0.08` |
| **Text + Speech + Face (`Phase 18D`)** | `1,108` | **60.47%** | 0.4241 | 0.5744 | 0.6515 | — | 18.86% | 1.0836 | Improves calibration (`NLL 1.205 -> 1.200`, `ECE 0.0847 -> 0.0839`), `65/1102` gated |

**MELD `dev` True Support (`N = 1,108`):** `neutral=469` (`42.3%`), `joy=163` (`14.7%`), `anger=153` (`13.8%`), `surprise=150` (`13.5%`), `sadness=111` (`10.0%`), `fear=40` (`3.6%`), `disgust=22` (`2.0%`).

---

## 7. Confirmed vs Suspected Matrix

| ID | Issue / Hypothesis | Status | Severity | Empirical Evidence | Proposed Safe Remediation | Verification After Fix |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **S1** | `Wav2Vec2` unweighted CE collapse on `disgust`/`fear` + MELD `42.3%` neutral dominance | **Confirmed** | **High** | `wav2vec2_model.py:173`; MELD `dev` `disgust/fear F1=0.0`, `51.49%` `<45%`, `mean ent=1.501` | Re-train / temperature-calibrate speech head with smoothed class weights + longer schedule + speaker-aware validation | Check MELD `dev` Macro F1, ECE, NLL, non-zero minority F1 |
| **S2** | `Wav2Vec2` `return_attention_mask: false` with `padding=True` in batch eval vs unpadded inference | **Confirmed** | **Medium** | `preprocessor_config.json:7`; `N=40` test: `mean diff=0.0436`, `max=0.2143`, `4/40` flips | Enable `return_attention_mask=True` & enforce consistent masking in eval and future retraining | Assert `max_abs_diff < 1e-5` between batched & single inference |
| **S3** | `Wav2Vec2` `8.0 s` training truncation vs `60.0 s` inference limit (`max_duration`) | **Confirmed** | **Medium** | `wav2vec2_model.py:38,103` vs `267`; `4.34%` dev `>8s`; `10s` silence shifts `dev_dia104_utt15` `surprise -> anger` | Apply `8.0 s` windowing or VAD energy-based active-speech trimming prior to feature extraction | Test `6s–15s` padded/long clips for prediction stability |
| **S4** | `silence_threshold_db` `-200 dB` (train) vs `-60 dB` (API `422` cliff) & `do_normalize` noise amplification | **Confirmed** | **Medium** | `dev_dia36_utt8 (-63.3 dB)` + `2` user session `422`s (`<=50 ms`); `10 dB` noise jumps conf to `0.615` | Document mic gain / RMS threshold in UI (`Too quiet: RMS < -60 dB`), trim leading/trailing silence before `do_normalize` | Test quiet (`-65` to `-45 dB`) & noisy clips for graceful UX guidance |
| **S5** | Missing training metadata (`config=None`, `dev_history=[]`) in `wav2vec2/training_metadata.json` | **Confirmed** | **Low** | Direct inspect (`train_samples=9988`, `dev_samples=1108`, empty history) | Ensure future speech training saves complete hyperparams, calibration temperature & per-epoch metrics | Schema check in unit tests |
| **F1** | `exp5_ce_weighted` smoothed class weights suppress `neutral` (`0.3969`) & flatten all posteriors | **Confirmed** | **High** | `exp5_ce_weighted_config.json:38-46`; `mean ent 1.6143` vs unweighted `1.0574`; `exp5` mean `neutral 0.2780` | Fit **temperature scaling** (`T < 1` or logit prior debiasing) on MELD `dev` OR re-evaluate milder weights / unweighted + focal tuning with multi-frame data | Check MELD `dev` ECE/NLL/entropy & top-1 confidence calibration |
| **F2** | `raw_all_detections_mean_prob` softmax averaging inflates entropy (`Jensen's inequality`) | **Confirmed** | **High** | `resnet_predictor.py:168-169`; `dev` mean ent `1.6364 -> 1.7336`; `98.19%` `<45%`; `disgust/fear F1=0.0`; single-frame peaks diluted by neutral/2nd faces | Switch runtime pooling (after dev validation & Candidate 4 regression check) to sharpness/track-weighted logit pooling or top-k confident track frames (`track_topk3_prob`) | Re-benchmark all `1,102` MELD `dev` videos + `phase18d` multimodal calibration |
| **F3** | Static `1`-frame `96x96` sitcom crop training with `"augment": false` (no tilt, scale, contrast, or glare invariance) | **Confirmed** | **High** | `run_exp4_face_detection.py:534-595`; `exp5` config; perturbation tests (`glare -> disgust 19.6%, ent 1.8635`) | Train multi-frame / quality-filtered crops (optionally `224x224` or stronger ResNet/AffectNet pretraining) with pitch/rotation, glare, brightness/contrast, and scale augmentation | Benchmark on MELD `dev` + webcam spectacles/tilt holdout set |
| **U1** | User's specific ~6s WAV & ~8s webcam video acoustic/visual properties (SNR, pitch angle, spectacles glare) | **Suspected** *(unverifiable directly)* | **Info** | Exact files not in workspace; controlled MELD perturbations match symptoms closely | Collect optional saved anonymized test samples (with explicit consent) to verify SNR & glare directly | Inspect RMS, VAD speech ratio, face pitch, & crop sharpness |
| **A1** | UI `"Low-confidence estimate (<45%)"` & `.toFixed(0)%` bar sum (`99%`) confusing users during live demos | **Confirmed (UX)** | **Low** | `App.tsx:116-135, 894, 916`; triggers on `51.5%` speech / `98.2%` face dev clips | Clarify banner copy ("Modality Uncertainty Notice — Standalone audio/facial cues are inherently noisy...") & match `toFixed(1)%` or largest-remainder rounding | Frontend unit + visual verification |

---

## 8. Prioritized Remediation Roadmap (No Changes Applied Yet)

> **Important Safeguard:** Do **not** alter [`api/fusion.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/fusion.py) Candidate 4 fusion behavior or any MELD `test` split constraints without explicit user approval and full `dev`-set parity verification.

1. **P0 — Immediate Safe Quick Wins (No Model Retraining Required):**
   - **Temporal Facial Pooling Upgrade (`F2`):** Evaluate replacing `raw_all_detections_mean_prob` with sharpness-weighted track pooling (`track_topk3_prob` / `track_mean_logit`, ignoring blurry/background faces) on MELD `dev` (`N = 1,108`) and webcam recordings.
   - **VAD Silence Trimming & `8s` Consistency (`S3`, `S4`):** Trim leading/trailing silence (with safe fallback) and align max duration handling with the `8.0 s` training regime, plus surface clear microphone level guidance when RMS `< -50 dB` or `< -60 dB` (`422` explanation).
   - **Frontend Calibration UX (`A1`):** Distinguish probabilistic uncertainty from errors (`Low-confidence estimate (<45%)` tooltip/note explaining why standalone prosody/facial models have lower unimodal confidence than text) and align `.toFixed(1)%` rounding across pills and bars.

2. **P1 — Calibration Without Changing Feature Extractors:**
   - **Post-hoc Temperature Scaling / Logit Calibration (`S1`, `F1`):** Fit scalar temperature $T$ and/or prior adjustment on MELD `dev` logits (evaluated via NLL, Brier, ECE, and held validation) so standalone predictions reflect calibrated class likelihoods rather than flattened weighted-CE artifacts (`S1`/`F1`).

3. **P2 — Model Retraining / Domain Generalization Improvements (Academic Cycle):**
   - **Speech (`Wav2Vec2`) V2 (`S1`, `S2`):** Fix `"return_attention_mask": true` in feature extraction, apply moderated smoothed class weights or focal loss so `disgust`/`fear` do not collapse while preserving `neutral`/`sadness` calibration, add speed/noise/room-reverb augmentation, and persist full `training_metadata.json`.
   - **Facial (`ResNet18`) V2 (`F3`):** Train across multiple quality-filtered frames per utterance (not a single arbitrary `50%` frame), upgrade input resolution / backbone regularization, and enable geometric + photometric augmentations (`±15°` rotation/pitch, scale jitter, specular glare overlays for spectacles, brightness/contrast shifts) so laptop webcam domain shift no longer collapses predictions into ~`19%` entropy plateaus.

