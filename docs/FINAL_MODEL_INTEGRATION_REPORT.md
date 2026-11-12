# Final Model Selection & Multimodal Integration Report

**Date:** 2026-11-12  
**Project:** Multimodal Emotion Recognition System (Final-Year Academic Project)  
**Canonical 7-Class Order:** `anger, disgust, fear, joy, neutral, sadness, surprise`  
**Status:** Verified Complete — Integrated in FastAPI Backend & React Frontend (`deployed_to_production = False`)

---

## 1. Final Model Selection & Checkpoint Registry

All active models load real trained checkpoints through [`api/config.py`](../api/config.py) and [`api/model_registry.py`](../api/model_registry.py) with SHA-256 verification on the ResNet18 facial checkpoints:

| Modality | Active Selected Version | Identifier | Checkpoint Path | Verified SHA-256 | Calibrated | Rollback Env Override |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Text** | DistilBERT | `DistilBertTextPredictor` | `models/text_emotion/distilbert/best/model.safetensors` | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | No | `EMOTION_API_TEXT_MODEL_DIR` |
| **Speech** | Wav2Vec2 V2 + L2 Bias | `speech-wav2vec2-v2` | `models/speech_emotion/wav2vec2_v2/best/model.safetensors` (+ `calibration.json`) | `fbbb902b5d77138178f07c8dde847c0f853aed6c5fde04aa73b4a260ca477478` | Yes (`T=1.4159`) | `EMOTION_API_AUDIO_MODEL_DIR=models/speech_emotion/wav2vec2/best` (V1+P1) or `EMOTION_API_USE_SPEECH_CALIBRATION=0` |
| **Facial** | ResNet18 V2 + MTCNN | `facial-resnet18-v2-la_ce_tau040` | `models/facial_emotion/resnet18_v2/resnet18_v2_best.pt` | `1822ba9dc13a7406c13ad4bd48a7c8be7d309ead286cbe40352525bf85c1211a` | No (raw logits hurt F1 in P1/P2) | `EMOTION_API_VIDEO_MODEL_PATH=models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` or `EMOTION_API_FACIAL_BACKEND=legacy_cnn` |

### Selection Rationale (Beyond Raw Confidence)
- **Speech (`Wav2Vec2_v2 + L2 Calibration`):** Strictly matched or outperformed both `V1 Raw` and `V1 + P1` across all 6 disjoint `dev_val` (`N=548`) metrics: Accuracy `47.26%`, Macro-F1 `26.26%` (vs `24.50%` V1 raw / `24.58%` V1+P1), Weighted-F1 `41.98%`, ECE `0.0537`, NLL `1.4510`, Brier `0.6791`, and `<45%` uncertainty rate `40.1%` (down `-11.5 pp`).
- **Facial (`ResNet18_v2`, uncalibrated, `raw_all_detections_mean_prob`):** Raised `dev_val` (`N=544`) Macro-F1 by `+3.56 pp` (`17.96% -> 21.52%`), Weighted-F1 by `+1.14 pp` (`32.17% -> 33.31%`), cut ECE nearly in half (`0.0934 -> 0.0473`), lifted mean confidence (`29.6% -> 38.6%`), and reduced `<45%` warnings (`97.8% -> 76.7%`) over V1 without V1's `Neutral` over-prediction.

### Protected Baseline Checksums Verified Intact
- `models/speech_emotion/wav2vec2/best/model.safetensors`: `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e`
- `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt`: `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64`
- `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt`: `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`
- `models/facial_emotion/facial_cnn_best.pt`: `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760`
- `api/fusion.py`: `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d`

---

## 2. Frozen Candidate 4 + Phase 18 Multimodal Fusion Architecture

- **Preserved Candidate 4 Parameters ([`api/fusion.py`](../api/fusion.py)):** `tau_a = 0.20`, `lambda_max = 0.40`, `kappa_a = 1.50`, `lambda_v = 0.00`.
- **Extended Three-Modality Logic ([`api/multimodal_extension.py`](../api/multimodal_extension.py)):**
  - Text + Audio (no video): delegates verbatim to `Candidate4FusionEngine`.
  - Text + Video / Text + Audio + Video: applies video fusion weight `w_v = 0.08` only when face detection coverage `>= 0.40` and prediction entropy `<= 1.85`; otherwise `w_v = 0.00` (advisory breakdown only, protecting Text).
  - Audio + Video (no text): audio primary (`0.92`) + quality-gated video (`0.08`) blend.
  - Single-modality inputs and invalid/missing optional uploads: returns real predictions or safely excludes failing optional streams without fabricating synthetic neutral outputs.

---

## 3. Verified Live Seven-Path Inference Check (`http://127.0.0.1:8000/predict/multimodal`, CUDA)

All 7 modality combinations were tested live against the running FastAPI backend using real MELD `dev` sample `dia0_utt0` audio/video plus utterance text:

| Path | Modalities Active | HTTP | Predicted Emotion | Confidence | Weights (`T / A / V`) | Fusion Identifier & Rule Applied | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| 1. Text only | `T` | 200 | `surprise` | 75.39% | `1.00 / 0.00 / 0.00` | `text_anchored_class_conditional_bounded_fusion_v1` (`text_anchor_default`) | 2160.3 *(cold)* |
| 2. Speech only | `A` | 200 | `joy` | 32.10% | `0.00 / 1.00 / 0.00` | `single_modality_audio_wav2vec2_v1` (`audio_only_fallback`) | 69.2 |
| 3. Facial video only | `V` | 200 | `neutral` | 32.45% | `0.00 / 0.00 / 1.00` | `single_modality_video_resnet18_v1` (`video_only_resnet18`) | 986.0 |
| 4. Text + Speech | `T + A` | 200 | `surprise` | 75.39% | `1.00 / 0.00 / 0.00` | `text_anchored_class_conditional_bounded_fusion_v1` (`text_anchor_default`) | 44.6 |
| 5. Text + Facial | `T + V` | 200 | `surprise` | 70.51% | `0.92 / 0.00 / 0.08` | `candidate4_plus_validated_resnet18_video_v1` (`text_anchor_default+validated_video_calibrated`) | 601.0 |
| 6. Speech + Facial | `A + V` | 200 | `joy` | 30.47% | `0.00 / 0.92 / 0.08` | `audio_primary_plus_validated_resnet18_video_v1` (`audio_video_calibrated_blend`) | 618.5 |
| 7. Text + Speech + Facial | `T + A + V` | 200 | `surprise` | 70.51% | `0.92 / 0.00 / 0.08` | `candidate4_plus_validated_resnet18_video_v1` (`text_anchor_default+validated_video_calibrated`) | 710.4 |

All 7 responses returned valid 7-class probability distributions in canonical order (`anger, disgust, fear, joy, neutral, sadness, surprise`) summing to `1.0`. Startup rollback configuration tests (`test_v1_plus_p1_env_rollback_configuration_and_registry` and `test_registry_supports_resnet18_default_and_legacy_rollback`) also pass in `pytest`.

