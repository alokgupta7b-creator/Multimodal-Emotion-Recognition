# Phase 20D — Model Storage & Inference Infrastructure

**Status:** Completed  
**Runtime Bundle Footprint:** 11 verified files, 736,848,769 bytes (**702.71 MiB**)

---

## 1. Runtime Storage & Bundle Strategy

1. **Exclusion of Experimental & Research Artifacts**:
   - Research training caches under `models/facial_emotion_phase*` and historical JSON/NPZ experiment outputs under `models/facial_emotion/resnet18/` are excluded from Docker build context via `.dockerignore` and skipped during staging deployment.
   - Staging mounts only the runtime checkpoint directories read-only (`:ro`) in [`docker-compose.staging.yml`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docker-compose.staging.yml#L41-L44).
   - [`scripts/validate_phase20d_models.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/validate_phase20d_models.py) includes `--export-bundle <dir>` to copy strictly the 11 verified runtime files for transfer to an external VPS or object bucket without copying training datasets or caches.

2. **Verified Runtime Checkpoints (SHA-256 & Sizes)**:

| Group | Artifact | Size (Bytes) | SHA-256 |
|---|---|---:|---|
| Text (`DistilBERT`) | `models/text_emotion/distilbert/best/model.safetensors` | 267,847,948 | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` |
| Text Config/Tokenizer | `config.json`, `tokenizer.json`, `tokenizer_config.json` | 712,806 | Recorded in [`phase20d_model_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase20d_model_validation.json) |
| Speech (`Wav2Vec2`) | `models/speech_emotion/wav2vec2/best/model.safetensors` | 378,307,452 | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` |
| Speech Configs | `config.json`, `preprocessor_config.json` | 2,981 | Recorded in [`phase20d_model_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase20d_model_validation.json) |
| Facial Primary (`ResNet18`) | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` | 44,797,666 | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` |
| Facial Fallback (`ResNet18`) | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` | 44,797,666 | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` |
| Facial Metadata | `models/facial_emotion/resnet18/manifest.json` | 2,122 | `a043f8fa0cadf1a905165f72eb7e06056a7c503408f2163e52cb899fb539d906` |
| Facial Legacy CNN | `models/facial_emotion/facial_cnn_best.pt` | 380,128 | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` |

---

## 2. Empirical Validation Results (`phase20d_model_validation.json`)

Executed via `scripts/validate_phase20d_models.py` (`task-4318`) using only MELD **development** fixtures (`dia0_utt0.wav`, `dia0_utt0.mp4`); held-out test splits were never accessed.

### A. CPU vs. CUDA Load, Memory & Inference

| Metric | CPU (`cpu`) | GPU (`cuda`, RTX 4050) |
|---|---:|---:|
| Total `ModelRegistry.load()` | 18,268.48 ms *(cold disk read)* | 2,336.49 ms |
| Process RSS After Load | 798.91 MiB | 1,317.84 MiB |
| CUDA Allocated / Peak Allocated | — | 671.50 MiB / 719.30 MiB |
| Text Inference (`"I cannot believe..."`) | `joy` (0.6966) — 116.41 ms | `joy` (0.6966) — 648.04 ms *(first CUDA call)* |
| Audio Inference (`dia0_utt0.wav`) | `surprise` (0.2976) — 450.75 ms | `surprise` (0.2977) — 468.23 ms |
| Video Inference (`dia0_utt0.mp4`, 8 frames / 21 faces) | `neutral` (0.3302) — 1,410.72 ms | `neutral` (0.3301) — 1,629.50 ms |

> [!NOTE]
> Process RSS after CPU model initialization (**798.91 MiB**) fits comfortably inside the staging Compose container's **4 GiB** (`mem_limit: 4g`) ceiling with `--workers 1` and single-instance startup loading.

### B. Fault Tolerance & Fallback Verification

1. **Corrupt Primary Checksum Fallback**: When `EMOTION_API_VIDEO_SHA256` fails verification against `exp5_ce_weighted_best.pt`, `ModelRegistry` automatically switches to weighted focal-loss fallback `exp5_focal_g3_wt_best.pt` (`active_role: "fallback"`, SHA-256 `4830ac84...`), returning valid video predictions (`neutral`, confidence `0.2522`).
2. **Legacy CNN Fallback**: With `EMOTION_API_FACIAL_BACKEND=legacy_cnn`, `ModelRegistry` loads `FacialEmotionPredictor` (`facial_cnn_best.pt`) and serves video inference (`anger`, confidence `0.1599`).

### C. Concurrent Request Handling (CPU Staging Config)

- **4 Concurrent Workers** (`ThreadPoolExecutor(max_workers=4)` hitting `/predict/text`, `/predict/audio`, `/predict/video`, and `/predict/multimodal` simultaneously):
  - Result: **4 / 4 HTTP 200 OK**, total batch wall time **3,190.31 ms** (text `327.43 ms`, audio `614.72 ms`, video `1,964.77 ms`, multimodal `3,175.35 ms`). Per-modality locks inside `ModelRegistry` serialized concurrent predictor calls cleanly without tensor/buffer corruption.

