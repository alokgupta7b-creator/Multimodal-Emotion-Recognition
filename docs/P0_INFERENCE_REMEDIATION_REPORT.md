# P0 Inference, Preprocessing & UX Remediation Report

**Workflow Stage:** Stage 2 (P0 — Inference & Preprocessing Fixes Without Model Retraining)  
**Evaluation Split Discipline:** MELD `dev` (`N = 1,108`, 114 dialogues; dialogue-grouped `dev_cal` `N = 560` / 57 dialogues, `dev_val` `N = 548` / 57 dialogues). Zero MELD `test` split tuning or optimization performed.  
**Protected Baselines Status:** DistilBERT, Wav2Vec2 v1 (`wav2vec2/best`), ResNet18 Primary (`exp5_ce_weighted_best.pt`), ResNet18 Fallback (`exp5_focal_g3_wt_best.pt`), Legacy CNN (`facial_cnn_best.pt`), and Candidate 4 fusion (`api/fusion.py`) weights and frozen constants untouched.

---

## 1. Stage 1 Protected Baseline & Recoverable Backup Summary

Before any code modification, a recoverable backup of 12 target files was saved with SHA-256 checksums to scratch (`pre_p0_p2_backup/backup_manifest.json`), and protected model/config hashes were verified:

| Protected Asset | Path | Verified SHA-256 |
| :--- | :--- | :--- |
| DistilBERT Text | `models/text_emotion/distilbert/best/model.safetensors` | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` |
| Wav2Vec2 Speech v1 | `models/speech_emotion/wav2vec2/best/model.safetensors` | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` |
| Facial ResNet18 Primary | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` |
| Facial ResNet18 Fallback | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` |
| Facial Legacy CNN | `models/facial_emotion/facial_cnn_best.pt` | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` |
| Candidate 4 Fusion | `api/fusion.py` | `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d` |

Canonical 7-class order preserved everywhere: `anger, disgust, fear, joy, neutral, sadness, surprise`.

---

## 2. Speech P0 Remediation (`speech_emotion/wav2vec2_model.py`, `api/app.py`)

### 2.1 Root-Cause Findings Verified Before Editing

1. **`attention_mask` Configuration & Batch Padding Mismatch:**  
   `models/speech_emotion/wav2vec2/best/preprocessor_config.json` specifies `"return_attention_mask": false` and `"do_normalize": true`. Inspecting HuggingFace `Wav2Vec2ForSequenceClassification.forward` confirmed that when `attention_mask=None`, `pooled_output = hidden_states.mean(dim=1)` averages **all** sequence frames including zero-padded tail frames in batched collators. On 32 MELD `dev` clips (`p0_speech_ablation.py`), batched padded inference without `attention_mask=True` diverged from unpadded single-clip inference by **0.9186 max abs logit** (`87.5%` top-1 agreement).
2. **Train-vs-Inference Duration Mismatch:**  
   `_AudioDataset` truncated waveforms at `8.0 s` (`128,000` samples at `16 kHz`), whereas `Wav2Vec2Predictor.predict_file` ran uncapped waveforms (up to `28.54 s` on MELD `dev`, with `48` clips `> 8.0 s`).
3. **Leading/Trailing Silence Sensitivity:**  
   Because `Wav2Vec2FeatureExtractor` performs per-utterance zero-mean/unit-variance normalization (`do_normalize=True`) over the entire waveform, leading/trailing dead air shrinks the variance normalization scale and adds silent CNN frames into global mean pooling.
4. **Scale-Invariant vs Fixed-dB VAD Ablation Discovery:**  
   Initial VAD testing with a hard `-55 dB` floor degraded `-18 dB` attenuated quiet speech agreement from `100.0%` (`6.3e-5` L1 drift) to `98.0%` (`0.0250` L1 drift). Removing the absolute floor and using a strictly **gain-invariant relative threshold** (`95th percentile frame dB - 32.0 dB` with `200 ms` safety margins, `min_kept_sec = 0.4 s`, `min_trimmed_sec = 0.25 s`) restored **100.0% quiet-speech invariance** while cutting padded silence drift by over 50%.

### 2.2 Speech P0 Ablation Results on MELD `dev` (`N = 1,107` valid; `dev_dia36_utt8` excluded at `-63.3 dB` RMS)

| Variant | Trimmed Clips | `dev` Acc | `dev` Macro-F1 | `dev` NLL | `dev` ECE | `>8s` Subset (`N=48`) Acc | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline (`uncapped`, `mask=False`) | 0 | **47.79%** | **25.34%** | 1.4742 | 0.0424 | **50.00%** | Raw baseline |
| `head_8s` + `attention_mask=True` | 0 | **47.79%** | 25.33% | 1.4741 | 0.0411 | **50.00%** | Exact train 8s bound |
| `peak_energy_8s` + `mask=True` | 0 | 47.52% | 24.95% | 1.4752 | 0.0448 | 43.75% | Worse on long clips |
| `chunked_8s_logit_mean` + `mask=True` | 0 | 47.61% | 25.05% | 1.4757 | 0.0398 | 45.83% | Late dialogue/laughter tail hurts |
| Scale-inv VAD `rel=30 dB` + `head_8s` | 11 | **47.79%** | 25.33% | **1.4740** | **0.0389** | **50.00%** | Strong MELD tightening |
| **Scale-inv VAD `rel=32 dB` + `head_8s` + `mask=True` (Selected P0)** | **5** | **47.79%** | **25.33%** | **1.4740** | **0.0397** | **50.00%** | Best padded-silence stability (`96%`) & zero quiet regression |
| Scale-inv VAD `rel=35 dB` + `head_8s` | 3 | **47.79%** | 25.33% | 1.4741 | 0.0397 | **50.00%** | Fewer dead-air trims |

#### Controlled Perturbation Robustness (`100` MELD `dev` clips `>= 0.8 s`)

| Condition | Metric | Baseline | Selected Speech P0 (`rel=32 dB`, `head_8s`, `mask=True`) |
| :--- | :--- | :---: | :---: |
| `+2.0s leading & +2.0s trailing -65 dB silence` | Top-1 Agreement vs Clean | 92.0% | **96.0%** (+4.0 pp) |
| `+2.0s leading & +2.0s trailing -65 dB silence` | Mean L1 Prob Drift | 0.1575 | **0.0773** (-50.9%) |
| `-18 dB` quiet speech scaling (`0.125x` gain) | Top-1 Agreement vs Clean | 100.0% | **100.0%** |
| `-18 dB` quiet speech scaling (`0.125x` gain) | Mean L1 Prob Drift | 0.000063 | **0.000064** |
| Pure silence (`< -60 dB` RMS) | Validation Result | Rejected (`422` / `silent_audio`) | Rejected (`422` / `silent_audio`) |

### 2.3 Speech P0 Production Before vs After Summary (`Wav2Vec2Predictor.predict_file`)

- **Shared helper introduced:** `trim_leading_trailing_silence()` and `preprocess_speech_waveform()` in `speech_emotion/wav2vec2_model.py`, used by both `_AudioDataset` and `Wav2Vec2Predictor.predict_file`, with `return_attention_mask=True` enforced in `_collator` and `Wav2Vec2Predictor`.
- **Same-sample comparison (`+4s` padded silence example):**
  - `dev_dia0_utt1` (`1.34 s`, true=`surprise`): Baseline clean `neutral (36.5%)` -> padded `neutral (54.6%)` (`+18.1 pp` drift); **P0** clean `neutral (36.5%)` -> padded trimmed to `1.77 s` (`neutral 45.7%`, halving drift).
  - `dev_dia1_utt1` (`1.64 s`, true=`joy`): P0 reduced padded L1 probability shift across the 100-clip suite from `0.1575` to `0.0773`.
- **Metrics & Latency (`dev`, `dev_cal`, `dev_val`):**
  - `dev` (`N=1,107`): Acc `47.79% -> 47.79%`, Macro-F1 `25.34% -> 25.33%` (`-0.01%`, Surprise F1 `36.2% -> 36.0%`, Anger F1 `43.3% -> 43.4%`), Weighted-F1 `42.58% -> 42.55%`, NLL `1.4742 -> 1.4740`, Brier `0.6805 -> 0.6804`, **ECE `0.0424 -> 0.0397`**, `<45%` rate `51.5% -> 51.3%`.
  - `dev_cal` (`N=560`): Acc `49.19% -> 49.19%`, Macro-F1 `26.24% -> 26.24%`, NLL `1.4506 -> 1.4491`.
  - `dev_val` (`N=547`): Acc `46.35% -> 46.35%`, Macro-F1 `24.51% -> 24.50%`, ECE `0.0645 -> 0.0638`.
  - **Latency & Parity:** Full `dev` CUDA mean inference latency dropped **`27.32 ms -> 17.12 ms` (-37.3%)**; 10-clip CPU mean latency dropped **`333.27 ms -> 200.80 ms` (-39.8%)**; CPU vs CUDA max probability diff = `3.00e-4`.
  - **Telemetry & Validation:** Added finite waveform checks, `min_duration=0.05 s`, and acoustic telemetry (`audio_duration_s`, `processed_duration_s`, `trimmed_silence_s`, `audio_rms_db`, `low_volume_warning` for `< -40 dB` flagging `33` quiet-but-valid dev clips, `prediction_entropy`) forwarded through `api/app.py` (`/predict/audio` and `/predict/multimodal`).

---

## 3. Facial P0 Temporal Pooling & Inference Benchmark (`facial_emotion/resnet_predictor.py`)

Using the cached MELD `dev` detections (`exp7_temporal/dev_temporal_unaligned_crops.npz`, `14,877` crops across `1,102 / 1,108` detected videos) and ResNet18 strict FP32 logits (`exp5_ce_weighted_best.pt`), all 9 pooling methods were evaluated across `dev`, `dev_cal`, `dev_val`, single-track vs multi-track subsets, Phase 18 downstream fusion, and drop-1 frame perturbation stability:

| Pooling Method | `dev` Acc | `dev` Macro-F1 | `dev` NLL | `dev` ECE | Mean Conf | `<45%` Rate | `1-Track (N=370)` Acc / MF1 | Drop-1 Top-1 Agree | P18 `dev_val` Acc (`active_v`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`raw_all_detections_mean_prob` (Kept Default)** | **38.29%** | **17.22%** | 1.6902 | 0.0944 | 29.8% | 98.2% | **35.41%** / **18.41%** | 95.16% | 58.58% (`1,037`) |
| `raw_mean_logit` (Added Option) | **39.84%** | 16.71% | **1.6729** | 0.0964 | 31.7% | 97.5% | 34.86% / 16.77% | **96.49%** | 58.58% (`1,070`) |
| `track_arith_mean_prob` | 34.66% | 15.89% | 1.7058 | 0.0505 | 30.4% | 97.5% | 35.41% / 18.41% | 91.26% | 58.76% (`1,032`) |
| `track_mean_logit` | 35.48% | 15.06% | 1.6914 | 0.0483 | 31.8% | 96.6% | 34.86% / 16.77% | 92.31% | 58.76% (`1,058`) |
| `track_topk3_prob` | 31.22% | 15.47% | 1.7357 | 0.0678 | 35.2% | 87.9% | 31.08% / 17.54% | 92.78% | 58.76% (`1,070`) |
| `track_topk3_logit` | 32.58% | 15.40% | 1.7279 | 0.0640 | 36.3% | 85.3% | — | 93.64% | 58.76% (`1,079`) |
| `track_quality_conf_weighted` | 33.21% | 15.47% | 1.7083 | **0.0394** | 31.0% | 96.9% | 35.14% / 18.48% | 90.98% | 58.76% (`1,036`) |
| `track_quality_weighted_logit` | 34.66% | 14.96% | 1.6956 | 0.0461 | 32.3% | 95.2% | — | 92.40% | 58.76% (`1,059`) |
| `single_frame_middle` | 30.04% | 17.15% | 1.7732 | 0.0663 | 34.4% | 88.1% | — | 74.83% | 58.76% (`1,050`) |

### Facial P0 Evidence-Based Decision
- **All single-track strategies degraded Accuracy and Macro-F1:** `track_topk3_prob` dropped MELD `dev` accuracy by **`-7.07 pp`** (`38.29% -> 31.22%`) and even lost **`-4.33 pp`** on single-person (`1-track`) clips (`35.41% -> 31.08%`). While `track_topk3_prob` lowered `<45%` warnings (`98.2% -> 87.9%`), that was artificial probability sharpening paired with **worse NLL** (`1.6902 -> 1.7357`) and **lower accuracy**.
- **`raw_mean_logit`** was added to `SUPPORTED_POOLING_METHODS` and `aggregate_crop_outputs` because it achieved the highest overall Accuracy (`39.84%`), best NLL (`1.6729`), and highest drop-1 stability (`96.49%`), though at a slight minority-class Macro-F1 trade-off (`17.22% -> 16.71%`, `1-track` Macro-F1 `18.41% -> 16.77%`, Anger F1 `13.3% -> 10.2%`).
- **P0 Decision:** Keep **`raw_all_detections_mean_prob`** as the P0 default (protecting minority-class Macro-F1 and avoiding unvalidated confidence inflation), and carry both `raw_all_detections_mean_prob` and `raw_mean_logit` forward into **Stage 3 (P1 calibration)** to test whether temperature/prior calibration resolves underconfidence cleanly without losing minority F1.

---

## 4. Frontend & API UX P0 Improvements (`frontend/src/`)

Without redesigning the visual layout or hiding the `<45%` warning:

1. **Consistent Probability Percentage Formatting (`media.ts`, `App.tsx`):**  
   Added `formatProbabilityPercent()` (`(value * 100).toFixed(1)%`) and updated `ProbabilityBars` so breakdown rows match confidence badges (`30.9% confidence` vs `30.9%` bar) instead of integer rounding mismatches.
2. **Low-Volume Microphone Guidance (`media.ts`, `LiveCapture.tsx`):**  
   Added `analyzeWaveformLevel()` and `measureWavRmsDbFromBuffer()` inspecting the actual captured 16 kHz PCM16 WAV bytes; `LiveAudioCapture` now displays live `"Very quiet signal — speak clearly closer to your microphone"` during capture when level < 5, plus post-capture RMS dB badges with clear warnings for `< -40 dB` (`isLowVolume`) and `< -58 dB` (`isNearSilent`).
3. **Clear Validation Failure vs Low-Confidence (`<45%`) Separation (`api.ts`, `App.tsx`):**  
   Added `classifyInputError()` so silent (`< -60 dB`), empty, corrupt, or face-free rejections (`422` / `415`) explain that invalid inputs are rejected rather than scored as low confidence. In `MultimodalResultCard`, `isSignalRejectionReason()` separates unattached optional media (`"Excluded"`) from rejected invalid uploads (`"Excluded (invalid input)"`). Preserved `"Low-confidence estimate (<45%)"` intact and appended contextual signal notes (`getSignalNote()`) from `QualityCard` metrics.
4. **Duplicate Request Protection (`App.tsx`):**  
   Added a synchronous `inFlightRef` lock in `runAnalysis` / `runGuarded` alongside disabled buttons to prevent duplicate network requests on rapid double-clicks on both submit and Retry buttons.

---

## 5. P0 Gate Verification & Known Limitations Carried Into P1 & P2

- **Backend tests (`pytest -q speech_emotion/tests/test_wav2vec2.py facial_emotion/tests/test_resnet_integration.py api/tests/test_api.py api/tests/test_multimodal_api.py`):** **54 passed**, 0 failed (includes 4 new P0 unit/integration tests for scale-invariant VAD, 8s/finite validation, `_collator` `attention_mask`, and `raw_mean_logit` pooling).
- **Frontend tests (`node --test frontend/tests/api.test.mjs`) & Build (`npm run build`):** **11/11 passed**, `tsc -b && vite build` succeeded cleanly in `4.62 s`.
- **Remaining Gaps Addressed in P1 & P2:**
  - Speech `v1`: `disgust` F1=`0.0%` (`0` pred), `fear` F1=`0.0%` (`0` pred), `<45%` rate=`51.3%`.
  - Facial `exp5_ce_weighted`: extreme underconfidence (`mean_conf = 29.8%` vs `38.29%` accuracy, `98.2%` `<45%`), `disgust` F1=`0.0%`, `fear` F1=`0.0%`, `neutral` over-prediction (`764 / 1,102`).
  - Proceeding to **Stage 3 (P1 Probability Calibration)**.

