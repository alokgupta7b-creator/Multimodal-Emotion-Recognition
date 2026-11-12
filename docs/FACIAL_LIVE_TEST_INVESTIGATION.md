# Facial Emotion Recognition Live Test & Video Pipeline Investigation

**Date:** 2026-11-12  
**Status:** Investigation Complete — Awaiting Explicit Approval Before Implementing Fixes  
**Scope:** Investigation-only root-cause analysis of Facial Emotion Recognition ([`ResNetFacialEmotionPredictor`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/resnet_predictor.py) + [`MTCNN`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/mtcnn_detector.py)) explaining why a real ~8s webcam recording with `100%` face detection coverage produced near-uniform probabilities (`Surprise 19.5%`, `entropy = 1.883`, `2,280 ms`).

---

## 1. User Live Test Evidence & Explicit Workspace Limitation

### User-Reported Live Facial Result (~8s Video)

- **Subject / Recording Conditions:** Visible frontal face wearing spectacles, slight downward head angle, subtle/calm expression, ~8 seconds duration.
- **API / UI Result Observed by User:**
  - Predicted emotion: **Surprise** (`19.5%` confidence header)
  - Class bar breakdown: `Anger 10%`, `Disgust 18%`, `Fear 8%`, `Joy 8%`, `Neutral 19%`, `Sadness 18%`, `Surprise 19%` *(UI rounding `.toFixed(0)%` vs header `.toFixed(1)%` makes `Neutral` and `Surprise` both display `19%`)*
  - Telemetry reported: `num_faces = 8`, `num_sampled_frames = 8`, **`coverage = 100%`**, **`prediction_entropy = 1.883`**, inference time **`2,280 ms`**

### Explicit Limitation

A workspace-wide search for non-dataset media files (`.mp4`, `.webm`, `.mov`, `.avi`, `.wav`, `.mp3`, `.ogg`) returned **0 files**. The user's exact ~8s webcam video is **not saved in the repository**, so its raw pixel crops, head-pose angles, illumination, and specular glare on lenses could not be inspected directly. All findings below combine:
1. Exact checkpoint, training-config, and runtime-code verification;
2. Full MELD `dev` (`N = 1,108`, `zero_test_split_access = True`) benchmark & pooling audits;
3. Frame-by-frame runtime inspections across real MELD `dev` videos + controlled visual perturbation tests approximating downward head tilt, spectacles glare, and subtle low contrast.

*(Note: An older webcam test video [`WIN_20261111_03_07_13_Pro.mp4`] outside the workspace was previously tested read-only in Phase 17D/18C (`11/13` valid track crops, `1,115.5 ms`), confirming finite non-crashing pipeline execution rather than high real-world generalization.)*

---

## 2. Runtime Checkpoint, Device & Preprocessing Contract Verification

Live `GET /models/status` telemetry (`task-4800`) and SHA-256 verification confirm the intended Phase 17 primary model is active on CUDA (`allow_tf32=False` IEEE FP32 guard active):

| Role | Model ID | Path | Verified SHA-256 | Architecture & Preprocessing | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Primary** | `facial-resnet18-ce-v1` | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | ResNet18 (`96x96` RGB ImageNet), `raw_all_detections_mean_prob` | **Loaded (`cuda`)** |
| Fallback (Standby) | `facial-resnet18-focal-v1` | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | ResNet18 (`96x96` RGB ImageNet), focal $\gamma=3.0$ | Unused (no fallback occurred) |
| Legacy CNN (Standby) | `FacialEmotionPredictor` | `models/facial_emotion/facial_cnn_best.pt` | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | Custom 4-block CNN (`48x48` grayscale `[-1,1]`) | Unused |

- **Why `num_faces = 8` / `100%` coverage happened cleanly:** [`MTCNN`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/mtcnn_detector.py) (`keep_all=True`, `min_face_size=24`, `DEFAULT_MIN_DET_PROB=0.90`, `DEFAULT_CROP_MARGIN=0.08`) is strictly a face bounding-box detector. Across a steady ~8s laptop webcam stream, detecting 1 frontal face on all `8/8` uniformly sampled frames (`100%` coverage) is expected—yet **face detection coverage measures box presence only, NOT expression signal clarity or calibration**.
- **Latency (`2,280 ms`) Breakdown:** On real MELD `dev` MP4s (`task-4783`), MTCNN CPU face detection takes `383.6–886.1 ms`, frame decoding takes `62.6–272.3 ms`, and ResNet18 CUDA forward inference takes only **`17.7–47.7 ms`**. On ~8s higher-resolution laptop webcam clips (or WebM streams triggering [`_sequential_subsample_capture`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/mtcnn_detector.py)), CPU decoding + MTCNN across 8 full frames readily accounts for `~2.28 s`.

---

## 3. Root Causes of Flat ~19% Probabilities and `entropy = 1.883`

Note that the theoretical **maximum entropy** of a uniform 7-class distribution is $\ln(7) = \mathbf{1.9459}$. An entropy of **`1.883`** represents **96.8%** of maximum uniform uncertainty. Four compounding design/dataset factors explain why `ResNet18 + MTCNN` produces this exact signature on calm webcam recordings:

### Root Cause 1: Smoothed Class-Weighted Cross-Entropy Prior Shift (`exp5_ce_weighted`)

Per [`models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_config.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_config.json), the approved `exp5_ce_weighted` checkpoint trained ResNet18 with aggressive smoothed square-root inverse-frequency class weights:

- `neutral`: **`0.3969`** *(suppressed by 2×–4× relative to non-neutral emotions)*
- `joy`: `0.7689` | `surprise`: `0.7826` | `anger`: `0.8173` | `sadness`: **`1.0386`** | `disgust`: **`1.6518`** | `fear`: **`1.6611`**

Comparing `exp5_ce_unweighted` vs `exp5_ce_weighted` vs `exp5_focal_g3_wt` on MELD `dev` (`exp5_results.json`):
- **Unweighted CE:** `Accuracy = 35.73%`, `Macro F1 = 0.1461`, **`mean neutral prob = 0.5154`**, **`mean entropy = 1.0574`**.
- **Primary (`exp5_ce_weighted`):** `Accuracy = 30.91%`, `Macro F1 = 0.1724`, **`mean neutral prob = 0.2780`**, **`mean entropy = 1.6143`** (`mean probs`: `anger=0.1682, joy=0.1573, sadness=0.1523, surprise=0.1508, disgust=0.0502, fear=0.0432`).
- **Fallback (`exp5_focal_g3_wt`):** `Accuracy = 32.91%`, `Macro F1 = 0.1669`, **`mean entropy = 1.7264`**.

**Consequence:** Because `neutral` carries weight `0.3969` while `sadness`, `disgust`, `fear`, `surprise`, and `anger` carry weights `2×–4.2×` higher, whenever facial features become subtle or domain-shifted, the classifier **systematically drains probability mass out of `Neutral` and spreads `10–20%` each across `Surprise`, `Sadness`, `Disgust`, and `Anger`**. That directly explains why the user saw `Neutral 19% / Surprise 19% / Disgust 18% / Sadness 18% / Anger 10%` instead of a confident `Neutral` (`50%+`) prediction.

### Root Cause 2: Post-Softmax `raw_all_detections_mean_prob` Inflates Entropy (Jensen's Inequality)

[`aggregate_crop_outputs`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/resnet_predictor.py#L168-L169) computes the plain arithmetic mean of **post-softmax probabilities** across all crops detected across all `8` sampled frames (`raw_all_detections_mean_prob`):

$$\bar{p} = \frac{1}{N} \sum_{i=1}^{N} p_i \implies H(\bar{p}) \ge \frac{1}{N} \sum_{i=1}^{N} H(p_i)$$

Because Shannon entropy $H(p)$ is strictly concave, averaging probabilities over frames where expression shifts (or blinks/micromovements/multiple faces appear) mathematically drives $\bar{p}$ toward uniform!
- Across MELD `dev` (`exp7_results.json` + Script 1, `N = 1,102`):
  - `single_frame_middle` (`mean entropy = 1.6364`): non-zero `disgust F1 = 0.0488`, `fear F1 = 0.0392`, `Macro F1 = 0.1715`.
  - **`raw_all_detections_mean_prob` (`mean entropy = 1.7336`, `median = 1.7513`, `max = 1.8949`)**: `mean confidence = 0.2982`, **`median = 0.2870`**, `90th percentile = 0.3639`, and **`98.19%` of all MELD `dev` videos fall below `45%` confidence**! `disgust` F1 and `fear` F1 both collapse to **`0.0000`**.

### Root Cause 3: Single-Static-Frame `96x96` Sitcom Training Without Geometry/Glare Augmentation

1. **Weak MELD static training data ([`scripts/run_exp4_face_detection.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/run_exp4_face_detection.py)):** Extracts only **`1` static frame per clip** (`[0.50, 0.25, 0.75, 0.12, 0.88]`, taking the first frame with an MTCNN detection). MELD emotions are annotated at the *utterance/text* level (`Friends` sitcom dialogues), so a `t=50%` frame frequently captures a listener, transitional mouth movement, or off-peak expression—even baseline matched `dev` accuracy is only **`31.07%` (`Macro F1 = 0.1739`)**.
2. **`"augment": false` in `exp5_ce_weighted_config.json:15` ([`phase17b_common.py:378-382`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/phase17b_common.py#L378-L382)):** Training applied **only** random horizontal flip (`p = 0.5`) at `96x96` (`margin = 0.08`) for `5` epochs—**zero** pitch/head-tilt rotation, scale jitter, contrast/gamma shift, blur, or eyeglass/eye-region occlusion augmentation.
3. **Webcam Domain Shift (Spectacles + Downward Head Tilt + Subtle Expression):**
   - Eyeglass frames and lens glare directly occlude or distort the periorbital/eyebrow region (`96x96` resolution amplifies rim contrast into high-frequency edge artifacts).
   - Slight downward head tilt foreshortens upper cheeks/jawline and makes upper eyelids appear lowered, overlapping with MELD's heavy `sadness` (`1.0386`) and `disgust` (`1.6518`) / `surprise` (`0.7826`) decision boundaries in un-rotated crops.

---

## 4. Frame-by-Frame & Perturbation Evidence (MELD `dev`, `task-4783`)

### 4.1 Real MELD `dev` Videos Inspected Frame-by-Frame (`ResNet18 Primary` vs `Fallback` vs `Legacy CNN`)

| MELD `dev` Clip (True Label) | `ResNet18` Primary Runtime (`raw_all_detections`) | Fallback (`focal g=3`) | Legacy CNN (`facial_cnn_best.pt`) | Frame / Track Dynamics Observed (`task-4783`) |
| :--- | :--- | :--- | :--- | :--- |
| **`dev_dia0_utt0`** (`sadness`) | `neutral (0.3301)`, `ent=1.7422`, `21` det / `3` tracks, `1028 ms` (`det=886ms`, `fwd=45ms`) | `neutral (0.2522)`, `ent=1.8320` | `anger (0.1599)`, `ent=1.9407` | Individual crop `mean ent=1.5905`, pooled `1.7422`. `f#9 det#2` peaked at `neutral (0.520, ent=1.273)`, `f#56 det#1` predicted `surprise (0.397)`, `det#0` predicted `sadness (0.315–0.447)`—averaging 3 people raised entropy `+0.152`. |
| **`dev_dia0_utt1`** (`surprise`) | `neutral (0.2943)`, `ent=1.7875`, `16` det / `2` tracks, `668 ms` (`fwd=48ms`) | `neutral (0.3476)`, `ent=1.7447` | `anger (0.1607)`, `ent=1.9404` | **Classic multi-face dilution:** On `f#22/26/31 det#1`, the actual expressive speaker strongly predicted **`surprise (0.550–0.632, ent=1.093–1.240)`**, while `f#17 det#0` hit `disgust (0.501)`, yet averaging both tracks flipped final prediction to `neutral (0.2943, ent=1.7875)`. |
| **`dev_dia74_utt19`** (`anger`) | `neutral (0.2614)`, `ent=1.8272`, `9` det / `2` tracks, `825 ms` (`fwd=18ms`) | `neutral (0.2385)`, `ent=1.8137` | `anger (0.1752)`, `ent=1.9335` | Frames `97, 292, 683` correctly predicted **`anger (0.366–0.421, ent=1.375–1.518)`**, while neutral/cutoff shots pulled the pooled average to `neutral (0.2614, ent=1.8272)`. |
| **`dev_dia10_utt3`** (`joy`, low-light `bright=27–33`) | `neutral (0.2771)`, `ent=1.8137`, `24` det / `3` tracks, `553 ms` (`fwd=40ms`) | `neutral (0.2848)`, `ent=1.7698` | `anger (0.1681)`, `ent=1.9374` | `f#145 det#0` predicted `joy (0.337, ent=1.521)`, diluted by 3 low-light faces into `neutral (0.2771, ent=1.8137)`. |
| **`dev_dia100_utt0`** (`neutral`) | `neutral (0.2212)`, `ent=1.7877`, `17` det / `4` tracks, `665 ms` (`fwd=25ms`) | `neutral (0.2373)`, `ent=1.7977` | `anger (0.1521)`, `ent=1.9433` | Near-flat distribution (`neutral 0.221, anger 0.205, sadness 0.198, joy 0.160, surprise 0.145`), nearly identical in flatness to user's webcam breakdown! |
| **`dev_dia5_utt9`** (`anger`) | **Runtime `ValueError`**: `No detectable faces found across 2 sampled frames (filtered_fp=2)` | Same rejection | Same rejection | Filtered 2 low-confidence/aspect-invalid FPs safely rather than fabricating `neutral`. |

*(Notice Legacy CNN sits near maximum entropy `1.933–1.943 / 1.9459` across every clip, confirming why Phase 17 replaced it, while Focal Fallback behaves almost identically flat (`ent = 1.745–1.832`), so switching to fallback would NOT solve webcam uncertainty.)*

### 4.2 Temporal Pooling Comparison on Inspected MELD `dev` Videos (`task-4783`)

| Pooling Method | `dev_dia0_utt0` (sad) | `dev_dia0_utt1` (surp) | `dev_dia74_utt19` (ang) | `dev_dia100_utt0` (neu) | Pattern Observed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`raw_all_detections_mean_prob` (Active)** | `neu 0.3301 (1.742)` | `neu 0.2943 (1.788)` | `neu 0.2614 (1.827)` | `neu 0.2212 (1.788)` | Consistently **lowest confidence & highest entropy** on every video |
| `single_frame_middle` | `neu 0.3820 (1.605)` | `disgust 0.5013 (1.395)` | **`anger 0.4213 (1.375)`** | `sadness 0.4419 (1.513)` | Sharper peaks, lower entropy (`-0.14` to `-0.45`), captures middle anger in `dia74_utt19` |
| `track_arith_mean_prob` | `neu 0.3548 (1.722)` | `neu 0.3253 (1.732)` | `neu 0.2347 (1.854)` | `sadness 0.3589 (1.621)` | Isolates primary track, reduces multi-person crosstalk |
| `track_mean_logit` | `neu 0.3820 (1.678)` | `neu 0.3441 (1.708)` | `neu 0.2771 (1.773)` | `neu 0.3272 (1.621)` | Logit pooling preserves sharper multiplicative evidence than prob averaging |
| **`track_topk3_prob`** | **`neu 0.3951 (1.615)`** | `neu 0.3342 (1.672)` | **`anger 0.2832 (1.660)`** | **`sadness 0.4519 (1.454)`** | Recovers correct `anger` on `dia74_utt19`, cuts entropy & lifts confidence `>45%` on strong segments |

### 4.3 Controlled Crop Perturbation Sensitivity on Clean Face (`dev_dia0_utt0 f#37`, `task-4783`)

To isolate the optical/geometric conditions described in the user's ~8s webcam test (spectacles, slight downward tilt, subtle contrast) without modifying code, four controlled perturbations were tested on crop `dev_dia0_utt0 f#37`:

| Controlled Crop Variant | Top Prediction & Conf. | Entropy $H(p)$ | Top-3 Class Probabilities | Diagnostic Match to User Live Test |
| :--- | :--- | :---: | :--- | :--- |
| **`baseline_crop`** | `neutral (38.20%)` | `1.6050` | `neutral 38.2%, anger 27.1%, joy 12.4%` | Reasonable peaked baseline |
| **`downward_pitch_proxy`** *(top-edge/vertical foreshorten shift)* | `neutral (39.35%)` | `1.7153` (`+0.110`) | `neutral 39.3%, anger 15.9%, joy 15.1%` | Pitch shift suppresses upper-face emotional cues (`anger` `-11.2 pts`) and inflates entropy |
| **`subtle_low_contrast`** *(compressed facial dynamic range)* | `neutral (27.89%)` | **`1.8373`** (`+0.232`) | `neutral 27.9%, anger 17.0%, joy 15.9%` | Calm/low-contrast expression slashes top confidence (`-10.3 pts`) near `1.85` gate |
| **`spectacles_glare_proxy`** *(periorbital contrast/glare disturbance)* | **`disgust (19.56%)`** | **`1.8635 > 1.85`** | **`disgust 19.6%, joy 18.4%, neutral 17.8%`** | **Reproduces user symptom closely (`19.56%` vs user `19.5%`; `ent 1.8635` vs `1.883`; `disgust` jump to `~18–20%`)!** |

**Interpretation:** In a model trained at `96x96` without eye-region occlusion or photometric/tilt augmentation, glare/frame edges across the spectacles band immediately flip class dominance (`disgust`/`surprise`) and spike entropy past `1.85`!

---

## 5. Multimodal Gating Behavior & Why Multimodal Mode Stays Protected

[`ExtendedMultimodalFusionEngine`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/multimodal_extension.py) enforces two strict video quality gates before assigning `VALIDATED_LAMBDA_VIDEO = 0.08`:
- `MIN_FACE_COVERAGE = 0.40`
- **`MAX_VIDEO_ENTROPY = 1.85`**

On MELD `dev` (`N = 1,102` detected videos, Script 1):
- `51 / 1,102` (**4.6%**) exceed `entropy > 1.85`
- `14 / 1,102` (**1.3%**) fall below `coverage < 0.40`
- `65 / 1,102` (**5.9%**) trigger a video quality gate and receive `w_v = 0.00` (advisory only)

For the user's ~8s test (`coverage = 100%`, `entropy = 1.883 > 1.85`), **the existing Phase 18D safety gate works as designed in Multimodal Studio**: it flags `video_high_entropy`, sets **`w_v = 0.00`**, and relies on `Candidate 4` (`Text + Speech`) so noisy uncertain facial scores do **not** corrupt fused predictions. However, when running the standalone **Face** tab, raw `ResNet18 + raw_all_detections_mean_prob` probabilities are surfaced unfiltered—explaining why standalone Face displays `"Surprise (19.5%) — Low-confidence estimate (<45%)"`.

---

## 6. Summary Conclusions & Next Recommended Steps (Awaiting Approval)

1. **Do NOT replace `exp5_ce_weighted` with Legacy CNN (`ent ~1.94`) or Focal Fallback (`exp5_focal_g3_wt`, dev `mean ent=1.7264`)** — neither solves standalone webcam flatness.
2. **Short-term fix (no retraining):** Benchmark replacing `raw_all_detections_mean_prob` with sharpness-weighted / top-k track pooling (`track_topk3_prob` or `track_mean_logit`) + post-hoc **temperature / neutral-prior calibration** fitted strictly on MELD `dev` logits (`zero_test_split_access = True`).
3. **Academic medium-term fix (model V2):** Retrain facial CNN/ResNet on multi-frame sharpness-selected MELD/facial crops (or stronger `224x224` resolution) **with** `"augment": true` enabling `±15°` rotation/pitch, scale jitter, brightness/contrast shifts, and eye-region spectacles glare/cutout augmentation, and calibrate output logits cleanly.

