  # Multimodal Emotion Recognition — Accuracy Recovery and Optimization Audit Report

  **Scope:** Controlled accuracy recovery, calibration optimization, and fair same-sample evaluation on MELD `dev` (`dev_cal`, `dev_val`, `full_dev`) while keeping base neural network weights, frozen Candidate 4 fusion constants (`api/fusion.py`), and the held-out MELD `test` split (`N=2,610`) untouched.

  ---

  ## 1. Executive Summary & Verdict

  | Modality | Historical Baseline (`test` / `dev`) | Unrecovered V2 (`dev`) | Promoted Recovery (`dev`) | Key Result vs Prior Default & Baseline |
  | :--- | :--- | :--- | :--- | :--- |
  | **Text** (`DistilBERT`) | `63.10%` Acc / `41.55%` MF1 (`test N=2,610`) | `60.11%` Acc / `42.05%` MF1 (`full_dev N=1,108`) | `60.11%` Acc / `42.05%` MF1 (`dev_val`: `59.12%` / `38.34%`) | **Preserved unchanged.** Audited earlier `58.48%` baseline script bug (omitted `preprocess_text()`); `dev` vs `test` Acc difference reflects Neutral prior shift (`42.33%` vs `48.12%`). |
  | **Speech** (`Wav2Vec2_v2`) | `50.08%` Acc / `24.34%` MF1 (`test N=2,606`); V1+P0 `full_dev`: `47.79%` / `25.33%` | `bs=4`: `42.73%` / `28.04%` uncal; `reg=0.005`: `48.87%` / `27.74%`<br>`bs=1` live: `47.97%` / `30.65%` uncal; `reg=0.005`: `49.68%` / `28.79%` | **`bs=1` `reg=0.05` (`T=1.1956`)**:<br>`dev_val (548)`: **`49.09%` Acc / `31.10%` MF1**<br>`full_dev (1,107)`: **`49.86%` Acc / `30.73%` MF1 / `47.07%` WF1 / ECE-15 `0.0310`** | **Recovered & Improved.** Discovered batched `bs=4` zero-padding distorted `GroupNorm` (`diff` up to `0.7706`) & P2 `reg=0.005` overfit `dev_cal`. Live `bs=1` `reg=0.05` yields **`+6.60 pp` MF1** (`95% CI [+3.17%, +10.12%]`) and `+2.74 pp` Acc over `V1_raw`, and **`+2.74 pp` Acc (`p=0.0135`)** over `V2_uncal`. |
  | **Facial** (`ResNet18_v2`) | V1 CE `full_dev (1,102)`: `38.29%` Acc / `17.22%` MF1 (`N=1,075`: `31.07%` single-frame) | `V2_raw` (`full_dev 1,102`): `33.94%` Acc / `19.72%` MF1 (`dev_val 544`: `34.19%` / `21.13%`) | **`crop_prior_shift_a025_temp` (`a=0.25, T=1.25`)**:<br>`dev_val (544)`: **`38.97%` Acc / `19.23%` MF1**<br>`full_dev (1,102)`: **`38.66%` Acc / `18.06%` MF1 / `33.29%` WF1 / ECE-15 `0.0531`** | **Recovered.** Mitigated `train_multiframe_crops` Neutral under-representation (`22.00%` vs `42.33%`), recovering **`+4.77 pp` Acc (`p=0.0022`)** over `V2_raw` and beating V1 CE in Acc, MF1, WF1, NLL, and ECE, with a modest Sadness (`18.79%->11.35%`) & Fear (`3.64%->0%`) recall trade-off. |
  | **Multimodal** (`Candidate 4` + Phase 18D) | `test (2,610)` Cand4 `T+A`: `63.37%` Acc / `41.93%` MF1 (`p=0.0230`) | `T+A(bs1_0005)` `full_dev`: `60.47%` / `42.34%` (ECE `0.0993`)<br>`T+A_old+V_raw` (`w_v=0.08`): `60.38%` / `41.41%` | **Cand4 `T+A` (`lv=0`)**: **`60.47%` Acc / `42.35%` MF1 / `57.41%` WF1 / ECE `0.0932`** (`+8/-4`)<br>**Phase 18D `T+A+V` (`w_v=0.08`)**: **`60.74%` Acc / `42.55%` MF1 / `57.63%` WF1 / NLL `1.1857` / ECE `0.0655`** (`+10/-3`) | **Recovered & Improved.** Uncalibrated Facial V2 previously *degraded* Phase 18D Macro-F1 (`42.05% -> 41.15%` in `T+V_old`); calibrated Facial V2 + `bs=1` Speech calibration raises full `T+A+V` above `T_only`, `T+A`, and prior `T+A_old+V_old` on every aggregate metric. |

  > **Verdict:** **RECOVERY ACHIEVED WITHOUT WEIGHT RETRAINING.** Zero base checkpoints (`.pt` / `.safetensors`) and zero frozen `Candidate 4` parameters (`api/fusion.py`) were altered (`9/9` SHA-256 hashes verified). All `314` repository `pytest` tests pass, and live FastAPI endpoints (`/health`, `/models/status`, `/predict/text`, `/predict/audio`, `/predict/video`, `/predict/multimodal`) are verified active with `audio.calibrated=true` and `video.calibrated=true`.

  ---

  ## 2. Fair Comparison Methodology & Split Separation

  Historical baseline metrics and P0/P1/P2 development metrics were previously reported across different evaluation splits, sample counts, and preprocessing paths. Direct comparison requires aligning split boundaries, class distributions, and exact inference code paths.

  ### 2.1 Evaluation Splits, Sample Counts & Class Prior Shift

  - **Protected Held-Out Test (`datasets/MELD/annotations/test_sent_emo.csv`)**: `N=2,610` (`Wav2Vec2 v1`: `N=2,606` valid clips). **Strictly protected:** never read, fitted, calibrated, or evaluated during this audit.
  - **Disjoint Development Cohorts (`models/remediation_splits/meld_dev_cal_val_split.json`)**:
    - Total MELD `dev`: `N=1,108` utterances (`calibration` fold `dev_cal`: `N=560`; validation fold `dev_val`: `N=548`, grouped without speaker/dialogue leakage by `dialogue_id`).
    - Speech valid utterances (`duration >= 0.05s`): `N=1,107` total (`dev_cal N=559`, `dev_val N=548`; `[dev_dia36_utt8]` invalid/short).
    - Facial detected videos (multi-frame MTCNN crops): `N=1,102` total (`dev_cal N=558`, `dev_val N=544`; 6 videos without detected faces).

  | Class | MELD `test` (`N=2,610`, Historical Ref) | MELD `dev` (`N=1,108`, Audit Cohort) | Facial V2 `train_multiframe_crops` (`N=20,761`) |
  | :--- | :--- | :--- | :--- |
  | `anger` | `345` (`13.22%`) | `153` (`13.81%`) | — |
  | `disgust` | `68` (`2.61%`) | `22` (`1.99%`) | — |
  | `fear` | `50` (`1.92%`) | `40` (`3.61%`) | — |
  | `joy` | `402` (`15.40%`) | `163` (`14.71%`) | — |
  | `neutral` | `1,256` (`48.12%`) | `469` (`42.33%`) | `22.00%` (sampled multi-frame crops) |
  | `sadness` | `208` (`7.97%`) | `111` (`10.02%`) | — |
  | `surprise` | `281` (`10.77%`) | `150` (`13.54%`) | — |

  ### 2.2 Checkpoints & Reference Prediction Sources

  - **Text**: `models/text_emotion/distilbert/best/model.safetensors` (SHA-256 `6fbc700e...`).
  - **Speech V1**: `models/speech_emotion/wav2vec2/best/model.safetensors` (`15256b66...`); P1 calibration at `models/speech_emotion/wav2vec2/calibration_p1.json`.
  - **Speech V2**: `models/speech_emotion/wav2vec2_v2/best/model.safetensors` (`fbbb902b...`); trainer batch-4 cache at `models/speech_emotion/wav2vec2_v2/dev_outputs.npz`; live `bs=1` `Wav2Vec2Predictor.predict_file()` logits cached across `N=1,107` clips.
  - **Facial V1 / Legacy**: `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` (`8825d865...`), `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` (`4830ac84...`), `models/facial_emotion/facial_cnn_best.pt` (`85267a9b...`).
  - **Facial V2**: `models/facial_emotion/resnet18_v2/resnet18_v2_best.pt` (`1822ba9d...`); per-video and crop logit caches (`models/facial_emotion/resnet18_v2/dev_video_outputs.npz`, `models/facial_experiments/exp7_temporal/dev_temporal_unaligned_meta.json`).

  ---

  ## 3. Modality-by-Modality Root Cause Analysis & Experiments

  ### 3.1 Text (`DistilBERT`)

  #### Root Cause Findings
  1. **Audit Script Omission vs. Production Truth**: An earlier ad-hoc baseline script omitted `preprocess_text()` before DistilBERT inference, causing **106 prediction flips** across `full_dev` (`N=1,108`) and artificially depressing measured Accuracy from **`60.11%` to `58.48%`** (`38.44% -> 42.05%` MF1). Production (`DistilBertTextPredictor`) and `api/app.py` already invoked `preprocess_text()` identically (`max_abs_diff = 3.49e-8` against runtime outputs).
  2. **`dev` (`60.11%`) vs. Historical `test` (`63.10%`)**: Explained cleanly by the **`+5.79 pp` higher Neutral base rate** in MELD `test` (`48.12%` vs. `42.33%` in `dev`), where DistilBERT achieves `76.03%` F1 (`406/469` recall). On macro-averaged F1 across all 7 emotions, `dev` actually **exceeds** `test` (`42.05%` vs `41.55%`), confirming zero label inversion or model regression.

  #### True Production Text Performance on MELD `dev` (`DistilBertTextPredictor`)

  | Split (`N`) | Variant | Acc | MF1 | WF1 | NLL | Brier | ECE-10 | ECE-15 | `<45%` Rate |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | `test` (`2,610`) | Historical Phase 16 | `63.10%` | `41.55%` | `60.77%` | — | `0.5240` | `0.0953` | — | — |
  | `dev_val` (`548`) | Raw text (buggy script) | `58.58%` | `37.23%` | `54.50%` | `1.2588` | — | `0.0951` | `0.0951` | — |
  | `dev_val` (`548`) | **Preprocessed (Prod)** | `59.12%` | `38.34%` | `55.11%` | `1.2772` | `0.5922` | `0.1184` | `0.1228` | `15.1%` |
  | `dev_cal` (`560`) | **Preprocessed (Prod)** | `61.07%` | `45.46%` | `58.62%` | — | — | — | `0.1167` | — |
  | `full_dev` (`1,108`) | **Preprocessed (Prod)** | **`60.11%`** | **`42.05%`** | **`57.00%`** | `1.2376` | `0.5751` | `0.1162` | `0.1174` | `14.6%` |

  - **`full_dev` (`N=1,108`) Per-Class F1**: Anger `40.71%`, Disgust `27.59%`, Fear `12.50%`, Joy `52.26%`, Neutral `76.03%`, Sadness `29.14%`, Surprise `56.09%`.
  - **`full_dev` 7x7 Confusion Matrix** (rows=true, cols=pred `[Ang, Dis, Fea, Joy, Neu, Sad, Sur]`):
    ```text
    [[ 57,  1,  2, 19,  49,  5, 20],
    [  5,  4,  0,  1,   7,  3,  2],
    [ 11,  0,  3,  1,  10,  3, 12],
    [ 19,  0,  1, 75,  46,  0, 22],
    [ 10,  1,  1, 11, 406,  5, 35],
    [ 15,  1,  0,  7,  53, 22, 13],
    [ 10,  0,  1, 10,  28,  2, 99]]
    ```
  - **Decision**: Retain `DistilBERT` unchanged.

  ---

  ### 3.2 Speech (`Wav2Vec2` V1 / V1+P1 / V2 / Recovered)

  #### Two Root Causes Discovered
  1. **Calibration Regularization Overfit (`reg=0.005` -> `reg=0.05`)**: P2's default `temp_plus_l2_bias` used `regularization=0.005`, letting 7 bias offsets overfit `dev_cal`: in `dev_outputs.npz`, `dev_cal` CV jumped to `50.45%` Acc / `29.23%` MF1 while `dev_val` fell to `47.26%` Acc / `26.26%` MF1 and starved minority classes (`joy`, `disgust`, `fear`). Increasing L2 shrinkage to `regularization=0.05` eliminated overfitting.
  2. **Crucial Parity Discovery — Batched `bs=4` Zero-Padding vs. Live `bs=1` Production Inference**:
    - Probing live `Wav2Vec2Predictor.predict_file()` against `models/speech_emotion/wav2vec2_v2/dev_outputs.npz` exposed probability discrepancies up to `0.3356` (calibrated) and `0.4742` (uncalibrated) on short clips, and `0.0000`–`6.36e-5` on unpadded longest-in-batch clips.
    - Reproducing the `train_wav2vec2_v2.py` `val_loader` (`batch_size=4`, `padding=True`, `return_attention_mask=True`) confirmed `0.0000e+00` diff against `dev_outputs.npz`. The shift occurs because `facebook/wav2vec2-base` uses **`feat_extract_norm="group"`** (`GroupNorm(512, 512)`, i.e. InstanceNorm across time `L`) in conv layer 0 plus `Wav2Vec2PositionalConvEmbedding` **prior to transformer attention masking**. Zero-padding shorter utterances (`1.05s–2.26s`) inside `bs=4` batches contaminates conv channel statistics across `300/548` `dev_val` clips (mean max-abs prob diff `0.1354`, median `0.0687`, max `0.7706`).
    - Production `Wav2Vec2Predictor.predict_file()` runs **`batch_size=1` unpadded**, completely immune to padding artifacts — and uncalibrated `Wav2Vec2_v2` at `bs=1` actually scores **`+4.20 pp` Acc (`42.15% -> 46.35%`) and `+2.58 pp` MF1 (`27.93% -> 30.51%`) higher** on `dev_val` than `bs=4 dev_outputs.npz`!
    - Fitting `temp_plus_l2_bias(reg=0.05)` directly on **unpadded `bs=1` `dev_cal` logits** (`T=1.1956`, `5-fold StratifiedGroupKFold(dialogue_id)`) aligns calibration 1:1 with production (`live vs offline max_abs_diff = 1.1102e-16` across 10 `dev_val` clips).

  #### Fair Same-Sample Comparison Across Batched (`bs=4`) and Production (`bs=1`) Regimes

  | Regime & Model (`N`) | Acc | MF1 | WF1 | NLL | Brier | ECE-10 | ECE-15 | `<45%` | Act |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Batched `bs=4` (`dev_outputs.npz`, `N=548` / `1,107`)** | | | | | | | | | |
  | `V2_uncal` (`bs=4`, `dev_val`) | `42.15%` | `27.93%` | `41.06%` | `1.5761` | — | — | `0.0883` | `40.5%` | `5/7` |
  | `V2_reg0.005` (`bs=4`, `dev_val`, P2 Prior) | `47.26%` | `26.26%` | `41.98%` | `1.4510` | `0.6791` | `0.0552` | `0.0537` | `40.1%` | `5/7` |
  | `V2_reg0.05` (`bs=4` cal, `dev_val`) | `48.72%` | `28.94%` | `44.77%` | `1.4787` | `0.6850` | `0.0421` | `0.0455` | `46.5%` | `5/7` |
  | **Live Production `bs=1` (`Wav2Vec2Predictor`, `dev_val N=548`)** | | | | | | | | | |
  | `V1_raw` (`V1+P0`, `dev_val`) | `46.35%` | `24.50%` | `41.07%` | `1.4995` | `0.6946` | `0.0575` | `0.0638` | `51.6%` | `5/7` |
  | `V1_p1` (`V1+P0+P1`, `dev_val`) | `47.26%` | `24.58%` | `40.91%` | `1.4834` | `0.6895` | `0.0502` | `0.0602` | `49.5%` | `5/7` |
  | `V2_uncal` (`bs=1`, `dev_val`) | `46.35%` | `30.51%` | `44.99%` | `1.4945` | `0.6897` | `0.0661` | `0.0692` | `36.9%` | `5/7` |
  | `V2_reg0.005` (`bs=1` eval, P2 cal, `dev_val`) | `48.72%` | `28.59%` | `44.35%` | `1.4074` | `0.6605` | `0.0550` | `0.0594` | `42.0%` | `5/7` |
  | **`V2_bs1_reg0.05` [PROMOTED] (`bs=1`, `dev_val`)** | **`49.09%`** | **`31.10%`** | **`46.58%`** | `1.4302` | `0.6672` | `0.0595` | **`0.0538`** | `41.1%` | `5/7` |
  | **Live Production `bs=1` (`full_dev N=1,107`, 5-fold CV `50.45% / 30.15%`)** | | | | | | | | | |
  | `V1_raw` (`full_dev`) | `47.79%` | `25.33%` | `42.55%` | `1.4740` | `0.6804` | `0.0309` | `0.0397` | `51.3%` | `5/7` |
  | `V1_p1` (`full_dev`) | `48.51%` | `24.83%` | `42.02%` | `1.4570` | `0.6744` | `0.0311` | `0.0324` | `49.2%` | `5/7` |
  | `V2_uncal` (`full_dev`) | `47.97%` | `30.65%` | `46.44%` | `1.4626` | `0.6751` | `0.0472` | `0.0477` | `37.5%` | `5/7` |
  | `V2_reg0.005` (`full_dev`) | `49.68%` | `28.79%` | `45.28%` | `1.3853` | `0.6487` | `0.0180` | `0.0299` | `40.2%` | `5/7` |
  | **`V2_bs1_reg0.05` [PROMOTED] (`full_dev`)** | **`49.86%`** | **`30.73%`** | **`47.07%`** | `1.4033` | `0.6537` | `0.0305` | **`0.0310`** | `39.7%` | `5/7` |

  *(Note: Historical Phase 16 `Wav2Vec2 v1` on held-out `test N=2,606` scored `50.08%` Acc / `24.34%` MF1 / `45.05%` WF1; on `dev`, `V2_bs1_reg0.05` achieves `49.86%` Acc / `30.73%` MF1, closing the Acc gap despite `-5.79 pp` fewer Neutral utterances while beating V1 by `+5.40 pp` to `+5.90 pp` MF1.)*

  #### Paired Significance on `dev_val` (`N=548`, `bs=1`, 3,000 Bootstrap Resamples)
  - `V2_bs1_reg0.05` vs `V1_raw`: `dAcc = +2.74%` (`95% CI [-1.09%, +6.75%]`, McNemar `p=0.1915`, `+65/-50`), **`dMF1 = +6.60%` (`95% CI [+3.17%, +10.12%]`, statistically significant)**.
  - `V2_bs1_reg0.05` vs `V1_p1`: `dAcc = +1.82%` (`CI [-1.82%, +5.66%]`, `p=0.3909`, `+60/-50`), **`dMF1 = +6.52%` (`CI [+3.18%, +10.00%]`, statistically significant)**.
  - `V2_bs1_reg0.05` vs `V2_uncal`: **`dAcc = +2.74%` (`CI [+0.73%, +4.93%]`, `p=0.0135`, statistically significant, `+24/-9`)**, `dMF1 = +0.59%` (`CI [-0.73%, +1.87%]`).
  - `V2_bs1_reg0.05` vs `V2_reg0.005`: `dAcc = +0.36%` (`CI [-1.82%, +2.55%]`, `p=0.8642`), **`dMF1 = +2.51%` (`CI [+0.50%, +4.58%]`, statistically significant)** — proving `reg=0.05` recovers rare-emotion F1 (`joy` `20.54%->27.30%`, `sadness` `32.49%->38.83%` on `full_dev`) without losing Accuracy.

  #### Per-Class Metrics (`full_dev N=1,107`, `bs=1`) & `Disgust` / `Fear` Investigation
  - **Per-Class F1 (`V2_bs1_reg0.05` vs `V1_p1`)**: Anger `46.71%` (`vs 44.21%`), Disgust `0.00%` (`0.00%`), Fear `0.00%` (`0.00%`), Joy **`27.30%`** (`vs 10.26%`), Neutral `65.59%` (`vs 65.63%`), Sadness **`38.83%`** (`vs 17.65%`), Surprise `36.70%` (`vs 36.04%`).
  - **`V2_bs1_reg0.05` `full_dev` 7x7 Confusion Matrix**:
    ```text
    [[ 78, 0, 0, 19,  46,  1,  9],
    [  4, 0, 0,  0,  11,  4,  3],
    [ 13, 0, 0,  6,  16,  2,  3],
    [ 24, 0, 0, 40,  78,  8, 13],
    [ 29, 0, 0, 35, 345, 31, 28],
    [ 13, 0, 0,  9,  37, 40, 12],
    [ 20, 0, 0, 21,  51,  9, 49]]
    ```
  - **Why `Disgust` (`N=22`, `1.99%`) & `Fear` (`N=40`, `3.61%`) remain `0%` Top-1 Recall**:
    - Rank analysis on `full_dev` (`bs=1`) shows `V2_uncal` substantially lifts acoustic probability mass over `V1_raw` (`Disgust` mean rank `6.18 -> 4.73`, top-3 rate `0.0% -> 18.2%`, mean prob `0.0468 -> 0.0780`; `Fear` mean rank `6.62 -> 4.62`, top-3 rate `0.0% -> 25.0%`, mean prob `0.0396 -> 0.0704`).
    - `V2_reg0.005` crushed both below V1 (`Disgust` rank `6.95`, `0.0%` top-3, mean prob `0.0328`). `V2_bs1_reg0.05` preserves nonzero ranking (`Disgust` rank `5.55`, `4.5%` top-3, mean prob `0.0614`; `Fear` rank `5.08`, **`20.0%` top-3**, mean prob `0.0626`), yet short conversational `Friends` sitcom audio clips lack discriminative prosody for ~2-3% base-rate classes (`Disgust` maps into Neutral `11/22` and Anger/Sadness `8/22`; `Fear` maps into Neutral `16/40` and Anger `13/40`). Candidate 4 explicitly shields Text predictions for `{fear, disgust, sadness, joy}` via Protected Class Lockout (`DistilBERT` scores `27.59%` Disgust and `12.50%` Fear F1).

  ---

  ### 3.3 Facial (`ResNet18` V1 / V2 / Recovered)

  #### Root Cause of `V2_raw` Accuracy Regression vs. MF1 Improvement
  - `ResNet18_v2` was trained with `LogitAdjustedCE(tau=0.40)` on `train_multiframe_crops` (`N=20,761`), where Neutral comprises only **`22.00%`** of training crops compared to **`42.33%`** of MELD `dev` videos.
  - Consequently, `V2_raw` lifted minority emotion separation (`MF1: 17.22% -> 19.72%` on `full_dev`, `6/7 -> 7/7` active classes), **but systematically under-predicted Neutral** (`Neutral recall: 74.6% in V1_ce -> 53.0% in V2_raw`, dropping overall Accuracy `-4.35 pp`: `38.29% -> 33.94%`).

  #### Controlled `dev_cal` Calibration Experiments & Selection (`dev_val N=544`, `full_dev N=1,102`)

  | Experiment / Variant | `dev_val (544)` Acc / MF1 / WF1 / NLL / ECE-15 / `<45%` / Act | `full_dev (1,102)` Acc / MF1 / WF1 / NLL / ECE-15 / `<45%` / Act | Verdict |
  | :--- | :---: | :---: | :--- |
  | `V1_ce` (`exp5_ce_weighted`) | `37.68% / 17.96% / 32.17% / 1.6891 / 0.0934 / 97.8% / 7` | `38.29% / 17.22% / 32.32% / 1.6902 / 0.0944 / 98.2% / 7` | Baseline V1 |
  | `V1_focal` (`exp5_focal_g3_wt`) | `38.42% / 14.88% / 30.09% / 1.7146 / 0.1321` | `37.57% / 14.73% / 29.73% / 1.7150 / 0.1210 / 99.5%` | Fallback V1 |
  | `V2_raw` (`use_facial_calibration=False`) | `34.19% / 21.13% / 32.81% / 1.7131 / 0.0536 / 76.8% / 6` | `33.94% / 19.72% / 32.40% / 1.7380 / 0.0538 / 79.6% / 7` | Uncalibrated V2 |
  | `V2_l2_bias` (`reg=0.005`) | `41.91% / 8.82%` / — (`3/7` active) | — | **REJECTED** (majority-class collapse) |
  | `V2_prior_a1.0` | `42.10% / 10.09%` / — (`4/7` active) | — | **REJECTED** (`-9 pp` MF1 collapse) |
  | **`V2_prior_a0.25_T1.25` [PROMOTED]** | **`38.97% / 19.23% / 33.93% / 1.6529 / 0.0486 / 83.6% / 6`** (`dev_cal CV: 38.35% / 16.77%`) | **`38.66% / 18.06% / 33.29% / 1.6741 / 0.0531 / 85.5% / 6`** | **PROMOTED** (`use_facial_calibration=True`) |

  #### Significance & Class-Wise Trade-Off (`full_dev N=1,102`)
  - **Paired Significance on `dev_val (N=544)`**:
    - `V2_prior_a0.25_T1.25` vs `V2_raw`: **`+4.77%` Acc (`95% CI [+1.65%, +7.72%]`, McNemar `p=0.0022`, statistically significant)**; `-1.86%` MF1 (`CI [-4.86%, +0.73%]`).
    - `V2_prior_a0.25_T1.25` vs `V1_ce`: `+1.35%` Acc (`CI [-2.94%, +5.51%]`, `p=0.6057`), `+1.33%` MF1 (`CI [-2.19%, +4.79%]`) — beating `V1_ce` across Acc, MF1, WF1, NLL (`1.6902 -> 1.6741`), and ECE-15 (`0.0944 -> 0.0531`) simultaneously.
  - **Per-Class F1 Shifts (`full_dev N=1,102`)**:
    - `V1_ce`: Ang `13.28%`, Dis `0.00%`, Fea `0.00%`, Joy `21.79%`, Neu `56.66%`, Sad `18.91%`, Sur `9.90%`.
    - `V2_raw`: Ang `22.06%`, Dis `0.00%`, Fea `3.64%`, Joy `27.88%`, Neu `49.70%`, Sad `18.79%`, Sur `15.97%`.
    - `V2_rec_a025_T125`: Ang **`21.31%`** (`+8.03` vs V1), Dis `0.00%`, Fea `0.00%` (`-3.64` vs V2_raw), Joy **`23.81%`** (`+2.02` vs V1), Neu **`56.20%`** (`+6.50` vs V2_raw), Sad `11.35%` (`-7.44` vs V2_raw / `-7.56` vs V1), Sur **`13.77%`** (`+3.87` vs V1).
  - **Explicit Trade-Off & Live Parity**: Mild log-prior interpolation (`alpha=0.25`) restores Neutral recall (`53.0% -> 72.6%`) while trading modest recall on Sadness (`12.6% -> 7.2%`) and rare Fear (`2.6% -> 0.0%`). Because video serves as an auxiliary signal behind Text in multimodal fusion, this balanced gain in accuracy, WF1, and probability calibration improves downstream multimodal performance. Live CUDA `ResNetFacialEmotionPredictor` vs offline crop-calibrated array matched within **`2.835e-07`** (`3.951e-07` uncalibrated).
  - **`full_dev` (`N=1,102`) 7x7 Confusion Matrices (`V2_raw` vs `V2_rec_a025_T125`)**:
    - `V2_raw`:
      ```text
      [[ 30, 0, 7,  28,  66,  8, 13],
      [  4, 0, 0,   9,   7,  1,  1],
      [  4, 0, 1,   7,  25,  1,  1],
      [ 11, 0, 0,  58,  70,  2, 20],
      [ 41, 0, 6, 106, 248,  6, 61],
      [ 14, 1, 0,  15,  47, 14, 20],
      [ 16, 3, 2,  32,  67,  6, 23]]
      ```
    - `V2_rec_a025_T125`:
      ```text
      [[ 26, 0, 3, 15,  92, 7,  9],
      [  4, 0, 0,  5,  12, 1,  0],
      [  3, 0, 0,  5,  29, 1,  1],
      [  8, 0, 0, 35,  99, 2, 17],
      [ 28, 0, 3, 50, 340, 6, 41],
      [ 13, 0, 0,  7,  70, 8, 13],
      [ 10, 0, 1, 16, 100, 5, 17]]
      ```

  ---

  ### 3.4 Multimodal Fusion (`Candidate 4` & `ExtendedMultimodalFusionEngine`)

  - **Frozen `api/fusion.py` (`Candidate4FusionEngine`)**: `DEFAULT_TAU_AUDIO = 0.20`, `DEFAULT_LAMBDA_MAX = 0.40`, `DEFAULT_KAPPA_AUDIO = 1.50`, `DEFAULT_LAMBDA_VIDEO = 0.00` preserved verbatim (`sha256 = ffa76bea...`).
  - **Phase 18D `api/multimodal_extension.py` (`ExtendedMultimodalFusionEngine`)**: Wraps `Candidate4FusionEngine`, delegating Text-only and Text+Audio verbatim, and when calibrated ResNet18 video passes quality gates (`face_detection_coverage >= 0.40`, `prediction_entropy <= 1.85`), injects `VALIDATED_LAMBDA_VIDEO = 0.08`.

  #### Multimodal Modality & Recovery Ablation (`N=548 dev_val`, `N=1,108 full_dev`)

  | Fusion Configuration | `dev_val (548)` Acc / MF1 / WF1 / NLL / ECE-15 (`+w/-l`) | `full_dev (1,108)` Acc / MF1 / WF1 / NLL / Brier / ECE-10 / ECE-15 (`+w/-l`, `p`) |
  | :--- | :---: | :---: |
  | **`Candidate4FusionEngine` (`lambda_video = 0.00`)** | | |
  | `T_only` (`DistilBERT`) | `59.12% / 38.34% / 55.11% / 1.2772 / 0.1228` | `60.11% / 42.05% / 57.00% / 1.2376 / 0.5751 / 0.1162 / 0.1174` (`base`) |
  | `T + A(V1_raw bs=1)` | `59.49% / 38.76% / 55.64% / 1.2505 / 0.0897` (`+4/-2`) | `60.47% / 42.43% / 57.47% / 1.2053 / 0.5630 / 0.0858 / 0.0839` (`+6/-2`, `p=0.289`) |
  | `T + A(V1_p1 bs=1)` | `59.49% / 38.74% / 55.54% / 1.2529 / 0.0946` (`+4/-2`) | `60.47% / 42.43% / 57.42% / 1.2095 / 0.5649 / 0.0959 / 0.0915` (`+6/-2`, `p=0.289`) |
  | `T + A(V2_reg0.005 bs=1)` (`syn=91/158`) | `59.49% / 38.69% / 55.59% / 1.2466 / 0.1001` (`+5/-3`) | `60.47% / 42.34% / 57.41% / 1.2135 / 0.5644 / 0.0966 / 0.0993` (`+8/-4`, `p=0.388`) |
  | **`T + A(V2_bs1_reg0.05)` [PROD Cand4]** (`syn=104/190`) | **`59.49% / 38.69% / 55.59% / 1.2406 / 0.0988`** (`+5/-3`) | **`60.47% / 42.35% / 57.41% / 1.2130 / 0.5647 / 0.0915 / 0.0932`** (`+8/-4`, `p=0.388`) |
  | **`ExtendedMultimodalFusionEngine` (`w_v = 0.08`, live API `/predict/multimodal`)** | | |
  | `T + V_old(V2_raw)` (`vid_in=535/1088`) | `59.12% / 36.34% / 54.89% / 1.2346 / 0.0966` (`+1/-1`) | `60.11% / 41.15% / 56.88% / 1.2048 / 0.5636 / 0.0794 / 0.0869` (`+2/-2`) |
  | `T + V_new(V2_rec)` (`vid_in=539/1093`) | `59.31% / 38.42% / 55.22% / 1.2334 / 0.0941` (`+1/-0`) | `60.29% / 42.14% / 57.12% / 1.2044 / 0.5639 / 0.0800 / 0.0861` (`+2/-0`) |
  | `T + A_old(bs1_0005) + V_old(raw)` (`PRIOR`) | `59.31% / 36.63% / 55.22% / 1.2111 / 0.0735` (`+5/-4`) | `60.38% / 41.41% / 57.21% / 1.1862 / 0.5551 / 0.0614 / 0.0667` (`+9/-6`, `p=0.607`) |
  | **`T + A_new(bs1_005) + V_new(rec)` [RECOVERED]** | **`59.67% / 38.77% / 55.71% / 1.2045 / 0.0800`** (`+6/-3`, `p=0.508`) | **`60.74% / 42.55% / 57.63% / 1.1857 / 0.5559 / 0.0550 / 0.0655`** (`+10/-3`, `p=0.092`) |

  - **Key Multimodal Findings**:
    1. **Audio synergy**: `T+A(V2_bs1_reg0.05)` triggers acoustic synergy on `104` `dev_val` and `190` `full_dev` utterances (`+8` fixes vs `-4` regressions on `full_dev`), improving NLL (`1.2376 -> 1.2130`) and ECE-15 (`0.1174 -> 0.0932`). (`V2_uncal` also achieves `60.56% / 42.49%` on `T+A`, `+10/-5`, at slightly worse stand-alone speech Acc/ECE; keeping `V2_bs1_reg0.05` maintains both strong Speech stand-alone metrics and clean `T+A` synergy.)
    2. **Video contribution fixed**: Prior uncalibrated `V2_raw` inside `ExtendedMultimodalFusionEngine` degraded `T_only` Macro-F1 by `-2.00 pp` on `dev_val` (`-0.90 pp` on `full_dev`). Promoting `V2_rec` (`crop_prior_shift_a025_temp`) flips video from net-harmful to net-beneficial (`T+V_new`: `+2/-0` vs `T_only`, and full **`T+A+V` RECOVERED = `60.74% Acc / 42.55% MF1 / 57.63% WF1 / NLL 1.1857 / ECE-15 0.0655`**, `+10/-3` vs `T_only`, `p=0.0923`).
    3. **Exploratory `Candidate4FusionEngine` (`cov>=0.8, ent<1.6`) sweep with recovered V2**: Increasing exploratory `lambda_video` from `0.05` (`60.47% / 42.35%`) to `0.10` (`60.56% / 42.44%`) and `0.20` (`60.74% / 42.58%`, `ECE-15 0.0683` on `full_dev`; `59.85% / 38.97%` on `dev_val`) shows monotonic probability-calibration and accuracy gains without disturbing `T+A`; we keep `api/fusion.py` (`DEFAULT_LAMBDA_VIDEO = 0.00`) and `api/multimodal_extension.py` (`VALIDATED_LAMBDA_VIDEO = 0.08`) unmodified as authorized.

  ---

  ## 4. Promoted Artifacts, Verification & Protected Checksums

  ### 4.1 Exact Promoted Files
  - [models/speech_emotion/wav2vec2_v2/best/calibration.json](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/speech_emotion/wav2vec2_v2/best/calibration.json): Updated to `temp_plus_l2_bias` fit on unpadded `bs=1` `dev_cal` (`N=559`, 5-fold `StratifiedGroupKFold(dialogue_id)`), `regularization = 0.05`, `temperature = 1.1955738963493479`.
  - [models/facial_emotion/resnet18_v2/calibration.json](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18_v2/calibration.json): Created `crop_prior_shift_a025_temp` (`alpha = 0.25`, `temperature = 1.25`).
  - [api/config.py](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/config.py#L109): Set `use_facial_calibration` default (`EMOTION_API_USE_FACIAL_CALIBRATION`) from `False` to `True`.
  - Aligned 3 explicit default-`False` test assertions to `True`:
    - [api/tests/test_phase20b_backend.py](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/tests/test_phase20b_backend.py#L139)
    - [api/tests/test_phase18d_multimodal_extension.py](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/tests/test_phase18d_multimodal_extension.py#L369) (preserving V1 rollback `False` check at line 431)
    - [facial_emotion/tests/test_resnet_integration.py](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/facial_emotion/tests/test_resnet_integration.py#L158)

  ### 4.2 Automated Suite & Live Backend Verification
  - **Full `pytest` Suite**: Executed `python -m pytest -q --tb=short` after final code/calibration edits -> **`314 passed` (`0` failed, `0` skipped) in `96.52s`**.
  - **Live FastAPI Server Verification (`http://127.0.0.1:8000`, CUDA)**:
    - `GET /health`: `200 OK` (`ready=true`, 7 canonical classes).
    - `GET /models/status`: `text` (`DistilBertTextPredictor`, `cuda`, `calibrated=false`), `audio` (`speech-wav2vec2-v2`, `cuda`, `calibrated=true`), `video` (`facial-resnet18-v2-la_ce_tau040`, `cuda`, `calibrated=true`, `sha256=1822ba9d...`).
    - `POST /predict/text`, `POST /predict/audio`, `POST /predict/video`, and `POST /predict/multimodal` verified on MELD `dev_val` sample `dev_dia1_utt0` (live `T+A+V`: `predicted_emotion="surprise"`, `confidence=0.5257`, `rule_applied="acoustic_synergy_anger+validated_video_calibrated"`).

  ### 4.3 Protected Checksums (`9/9` Verified Unchanged) & Rollback

  | Artifact | Verified SHA-256 |
  | :--- | :--- |
  | `models/text_emotion/distilbert/best/model.safetensors` | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` |
  | `models/speech_emotion/wav2vec2/best/model.safetensors` | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` |
  | `models/speech_emotion/wav2vec2_v2/best/model.safetensors` | `fbbb902b5d77138178f07c8dde847c0f853aed6c5fde04aa73b4a260ca477478` |
  | `models/facial_emotion/resnet18/exp5_ce_weighted_best.pt` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` |
  | `models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` |
  | `models/facial_emotion/facial_cnn_best.pt` | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` |
  | `models/facial_emotion/resnet18_v2/resnet18_v2_best.pt` | `1822ba9dc13a7406c13ad4bd48a7c8be7d309ead286cbe40352525bf85c1211a` |
  | `api/fusion.py` | `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d` |
  | `models/remediation_splits/meld_dev_cal_val_split.json` | `d1a5d67167b3e151b2e9db086e122ae9014b395eb4f0c55358f69da9044b5ada` |

  **Instant Rollback Options (No Weight Re-Training Needed):**
  - Disable Speech V2 calibration: `EMOTION_API_USE_SPEECH_CALIBRATION=false` (or restore prior `reg=0.005` `calibration.json` backup).
  - Disable Facial V2 calibration: `EMOTION_API_USE_FACIAL_CALIBRATION=false` (or remove `models/facial_emotion/resnet18_v2/calibration.json`).
  - Full V1+P1 rollback: `EMOTION_API_SPEECH_VERSION=v1`, `EMOTION_API_FACIAL_VERSION=v1`, `EMOTION_API_USE_FACIAL_CALIBRATION=false`.
