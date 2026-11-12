# Phase 17D — Final Facial Candidate Validation Report

**Date:** 2026-11-11  
**Script:** [`scripts/run_phase17d_validation.py`](../scripts/run_phase17d_validation.py)  
**Artifact:** [`models/facial_experiments/phase17d_validation.json`](../models/facial_experiments/phase17d_validation.json)  
**Final Overall Status:** **PASS** (11 / 11 validation gates passing; Staged/Shadow integration ready only — NOT deployed to production)

---

## 1. Validated Candidate Pair & Checksum Verification

| Role | ID / Checkpoint | Architecture | Expected SHA-256 | Verified SHA-256 | Match |
|---|---|---|---|---|---|
| **Primary** | `exp5_ce_weighted_best.pt` | `resnet18` (`imagenet`) | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | **TRUE** |
| **Fallback** | `exp5_focal_g3_wt_best.pt` | `resnet18` (`imagenet`) | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | **TRUE** |

### Protected Production & Baseline Assets (`all_ok = True`)
- `models/facial_emotion/facial_cnn_best.pt`: `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` (UNTOUCHED)
- `models/facial_experiments/exp1_corrected_cnn/exp1_corrected_cnn_best.pt`: `a2be816a67bb094026bdf43fb31884ee11812882de5f2c8108f26262b0f89613` (UNTOUCHED)
- `models/facial_experiments/exp2_weighted_cnn/exp2_weighted_cnn_best.pt`: `422a5934c5f1337733eeb9c5274f4bb52f19d1ad81b7d1656b0eb2d25cc8aa8a` (UNTOUCHED)
- `models/facial_experiments/exp3_resnet18/exp3_resnet18_best.pt`: `106b524e837417373dd904518da3ad389287c843d93c7f92596c5b1daf2b1446` (UNTOUCHED)
- `facial_emotion/model.py`: `08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b` (UNTOUCHED)
- `facial_emotion/detector.py`: `24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d` (UNTOUCHED)
- `facial_emotion/remediation.py`: `3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07` (UNTOUCHED)
- `api/fusion.py`: `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d` (UNTOUCHED)

---

## 2. Diagnostic Resolution & CPU/CUDA Numerical Parity

### Initial Validation Finding & Root Cause
- On the first Phase 17D execution (`task-2880`), `cpu_cuda_inference_parity` failed the `< 1e-4` threshold with `max_abs_probability_diff = 0.00034228` (`3.42e-4`), even though all `16 / 16` argmax predictions were identical.
- A targeted precision diagnostic (`scratch/check_tf32_parity.py`, `task-2897`) proved the divergence came solely from PyTorch's default `torch.backends.cudnn.allow_tf32 = True` on Ada Lovelace (`RTX 4050`):
  - Default `cudnn.allow_tf32 = True`: `max_diff = 0.00034228`, `argmax_match = True` (`16/16`)
  - Strict IEEE FP32 (`cudnn.allow_tf32 = False`, `cuda.matmul.allow_tf32 = False`): **`max_diff = 0.00000194` (`1.94e-6`)**, `argmax_match = True` (`16/16`)
- [`scripts/run_phase17d_validation.py`](../scripts/run_phase17d_validation.py) was updated to explicitly test **both** strict IEEE FP32 (`< 1e-4`) and default TF32 argmax parity (`16/16` and `< 1e-3`), then re-executed in full (`task-2909`).

---

## 3. Comprehensive Verification Gate Results (`task-2909`)

| Check Name | Outcome | Observed Measurement |
|---|---|---|
| `checkpoint_sha256_verified` | **PASS** | Primary (`8825d865...`) & Fallback (`4830ac84...`) exact match |
| `seven_canonical_classes_verified` | **PASS** | `['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']` |
| `imagenet_vs_legacy_preprocessing_parity` | **PASS** | Strict isolation verified between ImageNet RGB `(2,3,96,96)` and legacy `[-1, 1]` |
| `cpu_cuda_strict_fp32_parity` | **PASS** | `max_abs_diff = 1.94e-6 < 1e-4`; argmax `16/16` identical |
| `cpu_cuda_default_tf32_argmax_parity` | **PASS** | `max_abs_diff = 3.42e-4 < 1e-3`; argmax `16/16` identical (`cudnn.allow_tf32=True`) |
| `probabilities_finite_and_sum_to_one` | **PASS** | All finite & `>= 0`; `max_prob_sum_error = 1.19e-7 < 1e-5` |
| `single_image_inference_ok` | **PASS** | `16/16` MELD dev crop batch verified on CPU & CUDA |
| `zero_face_video_handled_safely` | **PASS** | Synthetic black video returned `detected=False` cleanly |
| `pytest_regression_suite_passed` | **PASS** | `python -m pytest -q`: **`271 passed, 6 warnings in 114.37s`** (`returncode=0`) |
| `protected_production_assets_untouched` | **PASS** | All 8 protected production files/checkpoints hash-verified |
| `zero_test_split_access` | **PASS** | MELD `test` split remained untouched throughout Phases 17B–17D |

### Single-Crop Latency & Memory (Final Phase 17D Run)
- **CUDA Latency / Peak Inference VRAM:** `7.28 ms/crop` (`60.66 MB` inference VRAM; note Exp 5 warm micro-benchmark achieved `2.58 ms/crop`)
- **CPU Latency:** `27.45 ms/crop`

---

## 4. Video & Edge-Case Stress Suite Results

| Stress Scenario | Input / Source | Frames (Total / Dec / Sampled) | Detections / Track Crops | Output / Prob Sum / Finite | Latency |
|---|---|---:|---:|---|---:|
| Live Camera Clip | `WIN_20261111_03_07_13_Pro.mp4` | `244 / 16 / 16` | `13` valid (`2` FP filtered), `2` tracks (`11` crops) | `sadness (0.2372)`, `sum=1.0000`, finite | `3,053.6 ms` |
| Zero-Face Synthetic | `blank_zero_face.mp4` (`320x240` black) | `15 / 8 / 8` | `0` valid | `detected=False`, no crash | `601.5 ms` |
| Zero-Face MELD Dev | `dev_dia5_utt9` | `2 / 2 / 2` | `0` valid (`2` FP filtered) | `detected=False`, no crash | `411.5 ms` |
| Multi-Face MELD Dev | `dev_dia0_utt0` (true `sadness`) | `67 / 8 / 8` | `21` valid, `3` tracks (`8` crops) | `neutral (0.3824)`, `sum=1.0000`, finite | `1,629.3 ms` |
| Ultra-Short MELD Dev | `dev_dia99_utt3` (`0.083s`, true `surprise`) | `2 / 2 / 2` | `2` valid, `1` track (`2` crops) | `neutral (0.6394)`, `sum=1.0000`, finite | `400.8 ms` |
| Long MELD Dev | `dev_dia74_utt19` (`28.53s`, true `anger`) | `684 / 8 / 8` | `9` valid, `2` tracks (`7` crops) | `anger (0.4211)`, `sum=1.0000`, finite | `1,384.7 ms` |
| Low-Light MELD Dev | `dev_dia10_utt3` (`luma=15.35`, true `joy`) | `146 / 8 / 8` | `24` valid, `9` tracks (`4` crops) | `joy (0.2297)`, `sum=1.0000`, finite | `1,356.1 ms` |
| Synthetic Degradation | `dev_dia0_utt0` (`orig`, `0.35x dark`, `blur sigma=1.5`) | Crop check | — | All `neutral`, all finite (`all_finite=True`) | — |

---

## 5. Remaining Limitations & Integration Prerequisites

1. **Deployment Status:** `deployed_to_production = False`. Live production [`facial_emotion/model.py`](../facial_emotion/model.py) and Candidate 4 multimodal fusion remain untouched pending explicit user review.
2. **Preprocessing Isolation Requirement:** Production integration MUST enforce model-aware preprocessing (`imagenet` RGB mean/std for `ResNet18` vs `[-1, 1]` for legacy CNN) using [`facial_emotion/remediation.py`](../facial_emotion/remediation.py) patterns so mismatched checkpoints never cross-contaminate.
3. **TF32 Precision Note:** If bit-tight (`< 1e-4`) CPU/CUDA probability parity is required on Ampere/Ada GPUs, disable `torch.backends.cudnn.allow_tf32` in the predictor; otherwise default TF32 introduces `~3.42e-4` rounding with identical top-1 class decisions (`16/16`).
4. **Detection Latency:** MTCNN + IoU tracking adds `~0.4–3.1s` per multi-frame video (`~165 ms/frame` at `360p` downscale, vs `28.4 ms/frame` for downscaled Haar) and requires GPU/CPU memory budgeting under concurrent requests.
5. **Minority-Class & Dataset Limits:** `disgust` (`n=22` dev) and `fear` (`n=40` dev) remain difficult on subtle sitcom frames (`disgust F1 = 0.0000` on selected single-frame `exp5_ce_weighted`), and facial cues reflect MELD-specific dataset distributions rather than verified clinical emotional ground truth.

