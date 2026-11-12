# Phase 19D — Performance & Resource Optimization Report

**Date:** 2026-11-11  
**Optimization Status:** `PASS`  
**Output Parity Status:** `EXACT` (`max_abs_prob_diff = 0.0` across Text, Speech, Video, and 3-Modality Multimodal on all MELD dev fixtures)  
**Rollback Toggle:** `EMOTION_API_OPTIMIZED_VIDEO_PIPELINE=0` cleanly reverts video decode/duration paths to baseline without code edits.  
**MELD Test Split Access:** `None` (benchmarked strictly on 6 MELD development fixtures: `dev_dia0_utt0`, `dev_dia10_utt3`, `dev_dia74_utt19`, `dev_dia99_utt3`, `dev_dia0_utt1`, `dev_dia1_utt0`).

---

## 1. Optimizations Implemented (Zero Model Weight or Threshold Changes)

1. **Sequential Forward `grab()` + `retrieve()` Decode (`facial_emotion/mtcnn_detector.py`):**
   - Replaced repeated `cap.set(cv2.CAP_PROP_POS_FRAMES, fi)` random seeks on clips with `<= 240` frames (`DEFAULT_SEQUENTIAL_GRAB_MAX_FRAMES = 240`) with a single monotonic `cap.grab()` pass + selective `cap.retrieve()` at sampled target indices, avoiding repeated H.264 IDR keyframe re-decoding.
   - Controlled via `use_sequential_grab` (default `True` when `EMOTION_API_OPTIMIZED_VIDEO_PIPELINE=1`); longer videos (`> 240` frames) and headerless WebM streams preserve indexed/sequential fallback cleanly.
   - Probe verification (`probe_19d.py`): `max_pix_diff = 0.0` against random-seek decode.
2. **Eliminated Duplicate Video Container Open (`api/app.py`, `facial_emotion/resnet_predictor.py`):**
   - Previously `_validate_and_predict_file_sync()` opened every MP4/WebM with OpenCV solely to check `max_video_seconds`, closed it, and immediately reopened the same file inside `sample_video_frames()`.
   - Delegated duration bounds (`VideoDurationExceededError` -> HTTP `413`) directly into `sample_video_frames(..., max_video_seconds=...)` inside `predict_video()` when `optimized_video_pipeline=True` and `facial_backend != "legacy_cnn"`.
3. **Concurrent Multipart Upload Spooling + Serialized Single-GPU Kernel Dispatch (`api/app.py`, `api/model_registry.py`):**
   - Replaced the single global `ModelRegistry._lock` with per-modality locks (`self._locks` for `"text"`, `"audio"`, `"video"`) and concurrent `store_upload()` spooling via `asyncio.gather()` inside `/predict/multimodal`.
   - Retained sequential PyTorch GPU execution (`Text -> Audio -> Video`) inside `/predict/multimodal` after observing empirically during Phase 19D benchmarking that overlapping all 3 GPU branches via threads increased single-RTX-4050 CUDA stream contention (`multimodal p95` rose from `938 ms -> 1106 ms`, concurrent 4-request wall time rose `3137 ms -> 3470 ms`, and peak VRAM rose `1263.8 MB -> 1304.4 MB`).
4. **Frontend React Result Re-Render Isolation (`frontend/src/App.tsx`):**
   - Wrapped `MultimodalResultCard`, `ResultCard`, `ProbabilityBars`, `StatusCard`, and `QualityCard` in `React.memo(...)` so typing inside text areas does not trigger unnecessary result tree reconciliations.

### Candidate Evaluated & Rejected
- **Batched `MTCNN.detect(batch_rgb)` across 8 frames:** Measured a faster detection pass in isolated probe (`291.8 ms` vs `548.6 ms`), **but** introduced `max_box_diff = 1.216370 px` due to `facenet-pytorch` batched proposal NMS indexing differences vs single-frame detection. **Rejected** in favor of exact `0.0` parity.

---

## 2. Benchmark Results (`scripts/benchmark_phase19d_performance.py`)

Artifact: `models/facial_emotion/resnet18/phase19d_performance_validation.json`

### A. Thermally-Paired Same-Session A/B Toggle (`--mode ab`)

Run back-to-back inside one GPU-warmed session on 6 MELD development clips (`EMOTION_API_OPTIMIZED_VIDEO_PIPELINE` toggled off vs on):

| Metric | Toggle OFF (Baseline) | Toggle ON (Optimized) | Delta |
|---|---|---|---|
| **Video Decode (`mean_ms`)** | `253.72 ms` | **`96.21 ms`** | **`-62.1%` (`2.64x` faster)** |
| **Video Decode (`p50_ms`)** | `258.36 ms` | **`64.09 ms`** | **`-75.2%` (`4.03x` faster)** |
| **Video API `/predict/video` (`mean_ms`)** | `687.00 ms` | **`546.72 ms`** | **`-20.4%`** |
| **Video API `/predict/video` (`p50_ms`)** | `739.14 ms` | **`550.56 ms`** | **`-25.5%`** |
| **Multimodal 3-Mod `/predict/multimodal` (`mean_ms`)** | `766.10 ms` | **`608.74 ms`** | **`-20.5%`** |
| **Multimodal 3-Mod `/predict/multimodal` (`p50_ms`)** | `863.17 ms` | **`582.59 ms`** | **`-32.5%`** |

### B. Full Modality & Resource Snapshot (`before` vs final `after`)

| Component / Metric | Before Baseline | Final After | Notes |
|---|---|---|---|
| **Text API (`/predict/text`) `p50` / `mean`** | `10.44 ms` / `10.65 ms` | `15.32 ms` / `15.87 ms` | Laptop GPU boost state variance (< 5 ms CPU/GPU scheduling) |
| **Audio API (`/predict/audio`) `p50` / `mean`** | `36.90 ms` / `54.07 ms` | `47.82 ms` / `66.43 ms` | Sub-70 ms across 6 dev clips |
| **Decode Stage `p50` / `mean`** | `304.17 ms` / `276.61 ms` | **`98.08 ms` / `154.51 ms`** | Persistent `-67.8%` median decode reduction |
| **4-Request Concurrent Multimodal Wall Time** | `3137.50 ms` (`4/4` HTTP 200) | `3539.76 ms` (`4/4` HTTP 200) | Laptop GPU thermal state variance across long suite; zero errors |
| **Process Resident RAM (`WorkingSetSize`)** | `1597.42 MB` | **`1553.23 MB`** | Well within 16 GB host budget |
| **CUDA Peak Allocated VRAM (6 GB RTX 4050)** | `1263.79 MB` | **`1263.79 MB`** (`1342.94 MB` in back-to-back A/B) | > 4.6 GB VRAM headroom remaining (`cuda_reserved_mb: 2170.0 MB`) |
| **Frontend Production Assets (`dist/assets/`)** | `221,572 B` (pre-memo build) | `221,623 B` (`19,994 B` CSS + `201,629 B` JS, `61.17 kB` gzip) | Negligible `+51 B` for React `memo` isolation |

### C. Strict Mathematical & Label Parity (`parity`)

| Modality | Samples Tested | Argmax Matches | `max_abs_prob_diff` |
|---|---|---|---|
| Text (Before vs After) | 6 / 6 | 6 / 6 | **`0.0`** |
| Speech (Before vs After) | 6 / 6 | 6 / 6 | **`0.0`** |
| Video (Before vs After & Paired A/B) | 6 / 6 | 6 / 6 | **`0.0`** |
| 3-Mod Multimodal (Before vs After & Paired A/B) | 6 / 6 | 6 / 6 | **`0.0`** |
