# Phase 18A — Integration Audit Report

**Date:** 2026-11-11  
**Scope:** End-to-end production readiness and compatibility audit across backend FastAPI service, model registry, facial checkpoints, MTCNN detection/tracking/pooling, Candidate 4 fusion, frontend studio UI, and test suites prior to Phase 18B–18E integration.  
**Gate Decision:** **PASS** (Ready to advance to Phase 18B — Facial Model Integration)

---

## 1. Checkpoint & Protected Asset Inventory (SHA-256 Verified)

All existing production and Phase 17 candidate artifacts were located and hash-verified prior to any code or model changes:

| Role | Relative Path | Size / Shape | Observed SHA-256 (Upper/Lower Match) | Status |
|---|---|---|---|---|
| **Primary Facial Candidate** | `models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_best.pt` | `44,797,666` bytes; `resnet18`, `fc.weight=(7, 512)`, `best_epoch=5` | `8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64` | **Verified** |
| **Fallback Facial Candidate** | `models/facial_experiments/exp5_focal_loss/exp5_focal_g3_wt_best.pt` | `44,797,666` bytes; `resnet18`, `fc.weight=(7, 512)`, `best_epoch=4` | `4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0` | **Verified** |
| **Legacy Facial CNN (Rollback)** | `models/facial_emotion/facial_cnn_best.pt` | `94,151` params (`facial-cnn-v1`) | `85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760` | **Protected** |
| **Production Text Model** | `models/text_emotion/distilbert/best/model.safetensors` | DistilBERT | `6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474` | **Protected** |
| **Production Speech Model** | `models/speech_emotion/wav2vec2/best/model.safetensors` | Wav2Vec2 | `15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e` | **Protected** |

### Phase 17 Candidate Checkpoint Internal Metadata Inspection (`weights_only=True`)

Both `exp5_ce_weighted_best.pt` and `exp5_focal_g3_wt_best.pt` self-describe their exact inference contract:
- **`classes`:** `["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]` (exact match to `MELD_CLASS_NAMES` / `facial_emotion.config.EMOTION_CLASSES`)
- **`architecture`:** `"resnet18"`
- **`input_normalization`:**
  - `mode`: `"imagenet"`
  - `color_order`: `"BGR_to_RGB"`
  - `image_size`: `96`
  - `scale`: `"1/255.0"`
  - `mean`: `[0.485, 0.456, 0.406]`
  - `std`: `[0.229, 0.224, 0.225]`
  - `detector`: `"mtcnn_640_fallback_native"`
  - `alignment`: `"square_unaligned_margin_0.08"`

**Note on expected checkpoint path vs repository layout:** The master specification referenced `phase17b_exp5_loss/exp5_ce_weighted_best.pt` and `phase17b_exp5_focal/exp5_focal_g3_wt_best.pt`; in the workspace both checkpoints are stored under `models/facial_experiments/exp5_focal_loss/`, with exact SHA-256 matches confirmed above. Before Phase 18B loads from a production directory, approved weights must be staged into a dedicated production path (e.g., `models/facial_emotion/resnet18/`) with manifest metadata while leaving both `models/facial_experiments/` and `models/facial_emotion/facial_cnn_best.pt` untouched.

---

## 2. Subsystem Audit Findings & Integration Gaps

### 2.1 Facial Predictor (`facial_emotion/`)
- **Current production (`facial_emotion/model.py`):**
  - `FacialEmotionPredictor` loads the 4-layer compact CNN (`create_model()`, `model_identifier="facial-cnn-v1"`), samples every 3rd frame (`sample_every=3`) with OpenCV Haar cascade (`FaceDetector`), and calls `preprocess_face_crops()` (`[-1.0, 1.0]`).
  - Contains the previously audited production bugs:
    1. `FacialEmotionPredictor.predict_faces()` divides by 2 + 0.5 when `normalize_input=False`, corrupting normalization if misconfigured.
    2. `num_sampled_frames` and `face_detection_coverage` in `FacialEmotionPredictor.predict_video()` estimate sampled frames from `round(len(faces)/3)` and clamp `min(1.0, len(faces)/num_sampled_frames)`, over-reporting coverage when frames contain multiple faces or zero faces.
- **Remediation utilities (`facial_emotion/remediation.py`):**
  - Already defines `NormalizationType` (`LEGACY_CNN`, `UNIT_INTERVAL`, `IMAGENET`), `IMAGENET_MEAN`, `IMAGENET_STD`, `remediated_preprocess_face_crops()`, and `VideoFrameAccounting`.
  - Still uses `OptimizedFaceDetector` (Haar) and `create_model()` (legacy CNN), not `ResNet18` or `MTCNN`.
- **Integration Plan (18B):**
  - Create an independent ResNet18 facial predictor (leaving `FacialEmotionPredictor` and `facial_cnn_best.pt` intact as configurable rollback) that enforces:
    - Startup SHA-256 integrity check + architecture/class-list metadata verification.
    - Strictly `IMAGENET` RGB `[0, 1]` scaled by `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)` on `96x96` uint8 BGR face crops.
    - Configurable strict IEEE FP32 toggle (`cudnn.allow_tf32` consideration) or documented `<1e-3` TF32 vs `<1e-4` strict FP32 CPU/CUDA parity guarantees.

### 2.2 MTCNN Detection, Square Cropping, Tracking, and Temporal Pooling
- **Verified Phase 17 implementation (`scripts/phase17b_common.py`, `scripts/run_exp7_temporal.py`):**
  - **Detection (`_detect_and_filter_frame`):** `MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709)`. Runs at `640px` width downscale first (`w > 640`), falling back to native resolution if no boxes are found. Filters false positives where `prob < 0.90`, `min(bw, bh) < 24.0`, or aspect ratio `bw / bh` outside `[0.40, 1.80]`.
  - **Crop Extraction (`crop_unaligned_square`):** Uses full-resolution original coordinates with `margin = 0.08`, square `side = max(bw, bh) * (1 + 2 * margin)`, clamped to frame bounds, resized to `96x96` (`INTER_CUBIC` for upscaling, `INTER_AREA` for downscaling).
  - **Tracking (`track_faces_across_frames`):** Uses original `(x, y, w, h)` boxes and frame diagonal for normalized center distance (`min_iou=0.08`, `max_center_dist=0.28`, max gap `3` frames), avoiding the legacy `96x96` resized-crop center collapse defect (`distance=0.0`). Selects primary track by `len(items) * sqrt(mean_area) * mean_prob`.
  - **Phase 17 Temporal Pooling Tradeoff (`N = 1,102` dev videos):**
    - `single_frame_middle`: `Acc=0.3004`, `Macro F1=0.1715`, `Weighted F1=0.2858`, `MinRec=0.1018`, `Active=7/7`.
    - `raw_all_detections_mean_prob`: `Acc=0.3829`, `Macro F1=0.1722`, `Weighted F1=0.3232`, `MinRec=0.0571`, `Active=7/7` (`disgust`/`fear` F1 drop to `0.0000`).
    - `track_arith_mean_prob`: `Acc=0.3466`, `Macro F1=0.1589`, `Weighted F1=0.3019`, `MinRec=0.0631`.
- **Integration Plan (18C):**
  - Promote clean production modules for MTCNN detection + normalized IoU/center tracking + uniform frame sampling (instead of reading all frames into memory when `CAP_PROP_FRAME_COUNT > 0`, with sequential fallback for WebM streams lacking reliable frame counts).
  - Preserve honest `VideoFrameAccounting` (`total_video_frames`, `frames_decoded`, `frames_sampled`, `frames_with_detected_faces`, `total_face_detections`, `false_positives_filtered`, `unique_tracks`, `selected_track_crops`), raise explicit non-fabricating `ValueError` (HTTP `422`) when zero valid faces exist, and benchmark pooling modes on the MELD `dev` split + 7 stress scenarios (normal, no-face, multi-face, short, long, low-light/degraded, invalid).

### 2.3 Model Registry, API Configuration, and Endpoints (`api/`)
- **Config (`api/config.py`):**
  - `ApiSettings` currently hardcodes `video_model_path = REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt"`, `device = os.environ.get("EMOTION_API_DEVICE", "cpu")`, `max_upload_bytes = 50MB`, `max_video_seconds = 60`.
  - **Gap:** Needs backend selector settings (`EMOTION_API_FACIAL_BACKEND`: ResNet18 default once Phase 18B/18C completes, legacy CNN rollback preserved; primary/fallback ResNet18 paths & expected SHA-256; temporal pooling config; strict FP32 toggle if configured).
- **Registry (`api/model_registry.py`):**
  - `ModelRegistry` protects `predict()` via `threading.RLock()` and exposes `status()` (`loaded`, `model_identifier`, `device`, `error`).
  - **Contract Check:** `test_models_status_endpoint` expects `set(models.keys()) == {"text", "audio", "video"}` with `loaded=True` for all three; `PredictionResponse` (`_response()`) checks `tuple(probabilities) == MELD_CLASS_NAMES`, `[0, 1]` range, sum within `1e-4` of `1.0`, and pulls quality fields `("num_faces", "num_sampled_frames", "face_detection_coverage", "prediction_entropy")`.
  - **Gap:** `ModelRegistry` must support additive model/pipeline metadata (`architecture`, `checkpoint_sha256`, `preprocessing`, `pooling_method`) and graceful primary -> fallback ResNet18 loading if primary checksum fails (while failing loudly if neither valid checkpoint matches!). Also check thread-safety/concurrency of `MTCNN` + `ResNet18` inside `_lock`.
- **Multimodal Endpoint & Fusion (`api/app.py`, `api/fusion.py`):**
  - `Candidate4FusionEngine` (`api/fusion.py`, SHA-256 `ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d`):
    - Implements Text-Anchored Class-Conditional Bounded Fusion (`tau_audio=0.20`, `lambda_max=0.40`, `kappa_audio=1.50`, `lambda_video=0.00`).
    - Protected classes (`fear`, `disgust`, `sadness`, `joy`) lock out auxiliary audio injection (`text weight = 1.0`, `audio weight = 0.0`).
    - Synergy classes (`anger`, `surprise`) inject audio up to `0.40` when `text_pred == "neutral"` and `m_a = p_a[pred_a] - p_a["neutral"] > 0.20`.
    - `test_video_outside_default_production_path` in `api/tests/test_fusion.py` explicitly tests default `Candidate4FusionEngine().fuse(...)` returning `exclusion_reasons["video"] == "outside_production_path"`.
  - `/predict/multimodal` (`api/app.py`):
    - Currently accepts JSON `{text}` or multipart form `text` + optional `file`/`audio`. Hardcodes `video: None` in `individual_predictions` and never reads a video upload!
    - Notice `test_multimodal_api.py` requires:
      - Text-only JSON/form returns `fusion_method == "text_anchored_class_conditional_bounded_fusion_v1"`, `modalities_available == {"text": True, "audio": False, "video": False}`, `modality_weights["text"] == 1.0`.
      - Text + Audio uses `Candidate4FusionEngine` identically to existing behavior.
      - Non-string / empty text is rejected with `422`.
      - Corrupt/silent auxiliary audio falls back gracefully (`200` with explicit `exclusion_reasons["audio"]`) and cleans up temp files.
  - **Integration Plan (18D):**
    - Preserve `Candidate4FusionEngine` (`Text` & `Text + Audio` exact outputs, rule names, weights, and `fusion_method` unchanged — regression-tested bit-for-bit!).
    - Accept optional `video` upload field alongside optional `file`/`audio` in `/predict/multimodal` (keeping backward compatibility cleanly: when no `video` is provided, response matches Candidate 4 `Text` / `Text + Audio` verbatim!).
    - Implement a **separately validated multimodal extension** when `video` is supplied (`Text + Video`, `Text + Audio + Video`, and standalone `Video` / `Audio + Video` if text is omitted or optional — carefully keeping empty/whitespace text `422` contract where `test_multimodal_api.py` requires it unless a distinct request mode explicitly omits text!). Empirical dev-split benchmarking (`dev`, never `test`) will determine whether/when facial predictions should receive positive weight `w_v` vs quality-gated small weight vs advisory display, ensuring weights are finite, non-negative, sum to `1.0`, and never degrade Text/Candidate 4 quality.

### 2.4 Frontend (`frontend/src/`)
- `frontend/src/api.ts`:
  - `predictFile("audio" | "video", file)` already posts to `/predict/audio` and `/predict/video`.
  - `predictMultimodal(text, audioFile?)` only sends `text` and `file` (`audioFile`), never a `video` file.
  - `MultimodalPrediction.individual_predictions.video` is typed `Prediction | null`, `exclusion_reasons.video` and `modalities_available.video` already exist in TS types.
- `frontend/src/App.tsx`:
  - Tab `"video"` currently says `"Facial CNN visual emotion inference"` in `tabs` hint and footer says `"Phase 14 · Adaptive Multimodal Studio"` / `"(DistilBERT + Wav2Vec2)"`.
  - `MultimodalInput` has utterance text + optional speech recording (`Upload WAV` or `LiveAudioCapture`), no optional video input yet.
  - `MultimodalResultCard` hardcodes `"Bypassed (w: 0.00)"` for `Video Modality` and shows a 2-column Individual Modality Breakdown (`DistilBERT Text` and `Wav2Vec2 Speech`), omitting Video breakdown & quality/exclusion details.
- `frontend/src/LiveCapture.tsx` & `frontend/src/media.ts`:
  - Already support live WebM/MP4 camera capture, playback, 16 kHz mono WAV audio conversion, and error handling.
- **Integration Plan (18D & 18E):**
  - Update video hints/status display to reflect ResNet18 + MTCNN (or dynamic `model_identifier` from `/models/status`).
  - Add optional video upload + live camera selector to `MultimodalInput`, extend `predictMultimodal()` to send `video` file, update `MultimodalResultCard` to display active/excluded Video status, weight, individual ResNet18 facial prediction, and video quality telemetry (`num_faces`, `num_sampled_frames`, `face_detection_coverage`).

### 2.5 Existing Test Suites
- Observed test suites:
  - `api/tests/test_api.py` (193 lines) — Note: lines 64 & 70 call `_media_path("test", ...)` for read-only `/predict/audio` and `/predict/video` smoke endpoint checks! We must NOT inspect or use MELD `test` split in any new tuning/benchmarking scripts (use `"dev"` exclusively for all Phase 18 validation/benchmarking).
  - `api/tests/test_fusion.py` (269 lines)
  - `api/tests/test_multimodal_api.py` (341 lines)
  - `facial_emotion/tests/test_facial.py` (145 lines)
  - `facial_emotion/tests/test_remediation.py` (103 lines)
  - Plus `text_emotion/tests/*` and `speech_emotion/tests/*`.

---

## 3. Compatibility & Verification Gate (Phase 18A)

| Gate Check | Result | Evidence |
|---|---|---|
| Primary & Fallback checkpoints present & SHA-256 match | **PASS** | `8825d865...` (`exp5_ce_weighted_best.pt`) and `4830ac84...` (`exp5_focal_g3_wt_best.pt`) verified |
| Seven canonical classes identical across all modules | **PASS** | `['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']` across checkpoints, `config.py`, `fusion.py`, `api.ts` |
| Checkpoint `fc.weight` output shape matches 7 classes | **PASS** | `(7, 512)` in both checkpoints |
| ImageNet RGB `96x96` preprocessing contract documented | **PASS** | Checked in `.pt` `input_normalization` & `remediation.py` |
| Protected assets & Candidate 4 fusion preserved | **PASS** | All production weights & `api/fusion.py` SHA-256 verified |
| Video/Frontend/API integration gaps mapped cleanly | **PASS** | Documented in Section 2 |

**Phase 18A Conclusion:** All compatibility checks pass. Advancing automatically to **Phase 18B — Facial Model Integration**.

