# Phase 9.5 Reliable Face Tracking and Versioned Cache Implementation

## 1. Scope and safeguards

Phase 9.5 implements a new, independent facial tracking/cache namespace. The
seven-class mapping remains unchanged:

`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.

The implementation does not modify or reuse-write the Phase 8,
Phase 9 v1, or Phase 9 v2 cache/checkpoint directories. No training process was
started, no model checkpoint was produced, and no held-out test data was
loaded or evaluated.

## 2. Architecture

- `FaceDetector.detect()` returns `FaceDetection` records containing the
  original-frame bounding box, resized BGR crop, and an optional detection
  confidence.
- `detect_and_align()` remains available for compatibility and returns only
  crops.
- `track_detections()` processes sampled frames while retaining original
  coordinates.
- The first track uses the largest detected face as a deterministic
  initialization heuristic only. It does not claim that the face belongs to
  the speaker.
- Later frames use gated bounding-box IoU and normalized center distance.
  Candidate scores combine IoU (60%) and normalized proximity (40%).
- A match is accepted when IoU is at least `0.05` or normalized center
  distance is at most `0.35`.
- A track can tolerate one missing frame by default
  (`max_missing_frames=1`). If the gate remains unsatisfied, the current track
  is terminated and a new track is initialized from the largest available
  detection.
- Missing detections are explicit. Clips below 75% coverage are excluded
  rather than silently padded.

## 3. Metadata

Each accepted entry records:

- cache version and sample identifier;
- configured tracking thresholds;
- per-frame detection count;
- selected track identifier;
- original bounding-box coordinates;
- detection confidence, when a detector supplies one;
- tracking confidence;
- missing-frame flags;
- coverage, observed/missing frame counts, multi-face count, and track
  switches;
- crop shape and whether detection confidence was available.

OpenCV Haar detection does not provide a calibrated confidence score, so its
confidence field is explicitly recorded as unavailable. Detection confidence
and tracking confidence are separate fields.

## 4. Versioned cache

- Version: `phase9_5_tracking_v1`
- Default location: `models/facial_emotion_phase9_5_tracking`
- Per-entry files:
  - `<sample_id>_4f.npy`
  - `<sample_id>_4f.json`
- Array contract: `uint8`, shape `(4, 96, 96, 3)`, BGR storage.
- Both array and JSON metadata writes use temporary files followed by
  `os.replace`, preventing partially written final files.
- Existing valid entries are reused after version, shape, dtype, and metadata
  validation.
- Missing or invalid entries are regenerated without deleting unrelated files.
- `--dev-only` resolves strictly to `train` and `dev`; it never loads a test
  manifest or creates a test cache. Test generation requires explicitly
  requesting `--split test`.
- The default Python API split set is also `("train", "dev")`, making
  development-only generation the safe default.

No Phase 9.5 cache generation was run during this implementation task, so no
new cache counts are claimed.

## 5. Tests and validation

Commands executed:

```powershell
python -m py_compile facial_emotion\detector.py facial_emotion\tracking_cache.py facial_emotion\tests\test_facial.py
python -m pytest facial_emotion/tests -q
```

Final result:

- Syntax compilation: passed.
- Focused facial suite: **13 passed**.
- Import smoke check for `FaceDetector`, `FaceDetection`, `CACHE_VERSION`,
  and `TrackingConfig`: passed.

Coverage includes:

- IoU and center-distance-gated matching;
- consistent identity assignment;
- multiple detections;
- missing detections and track termination;
- low-confidence/low-similarity rejection;
- cache version metadata;
- atomic/resumable cache writes;
- preservation of an unrelated pre-existing file;
- development-only split resolution excluding test.

The first test run found that the cache version was not written to metadata.
That defect was corrected, metadata writes were made atomic, and the suite was
rerun successfully.

## 6. Preservation and limitations

The implementation writes only to the new Phase 9.5 namespace. Tests verify
that an existing unrelated file is preserved and that valid entries are
reused. Existing Phase 8 and Phase 9 v1/v2 artifacts were not modified or
deleted.

Known limitations:

- IoU/center tracking improves temporal consistency but cannot identify the
  speaking person.
- Largest-face initialization is deterministic, not speaker-aware.
- Haar detection is sensitive to pose, scale, blur, occlusion, and lighting.
- A short missing-frame allowance can still produce track switches.
- Strict coverage filtering may reduce usable data and affect class balance.
- Uniform four-frame sampling may miss brief expressions.
- Tracking confidence is a geometric association score, not a calibrated
  identity probability.

## 7. Resource requirements

The implementation uses the existing OpenCV, NumPy, and pandas dependencies;
no new dependency or model download is required. Cache generation remains
video-decoding and Haar-detection bound. It should be run as a resumable
train/dev-only operation before any model experiment. The existing compact CNN
training resource estimates remain applicable; this implementation itself does
not allocate a training model or require GPU execution.

## 8. Explicit completion status

- Reliable bounding-box-aware tracking: implemented.
- Versioned resumable cache writer: implemented.
- Atomic array and metadata writes: implemented.
- Train/dev-only mode: implemented and tested.
- Training: **not started**.
- Held-out test evaluation: **not performed**.
- Phase 8 and Phase 9 v1/v2 artifacts: preserved.

The next step requires approval to generate the new train/dev cache. Training
should remain a separate, explicitly approved experiment after cache
integrity and quality summaries are reviewed.
