# Phase 9.4 Facial Data Quality and Identity Tracking Investigation

## Scope and safeguards

This investigation read the Phase 9.3 report, Phase 1-8 progress history, the
facial detector/training/model implementation, and only the train/dev
manifests and caches. The held-out test split was not loaded or evaluated. No
training run was started, no checkpoint was changed, and no existing cache
was deleted or overwritten.

## Existing pipeline findings

The current four-frame preparation:

1. Samples four uniformly spaced video frame indices.
2. Runs an OpenCV Haar frontal-face detector.
3. If multiple faces are detected, selects the middle detector result for the
   first sampled frame.
4. On later frames, attempts nearest-center selection.
5. Resizes the selected crop to 96x96 and pads missing detections with the
   last detected crop.
6. Stores four BGR `uint8` crops in a `.npy` cache.

The model converts BGR to RGB, normalizes to `[-1, 1]`, predicts each frame,
and averages frame logits. The seven-class mapping is unchanged and remains
correct.

## Multi-face selection problem

The current selector does not actually track a face. `detect_and_align()` drops
the original bounding-box coordinates and returns only resized crops. Every
returned crop therefore has the same 96x96 coordinate center. The later
nearest-center calculation compares identical crop centers and cannot
distinguish faces by motion or position. Detector ordering consequently
determines the selected face after the first frame.

A selected face cannot currently be associated with the speaking person.
MELD utterance labels identify the utterance emotion, but this implementation
has no speaker bounding boxes, active-speaker labels, audio-visual alignment,
or reliable identity metadata. The system must not claim that a selected face
belongs to the speaker.

## Train/dev detection audit

The audit sampled the first 500 usable train videos and all 1,108 usable dev
manifest rows, using the same four uniform frame positions and Haar detector.
It did not inspect the test split.

| Split/sample | Sampled frames | Zero-face frames | Multi-face frames | Mean detected face area |
|---|---:|---:|---:|---:|
| Train sample (500 videos) | 1,984 | 212 (10.69%) | 1,038 (52.32%) | 1.85% of frame |
| Dev (1,108 videos) | 4,417 | 492 (11.14%) | 2,143 (48.52%) | 1.99% of frame |

Detected face-count distributions:

- Train sample: 0: 212, 1: 734, 2: 526, 3: 286, 4: 125, 5: 71,
  6: 16, 7: 8, 8: 2, 9: 3, 11: 1.
- Dev: 0: 492, 1: 1,782, 2: 1,170, 3: 554, 4: 241, 5: 98, 6: 43,
  7: 23, 8: 9, 9: 3, 10: 1, 26: 1.

The existing v2 cache contains 9,810 train and 1,088 dev arrays. These are
valid reusable arrays, but cache validity does not mean every sampled frame
had a detection: the current pipeline pads missing detections. The manifest
to-cache usable proportions are 98.22% train and 98.19% dev.

## Crop quality findings

Representative cached crops from all seven classes contain human faces, but
the visual evidence is heterogeneous: profiles, blur, occlusion, partial
faces, small faces, and frames where the visible person may not be the
utterance speaker are present. Aggregate train/dev pixel statistics were
similar across classes, so no class-specific pixel corruption was found.

For the sampled detections:

- Median face-crop Laplacian variance was approximately 49.65 train and
  49.94 dev.
- The 10th percentile was approximately 18.05 train and 18.30 dev,
  indicating a meaningful low-sharpness tail.
- Mean detected face area was only 1.85% train and 1.99% dev of the full
  video frame.

These are quality indicators, not emotion-label correctness measurements.
Utterance-level MELD emotion remains noisy face-level supervision when a
conversation shows multiple people or when the visible face is not speaking.

## Tracking approach comparison

| Approach | Benefit | Limitation | Recommendation |
|---|---|---|---|
| Largest face | Stable simple heuristic | Largest person may not be speaker; can switch | Baseline only |
| Detector-order selection | No extra state | Unstable across frames and detector ordering | Do not use |
| Crop-center matching | Intended temporal consistency | Broken because crop coordinates are discarded | Replace |
| Bounding-box IoU + center distance | Lightweight temporal identity tracking | Still cannot prove speaker identity; misses can cause switches | Recommended |
| OpenCV tracker per identity | More temporal continuity | Tracker failure and dependency complexity; no speaker proof | Optional fallback |
| Active-speaker/audio-visual model | Could improve speaker association | Not available and substantially more complex | Out of scope |

## Proposed data-pipeline improvement

Implement a new cache namespace, preserving the existing Phase 8/v1/v2
caches:

1. Add a detector method returning crop plus original bounding box and
   detection confidence proxy.
2. Select the initial face deterministically, recording the choice. Use the
   largest face only as a documented initialization heuristic, not as a
   speaker claim.
3. Match subsequent detections by a gated cost combining bounding-box IoU and
   center distance. Reject matches outside the gate instead of silently
   switching identity.
4. Preserve the previous crop only for a small, explicit missing-frame limit.
   Record coverage, switches, unmatched frames, box area, and sharpness.
5. Mark low-coverage or low-quality clips as excluded/diagnostic rather than
   silently treating padded crops as observed faces.
6. Rebuild only the new cache namespace or entries whose tracking/preprocessing
   version changes. Never delete or overwrite the existing cache directories.
7. Keep train/dev selection and checkpointing isolated from the test split.

This pipeline still cannot identify the speaker. It only improves temporal
consistency of one detected face.

## Files requiring modification after approval

No source files were modified in Phase 9.4. A future implementation would
likely touch only:

- `facial_emotion/detector.py`: return bounding boxes and detection records.
- `facial_emotion/train.py`: versioned cache builder, IoU/center matching,
  quality metadata, and train/dev-only preparation.
- `facial_emotion/tests/test_facial.py`: tracking, missing-frame, cache
  versioning, and quality-gate regression tests.
- `facial_emotion/model.py`: only if the new metadata changes input handling.
- `docs/DEVELOPMENT_PROGRESS.md`: implementation results after approval.

Existing model artifacts and cache directories must remain untouched.

## Controlled experiment proposal

This is a proposal only; no experiment was started.

### Stage A: data-only validation

- Build a new versioned train/dev tracking cache.
- Use bounding-box IoU plus center-distance matching.
- Report coverage, unmatched frames, switches, sharpness, area, and per-class
  counts.
- Inspect representative crops from every class.
- Accept the cache only if it is complete, structurally valid, and its
  quality diagnostics are explicit.

### Stage B: model experiment

- Use the new cache and the unchanged seven-class mapping.
- Keep mean-logit aggregation as the baseline because it slightly exceeded
  individual-frame accuracy in Phase 9.3.
- Compare balanced sampling and class-weighted loss in separate runs.
- Use a modestly larger CNN only after the data-only stage passes.
- Use a lower initial learning rate, such as `3e-4`, a scheduler, and at least
  15-20 epochs with development macro-F1 early stopping.
- Log train/dev per-class metrics, prediction histograms, gradient norms, and
  parameter deltas every epoch.
- Do not load, cache, or evaluate test data during development.

### Acceptance criteria

Before any test evaluation is considered, require:

- All seven classes have nonzero development predictions at the selected
  checkpoint.
- Development macro-F1 materially exceeds the Phase 9 v2 value of 0.0963.
- No single class accounts for more than 70% of development predictions
  without an explicitly documented reason.
- Development loss decreases below the random seven-class baseline
  (`log(7) ~= 1.9459`) and remains stable.
- Tracking coverage and exclusions are reported rather than hidden by padding.

## Risks and limitations

- Haar detection is sensitive to pose, scale, occlusion, and lighting.
- IoU/center tracking preserves a face identity, not the speaking identity.
- MELD utterance labels are not direct face-expression labels.
- Uniform temporal sampling can miss brief expressions.
- Aggressive quality filtering may reduce data and amplify class imbalance.
- The small RTX 4050 VRAM budget favors compact models and batch size 16-32.

## Resource estimates

Based on the completed v2 run:

- Existing cached CNN training: approximately 114 seconds on CUDA.
- Existing cache generation: approximately 74 minutes for the train/dev
  preparation on this Windows environment.
- Versioned tracking-cache extraction should be budgeted for roughly
  75-120 minutes for train/dev until measured; this is an estimate because
  bounding-box metadata and quality metrics add work.
- A 15-20 epoch compact-CNN experiment should fit in approximately 2-8
  minutes after caching, depending on architecture and diagnostics.
- Expected active VRAM should remain below approximately 2 GiB for a compact
  96x96 model with batch size 16-32; this is an estimate, not a measured
  result for the proposed model.

## Validation record

- Focused facial tests: **8 passed in 5.08 seconds**.
- No source implementation was changed.
- No training was started.
- No test data was loaded or evaluated.
- No dataset or model was downloaded.
- Phase 8, Phase 9 v1, and Phase 9 v2 artifacts were preserved.
