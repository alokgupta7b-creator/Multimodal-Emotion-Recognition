# Phase 8 Report: Facial Emotion Model Training and Evaluation

## Scope

Phase 8 trained and evaluated the independent facial CNN on the validated
MELD video manifests. Text and speech artifacts were not modified, and
multimodal fusion was not implemented.

## Dataset and split integrity

The validated manifests contain 9,988 train, 1,108 development, and 2,610
test video records. The facial pipeline used the existing split boundaries and
the canonical class order:

`anger, disgust, fear, joy, neutral, sadness, surprise`

Face preparation used one representative detected face per video. The
detector sampled the first, middle, and final frames and selected the first
successful detected crop. Valid crops were cached under
`models/facial_emotion/face_cache`; existing valid cache files were reused.
No video was copied between splits and the test split was used only for the
final evaluation.

The resulting usable samples were:

| Split | Manifest rows | Usable face crops | Excluded: no detectable face |
|---|---:|---:|---:|
| Train | 9,988 | 9,756 | 232 |
| Dev | 1,108 | 1,080 | 28 |
| Test | 2,610 | 2,564 | 46 |

The two Phase 6 media failures remained excluded from the source manifests.
Additional Phase 8 exclusions are recorded in
`models/facial_emotion/training_metadata.json` with sample IDs and reason.

## Architecture and preprocessing

- OpenCV Haar-cascade detection and alignment.
- 96x96 RGB face crops.
- Pixel values scaled to `[0, 1]`.
- CNN: Conv2d 3->32, Conv2d 32->64, Conv2d 64->128, adaptive average
  pooling, and a seven-class linear head.
- Weighted cross-entropy based on the training crop distribution.
- AdamW optimizer, learning rate `1e-3`, weight decay `1e-4`.
- Deterministic seed: 42.
- DataLoader worker count: 0 for Windows/OpenCV reliability.
- Best checkpoint selected by development macro-F1.
- Early stopping patience: 2 epochs.

Command executed:

```text
python -m facial_emotion.train --prepared-root datasets\MELD_raw_acquisition\prepared --output models\facial_emotion --epochs 5 --batch-size 32
```

The run used the NVIDIA GeForce RTX 4050 Laptop GPU. Training time was
43.21 seconds and recorded peak allocated GPU memory was 163,538,432 bytes.

## Evaluation

The selected checkpoint was loaded and evaluated once on the held-out test
crop set. Actual results:

| Metric | Value |
|---|---:|
| Accuracy | 0.3846 |
| Macro precision | 0.1058 |
| Macro recall | 0.1482 |
| Macro F1 | 0.1163 |
| Weighted F1 | 0.3101 |
| Inference latency | 0.573 ms/sample |

Per-class precision, recall, F1, support, and the 7x7 confusion matrix are
stored in the generated metadata file. The model predominantly predicts
neutral and joy, indicating substantial class-imbalance and representation
limitations despite weighted loss.

## Artifacts

- Checkpoint: `models/facial_emotion/facial_cnn_best.pt`
- Checkpoint size: 380,128 bytes
- Metadata, history, metrics, exclusions, and confusion matrix:
  `models/facial_emotion/training_metadata.json`
- Face cache: `models/facial_emotion/face_cache`

## Validation

- Focused facial tests: `4 passed`.
- Full project suite: `198 passed, 5 warnings`.
- Python syntax validation for the changed training module passed.
- Probability normalization and model output behavior remain covered by the
  facial tests.

The warnings were existing non-fatal warnings from librosa, scikit-learn,
scikit-learn MLP convergence, and dependency deprecations. No test was
modified to hide a failure.

## Limitations

This is a compact baseline, not a production facial emotion system. It uses
one crop per video rather than a learned temporal or multi-face aggregation
strategy, and videos without a detected face are excluded rather than assigned
an inferred label. Haar detection can miss small, occluded, or profile faces.
The low macro-F1 demonstrates that further facial data handling and model
capacity work is required before comparing this model as a strong modality.
No new dataset was downloaded and no fusion model was started.
