# Phase 9.3 Root-Cause Investigation

## Scope and safeguards

This investigation used only the preserved Phase 9 v2 checkpoint, training
cache, development cache, and train/dev manifests. The MELD test split was not
loaded or evaluated. Phase 8, Phase 9 v1, and Phase 9 v2 artifacts were
preserved. No new training run was started.

## Executive finding

The training loop is active, the label mapping is consistent, the balanced
sampler is working, gradients are nonzero, and model weights changed. The
collapse is therefore not caused by a dead optimizer, invalid probabilities,
or a seven-class index mismatch.

The strongest supported causes are:

1. **Weak video-to-face supervision.** MELD labels describe the utterance, but
   the selected face crop is a heuristic crop from one detected face. The
   visual montage shows valid faces, but the sampled expression is not
   consistently diagnostic of the utterance label.
2. **Incorrect temporal face identity tracking.** `_faces_from_video()` stores
   only resized crops and then compares each crop's center. Every resized crop
   has the same center, so the claimed nearest-center tracking does not track a
   face across frames. In multi-face shots, the selected identity can change or
   default to detector order.
3. **Very small model and short optimization schedule.** The CNN has 94,151
   parameters, uses a `1e-3` AdamW learning rate, and ran for only five epochs.
   Training loss stayed essentially at the seven-class random-loss baseline.
4. **Severe source imbalance and limited minority data.** The sampler corrects
   exposure during one epoch, but several classes still have only 259--270
   usable training samples before replacement sampling.

These findings justify correcting face selection and the training objective
before another experiment. They do not justify accepting the v2 checkpoint.

## Dataset class counts

Canonical order:

`anger, disgust, fear, joy, neutral, sadness, surprise`

| Split | Anger | Disgust | Fear | Joy | Neutral | Sadness | Surprise | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train manifest | 1,109 | 271 | 268 | 1,743 | 4,709 | 683 | 1,205 | 9,988 |
| Train usable/cache | 1,090 | 270 | 259 | 1,716 | 4,618 | 673 | 1,184 | 9,810 |
| Dev manifest | 153 | 22 | 40 | 163 | 469 | 111 | 150 | 1,108 |
| Dev usable/cache | 152 | 21 | 38 | 156 | 464 | 111 | 146 | 1,088 |

Face exclusions were 178 train and 20 dev. All inspected cached arrays were
valid `uint8` arrays with shape `(4, 96, 96, 3)` and pixel values in `[0, 255]`.

## Balanced sampler verification

The implementation assigns each sample weight `1 / class_count` and draws
9,810 samples with replacement. With seed 42, one simulated epoch produced:

| Class | Sampled count | Frequency |
|---|---:|---:|
| Anger | 1,432 | 14.60% |
| Disgust | 1,364 | 13.90% |
| Fear | 1,377 | 14.04% |
| Joy | 1,400 | 14.27% |
| Neutral | 1,418 | 14.45% |
| Sadness | 1,433 | 14.61% |
| Surprise | 1,386 | 14.13% |

The sampler is therefore functioning and produces approximately uniform class
exposure. It does not create new visual information for minority classes.

## Crop and preprocessing findings

The crop montage was inspected across all seven labels. Crops contain valid
human faces, but framing and expression quality vary: some faces are profile
views, partially occluded, blurred, or not visibly expressive. This supports
weak/noisy visual supervision rather than a corrupted image pipeline.

Aggregate crop statistics were similar across classes:

- Train mean intensity: approximately 69.8--71.6
- Train mean standard deviation: approximately 41.5--42.1
- Dark pixels below 5: approximately 3.6--3.8%
- Bright pixels above 250: approximately 0.03--0.08%

The current preprocessing is internally consistent:

1. Detector crops are stored as OpenCV BGR `uint8`.
2. Training and inference reverse channels to RGB.
3. Values are scaled to `[0, 1]`.
4. Values are normalized with `(x - 0.5) / 0.5`.
5. Tensors are `[frames, channels, height, width]`, then batched as
   `[batch, frames, channels, height, width]`.

No normalization or tensor-dimension defect was found.

## Gradient and weight-update diagnostics

Using a fresh model and one cached training batch:

- Cross-entropy loss: `1.94945`
- Nonzero-gradient parameter tensors: `8/8`
- Gradient norm: `0.64872`
- Model parameters: `94,151`

Comparing a seed-42 fresh initialization with the saved best checkpoint:

- Changed parameter tensors: `8`
- Mean absolute parameter delta: `0.01797`
- Maximum tensor delta: `0.15932`

The optimizer did update the model. However, the loss remained close to the
seven-class random baseline `log(7) = 1.94591`, indicating that updates did
not produce useful class separation.

## Development metrics and prediction collapse

### Epoch history

| Epoch | Train loss | Dev loss | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.94684 | 1.97249 | 0.1461 | 0.0446 | 0.1457 | 0.0428 | 0.0428 |
| 2 | 1.94684 | 1.94308 | 0.1167 | 0.1827 | 0.1315 | 0.0613 | 0.0578 |
| 3 | 1.94567 | 1.92854 | 0.3465 | 0.0751 | 0.1342 | 0.0963 | 0.2476 |
| 4 | 1.94578 | 1.94025 | 0.1333 | 0.1076 | 0.1212 | 0.0846 | 0.1210 |
| 5 | 1.94542 | 1.95717 | 0.1048 | 0.0717 | 0.1327 | 0.0491 | 0.0431 |

The selected checkpoint was correctly chosen by highest development macro-F1:
epoch 3. Early stopping did not trigger because the run completed its five
configured epochs.

### Best epoch per-class metrics

At epoch 3:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Anger | 0.0000 | 0.0000 | 0.0000 |
| Disgust | 0.0000 | 0.0000 | 0.0000 |
| Fear | 0.0000 | 0.0000 | 0.0000 |
| Joy | 0.0000 | 0.0000 | 0.0000 |
| Neutral | 0.4177 | 0.7543 | 0.5376 |
| Sadness | 0.0000 | 0.0000 | 0.0000 |
| Surprise | 0.1080 | 0.1849 | 0.1364 |

Epoch 3 predicted only neutral (838) and surprise (250). Epoch 5 predicted
anger (11), neutral (7), sadness (738), and surprise (332), with zero
predictions for disgust, fear, and joy. The model remains collapsed.

## Temporal aggregation comparison

On the development cache using the saved best checkpoint:

- Mean-logit temporal accuracy: **34.65%**
- Individual-frame accuracy: **32.38%**
- Temporal predictions: neutral 838, surprise 250
- Flattened individual-frame predictions: neutral 2,917, surprise 1,431,
  sadness 4

Mean-logit aggregation is implemented correctly and provides a small accuracy
increase, but it does not solve class collapse. The aggregation strategy is
not the primary root cause.

## Optimization and capacity assessment

The compact CNN is suitable as a smoke-test baseline but weak for unconstrained
MELD video crops. Its adaptive-pooling architecture has only 94,151
parameters and no facial landmark alignment, attention, identity tracking, or
temporal feature extractor. The `1e-3` learning rate is plausible for a small
CNN, but five epochs and a nearly random training loss are insufficient
evidence that this configuration can learn the noisy task. Balanced sampling
without class-weighted loss also changes exposure but not label quality.

The v2 run completed on CUDA in 114.18 seconds. The recorded process used
approximately 1.3--1.4 GiB during active training on the RTX 4050; the model
itself is small, so a corrected experiment can use the same GPU and batch size.
Increasing model capacity or adding temporal/identity features should be
validated incrementally rather than combined with many simultaneous changes.

## Recommended corrective experiment

Do not train again until approved. The next controlled experiment should:

1. Preserve the existing caches and artifacts.
2. Store detector bounding boxes or centers with each sampled crop.
3. Track the same face using actual bounding-box centers, not resized crop
   centers.
4. Record face-detection coverage and reject or separately flag low-coverage
   clips.
5. Keep the train/dev-only boundary and select checkpoints by development
   macro-F1.
6. Compare class-weighted loss against the balanced sampler in separate runs.
7. Use a modestly larger CNN or an already-installed architecture without
   downloading weights.
8. Run more epochs with a learning-rate schedule and log gradient norms,
   parameter deltas, per-class train metrics, and prediction histograms.
9. Keep mean-logit aggregation as the baseline because it slightly outperformed
   individual-frame accuracy.

The first correction should be face identity selection and diagnostic logging.
If collapse persists with verified face tracking, then test model capacity and
optimization changes one at a time.

## Validation record

- Read-only train/dev diagnosis completed.
- Test split not loaded or evaluated.
- Phase 8 and Phase 9 v1/v2 artifacts preserved.
- Existing focused facial tests: 8 passed.
- No new training experiment started.
