# Phase 9 Report: Facial Emotion Recognition Improvement

## Status

Phase 9 was implemented experimentally and then paused. No validated
performance improvement over Phase 8 was demonstrated. No further training or
test-cache generation was performed during finalization.

## Objective and baseline limitations

The Phase 8 baseline trained a compact CNN from one representative crop per
video. It sampled the first, middle, and final frames, selected a detected
face from the first successful frame, and did not expose temporal coverage or
prediction entropy. Its held-out test result was accuracy `0.3846`, macro F1
`0.1163`, and weighted F1 `0.3101` on 2,564 usable face crops.

The Phase 8 training history showed a best development macro F1 of `0.0997`
at epoch 3, with unstable later validation behavior. The class distribution
was strongly imbalanced, particularly for neutral compared with disgust and
fear.

## Implementation

The Phase 9 implementation added:

- Four uniformly distributed candidate frames per video.
- Independent face detection for each sampled frame.
- A simple nearest-center continuity heuristic after the first detection.
- Reuse of the previous detected crop when a later sampled frame has no
  detection, while recording the observed coverage.
- Four-frame temporal mean-logit aggregation.
- RGB conversion and `[0, 1]` to `[-1, 1]` normalization.
- Horizontal-flip augmentation.
- A weighted sampler for class balancing.
- Development macro F1 checkpoint selection and early stopping.
- Confidence, entropy, sampled-frame count, and face-coverage fields in
  inference output.
- Separate Phase 9 output directories, preserving Phase 8 artifacts.

Phase 8 checkpoint loading remains compatible. Checkpoints without the new
`input_normalization` field use the original input range; Phase 9 checkpoints
record normalized preprocessing explicitly.

## Experiments

### Experiment 1: strict four-frame coverage

The first configuration required a detected face in all four sampled frames
and combined balanced sampling with weighted loss. This caused excessive
exclusions and model collapse:

- Train/dev/test usable: `7,229 / 805 / 1,840`
- Exclusions: `2,759 / 303 / 770`
- Best dev accuracy: `0.0335`
- Best dev macro F1: `0.0155`
- Test accuracy: `0.0179`
- Test macro F1: `0.0107`

The complete per-class test metrics and confusion matrix are preserved in
`models/facial_emotion_phase9/training_metadata.json`. This experiment is
retained as a negative result, not a candidate model.

### Experiment 2: Phase 9 v2 development-only configuration

The second configuration removed strict all-frame exclusion, padded missing
detections with the last available crop, retained balanced sampling, and
removed the additional class-weighted loss. Test evaluation was intentionally
disabled during this run.

- Train/dev usable: `9,810 / 1,088`
- Train/dev exclusions: `178 / 20`
- Best dev accuracy: `0.3465`
- Best dev macro F1: `0.0963` at epoch 3
- Training duration: `114.18 seconds`
- Peak allocated GPU memory: `551,609,856 bytes`

The best Phase 9 v2 development macro F1 (`0.0963`) is below the Phase 8
development peak (`0.0997`). Therefore Phase 9 v2 is not a demonstrated
improvement.

The selected v2 checkpoint and development metadata are preserved under
`models/facial_emotion_phase9_v2`. The existing
`final_test_evaluation.json` is incomplete: it contains an isolated
evaluation over only 1,585 cached test samples and is not a full held-out
test result. It must not be compared with the Phase 8 full-test metrics.

## Hardware and runtime

Training used the NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. The compact CNN
fit comfortably in memory, but OpenCV video decoding and face detection were
the dominant preprocessing cost. The known Windows duplicate OpenMP runtime
error occurred during one evaluation attempt:
`OMP: Error #15: Initializing libiomp5md.dll`. The unsafe
`KMP_DUPLICATE_LIB_OK=TRUE` workaround was not used.

## Limitations and future work

The continuity heuristic uses crop-local geometry rather than bounding-box
coordinates, so it is not reliable face tracking. Missing-frame padding can
repeat stale visual information. The compact CNN remains capacity-limited,
and the class sampler did not improve development macro F1. No pretrained
backbone or new data was used.

The full test cache was not completed, so no final Phase 9 full-test
comparison can be claimed. Future work should first fix robust tracking and
coverage accounting, then test one balancing strategy at a time with
development-only selection. Phase 10 and multimodal fusion were not started.
