# Phase 4 Facial Emotion Recognition

## Inspection

The repository contains visual reference validation utilities for MELD and
CMU-MOSEI, but no downloaded videos, face crops, FER-2013 CSV, AffectNet,
RAF-DB or trained facial artifact. `datasets/MELD` contains annotations and
manifests only. No facial dataset was downloaded in this phase.

The environment has OpenCV, PyTorch, torchvision and facenet-pytorch
available; MediaPipe is not installed. The implementation therefore uses
OpenCV's bundled Haar cascade for a dependency-light fallback detector.

## Implemented architecture

- Seven-class mapping is shared with the existing MELD mapping.
- `facial_emotion.data` validates FER-style CSV labels and pixels without
  creating or mixing splits.
- `facial_emotion.detector` detects and resizes faces with OpenCV and yields
  sampled video faces with deterministic resource cleanup.
- `facial_emotion.model` provides a compact CNN frame classifier, mean
  probability aggregation across faces, video inference, CPU fallback and
  CUDA selection.
- Model artifacts are isolated under `models/facial_emotion`.

## Training and results

No facial model was trained because no compatible local facial dataset or
manifest exists. Consequently, accuracy, precision, recall, F1, confusion
matrix, model size and latency for a trained facial model are not reported.
This avoids fabricating results. The training entry point refuses to download
data and requires an explicit FER-style CSV with disjoint `Training`,
`PublicTest` and `PrivateTest` usages.

GPU inference cannot be validated without a trained facial checkpoint. The
predictor includes CUDA support and normalized probability checks for when a
checkpoint is supplied.

The synthetic CUDA architecture smoke test was attempted but the process was
blocked by the machine's duplicate OpenMP runtime error
(`OMP: Error #15`). This is the same environment-level issue observed during
Phase 3 probing; no unsafe `KMP_DUPLICATE_LIB_OK` workaround was enabled.

Validation completed: Python syntax passed, the full suite passed with 194
tests, OpenCV blank-frame detection returned zero faces, and no files were
created under `models/facial_emotion`.

## Limitations and next step

OpenCV Haar detection is a baseline and is less robust than MediaPipe or a
modern detector under pose, occlusion and lighting changes. Before Phase 5,
obtain a licensed facial dataset, document its seven-class mapping, train the
CNN with train/dev/test isolation, and record the required evaluation metrics.
