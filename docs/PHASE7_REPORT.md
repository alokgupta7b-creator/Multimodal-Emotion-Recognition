# Phase 7 Wav2Vec2 Speech Emotion Recognition

## Dataset and protocol

Training used the 9,988 valid train WAVs. The 1,108 valid dev WAVs were used
for validation and checkpoint selection. The 2,610 test WAVs were evaluated
once after checkpoint selection. The two Phase 6 media failures remained
excluded and are preserved in the Phase 6 reports.

- Train/dev/test sample IDs: no overlap.
- Classes: anger, disgust, fear, joy, neutral, sadness, surprise.
- Audio validation: all selected files readable, mono and 16 kHz.
- Class distributions are recorded in the Phase 6 manifests and preflight
  logs; all seven classes occur in every split.

## Configuration

- Checkpoint: `facebook/wav2vec2-base`
- Maximum model input: 8 seconds, with truncation and dynamic padding
- Batch size: 2
- Evaluation batch size: 4
- Gradient accumulation: 4
- Learning rate: `1e-5`
- Epochs: 4
- Early-stopping patience: 2
- Feature encoder frozen
- FP16 enabled on CUDA
- Seed: 42
- Device: NVIDIA GeForce RTX 4050 Laptop GPU
- CUDA: PyTorch `2.6.0+cu124`, driver `592.82`

The pipeline was corrected during execution to accept valid long and
near-silent clips, truncate only at the model-input boundary, and disable
self-supervised masking that fails on short supervised-classification clips.

## Test results

The saved best checkpoint was evaluated on all 2,610 test samples:

| Metric | Result |
|---|---:|
| Accuracy | 0.5046 |
| Macro precision | 0.2905 |
| Macro recall | 0.2542 |
| Macro F1 | 0.2331 |
| Weighted F1 | 0.4409 |
| Inference latency | 17.21 ms/sample |

Per-class metrics and the confusion matrix are stored in
`models/speech_emotion/wav2vec2/test_evaluation.json`.

The observed GPU run used approximately 3,085 MiB VRAM at one sampling point;
this is not claimed as a peak measurement. The saved model artifacts occupy
378,400,910 bytes.

## Artifacts

- `models/speech_emotion/wav2vec2/best/model.safetensors`
- `models/speech_emotion/wav2vec2/best/config.json`
- `models/speech_emotion/wav2vec2/best/preprocessor_config.json`
- `models/speech_emotion/wav2vec2/test_evaluation.json`
- `models/speech_emotion/wav2vec2/training_metadata.json`

Existing MFCC, TF-IDF and DistilBERT artifacts were not modified.

## Limitations

The initial training implementation did not persist epoch-level training and
validation loss history before the test pass. The run did produce and save a
best checkpoint, and its held-out test evaluation is real, but loss curves
cannot be reconstructed without another full training run. No fabricated
curve or training-duration value is reported.

The PyTorch/OpenMP import issue remains environment-sensitive: the successful
training process used CUDA, while isolated `import torch` smoke tests can
still report duplicate OpenMP runtimes. `KMP_DUPLICATE_LIB_OK` was not used.
