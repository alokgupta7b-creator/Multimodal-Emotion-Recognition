# Facial Model Comparison

Phase 8 established the first trained facial model in this repository. There
is no earlier trained facial baseline, so a within-modality comparison is not
available yet.

## Current facial model

| Model | Test samples | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Compact facial CNN (`facial-cnn-v1`) | 2,564 | 0.3846 | 0.1163 | 0.3101 |

The model was selected using development macro-F1 and evaluated once on the
held-out test split. Full per-class metrics and the confusion matrix are in
`models/facial_emotion/training_metadata.json`.

The comparison is limited because the pipeline samples one representative
face crop from three candidate frames. It should not be interpreted as a
comparison against the text or speech models, which use different inputs and
sample availability.
