# Speech Model Comparison

All results below use the MELD test split with 2,610 samples.

| Model | Test accuracy | Macro F1 | Weighted F1 | Status |
|---|---:|---:|---:|---|
| MFCC Logistic Regression | 0.2318 | 0.1917 | 0.2676 | Existing selected baseline |
| MFCC SVM | — | — | — | Not run in Phase 5 |
| MFCC MLP | — | — | — | Not run in Phase 5 |
| Wav2Vec2 | 0.5046 | 0.2331 | 0.4409 | Phase 7 held-out evaluation |

Wav2Vec2 macro F1 exceeded the existing MFCC baseline by 0.0414 and weighted
F1 exceeded it by 0.1733 in this measured run. Macro precision was 0.2905 and
macro recall was 0.2542. Per-class metrics and the confusion matrix are in
`models/speech_emotion/wav2vec2/test_evaluation.json`.

Training and validation loss curves were not persisted by the original run
and are therefore not claimed.
