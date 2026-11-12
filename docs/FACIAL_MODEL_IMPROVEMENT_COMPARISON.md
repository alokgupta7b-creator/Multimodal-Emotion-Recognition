# Facial Model Improvement Comparison

All metrics below are reported from the recorded experiment metadata. Dev and
test metrics are kept separate and are not treated as equivalent.

| Configuration | Split | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Phase 8 compact CNN | Dev, best recorded epoch | 1,080 | 0.4241 | 0.0997 | 0.2526 |
| Phase 8 compact CNN | Test | 2,564 | 0.3846 | 0.1163 | 0.3101 |
| Phase 9 experiment 1 | Dev, best epoch | 805 | 0.0335 | 0.0155 | 0.0031 |
| Phase 9 experiment 1 | Test | 1,840 | 0.0179 | 0.0107 | 0.0017 |
| Phase 9 v2 | Dev, best epoch | 1,088 | 0.3465 | 0.0963 | 0.2476 |
| Phase 9 v2 | Test | Not available | Not reported | Not reported | Not reported |

Phase 9 v2 used a development-only run and did not evaluate the test split.
An earlier isolated evaluation file contains partial metrics for 1,585 cached
test samples, but it is incomplete and is intentionally excluded from this
comparison.

## Per-class evidence

The Phase 8 test per-class metrics are recorded in
`models/facial_emotion/training_metadata.json`. Experiment 1 per-class test
metrics are recorded in
`models/facial_emotion_phase9/training_metadata.json`. Phase 9 v2 has
development per-class metrics in
`models/facial_emotion_phase9_v2/training_metadata.json`, but no final test
per-class metrics.

## Conclusion

Phase 9 did not demonstrate an improvement using the primary metric,
development macro F1: `0.0963` for v2 versus the Phase 8 recorded peak of
`0.0997`. Experiment 1 suffered severe data loss and model collapse.
Accordingly, Phase 8 remains the reported facial baseline. No claim of
generalization improvement is made.
