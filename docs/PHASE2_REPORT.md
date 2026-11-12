# Phase 2 Stabilization Report

## Changes

This phase centralizes repository paths, makes the text baseline reproducible,
aligns text preprocessing across training/inference/export, fixes MLP training
with current scikit-learn, pins runtime dependencies, and documents the
trusted-artifact limitation of joblib.

## Files changed

- `project_config.py`
- `requirements.txt`
- `requirements-dev.txt`
- `text_emotion/train.py`, `predict.py`, `evaluate.py`, `tune.py`,
  `export_fusion_probabilities.py`, `model_io.py`
- `speech_emotion/model/classifier.py`
- Regression tests under `speech_emotion/tests` and `text_emotion/tests`
- Phase 1 and Phase 2 documentation under `docs`

## Validation results

Environment: Python 3.13.9, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.7.2, joblib 1.5.2, SciPy 1.16.3, librosa 0.11.0,
soundfile 0.13.1, h5py 3.15.1 and pytest 8.3.4.

- `python -m compileall -q .`: passed.
- Dependency import check: passed.
- `python -m pip check`: reported two unrelated pre-existing environment
  conflicts: `aiobotocore 2.25.0` requires `botocore<1.40.50,>=1.40.46`
  while `1.43.18` is installed, and `sphinx 8.2.3` requires
  `docutils>=0.20,<0.22` while `0.19` is installed. Neither package is a
  project dependency in the Phase 2 manifests.
- `python -m pytest -q`: **187 passed**, 3 warnings.
- MELD path validation: passed.
- Text baseline training: executed using only the official train split for
  fitting; dev and test were evaluation-only. The generated artifact is
  `text_emotion_model.joblib`, with metadata and label mapping sidecars.
- Text reload/sample inference: passed; seven probabilities were finite,
  non-negative and summed to 1.0.
- Text fusion export: passed for 1,109 dev and 2,610 test samples.
- Existing speech artifact reload: passed after restoring its missing
  `LogisticRegression.multi_class="auto"` compatibility attribute; output
  probabilities summed to 1.0.

The speech artifact emitted scikit-learn's version mismatch warning because it
was serialized with 1.9.1 and loaded with 1.7.2. This warning is retained and
an explicit log entry identifies the narrowly scoped compatibility adaptation.

## Security limitations

Joblib uses pickle deserialization and can execute code while loading a
malicious artifact. The application must load only locally generated or
otherwise verified artifacts; this phase does not make arbitrary joblib files
safe.
