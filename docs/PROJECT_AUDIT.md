# Phase 1 Repository Audit

## Scope

This audit records the repository state inspected before stabilization. It is
an evidence-based inventory, not a claim that documented features are
implemented.

## Implemented

- MELD and CMU-MOSEI dataset preparation, manifests, validation and alignment.
- Text TF-IDF word/character features with balanced Logistic Regression.
- Speech WAV loading, resampling, MFCC extraction and scikit-learn classifiers.
- Speech model persistence, probability prediction and unit tests.
- Text and speech fusion-probability CSV export utilities.

## Documented but not implemented

- DistilBERT/RoBERTa/DeBERTa text models.
- Wav2Vec2 or HuBERT speech models.
- Facial emotion recognition and face tracking.
- Trainable reliability-aware multimodal fusion.
- FastAPI/WebSocket services.
- MLflow tracking and production deployment artifacts.

## Datasets and artifacts

MELD annotations and manifests are under `datasets/MELD`. CMU-MOSEI
preparation code and synthetic fixtures are under `datasets/CMU-MOSEI`.
Existing speech joblib artifacts and reports are under `models/speech_emotion`.
Raw media, private datasets and large model files are intentionally excluded
from version control.

## Known issues entering Phase 2

- Dependency manifest omitted NumPy, pandas, SciPy, librosa, soundfile, h5py
  and pytest.
- Text scripts used inconsistent MELD locations and the text model artifact was
  absent.
- Text training, inference and fusion export did not share preprocessing.
- MLP early stopping failed with string targets on the installed scikit-learn.
- Joblib artifacts are pickle-based and must only be loaded from trusted files.

## Verification status

Before Phase 2, the test suite reported 182 passed and 1 failed. The failure
was the MLP early-stopping string-label issue described above. No transformer,
vision, fusion or backend implementation was verified.
