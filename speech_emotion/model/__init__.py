"""
speech_emotion.model
====================
Modular Speech Emotion Recognition (SER) models and prediction interfaces.
"""

from speech_emotion.model.label_mapping import (
    MELDLabelEncoder,
    MELD_CLASS_NAMES,
    NUM_MELD_CLASSES,
)
from speech_emotion.model.classifier import (
    SERClassifier,
    SERConfig,
)
from speech_emotion.model.predict import (
    SERPrediction,
    predict_emotion,
    predict_from_audio,
)

__all__ = [
    "MELDLabelEncoder",
    "MELD_CLASS_NAMES",
    "NUM_MELD_CLASSES",
    "SERClassifier",
    "SERConfig",
    "SERPrediction",
    "predict_emotion",
    "predict_from_audio",
]
