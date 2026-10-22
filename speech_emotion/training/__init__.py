"""
speech_emotion.training
=======================
Training, tuning, and evaluation suite for Speech Emotion Recognition (SER).
"""

from speech_emotion.training.config import DataConfig, TrainingConfig
from speech_emotion.training.data import DatasetSplit, MELDDataLoader, FEATURE_NAMES
from speech_emotion.training.metrics import EvaluationResult, compute_metrics
from speech_emotion.training.train import SERModelTrainer, train_single_model, evaluate_split
from speech_emotion.training.evaluate import evaluate_model_on_split
from speech_emotion.training.mosei_data import (
    COVAREP_FEATURE_DIM,
    COVAREP_FEATURE_NAMES,
    MOSEI_SENTIMENT_CLASSES,
    MOSEI_TO_CANONICAL_EMOTION,
    MOSEIDataConfig,
    MOSEIDataLoader,
    harmonize_mosei_emotion,
)

__all__ = [
    "COVAREP_FEATURE_DIM",
    "COVAREP_FEATURE_NAMES",
    "DataConfig",
    "DatasetSplit",
    "EvaluationResult",
    "FEATURE_NAMES",
    "MELDDataLoader",
    "MOSEI_SENTIMENT_CLASSES",
    "MOSEI_TO_CANONICAL_EMOTION",
    "MOSEIDataConfig",
    "MOSEIDataLoader",
    "SERModelTrainer",
    "TrainingConfig",
    "compute_metrics",
    "evaluate_model_on_split",
    "evaluate_split",
    "harmonize_mosei_emotion",
    "train_single_model",
]
