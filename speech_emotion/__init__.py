"""
speech_emotion
==============
Dedicated module for speech emotion preprocessing, audio integrity validation,
and acoustic feature extraction.

Part of the Multimodal Emotion Recognition research project.
"""

from speech_emotion.config import (
    AudioConfig,
    DEFAULT_SAMPLE_RATE,
    MELD_EMOTIONS,
    MELD_SPLITS,
    TOTAL_MFCC_FEATURES,
)
from speech_emotion.audio_loader import (
    AudioLoadStatus,
    AudioMetadata,
    load_audio,
    resample_signal,
    validate_audio_signal,
    write_wav,
)
from speech_emotion.feature_extraction import (
    MFCCExtractor,
    create_mel_filterbank,
    extract_mfcc_features,
)
from speech_emotion.preprocess_audio import (
    AuditReport,
    load_manifest_records,
    process_split,
)
from speech_emotion.model import (
    MELDLabelEncoder,
    MELD_CLASS_NAMES,
    NUM_MELD_CLASSES,
    SERClassifier,
    SERConfig,
    SERPrediction,
    predict_emotion,
    predict_from_audio,
)

__version__ = "0.1.0"

__all__ = [
    "AudioConfig",
    "AudioLoadStatus",
    "AudioMetadata",
    "AuditReport",
    "DEFAULT_SAMPLE_RATE",
    "MELDLabelEncoder",
    "MELD_CLASS_NAMES",
    "MELD_EMOTIONS",
    "MELD_SPLITS",
    "MFCCExtractor",
    "NUM_MELD_CLASSES",
    "SERClassifier",
    "SERConfig",
    "SERPrediction",
    "TOTAL_MFCC_FEATURES",
    "create_mel_filterbank",
    "extract_mfcc_features",
    "load_audio",
    "load_manifest_records",
    "predict_emotion",
    "predict_from_audio",
    "process_split",
    "resample_signal",
    "validate_audio_signal",
    "write_wav",
]
