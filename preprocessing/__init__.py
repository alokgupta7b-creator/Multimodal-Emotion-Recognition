"""
preprocessing — Cross-modal alignment and preprocessing module.

This package provides dataset-agnostic utilities for text, audio, and video
preprocessing and cross-modal alignment for multimodal emotion recognition.

Public API (import these from the package):
    config             — AlignmentConfig and dataset adapter constants
    preprocess_text    — normalize_text, preprocess_text_record, load_transcript_source
    preprocess_audio   — validate_audio_feature_array, resolve_audio_modality
    preprocess_video   — validate_visual_feature_array, resolve_visual_modality
    csd_reader         — CSDArchiveReader, CSDStatus
    align_modalities   — AlignmentRecord, align_meld_record, align_mosei_record
    validate_alignment — run_validation, AlignmentReport
"""

from preprocessing.csd_reader import CSDArchiveReader, CSDStatus
from preprocessing.preprocess_text import load_transcript_source

__version__ = "0.1.0"
__all__ = [
    "CSDArchiveReader",
    "CSDStatus",
    "load_transcript_source",
]
