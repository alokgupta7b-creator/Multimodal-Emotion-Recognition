"""
speech_emotion.config
=====================
Configuration and constants for audio preprocessing, feature extraction,
and dataset integration.

Design principles:
------------------
1. All parameters have explicit, documented defaults.
2. Supports the official MELD 40-feature MFCC convention (20 means + 20 stds).
3. Configurable sampling rate, filterbank size, FFT windows, and durations.
4. Independent of user machine paths and system usernames.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# MELD Dataset Constants
# ---------------------------------------------------------------------------

MELD_EMOTIONS: frozenset[str] = frozenset({
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
})

MELD_SPLITS: Tuple[str, ...] = ("train", "dev", "test")

# Standard MELD speech sampling rate
DEFAULT_SAMPLE_RATE: int = 16000

# Standard 20-coefficient MFCC configuration
DEFAULT_N_MFCC: int = 20
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_N_MELS: int = 128
DEFAULT_FMIN: float = 0.0
DEFAULT_FMAX: float = 8000.0

# 20 means + 20 standard deviations = 40 features
TOTAL_MFCC_FEATURES: int = 40


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AudioConfig:
    """
    Configuration parameters for audio loading and MFCC feature extraction.

    Parameters
    ----------
    sample_rate:
        Target sampling rate in Hz. Audio will be validated against this rate
        or resampled if permitted. Default: 16000.
    mono:
        Whether to convert multi-channel audio to mono. Default: True.
    n_mfcc:
        Number of MFCC coefficients to compute per frame. Default: 20.
    n_fft:
        FFT window length in samples. Default: 2048.
    hop_length:
        Hop length between successive frames in samples. Default: 512.
    n_mels:
        Number of Mel filter bands. Default: 128.
    fmin:
        Lowest frequency in Hz for the Mel filterbank. Default: 0.0.
    fmax:
        Highest frequency in Hz for the Mel filterbank. If None, defaults
        to sample_rate / 2.0. Default: 8000.0.
    min_duration:
        Minimum duration in seconds for a valid audio recording. Default: 0.1.
    max_duration:
        Maximum duration in seconds for a valid audio recording. Default: 60.0.
    silence_threshold_db:
        RMS energy threshold in dB below which a recording is considered
        silent. Default: -60.0.
    allow_resample:
        Whether to resample audio that does not match sample_rate. Default: True.
    feature_layout:
        Utterance pooling layout. "mean_std" concatenates n_mfcc means and
        n_mfcc standard deviations. Default: "mean_std".
    engine:
        Feature extraction backend: "auto", "numpy", or "librosa".
        "auto" uses librosa if available, otherwise pure numpy. Default: "auto".
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    mono: bool = True
    n_mfcc: int = DEFAULT_N_MFCC
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    n_mels: int = DEFAULT_N_MELS
    fmin: float = DEFAULT_FMIN
    fmax: Optional[float] = DEFAULT_FMAX
    min_duration: float = 0.1
    max_duration: float = 60.0
    silence_threshold_db: float = -60.0
    allow_resample: bool = True
    feature_layout: str = "mean_std"
    engine: str = "auto"
    pad_mode: str = "constant"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_mfcc <= 0:
            raise ValueError(f"n_mfcc must be positive, got {self.n_mfcc}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.fmin < 0:
            raise ValueError(f"fmin must be >= 0, got {self.fmin}")
        if self.fmax is None:
            self.fmax = float(self.sample_rate / 2.0)
        elif self.fmax <= self.fmin:
            raise ValueError(f"fmax ({self.fmax}) must be greater than fmin ({self.fmin})")
        if self.min_duration <= 0:
            raise ValueError(f"min_duration must be positive, got {self.min_duration}")
        if self.max_duration <= self.min_duration:
            raise ValueError(f"max_duration ({self.max_duration}) must exceed min_duration ({self.min_duration})")
        if self.feature_layout not in ("mean_std", "mean_only", "std_only"):
            raise ValueError(f"Unknown feature_layout: {self.feature_layout!r}")
        if self.engine not in ("auto", "numpy", "librosa"):
            raise ValueError(f"Unknown engine: {self.engine!r}")
        if self.pad_mode not in ("constant", "reflect", "symmetric", "edge"):
            raise ValueError(f"Unknown pad_mode: {self.pad_mode!r}")

    @property
    def total_features(self) -> int:
        """Total number of scalar features per utterance."""
        if self.feature_layout == "mean_std":
            return self.n_mfcc * 2
        return self.n_mfcc

    def get_feature_names(self) -> List[str]:
        """
        Return the standardized feature column names.

        For n_mfcc=20 and layout 'mean_std':
        ['mfcc_mean_01', ..., 'mfcc_mean_20', 'mfcc_std_01', ..., 'mfcc_std_20']
        """
        names: List[str] = []
        if self.feature_layout in ("mean_std", "mean_only"):
            names.extend([f"mfcc_mean_{j+1:02d}" for j in range(self.n_mfcc)])
        if self.feature_layout in ("mean_std", "std_only"):
            names.extend([f"mfcc_std_{j+1:02d}" for j in range(self.n_mfcc)])
        return names
