"""
speech_emotion.feature_extraction
=================================
Acoustic feature extraction: Mel-Frequency Cepstral Coefficients (MFCCs).

Implementation Details:
-----------------------
1. Standard MELD Convention:
   - 20 MFCC coefficients computed across time frames.
   - Utterance-level temporal pooling: 20 means + 20 standard deviations = 40 features.
   - Column layout:
     `mfcc_mean_01` .. `mfcc_mean_20`, `mfcc_std_01` .. `mfcc_std_20`.
2. Backends:
   - `librosa`: Used when available for exact floating-point reproduction of existing artifacts.
   - `numpy`: High-precision pure NumPy implementation of the Slaney Mel filterbank,
     STFT power spectrum, logarithmic compression, and orthogonal DCT-II.
     Guarantees zero-dependency execution anywhere NumPy is present.
3. Robustness:
   - Validates finite outputs (no NaN or Inf).
   - Reflection padding for signals shorter than n_fft window.
   - Deterministic and reproducible across repeated runs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from speech_emotion.audio_loader import AudioMetadata, load_audio
from speech_emotion.config import AudioConfig, DEFAULT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Pure NumPy Mel Filterbank & STFT Helpers
# ---------------------------------------------------------------------------

def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Convert Hz to Slaney Mel scale."""
    freqs = np.asanyarray(frequencies, dtype=np.float64)
    min_log_hz = 1000.0
    linear_step = 200.0 / 3.0
    logstep = math.log(6.4) / 27.0

    mels = np.zeros_like(freqs, dtype=np.float64)
    log_mask = freqs >= min_log_hz
    mels[~log_mask] = freqs[~log_mask] / linear_step
    mels[log_mask] = 15.0 + np.log(freqs[log_mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Convert Slaney Mel scale to Hz."""
    m = np.asanyarray(mels, dtype=np.float64)
    min_log_hz = 1000.0
    linear_step = 200.0 / 3.0
    logstep = math.log(6.4) / 27.0

    freqs = np.zeros_like(m, dtype=np.float64)
    log_mask = m >= 15.0
    freqs[~log_mask] = m[~log_mask] * linear_step
    freqs[log_mask] = min_log_hz * np.exp(logstep * (m[log_mask] - 15.0))
    return freqs


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Construct a Slaney Mel filterbank matrix of shape (n_mels, 1 + n_fft // 2).
    """
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fftfreqs = np.linspace(0.0, sr / 2.0, int(1 + n_fft // 2), dtype=np.float64)

    min_mel = _hz_to_mel_slaney(np.array([fmin]))[0]
    max_mel = _hz_to_mel_slaney(np.array([fmax]))[0]
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64)
    mel_f = _mel_to_hz_slaney(mel_points)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def _compute_stft_power(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Compute power spectrogram |STFT|^2 with periodic Hann window and padding.
    Default pad_mode="constant" (zero-padding) matches librosa's default.
    Returns array of shape (1 + n_fft // 2, num_frames).
    """
    pad_len = n_fft // 2
    if pad_mode == "constant":
        y_padded = np.pad(y, pad_len, mode="constant", constant_values=0)
    else:
        if len(y) < n_fft:
            pad_needed = n_fft - len(y)
            y = np.pad(y, (0, pad_needed), mode="symmetric")
        y_padded = np.pad(y, pad_len, mode=pad_mode)

    num_frames = 1 + (len(y_padded) - n_fft) // hop_length

    # Periodic Hann window: hann(M, sym=False)
    n = np.arange(n_fft, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)).astype(np.float32)

    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(num_frames, n_fft),
        strides=(y_padded.strides[0] * hop_length, y_padded.strides[0]),
    )
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=n_fft, axis=-1)
    power = (np.abs(stft) ** 2.0).T.astype(np.float32)  # (1 + n_fft//2, num_frames)
    return power


def _create_dct_ii_matrix(n_mfcc: int = 20, n_mels: int = 128) -> np.ndarray:
    """
    Orthogonal DCT-II matrix of shape (n_mfcc, n_mels).
    """
    matrix = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    n = np.arange(n_mels, dtype=np.float32)
    for k in range(n_mfcc):
        scale = math.sqrt(1.0 / n_mels) if k == 0 else math.sqrt(2.0 / n_mels)
        matrix[k] = scale * np.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * n_mels))
    return matrix


# ---------------------------------------------------------------------------
# Feature Extractor Class
# ---------------------------------------------------------------------------

class MFCCExtractor:
    """
    Mel-Frequency Cepstral Coefficients (MFCC) feature extractor.

    Parameters
    ----------
    config:
        AudioConfig specifying n_mfcc, n_fft, hop_length, n_mels, and layout.
    """

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()
        self._mel_filterbank: Optional[np.ndarray] = None
        self._dct_matrix: Optional[np.ndarray] = None

    def _get_filterbank(self, sr: int) -> np.ndarray:
        if self._mel_filterbank is None:
            fmax = self.config.fmax if self.config.fmax is not None else float(sr / 2.0)
            self._mel_filterbank = create_mel_filterbank(
                sr=sr,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=fmax,
            )
        return self._mel_filterbank

    def _get_dct_matrix(self) -> np.ndarray:
        if self._dct_matrix is None:
            self._dct_matrix = _create_dct_ii_matrix(
                n_mfcc=self.config.n_mfcc,
                n_mels=self.config.n_mels,
            )
        return self._dct_matrix

    def extract_frames(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Extract 2D MFCC matrix of shape (n_mfcc, num_frames).
        """
        if len(y) == 0:
            raise ValueError("Input audio waveform is empty (0 samples)")

        if not np.all(np.isfinite(y)):
            raise ValueError("Input audio waveform contains non-finite values (NaN/Inf)")

        sr = sr or self.config.sample_rate

        # 1. Check if librosa should be used
        use_librosa = False
        if self.config.engine == "librosa":
            use_librosa = True
        elif self.config.engine == "auto":
            try:
                import librosa
                use_librosa = True
            except ImportError:
                use_librosa = False

        if use_librosa:
            import librosa
            fmax = self.config.fmax if self.config.fmax is not None else float(sr / 2.0)
            mfcc_frames = librosa.feature.mfcc(
                y=y.astype(np.float32),
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=fmax,
            )
            return mfcc_frames

        # 2. Native NumPy Engine
        P = _compute_stft_power(
            y,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            pad_mode=self.config.pad_mode,
        )
        fb = self._get_filterbank(sr)
        M = np.dot(fb, P)

        # Power to dB (ref=1.0, amin=1e-10, top_db=80.0)
        amin = 1e-10
        top_db = 80.0
        log_M = 10.0 * np.log10(np.maximum(amin, M))
        log_M = np.maximum(log_M, log_M.max() - top_db)

        # DCT-II
        D = self._get_dct_matrix()
        mfcc_frames = np.dot(D, log_M)
        return mfcc_frames

    def extract(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Extract utterance-level pooled MFCC features.

        Returns
        -------
        1D numpy array of shape (40,) [for default layout: 20 means + 20 stds]
        """
        mfcc_frames = self.extract_frames(y, sr)

        if self.config.feature_layout == "mean_std":
            means = np.mean(mfcc_frames, axis=1)
            stds = np.std(mfcc_frames, axis=1)
            features = np.concatenate([means, stds]).astype(np.float32)
        elif self.config.feature_layout == "mean_only":
            features = np.mean(mfcc_frames, axis=1).astype(np.float32)
        elif self.config.feature_layout == "std_only":
            features = np.std(mfcc_frames, axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unknown feature_layout: {self.config.feature_layout}")

        if not np.all(np.isfinite(features)):
            raise ValueError("Extracted feature vector contains non-finite values (NaN/Inf)")

        return features

    def extract_from_file(
        self,
        file_path: Union[str, Path],
    ) -> Tuple[Optional[np.ndarray], AudioMetadata]:
        """
        Load an audio file and extract utterance-level features.

        Returns
        -------
        (features, metadata)
            features is None if audio is invalid.
        """
        waveform, meta = load_audio(
            file_path=file_path,
            target_sr=self.config.sample_rate,
            mono=self.config.mono,
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
            silence_threshold_db=self.config.silence_threshold_db,
            allow_resample=self.config.allow_resample,
        )

        if not meta.is_valid:
            return None, meta

        try:
            feats = self.extract(waveform, sr=meta.sample_rate)
            return feats, meta
        except Exception as exc:
            meta.is_valid = False
            meta.status = "extraction_error"
            meta.reason = f"Feature extraction failed: {exc}"
            return None, meta


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_mfcc_features(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    config: Optional[AudioConfig] = None,
) -> np.ndarray:
    """
    Convenience function to extract MFCC features from an audio waveform.
    """
    extractor = MFCCExtractor(config or AudioConfig(sample_rate=sr))
    return extractor.extract(y, sr=sr)
