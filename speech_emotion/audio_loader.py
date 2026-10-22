"""
speech_emotion.audio_loader
===========================
Robust audio loading, channel conversion, resampling, and validation.

Key capabilities:
-----------------
1. Pure Python + NumPy WAV decoding (16-bit, 24-bit, 32-bit float, 8-bit PCM)
   with zero required external C-library dependencies.
2. Optional fallback to `soundfile` or `librosa` if available.
3. Deterministic stereo-to-mono downmix by channel averaging: (L + R) / 2.
4. Resampling when input sampling rate differs from target rate.
5. Strict validation: detects missing files, empty files, corrupt headers,
   non-finite values (NaN/Inf), silent recordings, and invalid durations.
6. Returns rich `AudioMetadata` for auditing and exclusion logging.
"""

from __future__ import annotations

import dataclasses
import math
import struct
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Status Constants
# ---------------------------------------------------------------------------

class AudioLoadStatus:
    OK = "ok"
    FILE_NOT_FOUND = "file_not_found"
    EMPTY_FILE = "empty_file"
    UNREADABLE_HEADER = "unreadable_header"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNSUPPORTED_SAMPLE_RATE = "unsupported_sample_rate"
    ZERO_FRAMES = "zero_frames"
    NON_FINITE_VALUES = "non_finite_values"
    SILENT_AUDIO = "silent_audio"
    DURATION_TOO_SHORT = "duration_too_short"
    DURATION_TOO_LONG = "duration_too_long"
    CORRUPTED_STREAM = "corrupted_stream"


# ---------------------------------------------------------------------------
# Metadata Dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AudioMetadata:
    """
    Detailed audit information about a loaded audio recording.
    """

    file_path: str
    sample_rate: int = 0
    original_sample_rate: int = 0
    channels: int = 0
    num_frames: int = 0
    duration_seconds: float = 0.0
    is_mono: bool = True
    resampled: bool = False
    rms_db: float = -100.0
    is_valid: bool = False
    status: str = AudioLoadStatus.OK
    reason: str = ""


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def resample_signal(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1D audio waveform from orig_sr to target_sr.

    Tries `scipy.signal.resample_poly` or `librosa.resample` if available;
    otherwise falls back to high-quality linear interpolation in pure NumPy.
    """
    if orig_sr == target_sr or len(y) == 0:
        return y.astype(np.float32)

    # 1. Try scipy resample_poly
    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = resample_poly(y, up, down).astype(np.float32)
        return resampled
    except ImportError:
        pass

    # 2. Try librosa resample
    try:
        import librosa
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except (ImportError, Exception):
        pass

    # 3. Pure NumPy interpolation fallback
    duration = len(y) / float(orig_sr)
    target_len = int(round(duration * target_sr))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)

    orig_times = np.linspace(0.0, duration, len(y), endpoint=False)
    target_times = np.linspace(0.0, duration, target_len, endpoint=False)
    return np.interp(target_times, orig_times, y).astype(np.float32)


# ---------------------------------------------------------------------------
# Native WAV Decoder
# ---------------------------------------------------------------------------

def _decode_wav_pcm(
    file_path: Path,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Decode raw PCM audio frames from a WAV file using stdlib `wave`.

    Returns
    -------
    (data_array, channels, sample_rate, num_frames)
    """
    with wave.open(str(file_path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sample_rate = w.getframerate()
        num_frames = w.getnframes()

        if num_frames == 0:
            return np.zeros((0, channels), dtype=np.float32), channels, sample_rate, 0

        raw_bytes = w.readframes(num_frames)

    if sampwidth == 2:
        # Standard 16-bit signed PCM
        raw_arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        # 8-bit unsigned PCM
        raw_arr = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        # 32-bit PCM or float
        raw_arr = np.frombuffer(raw_bytes, dtype=np.float32)
    elif sampwidth == 3:
        # 24-bit PCM: 3 bytes per sample
        n_samples = len(raw_bytes) // 3
        # Unpack as little-endian signed 24-bit
        raw_arr = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            b0 = raw_bytes[i * 3]
            b1 = raw_bytes[i * 3 + 1]
            b2 = raw_bytes[i * 3 + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val & 0x800000:
                val -= 0x1000000
            raw_arr[i] = val / 8388608.0
    else:
        raise ValueError(f"Unsupported sample bit-depth width: {sampwidth} bytes")

    if channels > 1:
        # Reshape to (frames, channels)
        n_expected = num_frames * channels
        if len(raw_arr) >= n_expected:
            raw_arr = raw_arr[:n_expected].reshape(-1, channels)
        else:
            raw_arr = raw_arr.reshape(-1, channels)
    else:
        raw_arr = raw_arr.reshape(-1, 1)

    return raw_arr, channels, sample_rate, num_frames


# ---------------------------------------------------------------------------
# Signal Validation
# ---------------------------------------------------------------------------

def validate_audio_signal(
    y: np.ndarray,
    sr: int,
    min_duration: float = 0.1,
    max_duration: float = 60.0,
    silence_threshold_db: float = -60.0,
) -> Tuple[bool, str, str, float]:
    """
    Validate waveform numerical values, finiteness, duration, and energy.

    Returns
    -------
    (is_valid, status, reason, rms_db)
    """
    if len(y) == 0:
        return False, AudioLoadStatus.ZERO_FRAMES, "Audio recording contains 0 samples", -100.0

    if not np.all(np.isfinite(y)):
        return False, AudioLoadStatus.NON_FINITE_VALUES, "Audio waveform contains NaN or Inf values", -100.0

    duration = len(y) / float(sr)
    if duration < min_duration:
        return (
            False,
            AudioLoadStatus.DURATION_TOO_SHORT,
            f"Duration {duration:.3f}s is shorter than min_duration {min_duration:.3f}s",
            -100.0,
        )

    if duration > max_duration:
        return (
            False,
            AudioLoadStatus.DURATION_TOO_LONG,
            f"Duration {duration:.3f}s exceeds max_duration {max_duration:.3f}s",
            -100.0,
        )

    # Compute RMS energy
    mean_sq = np.mean(y ** 2)
    rms = math.sqrt(max(float(mean_sq), 1e-16))
    rms_db = 20.0 * math.log10(rms)

    if rms_db < silence_threshold_db:
        return (
            False,
            AudioLoadStatus.SILENT_AUDIO,
            f"RMS energy ({rms_db:.1f} dB) is below silence threshold ({silence_threshold_db:.1f} dB)",
            rms_db,
        )

    return True, AudioLoadStatus.OK, "", rms_db


# ---------------------------------------------------------------------------
# Main Public Audio Loader
# ---------------------------------------------------------------------------

def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
    min_duration: float = 0.1,
    max_duration: float = 60.0,
    silence_threshold_db: float = -60.0,
    allow_resample: bool = True,
) -> Tuple[np.ndarray, AudioMetadata]:
    """
    Load and validate an audio file with full integrity auditing.

    Parameters
    ----------
    file_path:
        Path to the audio file on disk.
    target_sr:
        Target sampling rate in Hz. Default: 16000.
    mono:
        Whether to convert stereo to mono by averaging channels. Default: True.
    min_duration:
        Minimum duration in seconds. Default: 0.1.
    max_duration:
        Maximum duration in seconds. Default: 60.0.
    silence_threshold_db:
        Silence threshold in dB. Default: -60.0.
    allow_resample:
        Whether to resample if input sampling rate != target_sr. Default: True.

    Returns
    -------
    (waveform, metadata)
        waveform: 1D numpy array of shape (N,) if mono=True, or (N, C)
        metadata: AudioMetadata with full audit details.
    """
    path = Path(file_path)
    meta = AudioMetadata(file_path=str(path))

    # 1. Existence check
    if not path.is_file():
        meta.is_valid = False
        meta.status = AudioLoadStatus.FILE_NOT_FOUND
        meta.reason = f"File does not exist: {path}"
        return np.zeros(0, dtype=np.float32), meta

    # 2. File size check
    try:
        file_size = path.stat().st_size
    except OSError as err:
        meta.is_valid = False
        meta.status = AudioLoadStatus.FILE_NOT_FOUND
        meta.reason = f"Cannot stat file: {err}"
        return np.zeros(0, dtype=np.float32), meta

    if file_size < 44:
        meta.is_valid = False
        meta.status = AudioLoadStatus.EMPTY_FILE
        meta.reason = f"File is empty or truncated ({file_size} bytes < 44 bytes minimum)"
        return np.zeros(0, dtype=np.float32), meta

    # 3. Decode audio
    loaded_arr = None
    orig_channels = 1
    orig_sr = target_sr
    num_frames = 0

    try:
        loaded_arr, orig_channels, orig_sr, num_frames = _decode_wav_pcm(path)
    except (wave.Error, struct.error, ValueError, EOFError, OSError) as wav_err:
        # Fallback to soundfile or librosa if available
        success = False
        try:
            import soundfile as sf
            data, sr_sf = sf.read(str(path), dtype="float32", always_2d=True)
            loaded_arr = data
            orig_channels = data.shape[1]
            orig_sr = sr_sf
            num_frames = data.shape[0]
            success = True
        except Exception:
            pass

        if not success:
            meta.is_valid = False
            meta.status = AudioLoadStatus.UNREADABLE_HEADER
            meta.reason = f"Unreadable WAV format or corrupted header: {wav_err}"
            return np.zeros(0, dtype=np.float32), meta
    except Exception as exc:
        meta.is_valid = False
        meta.status = AudioLoadStatus.CORRUPTED_STREAM
        meta.reason = f"Error reading audio stream: {exc}"
        return np.zeros(0, dtype=np.float32), meta

    meta.original_sample_rate = orig_sr
    meta.channels = orig_channels
    meta.num_frames = num_frames

    # 4. Check zero frames
    if loaded_arr is None or len(loaded_arr) == 0:
        meta.is_valid = False
        meta.status = AudioLoadStatus.ZERO_FRAMES
        meta.reason = "Decoded 0 audio frames"
        return np.zeros(0, dtype=np.float32), meta

    # 5. Convert to mono if requested
    if mono and orig_channels > 1:
        # Documented method: average channels across axis 1
        waveform = np.mean(loaded_arr, axis=1).astype(np.float32)
        meta.is_mono = True
    elif orig_channels == 1:
        waveform = loaded_arr.ravel().astype(np.float32)
        meta.is_mono = True
    else:
        waveform = loaded_arr.astype(np.float32)
        meta.is_mono = False

    # 6. Sample rate check & resampling
    if orig_sr != target_sr:
        if not allow_resample:
            meta.is_valid = False
            meta.sample_rate = orig_sr
            meta.status = AudioLoadStatus.UNSUPPORTED_SAMPLE_RATE
            meta.reason = f"Sample rate {orig_sr} Hz != target {target_sr} Hz and allow_resample is False"
            return waveform, meta
        waveform = resample_signal(waveform, orig_sr, target_sr)
        meta.sample_rate = target_sr
        meta.resampled = True
    else:
        meta.sample_rate = target_sr
        meta.resampled = False

    meta.duration_seconds = len(waveform) / float(meta.sample_rate)

    # 7. Validate numerical values and duration
    is_valid, status, reason, rms_db = validate_audio_signal(
        waveform,
        sr=meta.sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        silence_threshold_db=silence_threshold_db,
    )

    meta.rms_db = rms_db
    meta.is_valid = is_valid
    meta.status = status
    meta.reason = reason

    return waveform, meta


# ---------------------------------------------------------------------------
# Helper to write WAV files (useful for tests and synthetic audio)
# ---------------------------------------------------------------------------

def write_wav(
    file_path: Union[str, Path],
    waveform: np.ndarray,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Write a 16-bit PCM WAV file to disk.

    Parameters
    ----------
    file_path:
        Destination file path.
    waveform:
        1D array (mono) or 2D array of shape (samples, channels).
        Values should be in [-1.0, 1.0].
    sample_rate:
        Sampling rate in Hz. Default: 16000.
    channels:
        Number of channels (1 for mono, 2 for stereo). Default: 1.

    Returns
    -------
    pathlib.Path
    """
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.clip(waveform, -1.0, 1.0)
    int_samples = (y * 32767.0).astype(np.int16)

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(int_samples.tobytes())

    return out_path
