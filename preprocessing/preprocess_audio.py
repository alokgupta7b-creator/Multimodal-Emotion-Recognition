"""
preprocessing.preprocess_audio
================================
Audio modality validation and reference resolution utilities.

This module does **not** perform any signal processing or feature
extraction.  Its responsibility is:

1. Resolving which audio resource (WAV file or precomputed feature array)
   is associated with a given manifest record.
2. Validating the existence, shape, and numeric integrity of those resources
   without loading entire large archives into RAM.
3. Reporting detailed per-sample status strings that the alignment record
   and validation report can consume.

Dataset-specific behaviour
--------------------------
MELD:
    Audio files are WAV clips on disk under the configurable ``audio_root``
    directory.  Files are referenced by the ``audio_path`` column in the
    MELD manifest.  RIFF/WAVE header integrity is checked with stdlib only.

CMU-MOSEI (COVAREP):
    Precomputed 74-dimensional COVAREP acoustic features are stored in
    ``CMU_MOSEI_COVAREP.csd`` (an HDF5 archive).  This module resolves
    references to segments within the archive using the ``audio_path``
    column of the CMU-MOSEI manifest (``features/CMU_MOSEI_COVAREP.csd#<id>``).
    It validates that numpy arrays for the referenced segment are present
    and have non-zero shape when ``numpy`` is available.

    Because the ``.csd`` archive can be hundreds of MB and must remain
    open across many records, callers are expected to pass an open HDF5
    handle rather than letting this module open/close it per record.

Public API
----------
    AudioStatus              — typed status string constants
    resolve_audio_modality   — main entry point: returns (ref_str, status)
    validate_audio_array     — validate a numpy array representing audio features
    check_wav_header         — pure-Python RIFF/WAVE header check (no soundfile)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Status constants — used in alignment records and validation reports
# ---------------------------------------------------------------------------

class AudioStatus:
    OK_WAV             = "ok_wav"
    OK_NPY_FEATURE     = "ok_npy_feature"
    OK_CSD_COVAREP     = "ok_csd_covarep"
    MISSING_FILE       = "missing_file"
    EMPTY_FILE         = "empty_file"
    WAV_HEADER_INVALID = "wav_header_invalid"
    ARRAY_EMPTY_SHAPE  = "array_empty_shape"
    ARRAY_NAN_INF      = "array_nan_inf"
    ARRAY_WRONG_DIM    = "array_wrong_dim"
    NO_ROOT_PROVIDED   = "no_audio_root_provided"
    NOT_IN_CSD         = "video_not_in_covarep_csd"
    CSD_UNAVAILABLE    = "covarep_csd_unavailable"
    UNVERIFIED         = "present_unverified"


# ---------------------------------------------------------------------------
# RIFF/WAVE header check (pure Python, no soundfile/librosa dependency)
# ---------------------------------------------------------------------------

def check_wav_header(filepath: Path) -> Tuple[bool, str]:
    """
    Validate the RIFF/WAVE header of an audio file without reading the full
    signal.

    Checks:
      - File exists and is at least 44 bytes.
      - Bytes 0–3: ``RIFF`` marker.
      - Bytes 8–11: ``WAVE`` format marker.
      - Bytes 12–15: ``fmt `` sub-chunk marker.
      - fmt sub-chunk size >= 16 bytes (PCM standard).
      - Audio format code is one of PCM (1), IEEE float (3), A-law (6),
        μ-law (7), or WAVE_FORMAT_EXTENSIBLE (0xFFFE).

    Parameters
    ----------
    filepath:
        Absolute or relative path to a ``.wav`` file.

    Returns
    -------
    (ok, status_string)
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.exists():
        return False, AudioStatus.MISSING_FILE

    try:
        size = filepath.stat().st_size
    except OSError:
        return False, AudioStatus.MISSING_FILE

    if size < 44:
        return False, AudioStatus.WAV_HEADER_INVALID

    try:
        with open(filepath, "rb") as fh:
            riff = fh.read(4)
            if riff != b"RIFF":
                return False, AudioStatus.WAV_HEADER_INVALID
            fh.seek(8)
            wave_marker = fh.read(4)
            if wave_marker != b"WAVE":
                return False, AudioStatus.WAV_HEADER_INVALID
            fmt_marker = fh.read(4)
            if fmt_marker != b"fmt ":
                return False, AudioStatus.WAV_HEADER_INVALID
            (subchunk_size,) = struct.unpack("<I", fh.read(4))
            if subchunk_size < 16:
                return False, AudioStatus.WAV_HEADER_INVALID
            (audio_fmt,) = struct.unpack("<H", fh.read(2))
            valid_fmts = {1, 3, 6, 7, 0xFFFE}
            if audio_fmt not in valid_fmts:
                return False, AudioStatus.WAV_HEADER_INVALID
    except (OSError, struct.error):
        return False, AudioStatus.WAV_HEADER_INVALID

    return True, AudioStatus.OK_WAV


# ---------------------------------------------------------------------------
# Numpy array validation (gracefully degrades if numpy unavailable)
# ---------------------------------------------------------------------------

def validate_audio_array(
    array: Any,
    expected_feature_dim: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate a numpy array representing precomputed audio features.

    Parameters
    ----------
    array:
        A numpy ndarray of shape ``(T, D)`` where T is the number of
        time-frames and D is the feature dimensionality.  An array with
        ``T == 0`` is treated as empty/invalid.
    expected_feature_dim:
        When set, the last dimension of ``array`` must match this value.
        For CMU-MOSEI COVAREP features the expected value is 74.

    Returns
    -------
    (ok, status_string)
    """
    try:
        import numpy as np
    except ImportError:
        # Cannot validate without numpy; treat as unverified
        return True, AudioStatus.UNVERIFIED

    if array is None:
        return False, AudioStatus.MISSING_FILE

    arr = np.asarray(array)

    if arr.ndim == 0 or arr.size == 0:
        return False, AudioStatus.ARRAY_EMPTY_SHAPE

    if arr.ndim < 2:
        return False, AudioStatus.ARRAY_WRONG_DIM

    if expected_feature_dim is not None and arr.shape[-1] != expected_feature_dim:
        return False, AudioStatus.ARRAY_WRONG_DIM

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False, AudioStatus.ARRAY_NAN_INF

    return True, AudioStatus.OK_NPY_FEATURE


# ---------------------------------------------------------------------------
# MELD audio resolution
# ---------------------------------------------------------------------------

def resolve_meld_audio(
    sample_id: str,
    audio_path_field: str,
    audio_root: Optional[Path],
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Resolve and optionally validate a MELD audio WAV file.

    The MELD manifest stores a *relative* ``audio_path`` such as
    ``audio/train/dia0_utt0.wav``.  This function resolves that path
    against the provided ``audio_root``.

    Parameters
    ----------
    sample_id:
        Manifest sample identifier (used only for logging).
    audio_path_field:
        Value of the ``audio_path`` column from the MELD manifest row.
    audio_root:
        Base directory under which the audio files reside.  The path is
        formed as ``audio_root / audio_path_field``.  When ``None`` the
        function returns the manifest value as-is and records the modality
        as unavailable.
    verify:
        When ``True``, read the WAV header to check integrity.

    Returns
    -------
    (ref_str, has_audio, status)
        ref_str   — relative path string suitable for the alignment record.
        has_audio — ``True`` when the file exists and passes validation.
        status    — descriptive status string.
    """
    ref_str = audio_path_field or ""

    if not audio_root:
        return ref_str, False, AudioStatus.NO_ROOT_PROVIDED

    # Build candidate path: try the stored relative path joined to the root
    if ref_str:
        candidate = audio_root / ref_str
        # Fallback: strip leading 'audio/' prefix if root already includes it
        if not candidate.exists():
            parts = Path(ref_str).parts
            if parts and parts[0] == "audio":
                candidate = audio_root / Path(*parts[1:])
    else:
        return ref_str, False, AudioStatus.MISSING_FILE

    if not candidate.exists():
        return ref_str, False, AudioStatus.MISSING_FILE

    if verify:
        ok, status = check_wav_header(candidate)
        return ref_str, ok, status

    size = candidate.stat().st_size if candidate.exists() else 0
    if size == 0:
        return ref_str, False, AudioStatus.EMPTY_FILE

    return ref_str, True, AudioStatus.UNVERIFIED


# ---------------------------------------------------------------------------
# CMU-MOSEI COVAREP resolution
# ---------------------------------------------------------------------------

def resolve_mosei_audio(
    video_id: str,
    clip_id: str,
    start_time: float,
    end_time: float,
    audio_path_field: str,
    covarep_csd_path: Optional[Any],
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Resolve a CMU-MOSEI COVAREP audio feature reference.

    For CMU-MOSEI, precomputed COVAREP features are stored in
    ``CMU_MOSEI_COVAREP.csd`` (HDF5).  This function:

    1. Returns the manifest ``audio_path`` reference string unchanged.
    2. Uses a reusable ``CSDArchiveReader`` to inspect the archive for
       video coverage, intervals overlap, and feature dimensionality.

    Parameters
    ----------
    video_id, clip_id, start_time, end_time:
        Segment identifiers from the CMU-MOSEI manifest.
    audio_path_field:
        Value of the ``audio_path`` manifest column (may be empty when the
        pipeline ran without CSD).
    covarep_csd_path:
        ``CSDArchiveReader`` instance, ``pathlib.Path`` to the ``.csd`` archive,
        an open ``h5py.File`` handle, or ``None`` when unavailable.
    verify:
        When ``True``, inspect the CSD archive for coverage.

    Returns
    -------
    (ref_str, has_audio, status)
    """
    from preprocessing.csd_reader import CSDArchiveReader, CSDStatus

    ref_str = audio_path_field or (
        f"features/CMU_MOSEI_COVAREP.csd#{video_id}_{clip_id}"
        if video_id else ""
    )

    if covarep_csd_path is None:
        return ref_str, False, AudioStatus.CSD_UNAVAILABLE

    if not verify:
        return ref_str, bool(ref_str), AudioStatus.UNVERIFIED

    # If already a CSDArchiveReader instance
    if isinstance(covarep_csd_path, CSDArchiveReader):
        reader = covarep_csd_path
        if not reader.is_open:
            reader.open()
        ok, raw_status = reader.verify_segment(video_id, start_time, end_time)
    else:
        # One-off or synthetic handle: wrap in reader context manager
        try:
            with CSDArchiveReader(
                covarep_csd_path, modality="audio", expected_feature_dim=74
            ) as reader:
                ok, raw_status = reader.verify_segment(
                    video_id, start_time, end_time
                )
        except Exception:
            return ref_str, False, AudioStatus.CSD_UNAVAILABLE

    if raw_status == CSDStatus.NOT_IN_CSD:
        status = AudioStatus.NOT_IN_CSD
    elif raw_status == CSDStatus.CSD_UNAVAILABLE:
        status = AudioStatus.CSD_UNAVAILABLE
    elif raw_status == CSDStatus.OK_COVAREP:
        status = AudioStatus.OK_CSD_COVAREP
    else:
        status = raw_status

    return ref_str, ok, status


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def resolve_audio_modality(
    dataset: str,
    record: dict,
    audio_root: Optional[Path] = None,
    covarep_csd: Optional[Any] = None,
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Resolve the audio modality for a manifest record from either MELD or
    CMU-MOSEI, dispatching to the appropriate dataset adapter.

    Parameters
    ----------
    dataset:
        ``"meld"`` or ``"mosei"``.
    record:
        A manifest row dict.
    audio_root:
        For MELD: root directory containing WAV files.
    covarep_csd:
        For CMU-MOSEI: path to the COVAREP CSD archive or an open h5py handle.
    verify:
        Perform file-level integrity checks.

    Returns
    -------
    (ref_str, has_audio, status)
    """
    if dataset == "meld":
        return resolve_meld_audio(
            sample_id=record.get("sample_id", ""),
            audio_path_field=record.get("audio_path", ""),
            audio_root=audio_root,
            verify=verify,
        )
    elif dataset == "mosei":
        return resolve_mosei_audio(
            video_id=str(record.get("video_id", "")),
            clip_id=str(record.get("clip_id", "")),
            start_time=float(record.get("start_time", 0.0)),
            end_time=float(record.get("end_time", 0.0)),
            audio_path_field=record.get("audio_path", ""),
            covarep_csd_path=covarep_csd,
            verify=verify,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
