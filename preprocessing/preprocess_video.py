"""
preprocessing.preprocess_video
================================
Visual modality validation and reference resolution utilities.

This module handles two very different visual data sources:

MELD
    Raw MP4 video clips on disk.  Each utterance has its own clip file
    named ``dia{D}_utt{U}.mp4``.  The module validates the ISO base media
    format container (ftyp + moov atoms) using only stdlib ``struct``.
    It does **not** decode or extract any video frames.

CMU-MOSEI (VisualFacet 4.2)
    Precomputed facial Action Unit features from the OpenFace/FACET 4.2
    pipeline, stored in ``CMU_MOSEI_VisualFacet42.csd`` (HDF5).
    The 35-dimensional feature vectors per frame are *not* loaded; only
    the presence of the video ID and temporal overlap of intervals with the
    requested segment is verified when an open HDF5 handle is available.

Design constraints
------------------
* No frame extraction or video decoding is performed.
* Large CSD archives are not loaded into RAM.
* Missing visual data is recorded explicitly as a status string rather than
  raising an exception.

Public API
----------
    VisualStatus              — typed status string constants
    check_mp4_container       — pure-Python MP4 atom scanner (no ffprobe)
    validate_visual_array     — validate a numpy feature array for shape/NaN
    resolve_meld_visual       — MELD-specific video resolution
    resolve_mosei_visual      — CMU-MOSEI FACET 4.2 resolution
    resolve_visual_modality   — unified dispatcher
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

class VisualStatus:
    OK_MP4             = "ok_mp4"
    OK_NPY_FEATURE     = "ok_npy_feature"
    OK_CSD_FACET42     = "ok_csd_facet42"
    MISSING_FILE       = "missing_file"
    EMPTY_FILE         = "empty_file"
    CORRUPTED_NO_MOOV  = "corrupted_moov_atom_missing"
    CORRUPTED_NO_FTYP  = "corrupted_no_ftyp"
    MP4_PARSE_ERROR    = "mp4_parse_error"
    ARRAY_EMPTY_SHAPE  = "array_empty_shape"
    ARRAY_NAN_INF      = "array_nan_inf"
    ARRAY_WRONG_DIM    = "array_wrong_dim"
    NO_ROOT_PROVIDED   = "no_visual_root_provided"
    NOT_IN_CSD         = "video_not_in_facet_csd"
    CSD_UNAVAILABLE    = "facet_csd_unavailable"
    UNVERIFIED         = "present_unverified"


# ---------------------------------------------------------------------------
# Pure-Python MP4 container check
# ---------------------------------------------------------------------------

def check_mp4_container(filepath: Path) -> Tuple[bool, str]:
    """
    Validate the ISO base media file format (MP4/MOV) container by scanning
    top-level atoms (boxes) without decoding any video data.

    A valid MP4 container for multimodal research purposes must contain:
      - An ``ftyp`` (file type) box.
      - A ``moov`` (movie container) box.  Its absence typically means the
        file was not properly finalised (e.g., interrupted write).

    Parameters
    ----------
    filepath:
        Path to the MP4/MOV file.

    Returns
    -------
    (ok, status_string)
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.exists():
        return False, VisualStatus.MISSING_FILE

    try:
        size = filepath.stat().st_size
    except OSError:
        return False, VisualStatus.MISSING_FILE

    if size < 32:
        return False, VisualStatus.EMPTY_FILE

    found_ftyp = False
    found_moov = False

    try:
        with open(filepath, "rb") as fh:
            offset = 0
            while offset + 8 <= size:
                fh.seek(offset)
                header = fh.read(8)
                if len(header) < 8:
                    break
                atom_size, atom_type_bytes = struct.unpack(">I4s", header)
                try:
                    atom_type = atom_type_bytes.decode("latin-1")
                except Exception:
                    atom_type = ""

                if atom_type == "ftyp":
                    found_ftyp = True
                elif atom_type == "moov":
                    found_moov = True
                    break  # No need to scan further

                # Extended 64-bit box size
                if atom_size == 1:
                    ext = fh.read(8)
                    if len(ext) < 8:
                        break
                    atom_size = struct.unpack(">Q", ext)[0]
                    if atom_size < 16:
                        break
                elif atom_size == 0:
                    # Box extends to end of file
                    break
                elif atom_size < 8:
                    break

                offset += atom_size

    except (OSError, struct.error):
        return False, VisualStatus.MP4_PARSE_ERROR

    if not found_moov:
        return False, VisualStatus.CORRUPTED_NO_MOOV

    return True, VisualStatus.OK_MP4


# ---------------------------------------------------------------------------
# Numpy array validation
# ---------------------------------------------------------------------------

def validate_visual_array(
    array: Any,
    expected_feature_dim: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate a numpy array of precomputed visual features.

    Parameters
    ----------
    array:
        Expected shape ``(T, D)`` where T is frame count and D is the
        feature dimensionality.  For CMU-MOSEI VisualFacet 4.2 the
        expected ``D`` is 35.
    expected_feature_dim:
        When set, ``array.shape[-1]`` must match this value.

    Returns
    -------
    (ok, status_string)
    """
    try:
        import numpy as np
    except ImportError:
        return True, VisualStatus.UNVERIFIED

    if array is None:
        return False, VisualStatus.MISSING_FILE

    arr = np.asarray(array)

    if arr.ndim == 0 or arr.size == 0:
        return False, VisualStatus.ARRAY_EMPTY_SHAPE

    if arr.ndim < 2:
        return False, VisualStatus.ARRAY_WRONG_DIM

    if expected_feature_dim is not None and arr.shape[-1] != expected_feature_dim:
        return False, VisualStatus.ARRAY_WRONG_DIM

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False, VisualStatus.ARRAY_NAN_INF

    return True, VisualStatus.OK_NPY_FEATURE


# ---------------------------------------------------------------------------
# MELD visual resolution
# ---------------------------------------------------------------------------

def resolve_meld_visual(
    sample_id: str,
    video_path_field: str,
    video_root: Optional[Path],
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Resolve and optionally validate a MELD MP4 video clip.

    The MELD manifest stores a relative ``video_path`` such as
    ``train_splits/dia0_utt0.mp4``.

    Parameters
    ----------
    sample_id:
        Used for logging only.
    video_path_field:
        Value of the ``video_path`` column from the MELD manifest row.
    video_root:
        Base directory under which raw video splits reside.  When ``None``
        the resource is marked as unavailable.
    verify:
        When ``True``, scan the MP4 atoms to confirm container integrity.

    Returns
    -------
    (ref_str, has_visual, status)
    """
    ref_str = video_path_field or ""

    if not video_root:
        return ref_str, False, VisualStatus.NO_ROOT_PROVIDED

    if not ref_str:
        return ref_str, False, VisualStatus.MISSING_FILE

    candidate = video_root / ref_str
    if not candidate.exists():
        # Try stripping the leading split folder prefix
        parts = Path(ref_str).parts
        if len(parts) >= 2:
            candidate = video_root / Path(*parts[1:])

    if not candidate.exists():
        return ref_str, False, VisualStatus.MISSING_FILE

    if verify:
        ok, status = check_mp4_container(candidate)
        return ref_str, ok, status

    size = candidate.stat().st_size if candidate.exists() else 0
    if size == 0:
        return ref_str, False, VisualStatus.EMPTY_FILE

    return ref_str, True, VisualStatus.UNVERIFIED


# ---------------------------------------------------------------------------
# CMU-MOSEI VisualFacet 4.2 resolution
# ---------------------------------------------------------------------------

def resolve_mosei_visual(
    video_id: str,
    clip_id: str,
    start_time: float,
    end_time: float,
    visual_path_field: str,
    facet_csd_path: Optional[Any],
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Resolve a CMU-MOSEI VisualFacet 4.2 feature reference.

    Parameters
    ----------
    video_id, clip_id, start_time, end_time:
        Segment identifiers from the CMU-MOSEI manifest.
    visual_path_field:
        Value of the ``visual_path`` manifest column.
    facet_csd_path:
        ``CSDArchiveReader`` instance, ``pathlib.Path`` to the ``.csd`` archive,
        an open ``h5py.File`` handle, or ``None``.
    verify:
        Inspect CSD for temporal coverage.

    Returns
    -------
    (ref_str, has_visual, status)
    """
    from preprocessing.csd_reader import CSDArchiveReader, CSDStatus

    ref_str = visual_path_field or (
        f"features/CMU_MOSEI_VisualFacet42.csd#{video_id}_{clip_id}"
        if video_id else ""
    )

    if facet_csd_path is None:
        return ref_str, False, VisualStatus.CSD_UNAVAILABLE

    if not verify:
        return ref_str, bool(ref_str), VisualStatus.UNVERIFIED

    # If already a CSDArchiveReader instance
    if isinstance(facet_csd_path, CSDArchiveReader):
        reader = facet_csd_path
        if not reader.is_open:
            reader.open()
        ok, raw_status = reader.verify_segment(video_id, start_time, end_time)
    else:
        # One-off or synthetic handle: wrap in reader context manager
        try:
            with CSDArchiveReader(
                facet_csd_path, modality="visual", expected_feature_dim=35
            ) as reader:
                ok, raw_status = reader.verify_segment(
                    video_id, start_time, end_time
                )
        except Exception:
            return ref_str, False, VisualStatus.CSD_UNAVAILABLE

    if raw_status == CSDStatus.NOT_IN_CSD:
        status = VisualStatus.NOT_IN_CSD
    elif raw_status == CSDStatus.CSD_UNAVAILABLE:
        status = VisualStatus.CSD_UNAVAILABLE
    elif raw_status == CSDStatus.OK_FACET42:
        status = VisualStatus.OK_CSD_FACET42
    else:
        status = raw_status

    return ref_str, ok, status


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def resolve_visual_modality(
    dataset: str,
    record: dict,
    video_root: Optional[Path] = None,
    facet_csd: Optional[Any] = None,
    verify: bool = True,
) -> Tuple[str, bool, str]:
    """
    Unified visual modality resolver dispatching to the correct adapter.

    Parameters
    ----------
    dataset:
        ``"meld"`` or ``"mosei"``.
    record:
        Manifest row dict.
    video_root:
        For MELD: raw video clip directory.
    facet_csd:
        For CMU-MOSEI: path to VisualFacet CSD or open h5py.File.
    verify:
        Run file-level integrity check.

    Returns
    -------
    (ref_str, has_visual, status)
    """
    if dataset == "meld":
        return resolve_meld_visual(
            sample_id=record.get("sample_id", ""),
            video_path_field=record.get("video_path", ""),
            video_root=video_root,
            verify=verify,
        )
    elif dataset == "mosei":
        return resolve_mosei_visual(
            video_id=str(record.get("video_id", "")),
            clip_id=str(record.get("clip_id", "")),
            start_time=float(record.get("start_time", 0.0)),
            end_time=float(record.get("end_time", 0.0)),
            visual_path_field=record.get("visual_path", ""),
            facet_csd_path=facet_csd,
            verify=verify,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
