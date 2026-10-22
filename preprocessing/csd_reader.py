"""
preprocessing.csd_reader
========================
Safe, reusable HDF5 archive reader and context manager for CMU-MOSEI CSD files.

Design constraints:
-------------------
1. Deterministic resource management:
   Archives are opened once per processing run or batch, not once per record.
   All HDF5 file handles are deterministically closed on context exit or error.
2. Low memory footprint:
   Never loads large multi-gigabyte feature arrays into RAM. Reads only the
   small intervals array per query, and inspects dataset shapes without reading
   feature payloads.
3. Hierarchy flexibility:
   Supports both standard CMU-MultimodalSDK CSD structure:
       root_group (e.g. 'COVAREP', 'FACET 4.2') -> 'data' -> video_id
   and flat mock/synthetic test hierarchies:
       root_group / file -> video_id
4. Thread / process safety:
   HDF5 handles are not shared across processes. Each session owns its handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class CSDStatus:
    OK_COVAREP = "ok_csd_covarep"
    OK_FACET42 = "ok_csd_facet42"
    NOT_IN_CSD = "not_in_csd"
    CSD_UNAVAILABLE = "csd_unavailable"
    MISSING_DATASETS = "missing_intervals_or_features"
    EMPTY_SEQUENCE = "empty_sequence"
    INVALID_INTERVALS = "invalid_intervals_shape"
    ZERO_FRAMES_IN_INTERVAL = "zero_frames_in_interval"
    ARRAY_WRONG_DIM = "array_wrong_dim"


class CSDArchiveReader:
    """
    Safe reusable reader and context manager for CMU-MOSEI CSD (HDF5) archives.

    Parameters
    ----------
    csd_path_or_handle:
        Pathlib.Path, str path, or already-open HDF5 handle / dictionary.
    modality:
        "audio" (COVAREP) or "visual" (FACET 4.2).
    expected_feature_dim:
        Expected trailing feature dimension (74 for COVAREP, 35 for FACET 4.2).
    """

    def __init__(
        self,
        csd_path_or_handle: Optional[Union[Path, str, Any]],
        modality: str = "audio",
        expected_feature_dim: Optional[int] = None,
    ) -> None:
        self.csd_path_or_handle = csd_path_or_handle
        self.modality = modality.lower().strip()
        self.expected_feature_dim = (
            expected_feature_dim
            if expected_feature_dim is not None
            else (74 if self.modality == "audio" else 35)
        )
        self._file_handle: Any = None
        self._owns_handle: bool = False
        self._data_group: Any = None
        self._is_open: bool = False
        self.open_count: int = 0
        self.queries_count: int = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    def __enter__(self) -> CSDArchiveReader:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def open(self) -> CSDArchiveReader:
        """Open the CSD archive once if a file path was provided."""
        if self._is_open:
            return self

        if self.csd_path_or_handle is None:
            return self

        try:
            import h5py
        except ImportError:
            # h5py is missing; archive cannot be opened
            return self

        if isinstance(self.csd_path_or_handle, (str, Path)):
            path = Path(self.csd_path_or_handle)
            if not path.exists():
                return self
            try:
                self._file_handle = h5py.File(str(path), "r")
                self._owns_handle = True
            except Exception:
                return self
        else:
            # Passed an already open handle or dict
            self._file_handle = self.csd_path_or_handle
            self._owns_handle = False

        self._data_group = self._resolve_data_group(self._file_handle)
        self._is_open = True
        self.open_count += 1
        return self

    def close(self) -> None:
        """Deterministic cleanup of open handles."""
        if self._owns_handle and self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
        self._data_group = None
        self._is_open = False

    @staticmethod
    def _resolve_data_group(handle: Any) -> Any:
        """
        Locate the group containing video_id subgroups.

        Supports:
        1. Official CMU format: handle[root]['data'][video_id]
        2. Handle having 'data' directly: handle['data'][video_id]
        3. Flat mock format: handle[video_id]
        """
        if handle is None:
            return None

        # Check if handle itself has a 'data' subgroup (e.g. handle is fh[root])
        if hasattr(handle, "__contains__") and "data" in handle:
            return handle["data"]

        # Check top-level keys
        if hasattr(handle, "keys"):
            try:
                root_keys = list(handle.keys())
            except Exception:
                root_keys = []

            # Check if any top-level key contains a 'data' group (standard CMU CSD)
            for k in root_keys:
                try:
                    sub = handle[k]
                    if hasattr(sub, "__contains__") and "data" in sub:
                        return sub["data"]
                except Exception:
                    continue

        # Flat fallback: handle contains video_ids directly
        return handle

    def verify_segment(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> Tuple[bool, str]:
        """
        Verify video existence, interval overlap, and feature dimensions for a segment.

        Returns (is_valid, status_code).
        """
        self.queries_count += 1

        if not self._is_open or self._data_group is None:
            return False, CSDStatus.CSD_UNAVAILABLE

        # Video presence check
        try:
            if video_id not in self._data_group:
                return False, CSDStatus.NOT_IN_CSD
            vid_entry = self._data_group[video_id]
        except Exception:
            return False, CSDStatus.NOT_IN_CSD

        # Check dataset existence
        try:
            if "intervals" not in vid_entry or "features" not in vid_entry:
                return False, CSDStatus.MISSING_DATASETS
            intervals_ds = vid_entry["intervals"]
            features_ds = vid_entry["features"]
        except Exception:
            return False, CSDStatus.MISSING_DATASETS

        # Shape validation
        try:
            if intervals_ds.shape[0] == 0 or features_ds.shape[0] == 0:
                return False, CSDStatus.EMPTY_SEQUENCE

            if len(intervals_ds.shape) != 2 or intervals_ds.shape[1] != 2:
                return False, CSDStatus.INVALID_INTERVALS

            if (
                self.expected_feature_dim is not None
                and features_ds.shape[-1] != self.expected_feature_dim
            ):
                return False, CSDStatus.ARRAY_WRONG_DIM
        except Exception:
            return False, CSDStatus.INVALID_INTERVALS

        # Timestamp interval overlap check
        try:
            import numpy as np

            intervals = intervals_ds[:]
            # Overlap condition: frame_start < end_time and frame_end > start_time
            overlap = (intervals[:, 0] < end_time) & (intervals[:, 1] > start_time)
            if not overlap.any():
                status = (
                    "zero_audio_frames_in_interval"
                    if self.modality == "audio"
                    else "zero_visual_frames_in_interval"
                )
                return False, status
        except Exception:
            return False, CSDStatus.INVALID_INTERVALS

        ok_status = (
            CSDStatus.OK_COVAREP if self.modality == "audio" else CSDStatus.OK_FACET42
        )
        return True, ok_status
