#!/usr/bin/env python3
"""
preprocessing/tests/test_csd_reader.py
======================================
Regression tests for CSDArchiveReader:
- Hierarchical CSD lookup (root/data/video_id)
- Flat synthetic mock fallback (video_id)
- Missing video IDs, missing data groups, malformed archives
- Valid, empty, and non-overlapping timestamp intervals
- Repeated record access without repeatedly opening archives
- Resource cleanup after normal completion and exceptions
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from preprocessing.csd_reader import CSDArchiveReader, CSDStatus


class MockDataset:
    """Mock HDF5 Dataset supporting shape and slicing."""
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype

    def __getitem__(self, item: Any) -> np.ndarray:
        return self._data[item]


class MockGroup:
    """Mock HDF5 Group supporting dictionary-like access."""
    def __init__(self, items: Optional[Dict[str, Any]] = None):
        self._items = items or {}

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def keys(self):
        return self._items.keys()


class TestCSDArchiveReader(unittest.TestCase):

    def _make_hierarchical_csd(self, root_name="COVAREP", feature_dim=74):
        """Build standard CMU-MOSEI archive hierarchy: root -> data -> vid -> (intervals, features)."""
        intervals = np.array([
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ], dtype=np.float64)
        features = np.zeros((4, feature_dim), dtype=np.float32)

        vid_group = MockGroup({
            "intervals": MockDataset(intervals),
            "features": MockDataset(features),
        })

        data_group = MockGroup({"vid001": vid_group})
        root_group = MockGroup({"data": data_group})
        return MockGroup({root_name: root_group})

    def test_hierarchical_lookup_covarep(self):
        """Valid video in COVAREP/data/<video_id> is resolved and overlapping interval passes."""
        csd = self._make_hierarchical_csd(root_name="COVAREP", feature_dim=74)
        reader = CSDArchiveReader(csd, modality="audio", expected_feature_dim=74)
        with reader:
            ok, status = reader.verify_segment("vid001", start_time=0.5, end_time=1.5)
            self.assertTrue(ok)
            self.assertEqual(status, CSDStatus.OK_COVAREP)

    def test_hierarchical_lookup_facet(self):
        """Valid video in FACET 4.2/data/<video_id> is resolved and overlapping interval passes."""
        csd = self._make_hierarchical_csd(root_name="FACET 4.2", feature_dim=35)
        reader = CSDArchiveReader(csd, modality="visual", expected_feature_dim=35)
        with reader:
            ok, status = reader.verify_segment("vid001", start_time=2.2, end_time=3.5)
            self.assertTrue(ok)
            self.assertEqual(status, CSDStatus.OK_FACET42)

    def test_flat_mock_hierarchy_fallback(self):
        """Flat mock hierarchy (direct vid in archive) remains supported for synthetic fixtures."""
        intervals = np.array([[0.0, 2.0]], dtype=np.float64)
        features = np.zeros((1, 74), dtype=np.float32)
        flat_csd = MockGroup({"vid_flat": MockGroup({"intervals": MockDataset(intervals), "features": MockDataset(features)})})

        reader = CSDArchiveReader(flat_csd, modality="audio", expected_feature_dim=74)
        with reader:
            ok, status = reader.verify_segment("vid_flat", start_time=0.0, end_time=1.0)
            self.assertTrue(ok)
            self.assertEqual(status, CSDStatus.OK_COVAREP)

    def test_missing_video_id(self):
        """Video ID absent from data group returns NOT_IN_CSD."""
        csd = self._make_hierarchical_csd()
        reader = CSDArchiveReader(csd, modality="audio")
        with reader:
            ok, status = reader.verify_segment("nonexistent_vid", start_time=0.0, end_time=1.0)
            self.assertFalse(ok)
            self.assertEqual(status, CSDStatus.NOT_IN_CSD)

    def test_missing_data_group_empty_archive(self):
        """Empty archive without data group returns CSD_UNAVAILABLE."""
        reader = CSDArchiveReader(None, modality="audio")
        with reader:
            ok, status = reader.verify_segment("vid001", 0.0, 1.0)
            self.assertFalse(ok)
            self.assertEqual(status, CSDStatus.CSD_UNAVAILABLE)

    def test_zero_frames_in_interval(self):
        """Interval outside the video coverage returns zero_audio_frames_in_interval."""
        csd = self._make_hierarchical_csd()  # covers [0.0, 4.0]
        reader = CSDArchiveReader(csd, modality="audio")
        with reader:
            # Query interval [10.0, 15.0] does not overlap [0.0, 4.0]
            ok, status = reader.verify_segment("vid001", start_time=10.0, end_time=15.0)
            self.assertFalse(ok)
            self.assertEqual(status, "zero_audio_frames_in_interval")

    def test_empty_sequence(self):
        """0-length interval dataset returns EMPTY_SEQUENCE."""
        intervals = np.zeros((0, 2), dtype=np.float64)
        features = np.zeros((0, 74), dtype=np.float32)
        csd = MockGroup({"COVAREP": MockGroup({"data": MockGroup({"empty_vid": MockGroup({"intervals": MockDataset(intervals), "features": MockDataset(features)})})})})

        reader = CSDArchiveReader(csd, modality="audio")
        with reader:
            ok, status = reader.verify_segment("empty_vid", 0.0, 1.0)
            self.assertFalse(ok)
            self.assertEqual(status, CSDStatus.EMPTY_SEQUENCE)

    def test_invalid_intervals_shape(self):
        """1D intervals dataset returns INVALID_INTERVALS."""
        intervals = np.zeros(10, dtype=np.float64)
        features = np.zeros((10, 74), dtype=np.float32)
        csd = MockGroup({"COVAREP": MockGroup({"data": MockGroup({"bad_vid": MockGroup({"intervals": MockDataset(intervals), "features": MockDataset(features)})})})})

        reader = CSDArchiveReader(csd, modality="audio")
        with reader:
            ok, status = reader.verify_segment("bad_vid", 0.0, 1.0)
            self.assertFalse(ok)
            self.assertEqual(status, CSDStatus.INVALID_INTERVALS)

    def test_wrong_feature_dimension(self):
        """Feature dimensionality mismatch returns ARRAY_WRONG_DIM."""
        csd = self._make_hierarchical_csd(feature_dim=50)  # expected 74
        reader = CSDArchiveReader(csd, modality="audio", expected_feature_dim=74)
        with reader:
            ok, status = reader.verify_segment("vid001", 0.0, 1.0)
            self.assertFalse(ok)
            self.assertEqual(status, CSDStatus.ARRAY_WRONG_DIM)

    def test_repeated_access_reuses_open_handle(self):
        """Multiple queries on the same open reader reuse the handle (open_count == 1)."""
        csd = self._make_hierarchical_csd()
        reader = CSDArchiveReader(csd, modality="audio")
        with reader:
            self.assertEqual(reader.open_count, 1)
            for _ in range(5):
                ok, _ = reader.verify_segment("vid001", 0.5, 1.5)
                self.assertTrue(ok)
            self.assertEqual(reader.open_count, 1)
            self.assertEqual(reader.queries_count, 5)

    def test_deterministic_cleanup_on_exit_and_exception(self):
        """Context manager closes resource deterministically even when exception occurs."""
        csd = self._make_hierarchical_csd()
        reader = CSDArchiveReader(csd, modality="audio")

        try:
            with reader:
                self.assertTrue(reader.is_open)
                raise RuntimeError("Simulated processing error")
        except RuntimeError:
            pass

        self.assertFalse(reader.is_open)


if __name__ == "__main__":
    unittest.main()
