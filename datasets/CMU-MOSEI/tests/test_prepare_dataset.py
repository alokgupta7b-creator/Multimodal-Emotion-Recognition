#!/usr/bin/env python3
"""
Unit tests for the CMU-MOSEI Dataset Preparation Pipeline
=========================================================
Tests schema validation, deterministic manifest generation, duplicate detection,
video-level split isolation and leakage prevention, modality presence handling,
continuous preservation, discrete mapping, and exclusion logging.
"""

import csv
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Add datasets/CMU-MOSEI directory to sys.path to support hyphenated directory import
cmu_mosei_dir = Path(__file__).resolve().parent.parent
if str(cmu_mosei_dir) not in sys.path:
    sys.path.insert(0, str(cmu_mosei_dir))

from prepare_dataset import (
    OFFICIAL_EMOTIONS,
    check_mp4_integrity,
    check_wav_integrity,
    determine_dominant_emotion,
    discretize_sentiment,
    prepare_cmu_mosei,
    sanitize_path_display,
    validate_sample_record,
)


class TestCmuMoseiPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.fixtures_dir = Path(__file__).parent / "fixtures"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_wav(self, path: Path) -> None:
        """Creates a minimal valid 44-byte RIFF WAVE file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + 64))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<I", 16000))
            f.write(struct.pack("<I", 32000))
            f.write(struct.pack("<H", 2))
            f.write(struct.pack("<H", 16))
            f.write(b"data")
            f.write(struct.pack("<I", 64))
            f.write(b"\x00" * 64)

    def _create_synthetic_mp4(self, path: Path) -> None:
        """Creates a minimal valid MP4 file container with ftyp and moov boxes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00isommp42\x00\x00\x00\x08free\x00\x00\x00\x08moov")

    def test_schema_validation_valid_record(self):
        """Test that a compliant sample record passes validation and preserves continuous targets."""
        row = {
            "video_id": "vid_test_01",
            "clip_id": "0",
            "start_time": "1.5",
            "end_time": "4.2",
            "sentiment": "1.75",
            "happy": "2.0",
            "sad": "0.0",
            "anger": "0.0",
            "surprise": "0.5",
            "disgust": "0.0",
            "fear": "0.0",
            "split": "train",
            "text": "Valid utterance transcript.",
        }
        rec, err = validate_sample_record(row)
        self.assertIsNone(err)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["sample_id"], "vid_test_01_0")
        self.assertEqual(rec["sentiment"], 1.75)
        self.assertEqual(rec["sentiment_discrete"], "positive")
        self.assertEqual(rec["happy"], 2.0)
        self.assertEqual(rec["dominant_emotion"], "happy")
        self.assertEqual(rec["start_time"], 1.5)
        self.assertEqual(rec["end_time"], 4.2)

    def test_schema_validation_sentiment_out_of_bounds(self):
        """Test that sentiment scores outside [-3.0, 3.0] are detected and rejected."""
        row = {
            "video_id": "vid_err",
            "clip_id": "0",
            "start_time": "0.0",
            "end_time": "2.0",
            "sentiment": "3.5",  # Out of bounds (> 3.0)
            "happy": "0.0",
            "sad": "0.0",
            "anger": "0.0",
            "surprise": "0.0",
            "disgust": "0.0",
            "fear": "0.0",
            "split": "train",
        }
        rec, err = validate_sample_record(row)
        self.assertIsNone(rec)
        self.assertIn("invalid_sentiment_range", err)

    def test_schema_validation_emotion_out_of_bounds(self):
        """Test that emotion intensities outside [0.0, 3.0] are detected and rejected."""
        row = {
            "video_id": "vid_err",
            "clip_id": "0",
            "start_time": "0.0",
            "end_time": "2.0",
            "sentiment": "0.0",
            "happy": "-0.5",  # Out of bounds (< 0.0)
            "sad": "0.0",
            "anger": "0.0",
            "surprise": "0.0",
            "disgust": "0.0",
            "fear": "0.0",
            "split": "train",
        }
        rec, err = validate_sample_record(row)
        self.assertIsNone(rec)
        self.assertIn("invalid_emotion_range_happy", err)

    def test_schema_validation_invalid_timestamps(self):
        """Test that inverted timestamps (end_time < start_time) are rejected."""
        row = {
            "video_id": "vid_err",
            "clip_id": "0",
            "start_time": "5.0",
            "end_time": "2.0",  # Inverted
            "sentiment": "0.0",
            "happy": "0.0",
            "sad": "0.0",
            "anger": "0.0",
            "surprise": "0.0",
            "disgust": "0.0",
            "fear": "0.0",
            "split": "dev",
        }
        rec, err = validate_sample_record(row)
        self.assertIsNone(rec)
        self.assertEqual(err, "invalid_timestamps_negative_or_inverted")

    def test_discretize_sentiment_thresholds(self):
        """Test sentiment discretization with custom thresholds."""
        self.assertEqual(discretize_sentiment(0.5, threshold=0.0), "positive")
        self.assertEqual(discretize_sentiment(-0.5, threshold=0.0), "negative")
        self.assertEqual(discretize_sentiment(0.0, threshold=0.0), "neutral")
        # With threshold 0.2:
        self.assertEqual(discretize_sentiment(0.15, threshold=0.2), "neutral")
        self.assertEqual(discretize_sentiment(-0.15, threshold=0.2), "neutral")
        self.assertEqual(discretize_sentiment(0.25, threshold=0.2), "positive")
        self.assertEqual(discretize_sentiment(-0.25, threshold=0.2), "negative")

    def test_determine_dominant_emotion(self):
        """Test identifying dominant emotion with thresholds."""
        scores = {"happy": 1.5, "sad": 0.2, "anger": 0.0, "surprise": 0.0, "disgust": 0.0, "fear": 0.0}
        self.assertEqual(determine_dominant_emotion(scores), "happy")

        all_zero = {"happy": 0.0, "sad": 0.0, "anger": 0.0, "surprise": 0.0, "disgust": 0.0, "fear": 0.0}
        self.assertEqual(determine_dominant_emotion(all_zero), "none")

    def test_video_level_split_isolation_and_leakage_detection(self):
        """Test that allocating the same video ID to multiple splits is caught as leakage."""
        leak_csv = self.test_path / "leakage_test.csv"
        with open(leak_csv, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "clip_id", "start_time", "end_time", "sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear", "split"])
            writer.writerow(["vid_shared", "0", "0.0", "2.0", "1.0", "1.0", "0.0", "0.0", "0.0", "0.0", "0.0", "train"])
            writer.writerow(["vid_shared", "1", "2.0", "4.0", "0.5", "0.5", "0.0", "0.0", "0.0", "0.0", "0.0", "test"])

        stats = prepare_cmu_mosei(
            annotations_file=leak_csv,
            output_dir=self.test_path / "out_leak",
            verify_media=False,
        )

        audit = stats["split_isolation_audit"]
        self.assertTrue(audit["leakage_free"])
        # The second sample with the conflicting split must be routed to exclusions
        exclusions = stats["unassigned_exclusions"]
        self.assertEqual(len(exclusions), 1)
        self.assertIn("cross_split_leakage", exclusions[0]["exclusion_reason"])

    def test_duplicate_sample_id_detection(self):
        """Test that duplicate sample IDs are flagged and excluded."""
        dup_csv = self.test_path / "dup_test.csv"
        with open(dup_csv, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "clip_id", "start_time", "end_time", "sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear", "split"])
            writer.writerow(["vid_dup", "0", "0.0", "2.0", "1.0", "1.0", "0.0", "0.0", "0.0", "0.0", "0.0", "train"])
            writer.writerow(["vid_dup", "0", "0.0", "2.0", "1.0", "1.0", "0.0", "0.0", "0.0", "0.0", "0.0", "train"])

        stats = prepare_cmu_mosei(
            annotations_file=dup_csv,
            output_dir=self.test_path / "out_dup",
            verify_media=False,
        )

        self.assertEqual(stats["splits"]["train"]["total_samples"], 1)
        self.assertEqual(len(stats["unassigned_exclusions"]), 1)
        self.assertEqual(stats["unassigned_exclusions"][0]["exclusion_reason"], "duplicate_sample_id")

    def test_deterministic_manifest_output(self):
        """Test that repeated runs produce byte-for-byte identical manifest files."""
        anno_fixture = self.fixtures_dir / "synthetic_mosei_annotations.csv"
        out_run1 = self.test_path / "run1"
        out_run2 = self.test_path / "run2"

        prepare_cmu_mosei(annotations_file=anno_fixture, output_dir=out_run1, verify_media=False)
        prepare_cmu_mosei(annotations_file=anno_fixture, output_dir=out_run2, verify_media=False)

        for s in ["train", "dev", "test"]:
            f1 = (out_run1 / "manifests" / f"{s}_manifest.csv").read_bytes()
            f2 = (out_run2 / "manifests" / f"{s}_manifest.csv").read_bytes()
            self.assertEqual(f1, f2, f"Manifest for {s} split is not deterministic across runs.")

    def test_media_verification_and_missing_modality_handling(self):
        """Test that present media files are flagged usable and missing files are flagged unusable."""
        anno_csv = self.test_path / "media_test.csv"
        with open(anno_csv, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "clip_id", "start_time", "end_time", "sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear", "split"])
            writer.writerow(["vid_audio_ok", "0", "0.0", "2.0", "0.5", "0.5", "0.0", "0.0", "0.0", "0.0", "0.0", "train"])
            writer.writerow(["vid_audio_missing", "0", "0.0", "2.0", "-0.5", "0.0", "0.5", "0.0", "0.0", "0.0", "0.0", "train"])

        audio_dir = self.test_path / "audio"
        self._create_synthetic_wav(audio_dir / "vid_audio_ok_0.wav")
        # Do not create vid_audio_missing_0.wav

        stats = prepare_cmu_mosei(
            annotations_file=anno_csv,
            audio_dir=audio_dir,
            output_dir=self.test_path / "out_media",
            verify_media=True,
        )

        tr_stats = stats["splits"]["train"]
        self.assertEqual(tr_stats["total_samples"], 2)
        self.assertEqual(tr_stats["usable_samples"], 1)
        self.assertEqual(tr_stats["excluded_samples"], 1)
        self.assertIn("audio_missing_file", tr_stats["exclusions"][0]["exclusion_reason"])

    def test_path_sanitization(self):
        """Test that user home directories or absolute paths are sanitized."""
        personal_path = "C:/Users/TestUser/multimodal/data"
        sanitized = sanitize_path_display(personal_path)
        self.assertNotIn("TestUser", sanitized)

    def test_csd_features_verification(self):
        """Test that CSD computational sequence handles verify media presence and intervals."""
        import h5py
        import numpy as np

        mock_feat_dir = self.test_path / "mock_features"
        mock_feat_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create mock COVAREP CSD
        with h5py.File(mock_feat_dir / "CMU_MOSEI_COVAREP.csd", "w") as h5_aud:
            grp = h5_aud.create_group("COVAREP").create_group("data")
            v_grp = grp.create_group("vid_csd_01")
            v_grp.create_dataset("intervals", data=np.array([[0.0, 5.0]], dtype=np.float32))
            v_grp.create_dataset("features", data=np.ones((1, 74), dtype=np.float32))

        # 2. Create mock VisualFacet42 CSD
        with h5py.File(mock_feat_dir / "CMU_MOSEI_VisualFacet42.csd", "w") as h5_vis:
            grp = h5_vis.create_group("FACET 4.2").create_group("data")
            v_grp = grp.create_group("vid_csd_01")
            v_grp.create_dataset("intervals", data=np.array([[0.0, 5.0]], dtype=np.float32))
            v_grp.create_dataset("features", data=np.ones((1, 35), dtype=np.float32))

        anno_csv = self.test_path / "csd_test.csv"
        with open(anno_csv, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "clip_id", "start_time", "end_time", "sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear", "split", "text"])
            writer.writerow(["vid_csd_01", "0", "1.0", "3.0", "1.0", "1.0", "0.0", "0.0", "0.0", "0.0", "0.0", "train", "Mock utterance"])
            writer.writerow(["vid_csd_missing", "0", "1.0", "3.0", "-1.0", "0.0", "1.0", "0.0", "0.0", "0.0", "0.0", "train", "Mock missing utterance"])

        stats = prepare_cmu_mosei(
            annotations_file=anno_csv,
            features_dir=mock_feat_dir,
            output_dir=self.test_path / "out_csd",
            verify_media=True,
        )

        tr_stats = stats["splits"]["train"]
        self.assertEqual(tr_stats["total_samples"], 2)
        self.assertEqual(tr_stats["usable_samples"], 1)
        self.assertEqual(tr_stats["excluded_samples"], 1)
        self.assertEqual(tr_stats["modality_counts"]["has_audio"], 1)
        self.assertEqual(tr_stats["modality_counts"]["has_visual"], 1)


if __name__ == "__main__":
    unittest.main()
