import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.data import load_fer_csv
from facial_emotion.detector import FaceDetector
from facial_emotion.model import (
    create_model,
    preprocess_face_crops,
    temporal_logits,
    temporal_probabilities,
)
from facial_emotion.detector import FaceDetection
from facial_emotion.tracking_cache import (
    CACHE_VERSION,
    TrackingConfig,
    intersection_over_union,
    match_detection,
    resolve_cache_splits,
    track_detections,
    write_tracking_entry,
)


class TestFacialPreparation(unittest.TestCase):
    def test_canonical_classes(self):
        self.assertEqual(len(EMOTION_CLASSES), 7)
        self.assertEqual(EMOTION_CLASSES[0], "anger")

    def test_fer_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fer2013.csv"
            path.write_text("emotion,pixels,Usage\njoy," + " ".join(["0"] * 48 * 48) + ",Training\n", encoding="utf-8")
            rows = load_fer_csv(path)
            self.assertEqual(rows[0]["label_name"], "joy")
            self.assertEqual(len(rows[0]["pixels"]), 2304)

    def test_detector_handles_blank_frame(self):
        detector = FaceDetector()
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        self.assertEqual(detector.detect_and_align(frame), [])

    def test_invalid_video_path_is_rejected(self):
        detector = FaceDetector()
        with self.assertRaises(ValueError):
            list(__import__("facial_emotion.detector", fromlist=["iter_video_faces"]).iter_video_faces(Path("missing.mp4"), detector))

    def test_temporal_logits_shape(self):
        import torch

        model = create_model()
        frames = torch.zeros((2, 4, 3, 96, 96))
        logits = temporal_logits(model, frames)
        self.assertEqual(tuple(logits.shape), (2, 7))

    def test_temporal_probabilities_are_normalized(self):
        import torch

        model = create_model()
        frames = torch.zeros((1, 4, 3, 96, 96))
        probabilities = temporal_probabilities(model, frames, torch)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_preprocessing_converts_bgr_to_rgb_and_normalizes(self):
        import torch

        bgr = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
        tensor = preprocess_face_crops(bgr, torch)
        np.testing.assert_allclose(
            tensor.numpy()[0, :, 0, 0],
            np.array([1.0, 128 / 127.5 - 1.0, -1.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_temporal_probabilities_aggregate_logits_before_softmax(self):
        import torch

        class FixedFrameModel(torch.nn.Module):
            def forward(self, inputs):
                values = inputs[:, 0, 0, 0]
                zeros = torch.zeros_like(values)
                return torch.stack((values, -values, zeros, zeros, zeros, zeros, zeros), dim=1)

        model = FixedFrameModel()
        frames = torch.zeros((1, 2, 3, 1, 1))
        frames[0, 0, 0, 0, 0] = 2.0
        frames[0, 1, 0, 0, 0] = -2.0
        probabilities = temporal_probabilities(model, frames, torch)
        expected = torch.softmax(torch.zeros((1, 7)), dim=-1)
        torch.testing.assert_close(probabilities, expected)

    def test_tracking_geometry_and_identity(self):
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (2, 2, 10, 10)), 64 / 136)
        previous = FaceDetection((10, 10, 20, 20), np.zeros((96, 96, 3), dtype=np.uint8))
        detections = [
            FaceDetection((12, 11, 20, 20), np.zeros((96, 96, 3), dtype=np.uint8)),
            FaceDetection((80, 10, 20, 20), np.zeros((96, 96, 3), dtype=np.uint8)),
        ]
        self.assertEqual(match_detection(previous, detections, (100, 100))[0], 0)

    def test_tracking_handles_missing_and_terminates_track(self):
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
        frames = [
            (0, np.zeros((100, 100, 3), dtype=np.uint8), [FaceDetection((10, 10, 20, 20), crop)]),
            (1, np.zeros((100, 100, 3), dtype=np.uint8), []),
            (2, np.zeros((100, 100, 3), dtype=np.uint8), [FaceDetection((80, 10, 20, 20), crop)]),
        ]
        selected, metadata, summary = track_detections(frames, TrackingConfig(max_missing_frames=0))
        self.assertIsNotNone(selected[0])
        self.assertIsNone(selected[1])
        self.assertEqual(metadata[2].selected_track_id, 1)
        self.assertEqual(summary["track_switches"], 1)

    def test_tracking_rejects_low_confidence_match(self):
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
        previous = FaceDetection((10, 10, 20, 20), crop)
        far = FaceDetection((80, 80, 20, 20), crop)
        self.assertIsNone(match_detection(previous, [far], (100, 100), TrackingConfig(max_center_distance=0.1)))

    def test_versioned_cache_is_atomic_resumable_and_preserves_old_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.npy"
            old.write_bytes(b"preserve")
            crop = np.zeros((96, 96, 3), dtype=np.uint8)
            metadata = []
            summary = {"excluded": False, "coverage": 1.0}
            frame_metadata = []
            self.assertTrue(write_tracking_entry(root, "sample", [crop] * 4, frame_metadata, summary))
            self.assertTrue((root / "sample_4f.npy").is_file())
            payload = (root / "sample_4f.json").read_text(encoding="utf-8")
            self.assertIn(CACHE_VERSION, payload)
            self.assertEqual(old.read_bytes(), b"preserve")
            self.assertTrue(write_tracking_entry(root, "sample", [crop] * 4, frame_metadata, summary))

    def test_development_only_cache_splits_exclude_test(self):
        self.assertEqual(resolve_cache_splits(True, ["test"]), ("train", "dev"))
        self.assertEqual(resolve_cache_splits(False, None), ("train", "dev"))
        self.assertEqual(resolve_cache_splits(False, ["train", "dev"]), ("train", "dev"))
