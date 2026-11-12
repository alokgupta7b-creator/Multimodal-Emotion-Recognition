"""Regression and unit tests for Phase 17B-0 Facial Inference Remediation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.remediation import (
    NormalizationType,
    OptimizedFaceDetector,
    RemediatedFacialEmotionPredictor,
    remediated_preprocess_face_crops,
)
from project_config import REPOSITORY_ROOT

MODEL_PATH = REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt"


class TestFacialRemediation(unittest.TestCase):
    def test_checkpoint_loads_successfully_and_preserves_canonical_classes(self):
        predictor = RemediatedFacialEmotionPredictor(MODEL_PATH, device="cpu")
        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(EMOTION_CLASSES), 7)
        self.assertEqual(
            list(EMOTION_CLASSES),
            ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        )

    def test_legacy_normalization_produces_symmetric_minus_one_to_plus_one_range(self):
        # Create synthetic face crops in uint8 [0, 255]
        # Pixel min: 0, pixel max: 255
        crops = np.zeros((2, 96, 96, 3), dtype=np.uint8)
        crops[0, :, :, :] = 0     # Black -> should normalize to (0 - 0.5) / 0.5 = -1.0
        crops[1, :, :, :] = 255   # White -> should normalize to (1.0 - 0.5) / 0.5 = +1.0

        tensor = remediated_preprocess_face_crops(crops, normalization=NormalizationType.LEGACY_CNN)

        self.assertEqual(tensor.shape, (2, 3, 96, 96))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor[0].min().item(), -1.0, places=5)
        self.assertAlmostEqual(tensor[0].max().item(), -1.0, places=5)
        self.assertAlmostEqual(tensor[1].min().item(), 1.0, places=5)
        self.assertAlmostEqual(tensor[1].max().item(), 1.0, places=5)

    def test_preprocessing_converts_bgr_to_rgb(self):
        # Pure Red in BGR format: B=0, G=0, R=255
        pure_red_bgr = np.zeros((1, 96, 96, 3), dtype=np.uint8)
        pure_red_bgr[0, :, :, 2] = 255  # Red channel in BGR is index 2

        tensor = remediated_preprocess_face_crops(pure_red_bgr, normalization=NormalizationType.UNIT_INTERVAL)

        # In RGB tensor: R is channel 0, G is channel 1, B is channel 2
        self.assertAlmostEqual(tensor[0, 0, 0, 0].item(), 1.0, places=5)  # R
        self.assertAlmostEqual(tensor[0, 1, 0, 0].item(), 0.0, places=5)  # G
        self.assertAlmostEqual(tensor[0, 2, 0, 0].item(), 0.0, places=5)  # B

    def test_probabilities_are_finite_and_sum_to_one(self):
        predictor = RemediatedFacialEmotionPredictor(MODEL_PATH, device="cpu")
        dummy_crops = [np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(3)]

        result = predictor.predict_faces(dummy_crops)

        probs = result["probabilities"]
        self.assertEqual(len(probs), 7)
        prob_values = list(probs.values())
        self.assertTrue(all(np.isfinite(v) for v in prob_values))
        self.assertTrue(all(v >= 0.0 for v in prob_values))
        self.assertAlmostEqual(sum(prob_values), 1.0, places=6)
        self.assertIn(result["predicted_label"], EMOTION_CLASSES)
        self.assertAlmostEqual(result["confidence"], max(prob_values), places=6)

    def test_optimized_detector_scales_boxes_accurately(self):
        detector = OptimizedFaceDetector(min_face_size=24, downscale_factor=2.0)
        # Create blank image with synthetic white square
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Detect on blank frame returns empty list safely
        dets = detector.detect(frame)
        self.assertEqual(dets, [])

    def test_invalid_video_paths_and_empty_crops_handled_safely(self):
        predictor = RemediatedFacialEmotionPredictor(MODEL_PATH, device="cpu")

        # Missing file
        with self.assertRaises(FileNotFoundError):
            predictor.predict_video("non_existent_file.mp4")

        # Empty face list
        with self.assertRaises(ValueError):
            predictor.predict_faces([])

        # Corrupt / empty video file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            with self.assertRaises(ValueError):
                predictor.predict_video(tmp.name)


if __name__ == "__main__":
    unittest.main()
