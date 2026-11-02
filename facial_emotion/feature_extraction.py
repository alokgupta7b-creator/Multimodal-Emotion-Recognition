"""Facial landmark feature extraction for MELD videos."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from facial_emotion.config import FacialConfig


LANDMARK_COUNT = 478
COORDINATES_PER_LANDMARK = 2
FRAME_FEATURES = LANDMARK_COUNT * COORDINATES_PER_LANDMARK
TOTAL_FEATURES = FRAME_FEATURES * 2


class FacialFeatureExtractor:
    """Extract normalized Face Mesh landmarks from sampled video frames."""

    def __init__(self, config: Optional[FacialConfig] = None) -> None:
        self.config = config or FacialConfig()
        self._face_detector = None
        self._face_mesh = None

    def __enter__(self):
        self._initialize_models()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _initialize_models(self) -> None:
        """Initialize MediaPipe models once for this extractor."""
        if self._face_detector is None:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.config.face_detection_confidence,
            )

        if self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None

        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def _detect_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        """Detect faces and return square face crops."""
        self._initialize_models()

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._face_detector.process(rgb)
        crops = []

        if not results.detections:
            return crops

        for detection in results.detections:
            box = detection.location_data.relative_bounding_box

            x1 = max(0, int(box.xmin * width))
            y1 = max(0, int(box.ymin * height))
            x2 = min(width, int((box.xmin + box.width) * width))
            y2 = min(height, int((box.ymin + box.height) * height))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            side = int(
                max(x2 - x1, y2 - y1) * self.config.face_crop_padding
            )

            left = max(0, cx - side // 2)
            top = max(0, cy - side // 2)
            right = min(width, cx + (side + 1) // 2)
            bottom = min(height, cy + (side + 1) // 2)

            crop = frame[top:bottom, left:right]

            if crop.size == 0:
                continue

            crops.append(
                cv2.resize(
                    crop,
                    (
                        self.config.face_crop_size,
                        self.config.face_crop_size,
                    ),
                )
            )

        return crops

    def _extract_face_landmarks(
        self, face_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return normalized x/y coordinates for one face."""
        self._initialize_models()

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark

        coordinates = np.array(
            [[point.x, point.y] for point in landmarks],
            dtype=np.float32,
        )

        if coordinates.shape != (LANDMARK_COUNT, COORDINATES_PER_LANDMARK):
            return None

        if not np.all(np.isfinite(coordinates)):
            return None

        return coordinates.reshape(-1)

    def extract_from_video(
        self, video_path: str | Path
    ) -> Optional[np.ndarray]:
        """Extract pooled landmark features from one video."""
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._initialize_models()

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            capture.release()
            return None

        frame_indices = np.linspace(
            0,
            frame_count - 1,
            num=min(self.config.frame_sample_count, frame_count),
            dtype=int,
        )

        frame_features = []

        try:
            for frame_index in np.unique(frame_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()

                if not success or frame is None:
                    continue

                face_crops = self._detect_faces(frame)

                if not face_crops:
                    continue

                face_crop = max(
                    face_crops,
                    key=lambda crop: crop.shape[0] * crop.shape[1],
                )

                features = self._extract_face_landmarks(face_crop)

                if features is not None:
                    frame_features.append(features)

        finally:
            capture.release()

        if not frame_features:
            return None

        matrix = np.stack(frame_features, axis=0)

        pooled = np.concatenate(
            [
                np.mean(matrix, axis=0),
                np.std(matrix, axis=0),
            ]
        ).astype(np.float32)

        if pooled.shape != (TOTAL_FEATURES,):
            raise ValueError(
                f"Expected {TOTAL_FEATURES} features, got {pooled.shape}"
            )

        if not np.all(np.isfinite(pooled)):
            raise ValueError("Extracted facial features contain NaN or Inf")

        return pooled


def extract_facial_features(
    video_path: str | Path,
    config: Optional[FacialConfig] = None,
) -> Optional[np.ndarray]:
    """Convenience function for extracting one video's facial features."""
    with FacialFeatureExtractor(config) as extractor:
        return extractor.extract_from_video(video_path)