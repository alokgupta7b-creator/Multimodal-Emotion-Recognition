"""OpenCV face detection, alignment and video frame extraction."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    detection_confidence: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)


class FaceDetector:
    def __init__(self, min_face_size: int = 24) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.is_file():
            raise FileNotFoundError(f"OpenCV Haar cascade not found: {cascade_path}")
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        self.min_face_size = min_face_size

    def detect_and_align(self, frame: np.ndarray, size: int = 96) -> list[np.ndarray]:
        return [detection.crop for detection in self.detect(frame, size=size)]

    def detect(self, frame: np.ndarray, size: int = 96) -> list[FaceDetection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Expected a color image with shape (height, width, channels).")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size))
        detections: list[FaceDetection] = []
        for x, y, width, height in faces:
            crop = frame[y:y + height, x:x + width]
            if crop.size:
                detections.append(
                    FaceDetection(
                        bbox=(int(x), int(y), int(width), int(height)),
                        crop=cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA),
                    )
                )
        return detections


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """Prefer OpenCV's FFmpeg backend so browser-captured or audio-only streams never stall Windows MSMF."""
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    if ffmpeg_backend is not None:
        try:
            capture = cv2.VideoCapture(str(video_path), ffmpeg_backend)
            if capture.isOpened() or os.name == "nt":
                return capture
            capture.release()
        except TypeError:
            pass
    return cv2.VideoCapture(str(video_path))


def iter_video_faces(video_path: Path, detector: FaceDetector, sample_every: int = 1, size: int = 96) -> Iterator[np.ndarray]:
    """Yield aligned face crops from sampled video frames."""
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    capture = open_video_capture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every == 0:
                yield from detector.detect_and_align(frame, size=size)
            index += 1
    finally:
        capture.release()
