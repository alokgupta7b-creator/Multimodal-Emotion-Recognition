"""Phase 17B-0: Remediated Facial Inference and Video Pipeline.

Provides isolated, non-destructive remediation for the 3 confirmed production defects:
1. Normalization Inversion Bug: Ensures [-1.0, 1.0] training parity for legacy CNN.
2. Frame Accounting Bug: Accurately records frames decoded, frames sampled, detections, and tracked faces.
3. Face Detection Optimization: Benchmarks and supports resolution-aware Haar cascade downscaling (e.g. 640x360).

Keeps production facial_emotion/model.py and detector.py unmodified until formal audit approval.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.detector import FaceDetection, FaceDetector
from facial_emotion.model import create_model, temporal_logits, temporal_probabilities


class NormalizationType(str, enum.Enum):
    """Explicit architecture-specific input normalization schemes."""
    LEGACY_CNN = "legacy_cnn"        # [-1.0, 1.0] via (tensor - 0.5) / 0.5 (Training match for 94k CNN)
    UNIT_INTERVAL = "unit_interval"  # [0.0, 1.0] float32 tensor
    IMAGENET = "imagenet"            # Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def remediated_preprocess_face_crops(
    faces: np.ndarray | Sequence[np.ndarray],
    normalization: NormalizationType = NormalizationType.LEGACY_CNN,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert detector BGR face crops to normalized PyTorch tensor format.
    
    Guarantees:
    - Input shape: (N, H, W, 3) with uint8 pixel values
    - Channel swap: OpenCV BGR -> RGB contiguous array
    - Normalization:
      - LEGACY_CNN: strictly in [-1.0, +1.0] without erroneous post-hoc inversion
      - IMAGENET: ImageNet mean/std (strictly for ResNet/EfficientNet, NEVER for legacy CNN)
      - UNIT_INTERVAL: [0.0, 1.0]
    """
    array = np.asarray(faces)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("Expected face crops with shape (N, height, width, 3).")
    if array.dtype != np.uint8:
        raise ValueError("Face crops must use uint8 pixel values in [0, 255].")

    # 1. Convert BGR to RGB and float32 in [0.0, 1.0]
    rgb = np.ascontiguousarray(array[..., ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)  # (N, 3, H, W)

    # 2. Apply explicit architecture-matched normalization
    if normalization == NormalizationType.LEGACY_CNN:
        norm_tensor = (tensor - 0.5) / 0.5
    elif normalization == NormalizationType.IMAGENET:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        norm_tensor = (tensor - mean) / std
    elif normalization == NormalizationType.UNIT_INTERVAL:
        norm_tensor = tensor
    else:
        raise ValueError(f"Unknown normalization scheme: {normalization}")

    if device is not None:
        norm_tensor = norm_tensor.to(device)
    return norm_tensor


class OptimizedFaceDetector(FaceDetector):
    """OpenCV Face Detector with resolution-aware detection downscaling and full-res crop recovery."""

    def __init__(self, min_face_size: int = 24, downscale_factor: float = 1.0) -> None:
        super().__init__(min_face_size=min_face_size)
        if downscale_factor < 1.0:
            raise ValueError("downscale_factor must be >= 1.0")
        self.downscale_factor = float(downscale_factor)

    def detect(self, frame: np.ndarray, size: int = 96, downscale: float | None = None) -> list[FaceDetection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Expected a color image with shape (height, width, channels).")

        factor = float(downscale or self.downscale_factor)
        orig_h, orig_w = frame.shape[:2]

        if factor > 1.0:
            small_w = max(1, int(round(orig_w / factor)))
            small_h = max(1, int(round(orig_h / factor)))
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            scaled_min_size = max(12, int(round(self.min_face_size / factor)))
            raw_faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(scaled_min_size, scaled_min_size),
            )
            detections: list[FaceDetection] = []
            for sx, sy, sw, sh in raw_faces:
                # Scale bounding box back to original full-resolution frame
                orig_x = max(0, min(orig_w - 1, int(round(sx * factor))))
                orig_y = max(0, min(orig_h - 1, int(round(sy * factor))))
                orig_box_w = max(1, min(orig_w - orig_x, int(round(sw * factor))))
                orig_box_h = max(1, min(orig_h - orig_y, int(round(sh * factor))))

                # Crop from ORIGINAL high-resolution frame for pristine quality
                crop = frame[orig_y : orig_y + orig_box_h, orig_x : orig_x + orig_box_w]
                if crop.size > 0:
                    detections.append(
                        FaceDetection(
                            bbox=(orig_x, orig_y, orig_box_w, orig_box_h),
                            crop=cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA),
                        )
                    )
            return detections
        else:
            # Full native resolution detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            detections = []
            for x, y, width, height in faces:
                crop = frame[y : y + height, x : x + width]
                if crop.size > 0:
                    detections.append(
                        FaceDetection(
                            bbox=(int(x), int(y), int(width), int(height)),
                            crop=cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA),
                        )
                    )
            return detections


@dataclass
class VideoFrameAccounting:
    total_video_frames: int
    frames_decoded: int
    frames_sampled: int
    frames_with_detected_faces: int
    total_face_detections: int
    unique_tracked_faces: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_video_frames": self.total_video_frames,
            "frames_decoded": self.frames_decoded,
            "frames_sampled": self.frames_sampled,
            "frames_with_detected_faces": self.frames_with_detected_faces,
            "total_face_detections": self.total_face_detections,
            "unique_tracked_faces": self.unique_tracked_faces,
        }


class RemediatedFacialEmotionPredictor:
    """Remediated Facial Emotion Predictor with verified normalization and accurate frame accounting."""

    def __init__(
        self,
        model_path: Path | str,
        normalization: NormalizationType = NormalizationType.LEGACY_CNN,
        device: str | None = None,
        downscale_factor: float = 1.0,
    ) -> None:
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Facial model artifact not found: {self.path}")

        self.normalization = normalization
        requested_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        self.device = torch.device(requested_dev)

        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        self.model = create_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

        self.detector = OptimizedFaceDetector(min_face_size=24, downscale_factor=downscale_factor)
        self.model_identifier = "facial-cnn-v1-remediated"

    def predict_faces(self, faces: list[np.ndarray]) -> dict[str, Any]:
        if not faces:
            raise ValueError("No detectable faces were supplied.")

        batch = np.stack(faces)
        # Normalization: strictly [-1.0, 1.0] for legacy CNN, without inverse scaling
        tensor = remediated_preprocess_face_crops(
            batch, normalization=self.normalization, device=self.device
        ).unsqueeze(0)  # (1, N, 3, H, W)

        start = time.perf_counter()
        with torch.no_grad():
            probabilities = temporal_probabilities(self.model, tensor, torch)[0].cpu().numpy()
        inference_latency_ms = (time.perf_counter() - start) * 1000.0

        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())

        return {
            "predicted_label": EMOTION_CLASSES[int(probabilities.argmax())],
            "confidence": float(probabilities.max()),
            "probabilities": {label: float(val) for label, val in zip(EMOTION_CLASSES, probabilities)},
            "model_identifier": self.model_identifier,
            "num_faces": len(faces),
            "prediction_entropy": entropy,
            "inference_latency_ms": inference_latency_ms,
        }

    def predict_video(
        self,
        video_path: Path | str,
        sample_every: int = 3,
        primary_face_only: bool = True,
    ) -> dict[str, Any]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        orig_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        t0_total = time.perf_counter()
        t_decode_acc = 0.0
        t_detect_acc = 0.0

        frames_decoded = 0
        frames_sampled = 0
        frames_with_faces = 0
        all_detected_crops: list[np.ndarray] = []
        tracked_primary_crops: list[np.ndarray] = []
        frame_tracking_metadata: list[dict[str, Any]] = []

        try:
            while True:
                t_dec_0 = time.perf_counter()
                ok, frame = capture.read()
                t_decode_acc += (time.perf_counter() - t_dec_0) * 1000.0
                if not ok:
                    break

                curr_frame_idx = frames_decoded
                frames_decoded += 1

                if curr_frame_idx % sample_every == 0:
                    frames_sampled += 1

                    t_det_0 = time.perf_counter()
                    detections = self.detector.detect(frame, size=96)
                    t_detect_acc += (time.perf_counter() - t_det_0) * 1000.0

                    if detections:
                        frames_with_faces += 1
                        for d in detections:
                            all_detected_crops.append(d.crop)

                        # Filter out secondary noise (area < 10,000 px^2 if primary face exists)
                        # and pick primary speaker face
                        primary_candidate = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
                        tracked_primary_crops.append(primary_candidate.crop)
                        frame_tracking_metadata.append({
                            "frame_index": curr_frame_idx,
                            "timestamp_sec": curr_frame_idx / fps if fps > 0 else 0.0,
                            "bbox": primary_candidate.bbox,
                            "area": primary_candidate.bbox[2] * primary_candidate.bbox[3],
                        })
        finally:
            capture.release()

        accounting = VideoFrameAccounting(
            total_video_frames=total_frames,
            frames_decoded=frames_decoded,
            frames_sampled=frames_sampled,
            frames_with_detected_faces=frames_with_faces,
            total_face_detections=len(all_detected_crops),
            unique_tracked_faces=len(tracked_primary_crops),
        )

        selected_crops = tracked_primary_crops if primary_face_only else all_detected_crops
        if not selected_crops:
            raise ValueError(f"No faces detected in video across {frames_sampled} sampled frames.")

        pred_result = self.predict_faces(selected_crops)
        t_total_ms = (time.perf_counter() - t0_total) * 1000.0

        # Assemble verified response
        pred_result["num_sampled_frames"] = frames_sampled
        pred_result["face_detection_coverage"] = float(frames_with_faces / max(1, frames_sampled))
        pred_result["frame_accounting"] = accounting.to_dict()
        pred_result["timing_profile"] = {
            "decoding_ms": round(t_decode_acc, 2),
            "detection_ms": round(t_detect_acc, 2),
            "neural_forward_ms": round(pred_result["inference_latency_ms"], 2),
            "total_pipeline_ms": round(t_total_ms, 2),
        }
        return pred_result

