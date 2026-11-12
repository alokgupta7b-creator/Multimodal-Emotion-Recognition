"""Compact seven-class facial frame classifier and video inference."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from facial_emotion.config import EMOTION_CLASSES, FACIAL_MODEL_ROOT
from facial_emotion.detector import FaceDetector, iter_video_faces


def _torch() -> Any:
    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as exc:
        raise RuntimeError("Facial inference requires a working PyTorch installation.") from exc
    return torch, nn


def create_model() -> Any:
    torch, nn = _torch()
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(128, len(EMOTION_CLASSES)),
    )


def preprocess_face_crops(faces: np.ndarray, torch: Any) -> Any:
    """Convert detector BGR crops to the normalized model tensor format."""
    array = np.asarray(faces)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("Expected face crops with shape (frames, height, width, 3).")
    if array.dtype != np.uint8:
        raise ValueError("Face crops must use uint8 pixel values.")
    rgb = np.ascontiguousarray(array[..., ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    return (tensor - 0.5) / 0.5


def temporal_logits(model: Any, frames: Any) -> Any:
    """Return mean frame logits for [batch, frames, channels, height, width]."""
    if frames.ndim == 4:
        return model(frames)
    if frames.ndim != 5:
        raise ValueError("Expected facial input with four or five dimensions.")
    batch, count, channels, height, width = frames.shape
    logits = model(frames.reshape(batch * count, channels, height, width))
    return logits.reshape(batch, count, -1).mean(dim=1)


def temporal_probabilities(model: Any, frames: Any, torch: Any) -> Any:
    """Return probabilities after temporal mean-logit aggregation."""
    return torch.softmax(temporal_logits(model, frames), dim=-1)


class FacialEmotionPredictor:
    def __init__(self, model_path: Path, device: str | None = None) -> None:
        torch, _ = _torch()
        self.torch = torch
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Facial model artifact not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model = create_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.normalize_input = bool(checkpoint.get("input_normalization", False))
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        self.device = torch.device(requested)
        self.model.to(self.device).eval()
        self.detector = FaceDetector()

    def predict_faces(self, faces: list[np.ndarray]) -> dict[str, Any]:
        if not faces:
            raise ValueError("No detectable faces were supplied.")
        batch = np.stack(faces)
        tensor = preprocess_face_crops(batch, self.torch).unsqueeze(0).to(self.device)
        if not self.normalize_input:
            tensor = tensor * 0.5 + 0.5
        start = time.perf_counter()
        with self.torch.no_grad():
            probabilities = temporal_probabilities(self.model, tensor, self.torch)[0].cpu().numpy()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        return {
            "predicted_label": EMOTION_CLASSES[int(probabilities.argmax())],
            "confidence": float(probabilities.max()),
            "probabilities": {label: float(value) for label, value in zip(EMOTION_CLASSES, probabilities)},
            "model_identifier": "facial-cnn-v1",
            "num_faces": len(faces),
            "num_sampled_frames": len(faces),
            "face_detection_coverage": 1.0,
            "prediction_entropy": entropy,
            "inference_latency_ms": (time.perf_counter() - start) * 1000,
        }

    def predict_video(self, video_path: Path, sample_every: int = 3) -> dict[str, Any]:
        faces = list(iter_video_faces(Path(video_path), self.detector, sample_every=sample_every))
        result = self.predict_faces(faces)
        result["num_sampled_frames"] = max(1, int(round(len(faces) / max(1, sample_every))))
        result["face_detection_coverage"] = min(1.0, len(faces) / result["num_sampled_frames"])
        return result
