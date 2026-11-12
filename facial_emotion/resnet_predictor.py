"""Independent Phase 17 ResNet18 + MTCNN facial emotion predictor for Phase 18."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from calibration_utils import (
    apply_logit_calibration,
    load_logit_calibration,
    resolve_calibration_path,
)
from facial_emotion.config import DEFAULT_IMAGE_SIZE, EMOTION_CLASSES, FACIAL_MODEL_ROOT
from facial_emotion.mtcnn_detector import (
    MtcnnFaceDetector,
    sample_video_frames,
    track_mtcnn_detections,
)
from facial_emotion.remediation import NormalizationType, remediated_preprocess_face_crops

logger = logging.getLogger(__name__)

PRIMARY_CHECKPOINT_SHA256 = "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64"
FALLBACK_CHECKPOINT_SHA256 = "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0"
PRIMARY_MODEL_IDENTIFIER = "facial-resnet18-ce-v1"
FALLBACK_MODEL_IDENTIFIER = "facial-resnet18-focal-v1"
DEFAULT_RESNET_DIR = FACIAL_MODEL_ROOT / "resnet18"
DEFAULT_PRIMARY_CHECKPOINT = DEFAULT_RESNET_DIR / "exp5_ce_weighted_best.pt"
DEFAULT_FALLBACK_CHECKPOINT = DEFAULT_RESNET_DIR / "exp5_focal_g3_wt_best.pt"
DEFAULT_TEMPORAL_POOLING = "raw_all_detections_mean_prob"
SUPPORTED_POOLING_METHODS = (
    "raw_all_detections_mean_prob",
    "raw_mean_logit",
    "track_arith_mean_prob",
    "track_mean_logit",
    "track_topk3_prob",
    "track_quality_conf_weighted",
    "single_frame_middle",
)


def compute_sha256(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def build_resnet18_classifier(num_classes: int = len(EMOTION_CLASSES)) -> nn.Module:
    """Build a 7-class ResNet18 classifier without network weight downloads."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def preprocess_resnet_face_crops(
    faces: np.ndarray | Sequence[np.ndarray],
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Resize if needed, swap BGR->RGB, and apply strict ImageNet normalization."""
    if isinstance(faces, np.ndarray):
        if faces.ndim != 4 or faces.shape[-1] != 3:
            raise ValueError("Expected face crops array with shape (N, height, width, 3).")
        crop_seq: Sequence[np.ndarray] = [faces[i] for i in range(faces.shape[0])]
    else:
        crop_seq = list(faces)

    if not crop_seq:
        raise ValueError("No detectable faces were supplied.")

    standardized: list[np.ndarray] = []
    for idx, crop in enumerate(crop_seq):
        arr = np.asarray(crop)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Face crop at index {idx} must have shape (H, W, 3).")
        if arr.dtype != np.uint8:
            raise ValueError(f"Face crop at index {idx} must use uint8 pixel values.")
        if arr.shape[0] != image_size or arr.shape[1] != image_size:
            interp = cv2.INTER_CUBIC if arr.shape[0] < image_size or arr.shape[1] < image_size else cv2.INTER_AREA
            arr = cv2.resize(arr, (image_size, image_size), interpolation=interp)
        standardized.append(arr)

    batch = np.stack(standardized, axis=0)
    return remediated_preprocess_face_crops(
        batch,
        normalization=NormalizationType.IMAGENET,
        device=device,
    )


def _validate_and_load_checkpoint(
    path: Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    """Verify file presence, SHA-256 hash, architecture, and 7-class mapping before loading weights."""
    if not path.is_file():
        raise FileNotFoundError(f"ResNet18 facial checkpoint not found: {path}")

    actual_sha256 = compute_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.strip().lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload in {path}: missing 'state_dict'.")

    classes = tuple(checkpoint.get("classes", ()))
    if classes != tuple(EMOTION_CLASSES):
        raise ValueError(f"Checkpoint classes {classes!r} do not match canonical {tuple(EMOTION_CLASSES)!r}.")

    arch = checkpoint.get("architecture")
    if arch != "resnet18":
        raise ValueError(f"Expected ResNet18 checkpoint architecture, got {arch!r}.")

    norm_cfg = checkpoint.get("input_normalization")
    if isinstance(norm_cfg, dict) and norm_cfg.get("mode") != "imagenet":
        raise ValueError(f"Expected 'imagenet' normalization in checkpoint metadata, got {norm_cfg.get('mode')!r}.")

    return checkpoint, actual_sha256


@contextmanager
def _strict_fp32_guard(enabled: bool, device: torch.device) -> Iterator[None]:
    """Ensure deterministic IEEE FP32 precision on CUDA Ada/Ampere GPUs when enabled."""
    if not enabled or device.type != "cuda":
        yield
        return

    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def aggregate_crop_outputs(
    logits: np.ndarray,
    method: str = DEFAULT_TEMPORAL_POOLING,
    track_positions: Sequence[int] | None = None,
    track_det_probs: Sequence[float] | None = None,
    track_sharpness: Sequence[float] | None = None,
) -> np.ndarray:
    """Aggregate per-crop logits/probabilities using the requested temporal strategy."""
    if method not in SUPPORTED_POOLING_METHODS:
        raise ValueError(f"Unsupported pooling method {method!r}; expected one of {SUPPORTED_POOLING_METHODS}")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != len(EMOTION_CLASSES):
        raise ValueError(f"Expected logits with shape (N, {len(EMOTION_CLASSES)}), got {logits.shape}.")

    all_probs = torch.softmax(torch.from_numpy(logits.astype(np.float64)), dim=-1).numpy()

    if method == "raw_all_detections_mean_prob":
        pooled = np.mean(all_probs, axis=0)
    elif method == "raw_mean_logit":
        pooled = torch.softmax(torch.from_numpy(np.mean(logits.astype(np.float64), axis=0)), dim=-1).numpy()
    else:
        indices = list(range(logits.shape[0])) if track_positions is None else list(track_positions)
        if not indices:
            raise ValueError("Selected primary face track is empty.")
        tr_logits = logits[indices]
        tr_probs = all_probs[indices]

        if method == "single_frame_middle":
            pooled = tr_probs[len(indices) // 2]
        elif method == "track_arith_mean_prob":
            pooled = np.mean(tr_probs, axis=0)
        elif method == "track_mean_logit":
            mean_logit = torch.from_numpy(np.mean(tr_logits, axis=0))
            pooled = torch.softmax(mean_logit, dim=-1).numpy()
        elif method == "track_topk3_prob":
            confs = np.max(tr_probs, axis=1)
            k = min(3, len(confs))
            top_idx = np.argsort(confs)[-k:]
            pooled = np.mean(tr_probs[top_idx], axis=0)
        elif method == "track_quality_conf_weighted":
            confs = np.max(tr_probs, axis=1)
            det_p = np.ones(len(indices), dtype=np.float64) if track_det_probs is None else np.asarray(track_det_probs, dtype=np.float64)
            sharp = np.ones(len(indices), dtype=np.float64) if track_sharpness is None else np.asarray(track_sharpness, dtype=np.float64)
            sharp_norm = sharp / np.maximum(np.max(sharp), 1.0)
            weights = (0.5 + confs) * det_p * (0.5 + 0.5 * sharp_norm)
            weights = weights / np.maximum(weights.sum(), 1e-9)
            pooled = np.sum(tr_probs * weights[:, None], axis=0)
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported pooling method: {method!r}")

    pooled = np.asarray(pooled, dtype=np.float64)
    if not np.isfinite(pooled).all():
        raise RuntimeError("Facial model produced non-finite probability values.")
    pooled = np.clip(pooled, 0.0, None)
    total = float(pooled.sum())
    if total <= 0.0:
        raise RuntimeError("Facial model produced zero probability mass.")
    return pooled / total


class ResNetFacialEmotionPredictor:
    """Verified ResNet18 + MTCNN predictor preserving canonical 7-class outputs."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_PRIMARY_CHECKPOINT,
        expected_sha256: str | None = PRIMARY_CHECKPOINT_SHA256,
        fallback_model_path: Path | str | None = DEFAULT_FALLBACK_CHECKPOINT,
        fallback_expected_sha256: str | None = FALLBACK_CHECKPOINT_SHA256,
        device: str | None = None,
        strict_fp32: bool = True,
        default_pooling: str = DEFAULT_TEMPORAL_POOLING,
        n_sample_frames: int = 8,
        calibration_path: Path | str | None = None,
        use_calibration: bool = True,
    ) -> None:
        if default_pooling not in SUPPORTED_POOLING_METHODS:
            raise ValueError(f"Unsupported default_pooling={default_pooling!r}")

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for ResNetFacialEmotionPredictor but CUDA is unavailable.")

        self.device = torch.device(requested_device)
        self.strict_fp32 = bool(strict_fp32)
        self.default_pooling = default_pooling
        self.n_sample_frames = int(n_sample_frames)
        self._lock = threading.RLock()

        primary_path = Path(model_path)
        fallback_path = Path(fallback_model_path) if fallback_model_path is not None else None

        try:
            checkpoint, verified_sha = _validate_and_load_checkpoint(primary_path, expected_sha256)
            self.path = primary_path
            self.active_role = "primary"
            self.model_identifier = str(checkpoint.get("model_identifier") or PRIMARY_MODEL_IDENTIFIER)
            self.fallback_used = False
            self.primary_error: str | None = None
        except Exception as primary_exc:
            if fallback_path is None:
                raise
            logger.warning(
                "Primary ResNet18 facial checkpoint %s failed verification (%s); attempting verified fallback %s.",
                primary_path,
                primary_exc,
                fallback_path,
            )
            checkpoint, verified_sha = _validate_and_load_checkpoint(fallback_path, fallback_expected_sha256)
            self.path = fallback_path
            self.active_role = "fallback"
            self.model_identifier = FALLBACK_MODEL_IDENTIFIER
            self.fallback_used = True
            self.primary_error = str(primary_exc)

        self.checkpoint_sha256 = verified_sha
        self.architecture = "resnet18"
        self.classes = tuple(EMOTION_CLASSES)
        self.model = build_resnet18_classifier(num_classes=len(EMOTION_CLASSES))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        resolved_cal = (
            resolve_calibration_path(self.path, explicit_path=calibration_path, use_calibration=use_calibration)
            if (not self.fallback_used or calibration_path is not None)
            else None
        )
        self.calibration = load_logit_calibration(resolved_cal, EMOTION_CLASSES) if resolved_cal is not None else None

        self.detector = MtcnnFaceDetector(device=self.device)

    def infer_crop_logits(self, faces: np.ndarray | Sequence[np.ndarray], batch_size: int = 64) -> tuple[np.ndarray, float]:
        """Run preprocessed ResNet18 forward pass on uint8 BGR face crops."""
        crops_tensor = preprocess_resnet_face_crops(faces, image_size=DEFAULT_IMAGE_SIZE, device=None)
        num_crops = crops_tensor.shape[0]
        chunks: list[np.ndarray] = []

        t0 = time.perf_counter()
        with self._lock:
            with torch.no_grad():
                with _strict_fp32_guard(self.strict_fp32, self.device):
                    for start in range(0, num_crops, batch_size):
                        batch_tensor = crops_tensor[start : start + batch_size].to(self.device, non_blocking=False)
                        logits = self.model(batch_tensor)
                        chunks.append(logits.detach().cpu().numpy())
        neural_ms = (time.perf_counter() - t0) * 1000.0
        return np.concatenate(chunks, axis=0), float(neural_ms)

    def predict_faces(
        self,
        faces: np.ndarray | Sequence[np.ndarray],
        pooling: str = "raw_all_detections_mean_prob",
    ) -> dict[str, Any]:
        """Predict canonical 7-class emotion probabilities from pre-cropped uint8 BGR faces."""
        logits, neural_ms = self.infer_crop_logits(faces)
        probabilities = aggregate_crop_outputs(
            apply_logit_calibration(logits, self.calibration),
            method=pooling,
        )
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        pred_idx = int(probabilities.argmax())

        return {
            "predicted_label": EMOTION_CLASSES[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "probabilities": {label: float(val) for label, val in zip(EMOTION_CLASSES, probabilities)},
            "model_identifier": self.model_identifier,
            "architecture": self.architecture,
            "checkpoint_sha256": self.checkpoint_sha256,
            "active_role": self.active_role,
            "preprocessing": "imagenet_rgb_96x96",
            "pooling_method": pooling,
            "calibrated": self.calibration is not None,
            "calibration_method": self.calibration.method if self.calibration is not None else None,
            "num_faces": int(logits.shape[0]),
            "prediction_entropy": entropy,
            "inference_latency_ms": float(neural_ms),
        }

    def predict_image(self, frame_bgr: np.ndarray, pooling: str = "raw_all_detections_mean_prob") -> dict[str, Any]:
        """Detect face(s) in one BGR frame using MTCNN and classify emotion."""
        t0 = time.perf_counter()
        with self._lock:
            detections, filtered_fp = self.detector.detect_frame(frame_bgr, size=DEFAULT_IMAGE_SIZE)
        if not detections:
            raise ValueError(f"No detectable faces were found in the image (filtered_fp={filtered_fp}).")

        crops = [d.crop for d in detections]
        logits, neural_ms = self.infer_crop_logits(crops)
        probabilities = aggregate_crop_outputs(
            apply_logit_calibration(logits, self.calibration),
            method=pooling,
            track_positions=list(range(len(detections))),
            track_det_probs=[d.confidence for d in detections],
            track_sharpness=[d.sharpness for d in detections],
        )
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        pred_idx = int(probabilities.argmax())

        return {
            "predicted_label": EMOTION_CLASSES[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "probabilities": {label: float(val) for label, val in zip(EMOTION_CLASSES, probabilities)},
            "model_identifier": self.model_identifier,
            "architecture": self.architecture,
            "checkpoint_sha256": self.checkpoint_sha256,
            "active_role": self.active_role,
            "preprocessing": "imagenet_rgb_96x96",
            "pooling_method": pooling,
            "calibrated": self.calibration is not None,
            "calibration_method": self.calibration.method if self.calibration is not None else None,
            "num_faces": len(detections),
            "false_positives_filtered": int(filtered_fp),
            "prediction_entropy": entropy,
            "inference_latency_ms": float((time.perf_counter() - t0) * 1000.0),
            "neural_forward_ms": float(neural_ms),
        }

    def predict_video(
        self,
        video_path: Path | str,
        n_sample_frames: int | None = None,
        pooling: str | None = None,
        max_video_seconds: float | None = None,
        use_sequential_grab: bool = True,
    ) -> dict[str, Any]:
        """Run uniform frame sampling, MTCNN detection, IoU tracking, and temporal pooling."""
        total_t0 = time.perf_counter()
        target_samples = int(self.n_sample_frames if n_sample_frames is None else n_sample_frames)
        chosen_pooling = self.default_pooling if pooling is None else pooling
        if chosen_pooling not in SUPPORTED_POOLING_METHODS:
            raise ValueError(f"Unsupported pooling method {chosen_pooling!r}")

        t_dec0 = time.perf_counter()
        decoded = sample_video_frames(
            video_path,
            n_sample_frames=target_samples,
            max_video_seconds=max_video_seconds,
            use_sequential_grab=use_sequential_grab,
        )
        decoding_ms = (time.perf_counter() - t_dec0) * 1000.0

        sampled_frames: list[tuple[int, np.ndarray]] = decoded["sampled_frames"]
        frame_w = decoded["frame_w"]
        frame_h = decoded["frame_h"]

        t_det0 = time.perf_counter()
        per_frame_dets = []
        raw_crops: list[np.ndarray] = []
        det_crop_pos: list[list[int]] = []
        frames_with_faces = 0
        multi_face_frames = 0
        false_positives_filtered = 0

        with self._lock:
            for _fi, frame_bgr in sampled_frames:
                dets, fp_count = self.detector.detect_frame(frame_bgr, size=DEFAULT_IMAGE_SIZE)
                false_positives_filtered += fp_count
                if dets:
                    frames_with_faces += 1
                    if len(dets) > 1:
                        multi_face_frames += 1
                pos_list: list[int] = []
                for det in dets:
                    pos_list.append(len(raw_crops))
                    raw_crops.append(det.crop)
                det_crop_pos.append(pos_list)
                per_frame_dets.append(dets)
        detection_ms = (time.perf_counter() - t_det0) * 1000.0

        if not raw_crops:
            raise ValueError(
                f"No detectable faces found across {decoded['frames_sampled']} sampled frames "
                f"(filtered_fp={false_positives_filtered})."
            )

        track_result = track_mtcnn_detections(per_frame_dets, frame_w=frame_w, frame_h=frame_h)
        selected_items = track_result["selected_items"]
        track_positions = [det_crop_pos[t_idx][det_idx] for t_idx, det_idx, _det in selected_items]
        track_det_probs = [det.confidence for _t_idx, _det_idx, det in selected_items]
        track_sharpness = [det.sharpness for _t_idx, _det_idx, det in selected_items]

        logits, neural_ms = self.infer_crop_logits(raw_crops)
        probabilities = aggregate_crop_outputs(
            apply_logit_calibration(logits, self.calibration),
            method=chosen_pooling,
            track_positions=track_positions,
            track_det_probs=track_det_probs,
            track_sharpness=track_sharpness,
        )
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        pred_idx = int(probabilities.argmax())
        total_ms = (time.perf_counter() - total_t0) * 1000.0

        num_sampled = int(decoded["frames_sampled"])
        coverage = float(frames_with_faces / max(1, num_sampled))
        primary_track_crops = len(track_positions)
        total_detections = len(raw_crops)

        return {
            "predicted_label": EMOTION_CLASSES[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "probabilities": {label: float(val) for label, val in zip(EMOTION_CLASSES, probabilities)},
            "model_identifier": self.model_identifier,
            "architecture": self.architecture,
            "checkpoint_sha256": self.checkpoint_sha256,
            "active_role": self.active_role,
            "preprocessing": "imagenet_rgb_96x96",
            "pooling_method": chosen_pooling,
            "calibrated": self.calibration is not None,
            "calibration_method": self.calibration.method if self.calibration is not None else None,
            "num_faces": primary_track_crops if chosen_pooling != "raw_all_detections_mean_prob" else total_detections,
            "num_sampled_frames": num_sampled,
            "face_detection_coverage": coverage,
            "prediction_entropy": entropy,
            "inference_latency_ms": float(total_ms),
            "frame_accounting": {
                "total_video_frames": decoded["total_video_frames"],
                "frames_decoded": decoded["frames_decoded"],
                "frames_sampled": num_sampled,
                "frames_with_detected_faces": frames_with_faces,
                "multi_face_frames": multi_face_frames,
                "total_face_detections": total_detections,
                "false_positives_filtered": false_positives_filtered,
                "unique_tracks": int(track_result["track_count"]),
                "primary_track_crops": primary_track_crops,
                "mean_adjacent_iou": track_result["mean_adjacent_iou"],
            },
            "timing_profile": {
                "decoding_ms": round(decoding_ms, 2),
                "detection_ms": round(detection_ms, 2),
                "neural_forward_ms": round(neural_ms, 2),
                "total_pipeline_ms": round(total_ms, 2),
            },
        }

