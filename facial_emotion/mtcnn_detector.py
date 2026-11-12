"""Production MTCNN detection, square-unaligned cropping, and IoU face tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from facial_emotion.config import DEFAULT_IMAGE_SIZE
from facial_emotion.detector import open_video_capture

DEFAULT_MIN_FACE_SIZE = 24
DEFAULT_MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)
DEFAULT_MTCNN_FACTOR = 0.709
DEFAULT_MIN_DET_PROB = 0.90
DEFAULT_CROP_MARGIN = 0.08
DEFAULT_DOWNSCALE_WIDTH = 640
DEFAULT_MIN_ASPECT_RATIO = 0.40
DEFAULT_MAX_ASPECT_RATIO = 1.80
DEFAULT_MAX_SEQ_DECODE_FRAMES = 3600


@dataclass(frozen=True)
class MtcnnFaceDetection:
    """Single validated face detection in original video/image frame coordinates."""

    bbox: tuple[int, int, int, int]
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    crop: np.ndarray
    confidence: float
    area: float
    sharpness: float
    brightness: float


def compute_crop_quality(crop_bgr: np.ndarray) -> dict[str, float]:
    """Compute crop sharpness (Laplacian variance) and mean luminance."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return {"sharpness": sharpness, "brightness": brightness}


def crop_unaligned_square(
    frame_bgr: np.ndarray,
    box_xyxy: Sequence[float] | np.ndarray,
    size: int = DEFAULT_IMAGE_SIZE,
    margin: float = DEFAULT_CROP_MARGIN,
) -> np.ndarray | None:
    """Extract a clean square-padded BGR crop around an (x1, y1, x2, y2) detection box."""
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    nx1 = int(max(0, np.floor(cx - side / 2.0)))
    ny1 = int(max(0, np.floor(cy - side / 2.0)))
    nx2 = int(min(w_img, np.ceil(cx + side / 2.0)))
    ny2 = int(min(h_img, np.ceil(cy + side / 2.0)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None

    interp = cv2.INTER_CUBIC if crop.shape[0] < size or crop.shape[1] < size else cv2.INTER_AREA
    return cv2.resize(crop, (size, size), interpolation=interp)


def bbox_iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union of two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return float(inter / max(aw * ah + bw * bh - inter, 1e-9))


def normalized_center_distance_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> float:
    """Euclidean distance between box centers normalized by full frame diagonal."""
    acx, acy = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bcx, bcy = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    diag = float(np.hypot(max(1, frame_w), max(1, frame_h)))
    return float(np.hypot(acx - bcx, acy - bcy) / max(diag, 1.0))


class MtcnnFaceDetector:
    """MTCNN face detector matching Phase 17 Exp 4/5/7 detection and cropping settings."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
        thresholds: Sequence[float] = DEFAULT_MTCNN_THRESHOLDS,
        factor: float = DEFAULT_MTCNN_FACTOR,
        min_prob: float = DEFAULT_MIN_DET_PROB,
        downscale_width: int = DEFAULT_DOWNSCALE_WIDTH,
        crop_margin: float = DEFAULT_CROP_MARGIN,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
        self.device = resolved_device
        self.min_face_size = int(min_face_size)
        self.min_prob = float(min_prob)
        self.downscale_width = int(downscale_width)
        self.crop_margin = float(crop_margin)
        self.image_size = int(image_size)
        self._mtcnn = MTCNN(
            keep_all=True,
            min_face_size=self.min_face_size,
            thresholds=list(thresholds),
            factor=float(factor),
            device=self.device,
        )

    def _run_scale(
        self,
        frame_bgr: np.ndarray,
        target_width: int | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        h, w = frame_bgr.shape[:2]
        if target_width is not None and w > target_width:
            scale = float(target_width) / float(w)
            scaled_h = max(1, int(round(h * scale)))
            work = cv2.resize(frame_bgr, (target_width, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work = frame_bgr

        if min(work.shape[:2]) < self.min_face_size:
            return None, None, scale

        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            boxes, probs = self._mtcnn.detect(rgb, landmarks=False)
        return boxes, probs, scale

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        size: int | None = None,
        min_prob: float | None = None,
    ) -> tuple[list[MtcnnFaceDetection], int]:
        """Detect and filter faces in one BGR frame, returning (valid_detections, filtered_count)."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR color image with shape (height, width, 3).")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR image pixels.")

        target_size = int(size or self.image_size)
        prob_threshold = float(self.min_prob if min_prob is None else min_prob)
        h, w = frame_bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return [], 0

        boxes, probs, scale = self._run_scale(frame_bgr, self.downscale_width)
        if boxes is None or len(boxes) == 0:
            if w > self.downscale_width:
                boxes, probs, scale = self._run_scale(frame_bgr, None)
        if boxes is None or len(boxes) == 0:
            return [], 0

        valid: list[MtcnnFaceDetection] = []
        filtered_fp = 0
        for idx in range(len(boxes)):
            prob = None if probs is None else probs[idx]
            if prob is None or not np.isfinite(float(prob)) or float(prob) < prob_threshold:
                filtered_fp += 1
                continue

            bx1, by1, bx2, by2 = (float(v) / scale for v in boxes[idx])
            bx1 = max(0.0, min(float(w - 1), bx1))
            by1 = max(0.0, min(float(h - 1), by1))
            bx2 = max(bx1 + 1.0, min(float(w), bx2))
            by2 = max(by1 + 1.0, min(float(h), by2))
            bw, bh = bx2 - bx1, by2 - by1
            aspect = bw / max(1e-3, bh)

            if (
                min(bw, bh) < float(self.min_face_size)
                or aspect < DEFAULT_MIN_ASPECT_RATIO
                or aspect > DEFAULT_MAX_ASPECT_RATIO
            ):
                filtered_fp += 1
                continue

            crop = crop_unaligned_square(
                frame_bgr,
                (bx1, by1, bx2, by2),
                size=target_size,
                margin=self.crop_margin,
            )
            if crop is None:
                filtered_fp += 1
                continue

            quality = compute_crop_quality(crop)
            int_bbox = (
                int(round(bx1)),
                int(round(by1)),
                max(1, int(round(bw))),
                max(1, int(round(bh))),
            )
            valid.append(
                MtcnnFaceDetection(
                    bbox=int_bbox,
                    bbox_xywh=(bx1, by1, bw, bh),
                    bbox_xyxy=(bx1, by1, bx2, by2),
                    crop=crop,
                    confidence=float(prob),
                    area=float(bw * bh),
                    sharpness=float(quality["sharpness"]),
                    brightness=float(quality["brightness"]),
                )
            )

        return valid, filtered_fp


def track_mtcnn_detections(
    frame_detections: list[list[MtcnnFaceDetection]],
    frame_w: int,
    frame_h: int,
    min_iou: float = 0.08,
    max_center_dist: float = 0.28,
    max_frame_gap: int = 3,
) -> dict[str, Any]:
    """Associate detections across sampled frames in original frame coordinates."""
    tracks: list[dict[str, Any]] = []
    next_tid = 0

    for t_idx, dets in enumerate(frame_detections):
        if not dets:
            continue

        candidates: list[tuple[float, int, int]] = []
        for tr_idx, tr in enumerate(tracks):
            if (t_idx - tr["last_t"]) > max_frame_gap:
                continue
            for det_idx, det in enumerate(dets):
                iou = bbox_iou_xywh(tr["last_bbox"], det.bbox_xywh)
                cdist = normalized_center_distance_xywh(tr["last_bbox"], det.bbox_xywh, frame_w, frame_h)
                if iou >= min_iou or cdist <= max_center_dist:
                    score = iou + max(0.0, 1.0 - cdist / max(max_center_dist, 1e-6)) * 0.5
                    candidates.append((score, tr_idx, det_idx))

        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _score, tr_idx, det_idx in candidates:
            if tr_idx in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(tr_idx)
            used_dets.add(det_idx)
            det = dets[det_idx]
            tracks[tr_idx]["items"].append((t_idx, det_idx, det))
            tracks[tr_idx]["last_t"] = t_idx
            tracks[tr_idx]["last_bbox"] = det.bbox_xywh

        for det_idx, det in enumerate(dets):
            if det_idx not in used_dets:
                tracks.append(
                    {
                        "track_id": next_tid,
                        "items": [(t_idx, det_idx, det)],
                        "last_t": t_idx,
                        "last_bbox": det.bbox_xywh,
                    }
                )
                next_tid += 1

    if not tracks:
        return {
            "selected_track_id": None,
            "track_count": 0,
            "selected_items": [],
            "ious": [],
            "mean_adjacent_iou": None,
        }

    def _track_score(tr: dict[str, Any]) -> float:
        items = tr["items"]
        mean_area = float(np.mean([it[2].area for it in items]))
        mean_prob = float(np.mean([it[2].confidence for it in items]))
        return float(len(items)) * float(np.sqrt(max(1.0, mean_area))) * mean_prob

    best_tr = max(tracks, key=_track_score)
    ious: list[float] = []
    for k in range(1, len(best_tr["items"])):
        prev_b = best_tr["items"][k - 1][2].bbox_xywh
        curr_b = best_tr["items"][k][2].bbox_xywh
        ious.append(bbox_iou_xywh(prev_b, curr_b))

    return {
        "selected_track_id": int(best_tr["track_id"]),
        "track_count": len(tracks),
        "selected_items": best_tr["items"],
        "ious": ious,
        "mean_adjacent_iou": float(np.mean(ious)) if ious else None,
    }


def _sequential_subsample_capture(
    video_path: Path,
    n_sample_frames: int,
    max_decode_frames: int = DEFAULT_MAX_SEQ_DECODE_FRAMES,
) -> tuple[list[tuple[int, np.ndarray]], int, int, int, int, float]:
    """Decode frames sequentially and uniformly pick up to n_sample_frames."""
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        raw_frames: list[tuple[int, np.ndarray]] = []
        decoded_count = 0
        while decoded_count < max_decode_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_w <= 0 or frame_h <= 0:
                frame_h, frame_w = frame.shape[:2]
            raw_frames.append((decoded_count, frame))
            decoded_count += 1
    finally:
        cap.release()

    if not raw_frames:
        return [], decoded_count, total_vf, frame_w, frame_h, fps

    pick = sorted(
        set(np.linspace(0, len(raw_frames) - 1, min(n_sample_frames, len(raw_frames)), dtype=int).tolist())
    )
    sampled = [raw_frames[idx] for idx in pick]
    return sampled, decoded_count, max(total_vf, decoded_count), frame_w, frame_h, fps


class VideoDurationExceededError(ValueError):
    """Raised when a video clip exceeds the configured duration limit."""


DEFAULT_SEQUENTIAL_GRAB_MAX_FRAMES = 240


def sample_video_frames(
    video_path: Path | str,
    n_sample_frames: int = 8,
    max_seq_decode_frames: int = DEFAULT_MAX_SEQ_DECODE_FRAMES,
    max_video_seconds: float | None = None,
    use_sequential_grab: bool = True,
    sequential_grab_max_frames: int = DEFAULT_SEQUENTIAL_GRAB_MAX_FRAMES,
) -> dict[str, Any]:
    """Read uniformly sampled frames from an MP4/WebM clip with sequential fallback."""
    if n_sample_frames < 1:
        raise ValueError("n_sample_frames must be >= 1.")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = open_video_capture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    sampled: list[tuple[int, np.ndarray]] = []
    decoded_count = 0
    try:
        total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if max_video_seconds is not None and fps > 0 and total_vf > 0 and (total_vf / fps) > max_video_seconds:
            raise VideoDurationExceededError("Video exceeds the configured duration limit.")

        if 0 < total_vf <= max_seq_decode_frames:
            idxs = sorted(
                set(np.linspace(0, max(total_vf - 1, 0), min(n_sample_frames, total_vf), dtype=int).tolist())
            )
            if use_sequential_grab and total_vf <= sequential_grab_max_frames and idxs:
                # Single forward H.264/MP4 pass via grab()+retrieve() skips repeated IDR keyframe seeks.
                target_set = set(idxs)
                last_idx = idxs[-1]
                fi = 0
                while fi <= last_idx:
                    if not cap.grab():
                        break
                    if fi in target_set:
                        ok, frame = cap.retrieve()
                        if ok and frame is not None:
                            decoded_count += 1
                            if frame_w <= 0 or frame_h <= 0:
                                frame_h, frame_w = frame.shape[:2]
                            sampled.append((int(fi), frame))
                    fi += 1
            else:
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        decoded_count += 1
                        if frame_w <= 0 or frame_h <= 0:
                            frame_h, frame_w = frame.shape[:2]
                        sampled.append((int(fi), frame))
    finally:
        cap.release()

    if not sampled:
        # WebM streams (e.g. MediaRecorder output) often have unknown/zero CAP_PROP_FRAME_COUNT or fail seeks
        sampled, decoded_count, total_vf, frame_w, frame_h, fps = _sequential_subsample_capture(
            path,
            n_sample_frames=n_sample_frames,
            max_decode_frames=max_seq_decode_frames,
        )

    if not sampled:
        raise ValueError(f"Video contains no decodable frames: {path}")

    return {
        "sampled_frames": sampled,
        "total_video_frames": int(total_vf),
        "frames_decoded": int(decoded_count),
        "frames_sampled": int(len(sampled)),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "fps": float(fps),
    }

