"""Versioned, resumable facial tracking cache generation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.detector import FaceDetection, FaceDetector
from project_config import MELD_PREPARED_ROOT

CACHE_VERSION = "phase9_5_tracking_v1"
SUPPORTED_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class TrackingConfig:
    frame_count: int = 4
    image_size: int = 96
    min_iou: float = 0.05
    max_center_distance: float = 0.35
    max_missing_frames: int = 1
    min_coverage: float = 0.75


@dataclass(frozen=True)
class TrackedFrame:
    frame_index: int
    detections: int
    selected_track_id: int | None
    bbox: tuple[int, int, int, int] | None
    detection_confidence: float | None
    tracking_confidence: float
    missing: bool


def intersection_over_union(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return float(intersection / union) if union else 0.0


def normalized_center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> float:
    height, width = frame_shape[:2]
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    distance = np.hypot((lx + lw / 2) - (rx + rw / 2), (ly + lh / 2) - (ry + rh / 2))
    return float(distance / max(1.0, np.hypot(width, height)))


def match_detection(
    previous: FaceDetection,
    detections: list[FaceDetection],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: TrackingConfig = TrackingConfig(),
) -> tuple[int, float] | None:
    candidates: list[tuple[float, int]] = []
    for index, detection in enumerate(detections):
        iou = intersection_over_union(previous.bbox, detection.bbox)
        distance = normalized_center_distance(previous.bbox, detection.bbox, frame_shape)
        if iou >= config.min_iou or distance <= config.max_center_distance:
            score = 0.6 * iou + 0.4 * max(0.0, 1.0 - distance / max(config.max_center_distance, 1e-9))
            candidates.append((score, index))
    if not candidates:
        return None
    score, index = max(candidates)
    return index, float(score)


def track_detections(
    frames: Iterable[tuple[int, np.ndarray, list[FaceDetection]]],
    config: TrackingConfig = TrackingConfig(),
) -> tuple[list[FaceDetection | None], list[TrackedFrame], dict[str, Any]]:
    selected: list[FaceDetection | None] = []
    metadata: list[TrackedFrame] = []
    previous: FaceDetection | None = None
    track_id: int | None = None
    next_track_id = 0
    missing = 0
    switches = 0
    for frame_index, frame, detections in frames:
        chosen: FaceDetection | None = None
        confidence = 0.0
        if previous is None and detections:
            chosen = max(detections, key=lambda detection: detection.bbox[2] * detection.bbox[3])
            track_id = next_track_id
            next_track_id += 1
            confidence = 1.0
        elif previous is not None:
            match = match_detection(previous, detections, frame.shape, config)
            if match is not None:
                index, confidence = match
                chosen = detections[index]
            elif detections:
                missing += 1
                if missing > config.max_missing_frames:
                    chosen = max(detections, key=lambda detection: detection.bbox[2] * detection.bbox[3])
                    track_id = next_track_id
                    next_track_id += 1
                    switches += 1
                    missing = 0
                    confidence = 1.0
        elif previous is not None:
            missing += 1
        if chosen is not None:
            previous = chosen
            missing = 0
        selected.append(chosen)
        metadata.append(
            TrackedFrame(
                frame_index=frame_index,
                detections=len(detections),
                selected_track_id=track_id,
                bbox=chosen.bbox if chosen else None,
                detection_confidence=chosen.detection_confidence if chosen else None,
                tracking_confidence=confidence,
                missing=chosen is None,
            )
        )
    observed = sum(item is not None for item in selected)
    summary = {
        "cache_version": CACHE_VERSION,
        "frame_count": len(selected),
        "observed_frames": observed,
        "missing_frames": len(selected) - observed,
        "coverage": observed / max(1, len(selected)),
        "multi_face_frames": sum(item.detections > 1 for item in metadata),
        "track_switches": switches,
        "selected_track_id": next((item.selected_track_id for item in metadata if item.selected_track_id is not None), None),
        "excluded": observed / max(1, len(selected)) < config.min_coverage,
    }
    return selected, metadata, summary


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _valid_cache(array_path: Path, metadata_path: Path, config: TrackingConfig) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        array = np.load(array_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and tuple(array.shape) == (config.frame_count, config.image_size, config.image_size, 3)
        and array.dtype == np.uint8
    )


def write_tracking_entry(
    output_dir: Path,
    sample_id: str,
    crops: list[np.ndarray],
    frame_metadata: list[TrackedFrame],
    summary: dict[str, Any],
    config: TrackingConfig = TrackingConfig(),
) -> bool:
    if summary["excluded"] or len(crops) != config.frame_count:
        return False
    array_path = output_dir / f"{sample_id}_{config.frame_count}f.npy"
    metadata_path = output_dir / f"{sample_id}_{config.frame_count}f.json"
    if _valid_cache(array_path, metadata_path, config):
        return True
    array = np.stack(crops).astype(np.uint8, copy=False)
    if array.shape != (config.frame_count, config.image_size, config.image_size, 3):
        raise ValueError(f"Unexpected tracked crop shape for {sample_id}: {array.shape}")
    _atomic_save_array(array_path, array)
    metadata = {
        "cache_version": CACHE_VERSION,
        **summary,
        "sample_id": sample_id,
        "config": asdict(config),
        "frames": [asdict(item) for item in frame_metadata],
        "crop_shape": list(array.shape),
        "detection_confidence_available": False,
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
    return True


def _load_rows(split: str, prepared_root: Path) -> pd.DataFrame:
    path = prepared_root / "reports" / split / "video_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Validated video manifest not found: {path}")
    rows = pd.read_csv(path)
    rows = rows[rows["status"].eq("ok")].copy()
    mapping = {name: index for index, name in enumerate(EMOTION_CLASSES)}
    rows["label"] = rows["emotion"].astype(str).str.lower().map(mapping)
    if rows["label"].isna().any():
        raise ValueError(f"Unknown emotion label in {split} manifest")
    return rows.reset_index(drop=True)


def generate_tracking_cache(
    prepared_root: Path = MELD_PREPARED_ROOT,
    output_root: Path = Path("models/facial_emotion_phase9_5_tracking"),
    splits: tuple[str, ...] = ("train", "dev"),
    config: TrackingConfig = TrackingConfig(),
) -> dict[str, int]:
    """Generate resumable tracking caches for explicitly requested splits."""
    detector = FaceDetector(min_face_size=24)
    counts: dict[str, int] = {}
    for split in splits:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        split_output = output_root / split
        count = 0
        for row in _load_rows(split, prepared_root).to_dict("records"):
            sample_id = str(row["sample_id"])
            array_path = split_output / f"{sample_id}_{config.frame_count}f.npy"
            metadata_path = split_output / f"{sample_id}_{config.frame_count}f.json"
            if _valid_cache(array_path, metadata_path, config):
                count += 1
                continue
            capture = cv2.VideoCapture(str(row["resolved_video_path"]))
            if not capture.isOpened():
                continue
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    continue
                indices = sorted({round(i * (frame_count - 1) / max(1, config.frame_count - 1)) for i in range(config.frame_count)})
                frames: list[tuple[int, np.ndarray, list[FaceDetection]]] = []
                for index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if ok:
                        frames.append((index, frame, detector.detect(frame, size=config.image_size)))
            finally:
                capture.release()
            selected, frame_metadata, summary = track_detections(frames, config)
            if summary["excluded"] or any(item is None for item in selected):
                continue
            if write_tracking_entry(split_output, sample_id, [item.crop for item in selected if item], frame_metadata, summary, config):
                count += 1
        counts[split] = count
    return counts


def resolve_cache_splits(dev_only: bool, requested: list[str] | None) -> tuple[str, ...]:
    if dev_only or not requested:
        return ("train", "dev")
    return tuple(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate versioned facial tracking caches.")
    parser.add_argument("--prepared-root", type=Path, default=MELD_PREPARED_ROOT)
    parser.add_argument("--output", type=Path, default=Path("models/facial_emotion_phase9_5_tracking"))
    parser.add_argument("--dev-only", action="store_true", help="Generate train and dev caches; never access test data.")
    parser.add_argument("--split", action="append", choices=SUPPORTED_SPLITS)
    args = parser.parse_args()
    splits = resolve_cache_splits(args.dev_only, args.split)
    generate_tracking_cache(args.prepared_root, args.output, splits=splits)


if __name__ == "__main__":
    main()
