"""Focused Phase 18C unit tests for MTCNN tracking, WebM/uniform sampling, and pooling."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.mtcnn_detector import (
    MtcnnFaceDetection,
    bbox_iou_xywh,
    normalized_center_distance_xywh,
    sample_video_frames,
    track_mtcnn_detections,
)
from facial_emotion.resnet_predictor import (
    SUPPORTED_POOLING_METHODS,
    ResNetFacialEmotionPredictor,
    aggregate_crop_outputs,
)


def _det(x: float, y: float, w: float, h: float, conf: float = 0.99, sharp: float = 100.0) -> MtcnnFaceDetection:
    return MtcnnFaceDetection(
        bbox=(int(x), int(y), int(w), int(h)),
        bbox_xywh=(x, y, w, h),
        bbox_xyxy=(x, y, x + w, y + h),
        crop=np.full((96, 96, 3), 128, dtype=np.uint8),
        confidence=conf,
        area=w * h,
        sharpness=sharp,
        brightness=128.0,
    )


def test_tracking_preserves_original_frame_separation_and_picks_primary_track():
    # Two separated faces across 4 frames: left prominent speaker (larger, 4 frames) vs right background (smaller, 2 frames)
    frames = [
        [_det(120, 100, 140, 160, 0.99), _det(900, 120, 70, 80, 0.95)],
        [_det(126, 102, 140, 160, 0.98), _det(904, 121, 70, 80, 0.94)],
        [_det(130, 104, 140, 160, 0.99)],
        [_det(134, 105, 140, 160, 0.97)],
    ]
    assert bbox_iou_xywh(frames[0][0].bbox_xywh, frames[0][1].bbox_xywh) == 0.0
    assert normalized_center_distance_xywh(frames[0][0].bbox_xywh, frames[0][1].bbox_xywh, 1280, 720) > 0.45

    tr = track_mtcnn_detections(frames, frame_w=1280, frame_h=720)
    assert tr["track_count"] == 2
    assert len(tr["selected_items"]) == 4
    assert tr["mean_adjacent_iou"] is not None and tr["mean_adjacent_iou"] > 0.85


def test_all_supported_pooling_methods_return_finite_normalized_seven_class_probs():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(6, len(EMOTION_CLASSES))).astype(np.float32)
    for method in SUPPORTED_POOLING_METHODS:
        probs = aggregate_crop_outputs(
            logits,
            method=method,
            track_positions=[0, 1, 3, 4],
            track_det_probs=[0.99, 0.95, 0.98, 0.92],
            track_sharpness=[120.0, 90.0, 150.0, 60.0],
        )
        assert probs.shape == (7,)
        assert np.isfinite(probs).all()
        assert np.all(probs >= 0.0)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_corrupt_and_blank_video_handling(tmp_path):
    predictor = ResNetFacialEmotionPredictor(device="cpu")

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"corrupt header")
    with pytest.raises(ValueError):
        predictor.predict_video(corrupt)

    blank = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
    for _ in range(12):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    with pytest.raises(ValueError, match="No detectable faces"):
        predictor.predict_video(blank)


def test_webm_sequential_sampling_works_without_seek(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    mp4 = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15, (160, 120))
    for i in range(15):
        frame = np.full((120, 160, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    webm = tmp_path / "sample.webm"
    subprocess.run([ffmpeg, "-y", "-i", str(mp4), "-c:v", "libvpx", "-an", str(webm)], check=True, capture_output=True)
    decoded = sample_video_frames(webm, n_sample_frames=6)
    assert decoded["frames_sampled"] == 6
    assert decoded["frames_decoded"] >= 6
    assert decoded["frame_w"] == 160
    assert decoded["frame_h"] == 120

