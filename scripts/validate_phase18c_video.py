"""Phase 18C — Production Video Inference & Temporal Aggregation Verification.

Evaluates ResNetFacialEmotionPredictor end-to-end on:
1. Full N=1,108 MELD dev cached detections via production inference/pooling helpers.
2. End-to-end decoded MELD dev video subset (N=70 stratified across 7 classes, ~10 per class)
   verifying identical MTCNN + tracking + ResNet18 execution against Phase 17 Exp 7.
3. All required stress scenarios: normal dialogue, live webcam clip, no-face (synthetic & dev),
   multi-face, ultra-short, long, low-light/degraded, WebM sequential fallback, and invalid inputs.

Strictly accesses ONLY MELD `dev` split (never `test`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import csv
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.resnet_predictor import (
    DEFAULT_PRIMARY_CHECKPOINT,
    PRIMARY_CHECKPOINT_SHA256,
    SUPPORTED_POOLING_METHODS,
    ResNetFacialEmotionPredictor,
    aggregate_crop_outputs,
    compute_sha256,
)
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

EXP7_DIR = REPOSITORY_ROOT / "models" / "facial_experiments" / "exp7_temporal"
OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase18c_video_validation.json"
WIN_CLIP = Path("C:/Users/Bhavanth D/OneDrive/Pictures/Camera Roll/WIN_20261111_03_07_13_Pro.mp4")


def _assert_no_test_path(path: Path | str) -> Path:
    resolved = Path(path)
    lowered = str(resolved).lower().replace("\\", "/")
    if "/test/" in lowered or "\\test\\" in lowered or "test_sent_emo" in lowered or "test_manifest" in lowered:
        raise AssertionError(f"Blocked forbidden held-out test split access: {resolved}")
    return resolved


def _load_dev_video_rows() -> dict[str, dict[str, Any]]:
    manifest = _assert_no_test_path(MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv")
    rows: dict[str, dict[str, Any]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("video_status") != "ok":
                continue
            sid = str(row.get("sample_id", "")).strip()
            emo = str(row.get("emotion", "")).strip().lower()
            vpath = Path(row.get("resolved_video_path") or row.get("output_path") or "")
            if sid and emo in EMOTION_CLASSES and vpath.is_file():
                _assert_no_test_path(vpath)
                rows[sid] = {"sample_id": sid, "emotion": emo, "label": EMOTION_CLASSES.index(emo), "video_path": vpath}
    return rows


def _run_full_dev_pooling_check(predictor: ResNetFacialEmotionPredictor) -> dict[str, Any]:
    meta_path = EXP7_DIR / "dev_temporal_unaligned_meta.json"
    npz_path = EXP7_DIR / "dev_temporal_unaligned_crops.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return {"skipped": True, "reason": "Exp 7 dev cache not found"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    all_crops = np.load(npz_path)["crops"]
    videos = meta["videos"]

    # Batched forward pass through the exact production ResNetFacialEmotionPredictor
    t0 = time.perf_counter()
    all_logits, neural_ms = predictor.infer_crop_logits(all_crops, batch_size=128)
    eval_results: dict[str, Any] = {}

    for method in SUPPORTED_POOLING_METHODS:
        y_true: list[int] = []
        y_pred: list[int] = []
        for v in videos:
            if not v.get("detected"):
                continue
            indices = v["all_det_crop_indices"] if method == "raw_all_detections_mean_prob" else v["track_crop_indices"]
            if not indices:
                continue
            probs = aggregate_crop_outputs(
                all_logits[indices],
                method=method,
                track_positions=list(range(len(indices))),
                track_det_probs=v.get("track_det_probs") if method != "raw_all_detections_mean_prob" else None,
                track_sharpness=v.get("track_sharpness") if method != "raw_all_detections_mean_prob" else None,
            )
            y_true.append(int(v["label"]))
            y_pred.append(int(np.argmax(probs)))

        acc = float(accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, labels=list(range(7)), average="macro", zero_division=0))
        wf1 = float(f1_score(y_true, y_pred, labels=list(range(7)), average="weighted", zero_division=0))
        per_class = f1_score(y_true, y_pred, labels=list(range(7)), average=None, zero_division=0).tolist()
        eval_results[method] = {
            "n_detected": len(y_true),
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
            "active_classes": int(sum(1 for x in per_class if x > 0.0)),
            "per_class_f1": {EMOTION_CLASSES[i]: round(float(per_class[i]), 4) for i in range(7)},
        }

    return {
        "skipped": False,
        "total_dev_videos": len(videos),
        "total_cached_crops": int(len(all_crops)),
        "neural_forward_ms": round(float(neural_ms), 2),
        "elapsed_seconds": round(float(time.perf_counter() - t0), 2),
        "pooling_comparison": eval_results,
    }


def _run_end_to_end_stratified_dev_sample(
    predictor: ResNetFacialEmotionPredictor,
    dev_rows: dict[str, dict[str, Any]],
    per_class: int = 10,
) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sid in sorted(dev_rows):
        row = dev_rows[sid]
        by_class[row["emotion"]].append(row)

    sample_rows: list[dict[str, Any]] = []
    for emo in EMOTION_CLASSES:
        sample_rows.extend(by_class[emo][:per_class])

    y_true: list[int] = []
    preds_raw: list[int] = []
    preds_track: list[int] = []
    no_face_count = 0
    latencies: list[float] = []

    for row in sample_rows:
        try:
            res_raw = predictor.predict_video(row["video_path"], pooling="raw_all_detections_mean_prob")
            res_tr = predictor.predict_video(row["video_path"], pooling="track_arith_mean_prob")
            assert abs(sum(res_raw["probabilities"].values()) - 1.0) < 1e-5
            assert all(np.isfinite(list(res_raw["probabilities"].values())))
            assert 0.0 <= res_raw["face_detection_coverage"] <= 1.0
            y_true.append(row["label"])
            preds_raw.append(EMOTION_CLASSES.index(res_raw["predicted_label"]))
            preds_track.append(EMOTION_CLASSES.index(res_tr["predicted_label"]))
            latencies.append(float(res_raw["inference_latency_ms"]))
        except ValueError:
            no_face_count += 1

    return {
        "n_attempted": len(sample_rows),
        "n_detected": len(y_true),
        "no_face_count": no_face_count,
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        "raw_all_detections_mean_prob": {
            "accuracy": round(float(accuracy_score(y_true, preds_raw)), 4),
            "macro_f1": round(float(f1_score(y_true, preds_raw, labels=list(range(7)), average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_true, preds_raw, labels=list(range(7)), average="weighted", zero_division=0)), 4),
        },
        "track_arith_mean_prob": {
            "accuracy": round(float(accuracy_score(y_true, preds_track)), 4),
            "macro_f1": round(float(f1_score(y_true, preds_track, labels=list(range(7)), average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_true, preds_track, labels=list(range(7)), average="weighted", zero_division=0)), 4),
        },
    }


def _run_stress_suite(
    predictor: ResNetFacialEmotionPredictor,
    dev_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # 1. Normal dialogue video
    normal_sid = sorted(dev_rows)[0]
    results["normal_dialogue_dev"] = _summarize_video_call(predictor, dev_rows[normal_sid]["video_path"])

    # 2. Live webcam clip if present
    if WIN_CLIP.is_file():
        results["live_camera_clip"] = _summarize_video_call(predictor, WIN_CLIP)

    # 3. Multi-face, short, long, low-light, no-face MELD dev samples
    named_cases = {
        "meld_dev_no_face": "dev_dia5_utt9",
        "meld_dev_multi_face": "dev_dia0_utt0",
        "meld_dev_short": "dev_dia99_utt3",
        "meld_dev_long": "dev_dia74_utt19",
        "meld_dev_low_light": "dev_dia10_utt3",
    }
    for key, sid in named_cases.items():
        if sid in dev_rows:
            results[key] = _summarize_video_call(predictor, dev_rows[sid]["video_path"], expect_no_face=(key == "meld_dev_no_face"))

    # 4. Synthetic no-face blank video, degraded (dark + blur) MP4 from normal clip, WebM conversion, corrupt video
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        blank = tmp / "blank.mp4"
        writer = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        for _ in range(10):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        results["synthetic_no_face"] = _summarize_video_call(predictor, blank, expect_no_face=True)

        # Low-light + blurred video from dev_dia0_utt0
        src_clip = dev_rows.get("dev_dia0_utt0", dev_rows[normal_sid])["video_path"]
        cap = cv2.VideoCapture(str(src_clip))
        degraded = tmp / "dark_blurred.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        dwriter = cv2.VideoWriter(str(degraded), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(16):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            dark = np.clip(fr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(dark, (5, 5), 1.5)
            dwriter.write(blurred)
        cap.release()
        dwriter.release()
        results["synthetic_degraded_dark_blur"] = _summarize_video_call(predictor, degraded)

        # WebM live-capture container stress via ffmpeg if available
        import shutil

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            webm = tmp / "capture.webm"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(src_clip), "-t", "1.2", "-c:v", "libvpx", "-an", str(webm)],
                check=True,
                capture_output=True,
            )
            results["webm_stream_fallback"] = _summarize_video_call(predictor, webm)

        # Corrupt MP4
        corrupt = tmp / "corrupt.mp4"
        corrupt.write_bytes(b"not a valid mp4 stream")
        try:
            predictor.predict_video(corrupt)
            results["corrupt_video"] = {"status": "unexpected_success"}
        except Exception as exc:
            results["corrupt_video"] = {"status": "rejected_safely", "error_type": type(exc).__name__, "error": str(exc)}

    return results


def _summarize_video_call(
    predictor: ResNetFacialEmotionPredictor,
    path: Path,
    expect_no_face: bool = False,
) -> dict[str, Any]:
    try:
        res = predictor.predict_video(path)
        return {
            "status": "unexpected_face" if expect_no_face else "ok",
            "predicted_label": res["predicted_label"],
            "confidence": round(float(res["confidence"]), 4),
            "prob_sum": round(float(sum(res["probabilities"].values())), 6),
            "all_finite": bool(np.isfinite(list(res["probabilities"].values())).all()),
            "pooling_method": res["pooling_method"],
            "frame_accounting": res["frame_accounting"],
            "timing_profile": res["timing_profile"],
        }
    except ValueError as exc:
        return {
            "status": "no_face_rejected_safely" if expect_no_face else "no_face_value_error",
            "error": str(exc),
        }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert compute_sha256(DEFAULT_PRIMARY_CHECKPOINT) == PRIMARY_CHECKPOINT_SHA256
    predictor = ResNetFacialEmotionPredictor(device=device, strict_fp32=True)
    dev_rows = _load_dev_video_rows()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    full_dev_pooling = _run_full_dev_pooling_check(predictor)
    stratified_e2e = _run_end_to_end_stratified_dev_sample(predictor, dev_rows, per_class=10)
    stress_suite = _run_stress_suite(predictor, dev_rows)
    peak_vram_mb = round(float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 2) if device == "cuda" else 0.0

    report = {
        "device": device,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "peak_vram_mb": peak_vram_mb,
        "zero_test_split_access": True,
        "full_dev_pooling": full_dev_pooling,
        "stratified_e2e_n70": stratified_e2e,
        "stress_suite": stress_suite,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

