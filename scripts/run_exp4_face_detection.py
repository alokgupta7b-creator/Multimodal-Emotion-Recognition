"""Phase 17B Experiment 4 — Improved Face Detection and Landmark Alignment.

Evaluates:
1. Phase 17A Exp 3 ResNet18 baseline parity on original Haar dev crops (N=1,080).
2. Controlled detector benchmark on a 140-video stratified MELD dev cohort + WIN_20261111_03_07_13_Pro.mp4:
   - Haar Native (1280x720) vs Haar Downscaled (640x360) vs MTCNN Native vs MTCNN Downscaled
   - False positives, missed detections, size/lighting/pose stratification, latency, memory, crop quality.
3. Extraction of isolated train & dev MTCNN caches (both square-unaligned and 5-pt landmark-aligned)
   with strict anti-test-leakage guards (zero access to test split).
4. Zero-shot Exp 3 ResNet18 evaluation + controlled ResNet18 training on MTCNN-Unaligned (Exp 4A) and
   MTCNN-Aligned (Exp 4B) vs Exp 3 Haar baseline.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from facenet_pytorch import MTCNN  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from facial_emotion.config import EMOTION_CLASSES  # noqa: E402
from facial_emotion.detector import FaceDetector, open_video_capture  # noqa: E402
from facial_emotion.remediation import OptimizedFaceDetector  # noqa: E402
from project_config import REPOSITORY_ROOT  # noqa: E402
from scripts.phase17b_common import (  # noqa: E402
    EXP3_BASELINE_CKPT,
    EXP_ROOT,
    IMAGE_SIZE,
    SEED,
    ArrayFaceDataset,
    align_face_robust,
    assert_no_test_path,
    assert_valid_split,
    build_backbone,
    collect_environment_info,
    compute_crop_quality,
    compute_smoothed_class_weights,
    crop_unaligned_square,
    evaluate_model,
    get_preprocessing_config,
    load_haar_cached_split,
    load_video_manifest,
    measure_inference_latency_and_memory,
    seed_everything,
    sha256_file,
    train_backbone_experiment,
    verify_protected_assets,
)

EXP4_DIR = EXP_ROOT / "exp4_face_detection"
CACHE_DIR = EXP4_DIR / "cache"
WIN_DIAGNOSTIC_VIDEO = REPOSITORY_ROOT / "WIN_20261111_03_07_13_Pro.mp4"


def _box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-9))


def _detect_mtcnn_scaled(
    mtcnn: MTCNN,
    frame_bgr: np.ndarray,
    target_width: int | None = 640,
    min_prob: float = 0.90,
) -> list[dict[str, Any]]:
    h, w = frame_bgr.shape[:2]
    if target_width is not None and w > target_width:
        scale = float(target_width) / float(w)
        work = cv2.resize(frame_bgr, (target_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        work = frame_bgr

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    boxes, probs, points = mtcnn.detect(rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return []

    out: list[dict[str, Any]] = []
    for idx in range(len(boxes)):
        prob = None if probs is None else probs[idx]
        if prob is None or float(prob) < min_prob:
            continue
        bx1, by1, bx2, by2 = [float(v) / scale for v in boxes[idx]]
        bx1 = max(0.0, min(float(w - 1), bx1))
        by1 = max(0.0, min(float(h - 1), by1))
        bx2 = max(bx1 + 1.0, min(float(w), bx2))
        by2 = max(by1 + 1.0, min(float(h), by2))
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 16.0 or bh < 16.0:
            continue
        lm = None if points is None else (np.asarray(points[idx], dtype=np.float32) / scale)
        yaw_asym = 0.0
        roll_deg = 0.0
        if lm is not None and lm.shape == (5, 2):
            le, re, nose, _lm, _rm = lm
            eye_mid_x = 0.5 * (le[0] + re[0])
            eye_dist = max(1e-3, float(np.hypot(re[0] - le[0], re[1] - le[1])))
            yaw_asym = float(abs(nose[0] - eye_mid_x) / eye_dist)
            roll_deg = float(np.degrees(np.arctan2(re[1] - le[1], max(1e-3, abs(re[0] - le[0])))))
        out.append(
            {
                "bbox_xyxy": (bx1, by1, bx2, by2),
                "bbox_xywh": (int(round(bx1)), int(round(by1)), int(round(bw)), int(round(bh))),
                "prob": float(prob),
                "landmarks": lm,
                "min_side": float(min(bw, bh)),
                "area": float(bw * bh),
                "yaw_asym": yaw_asym,
                "roll_deg": roll_deg,
            }
        )
    return out


def _select_salient_face(detections: list[dict[str, Any]], frame_shape: tuple[int, int, ...]) -> dict[str, Any]:
    h, w = frame_shape[:2]
    cx_img, cy_img = 0.5 * w, 0.5 * h
    diag = float(np.hypot(w, h))

    def _score(d: dict[str, Any]) -> float:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        fcx, fcy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        center_dist = float(np.hypot(fcx - cx_img, fcy - cy_img)) / max(diag, 1.0)
        return float(d["prob"]) * float(np.sqrt(max(1.0, d["area"]))) * (1.0 - 0.15 * center_dist)

    return max(detections, key=_score)


def _sample_video_frames_for_diagnostic(video_path: Path, n_frames: int = 6) -> list[tuple[int, np.ndarray]]:
    assert_no_test_path(video_path)
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames: list[tuple[int, np.ndarray]] = []
    try:
        if total > 0:
            idxs = sorted(set(np.linspace(0, max(total - 1, 0), min(n_frames, total), dtype=int).tolist()))
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append((int(idx), fr))
        else:
            idx = 0
            while len(frames) < n_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append((idx, fr))
                idx += 5
    finally:
        cap.release()
    return frames


def run_detector_benchmark(dev_rows: list[dict[str, Any]], mtcnn: MTCNN) -> dict[str, Any]:
    """Compare Haar vs MTCNN across stratified MELD dev videos + WIN_20261111_03_07_13_Pro.mp4."""
    print("\n--- Part 2: Detector & Landmark Diagnostic Benchmark (140 stratified dev videos) ---")
    rng = np.random.default_rng(SEED)
    sampled_rows: list[dict[str, Any]] = []
    for cls_name in EMOTION_CLASSES:
        cls_rows = [r for r in dev_rows if r["emotion"] == cls_name]
        take = min(20, len(cls_rows))
        chosen_idx = sorted(rng.choice(len(cls_rows), size=take, replace=False).tolist())
        sampled_rows.extend([cls_rows[i] for i in chosen_idx])

    frames_data: list[dict[str, Any]] = []
    for row in sampled_rows:
        for fidx, fr in _sample_video_frames_for_diagnostic(Path(row["video_path"]), n_frames=6):
            luma = float(np.mean(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
            frames_data.append(
                {
                    "sample_id": row["sample_id"],
                    "emotion": row["emotion"],
                    "frame_idx": fidx,
                    "frame": fr,
                    "luma": luma,
                }
            )

    print(f"  Loaded {len(frames_data)} frames across {len(sampled_rows)} stratified dev videos.")

    haar_native = FaceDetector(min_face_size=24)
    haar_down = OptimizedFaceDetector(min_face_size=24, downscale_factor=2.0)

    # Warmup MTCNN
    if frames_data:
        _ = _detect_mtcnn_scaled(mtcnn, frames_data[0]["frame"], target_width=640)

    detectors_meta: dict[str, dict[str, Any]] = {}
    per_frame_outputs: dict[str, list[list[dict[str, Any]]]] = {}

    def _run_detector(name: str, fn) -> None:
        tracemalloc.start()
        if torch.cuda.is_available() and "mtcnn" in name:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        all_dets: list[list[dict[str, Any]]] = []
        for item in frames_data:
            all_dets.append(fn(item["frame"]))
        if torch.cuda.is_available() and "mtcnn" in name:
            torch.cuda.synchronize()
            vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        else:
            vram_mb = 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _cur, peak_cpu_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        frames_with_face = sum(1 for d in all_dets if len(d) > 0)
        total_boxes = sum(len(d) for d in all_dets)
        vids_with_face = len({frames_data[i]["sample_id"] for i, d in enumerate(all_dets) if len(d) > 0})
        per_frame_outputs[name] = all_dets
        detectors_meta[name] = {
            "frames_evaluated": len(frames_data),
            "videos_evaluated": len(sampled_rows),
            "videos_with_face": vids_with_face,
            "frames_with_face": frames_with_face,
            "frame_detection_rate": float(frames_with_face / max(1, len(frames_data))),
            "total_detections": total_boxes,
            "latency_ms_per_frame": float(elapsed_ms / max(1, len(frames_data))),
            "peak_gpu_memory_mb": vram_mb,
            "peak_cpu_traced_mb": float(peak_cpu_bytes / (1024 ** 2)),
        }
        print(
            f"  [{name}] frames={frames_with_face}/{len(frames_data)} ({frames_with_face/len(frames_data):.1%}), "
            f"videos={vids_with_face}/{len(sampled_rows)}, boxes={total_boxes}, "
            f"latency={elapsed_ms/len(frames_data):.2f} ms/frame, gpu_vram={vram_mb:.1f} MB"
        )

    def _haar_wrap(det, fr: np.ndarray) -> list[dict[str, Any]]:
        raw = det.detect(fr, size=IMAGE_SIZE)
        out = []
        for r in raw:
            x, y, w, h = r.bbox
            out.append(
                {
                    "bbox_xyxy": (float(x), float(y), float(x + w), float(y + h)),
                    "bbox_xywh": (int(x), int(y), int(w), int(h)),
                    "prob": None,
                    "landmarks": None,
                    "min_side": float(min(w, h)),
                }
            )
        return out

    _run_detector("haar_native_720p", lambda fr: _haar_wrap(haar_native, fr))
    _run_detector("haar_downscaled_360p", lambda fr: _haar_wrap(haar_down, fr))
    _run_detector("mtcnn_native_720p", lambda fr: _detect_mtcnn_scaled(mtcnn, fr, target_width=None, min_prob=0.90))
    _run_detector("mtcnn_downscaled_360p", lambda fr: _detect_mtcnn_scaled(mtcnn, fr, target_width=640, min_prob=0.90))

    # False-positive & missed-face cross-comparison + size/lighting/pose stratification
    hn_list = per_frame_outputs["haar_native_720p"]
    hd_list = per_frame_outputs["haar_downscaled_360p"]
    mn_list = per_frame_outputs["mtcnn_native_720p"]
    md_list = per_frame_outputs["mtcnn_downscaled_360p"]

    haar_missed_where_mtcnn_confident = 0
    mtcnn_missed_where_haar_detected = 0
    haar_unmatched_boxes = 0
    haar_total_boxes = 0
    mtcnn_highconf_total = 0

    strat = {
        "face_size": {"small_lt64": [0, 0, 0], "medium_64_128": [0, 0, 0], "large_ge128": [0, 0, 0]},  # [n, haar_hit, mtcnn_down_hit]
        "lighting": {"dark_lt75": [0, 0, 0], "normal_75_155": [0, 0, 0], "bright_ge155": [0, 0, 0]},
        "pose_from_landmarks": {"frontal": [0, 0, 0], "angled_or_tilted": [0, 0, 0]},
    }

    qual_haar: list[dict[str, float]] = []
    qual_mtcnn_u: list[dict[str, float]] = []
    qual_mtcnn_a: list[dict[str, float]] = []
    eyes_y_unaligned: list[float] = []
    eyes_y_aligned: list[float] = []
    rolls_before: list[float] = []
    visual_examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for i, item in enumerate(frames_data):
        fr = item["frame"]
        luma = item["luma"]
        hn, mn, md = hn_list[i], mn_list[i], md_list[i]

        lbin = "dark_lt75" if luma < 75 else ("normal_75_155" if luma < 155 else "bright_ge155")
        strat["lighting"][lbin][0] += 1
        if hn:
            strat["lighting"][lbin][1] += 1
        if md:
            strat["lighting"][lbin][2] += 1

        mn_hi = [d for d in mn if (d["prob"] or 0.0) >= 0.95]
        mtcnn_highconf_total += len(mn_hi)
        if not hn and any((d["prob"] or 0.0) >= 0.95 for d in mn):
            haar_missed_where_mtcnn_confident += 1
        if hn and not mn:
            mtcnn_missed_where_haar_detected += 1

        for hb in hn:
            haar_total_boxes += 1
            best_iou = max((_box_iou_xyxy(hb["bbox_xyxy"], mb["bbox_xyxy"]) for mb in mn), default=0.0)
            if best_iou < 0.15:
                haar_unmatched_boxes += 1

        if mn:
            primary_ref = max(mn, key=lambda d: d["area"])
            ms = primary_ref["min_side"]
            sbin = "small_lt64" if ms < 64 else ("medium_64_128" if ms < 128 else "large_ge128")
            strat["face_size"][sbin][0] += 1
            if any(_box_iou_xyxy(hb["bbox_xyxy"], primary_ref["bbox_xyxy"]) >= 0.25 for hb in hn):
                strat["face_size"][sbin][1] += 1
            if any(_box_iou_xyxy(mb["bbox_xyxy"], primary_ref["bbox_xyxy"]) >= 0.25 for mb in md):
                strat["face_size"][sbin][2] += 1

            is_angled = abs(primary_ref["roll_deg"]) > 10.0 or primary_ref["yaw_asym"] > 0.28
            pbin = "angled_or_tilted" if is_angled else "frontal"
            strat["pose_from_landmarks"][pbin][0] += 1
            if any(_box_iou_xyxy(hb["bbox_xyxy"], primary_ref["bbox_xyxy"]) >= 0.25 for hb in hn):
                strat["pose_from_landmarks"][pbin][1] += 1
            if any(_box_iou_xyxy(mb["bbox_xyxy"], primary_ref["bbox_xyxy"]) >= 0.25 for mb in md):
                strat["pose_from_landmarks"][pbin][2] += 1

        # Crop quality comparison on matched frames
        if hn and md:
            best_m = _select_salient_face(md, fr.shape)
            best_h = max(hn, key=lambda d: _box_iou_xyxy(d["bbox_xyxy"], best_m["bbox_xyxy"]))
            if _box_iou_xyxy(best_h["bbox_xyxy"], best_m["bbox_xyxy"]) >= 0.25 and best_m["landmarks"] is not None:
                xh, yh, wh, hh = best_h["bbox_xywh"]
                xh0, yh0 = max(0, xh), max(0, yh)
                xh1, yh1 = min(fr.shape[1], xh + wh), min(fr.shape[0], yh + hh)
                raw_h = fr[yh0:yh1, xh0:xh1]
                if raw_h.size == 0:
                    continue
                crop_h = cv2.resize(raw_h, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                crop_u = crop_unaligned_square(fr, best_m["bbox_xyxy"], size=IMAGE_SIZE)
                crop_a, a_meta = align_face_robust(fr, best_m["bbox_xyxy"], best_m["landmarks"], size=IMAGE_SIZE)
                if crop_u is not None and crop_a is not None and a_meta.get("valid"):
                    qual_haar.append(compute_crop_quality(crop_h))
                    qual_mtcnn_u.append(compute_crop_quality(crop_u))
                    qual_mtcnn_a.append(compute_crop_quality(crop_a))
                    # Eye height normalization before vs after alignment
                    x1, y1, x2, y2 = best_m["bbox_xyxy"]
                    le, re = best_m["landmarks"][0], best_m["landmarks"][1]
                    eyes_y_unaligned.append(float((0.5 * (le[1] + re[1]) - y1) / max(1.0, y2 - y1) * IMAGE_SIZE))
                    eyes_y_aligned.append(float(a_meta["eyes_y_aligned"]))
                    rolls_before.append(float(abs(a_meta["roll_deg"])))
                    if len(visual_examples) < 6 and abs(a_meta["roll_deg"]) > 7.0:
                        visual_examples.append((item["sample_id"], crop_h, crop_u, crop_a))

    # Save visual crop comparison figure
    if visual_examples:
        nrows = len(visual_examples)
        fig, axes = plt.subplots(nrows, 3, figsize=(7.5, 2.3 * nrows), dpi=140)
        if nrows == 1:
            axes = np.asarray([axes])
        cols = ["Haar Tight Crop", "MTCNN Square Unaligned", "MTCNN Landmark Aligned"]
        for r_idx, (sid, ch, cu, ca) in enumerate(visual_examples):
            for c_idx, img in enumerate([ch, cu, ca]):
                ax = axes[r_idx, c_idx]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_xticks([])
                ax.set_yticks([])
                if r_idx == 0:
                    ax.set_title(cols[c_idx], fontsize=10)
                if c_idx == 0:
                    ax.set_ylabel(sid, fontsize=8)
        fig.suptitle("Experiment 4: Detector & Landmark Alignment Crop Comparison (MELD Dev)", fontsize=11)
        fig.tight_layout()
        fig.savefig(EXP4_DIR / "exp4_crop_comparison.png", bbox_inches="tight")
        plt.close(fig)

    # WIN_20261111_03_07_13_Pro.mp4 diagnostic comparison (if present)
    win_diag: dict[str, Any] = {"present": WIN_DIAGNOSTIC_VIDEO.is_file()}
    if WIN_DIAGNOSTIC_VIDEO.is_file():
        cap = open_video_capture(WIN_DIAGNOSTIC_VIDEO)
        win_frames = []
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % 3 == 0:
                win_frames.append((idx, fr))
            idx += 1
        cap.release()
        w_haar_boxes, w_haar_frames = 0, 0
        w_mtcnn_boxes, w_mtcnn_frames, w_mtcnn_fp = 0, 0, 0
        w_rolls = []
        for _, fr in win_frames:
            hb = _haar_wrap(haar_down, fr)
            if hb:
                w_haar_frames += 1
                w_haar_boxes += len(hb)
            mb = _detect_mtcnn_scaled(mtcnn, fr, target_width=640, min_prob=0.90)
            if mb:
                w_mtcnn_frames += 1
                w_mtcnn_boxes += len(mb)
                sal = _select_salient_face(mb, fr.shape)
                if sal["landmarks"] is not None:
                    _, am = align_face_robust(fr, sal["bbox_xyxy"], sal["landmarks"])
                    if am.get("valid"):
                        w_rolls.append(am["roll_deg"])
        win_diag.update(
            {
                "frames_sampled": len(win_frames),
                "haar_downscaled_frames_with_face": w_haar_frames,
                "haar_downscaled_total_boxes": w_haar_boxes,
                "mtcnn_downscaled_frames_with_face": w_mtcnn_frames,
                "mtcnn_downscaled_total_boxes": w_mtcnn_boxes,
                "mtcnn_extra_boxes_over_1_per_frame": max(0, w_mtcnn_boxes - w_mtcnn_frames),
                "mean_abs_roll_deg": float(np.mean(np.abs(w_rolls))) if w_rolls else 0.0,
            }
        )

    def _summarize_strat(group: dict[str, list[int]]) -> dict[str, Any]:
        res = {}
        for k, (tot, hh, mh) in group.items():
            res[k] = {
                "count": tot,
                "haar_native_hits": hh,
                "haar_native_rate": float(hh / max(1, tot)),
                "mtcnn_downscaled_hits": mh,
                "mtcnn_downscaled_rate": float(mh / max(1, tot)),
            }
        return res

    def _mean_qual(qs: list[dict[str, float]]) -> dict[str, float]:
        if not qs:
            return {"count": 0, "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
        return {
            "count": len(qs),
            "sharpness": float(np.mean([q["sharpness"] for q in qs])),
            "brightness": float(np.mean([q["brightness"] for q in qs])),
            "contrast": float(np.mean([q["contrast"] for q in qs])),
        }

    return {
        "detectors": detectors_meta,
        "missed_and_false_positives": {
            "haar_missed_where_mtcnn_conf_ge_0_95_frames": haar_missed_where_mtcnn_confident,
            "mtcnn_missed_where_haar_detected_frames": mtcnn_missed_where_haar_detected,
            "haar_total_boxes": haar_total_boxes,
            "haar_unmatched_to_mtcnn_boxes": haar_unmatched_boxes,
            "haar_unmatched_rate": float(haar_unmatched_boxes / max(1, haar_total_boxes)),
            "note": "Unmatched Haar boxes (IoU < 0.15 vs any MTCNN box on the same frame) serve as a candidate background false-positive rate.",
        },
        "stratified_recall": {
            "face_size": _summarize_strat(strat["face_size"]),
            "lighting": _summarize_strat(strat["lighting"]),
            "pose_from_landmarks": _summarize_strat(strat["pose_from_landmarks"]),
        },
        "crop_quality": {
            "haar_native": _mean_qual(qual_haar),
            "mtcnn_unaligned": _mean_qual(qual_mtcnn_u),
            "mtcnn_aligned": _mean_qual(qual_mtcnn_a),
            "eye_y_std_before_alignment_px": float(np.std(eyes_y_unaligned)) if eyes_y_unaligned else 0.0,
            "eye_y_std_after_alignment_px": float(np.std(eyes_y_aligned)) if eyes_y_aligned else 0.0,
            "mean_abs_roll_corrected_deg": float(np.mean(rolls_before)) if rolls_before else 0.0,
        },
        "win_diagnostic_video": win_diag,
    }


# ---------------------------------------------------------------------------
# Full Train/Dev MTCNN Extraction (Unaligned & Landmark-Aligned)
# ---------------------------------------------------------------------------


def extract_or_load_mtcnn_cache(split: str, rows: list[dict[str, Any]], mtcnn: MTCNN) -> dict[str, Any]:
    """Extract (or load) salient MTCNN unaligned and landmark-aligned 96x96 crops for train or dev."""
    assert_valid_split(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = CACHE_DIR / f"{split}_mtcnn_crops.npz"
    meta_path = CACHE_DIR / f"{split}_mtcnn_meta.json"
    if npz_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  Loaded existing Exp 4 MTCNN cache for '{split}': {meta['detected_videos']}/{meta['total_videos']} videos")
        return {"npz_path": str(npz_path), "meta": meta}

    print(f"\n--- Extracting MTCNN unaligned + landmark-aligned cache for '{split}' ({len(rows)} videos) ---")
    t0 = time.perf_counter()

    sample_ids: list[str] = []
    labels: list[int] = []
    emotions: list[str] = []
    unaligned_list: list[np.ndarray] = []
    aligned_list: list[np.ndarray] = []
    per_video_meta: list[dict[str, Any]] = []
    excluded_ids: list[str] = []
    native_fallback_hits = 0

    for idx, row in enumerate(rows):
        vpath = assert_no_test_path(row["video_path"])
        cap = open_video_capture(vpath)
        if not cap.isOpened():
            excluded_ids.append(row["sample_id"])
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        chosen_det: dict[str, Any] | None = None
        chosen_frame: np.ndarray | None = None
        chosen_fidx = -1

        try:
            if total > 0:
                fractions = [0.50, 0.25, 0.75, 0.12, 0.88]
                cand_indices = []
                seen = set()
                for frac in fractions:
                    fi = int(np.clip(round(frac * (total - 1)), 0, max(0, total - 1)))
                    if fi not in seen:
                        seen.add(fi)
                        cand_indices.append(fi)
                for fi in cand_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        continue
                    dets = _detect_mtcnn_scaled(mtcnn, fr, target_width=640, min_prob=0.90)
                    if not dets:
                        # Native resolution fallback only when downscaled misses
                        dets = _detect_mtcnn_scaled(mtcnn, fr, target_width=None, min_prob=0.90)
                        if dets:
                            native_fallback_hits += 1
                    if dets:
                        chosen_det = _select_salient_face(dets, fr.shape)
                        chosen_frame = fr
                        chosen_fidx = fi
                        break
            else:
                fi = 0
                while True:
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    if fi % 8 == 0:
                        dets = _detect_mtcnn_scaled(mtcnn, fr, target_width=640, min_prob=0.90)
                        if dets:
                            chosen_det = _select_salient_face(dets, fr.shape)
                            chosen_frame = fr
                            chosen_fidx = fi
                            break
                    fi += 1
        finally:
            cap.release()

        if chosen_det is None or chosen_frame is None:
            excluded_ids.append(row["sample_id"])
            continue

        crop_u = crop_unaligned_square(chosen_frame, chosen_det["bbox_xyxy"], size=IMAGE_SIZE, margin=0.08)
        if chosen_det["landmarks"] is not None:
            crop_a, a_meta = align_face_robust(chosen_frame, chosen_det["bbox_xyxy"], chosen_det["landmarks"], size=IMAGE_SIZE)
        else:
            crop_a, a_meta = None, {"valid": False}

        if crop_u is None or crop_a is None or not a_meta.get("valid"):
            excluded_ids.append(row["sample_id"])
            continue

        sample_ids.append(row["sample_id"])
        labels.append(int(row["label"]))
        emotions.append(row["emotion"])
        unaligned_list.append(crop_u)
        aligned_list.append(crop_a)
        per_video_meta.append(
            {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "emotion": row["emotion"],
                "frame_idx": chosen_fidx,
                "bbox_xyxy": [round(float(x), 2) for x in chosen_det["bbox_xyxy"]],
                "prob": round(float(chosen_det["prob"]), 5),
                "roll_deg": round(float(a_meta["roll_deg"]), 2),
                "yaw_asym": round(float(chosen_det["yaw_asym"]), 3),
            }
        )

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(rows):
            dt = time.perf_counter() - t0
            print(f"    [{split}] {idx + 1}/{len(rows)} processed | detected={len(sample_ids)} | excluded={len(excluded_ids)} | elapsed={dt:.1f}s")

    unaligned_arr = np.stack(unaligned_list, axis=0).astype(np.uint8)
    aligned_arr = np.stack(aligned_list, axis=0).astype(np.uint8)
    labels_arr = np.asarray(labels, dtype=np.int64)
    sample_ids_arr = np.asarray(sample_ids)

    np.savez_compressed(
        npz_path,
        sample_ids=sample_ids_arr,
        labels=labels_arr,
        unaligned=unaligned_arr,
        aligned=aligned_arr,
    )
    meta = {
        "split": split,
        "total_videos": len(rows),
        "detected_videos": len(sample_ids),
        "excluded_videos": len(excluded_ids),
        "coverage_rate": float(len(sample_ids) / max(1, len(rows))),
        "native_fallback_hits": native_fallback_hits,
        "extraction_seconds": float(time.perf_counter() - t0),
        "excluded_sample_ids": excluded_ids,
        "records": per_video_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"npz_path": str(npz_path), "meta": meta}


@torch.no_grad()
def evaluate_checkpoint_on_arrays(
    ckpt_path: Path,
    crops: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_backbone("resnet18", pretrained=False, num_classes=len(EMOTION_CLASSES)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    ds = ArrayFaceDataset(crops, labels, preprocessing_mode="imagenet", is_training=False, augment=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    _, metrics, _, _ = evaluate_model(model, dl, nn.CrossEntropyLoss(), device)
    del model
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    EXP4_DIR.mkdir(parents=True, exist_ok=True)
    t_exp_start = time.perf_counter()
    protected_before = verify_protected_assets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = load_video_manifest("train")
    dev_manifest = load_video_manifest("dev")
    print(f"Manifest loaded (NO test): train={len(train_manifest)}, dev={len(dev_manifest)}")

    # 1. Verify Phase 17A Exp 3 Baseline on Haar face_cache/dev (N=1,080)
    print("\n--- Part 1: Verify Exp 3 ResNet18 Baseline Protocol on Haar Dev Cache ---")
    haar_dev_items = load_haar_cached_split("dev")
    haar_dev_crops = np.stack([np.load(it["crop_path"]) for it in haar_dev_items], axis=0)
    haar_dev_labels = np.asarray([it["label"] for it in haar_dev_items], dtype=np.int64)
    haar_dev_ids = [it["sample_id"] for it in haar_dev_items]

    exp3_baseline_metrics = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, haar_dev_crops, haar_dev_labels, device)
    exp3_model_lat = build_backbone("resnet18", pretrained=False, num_classes=7).to(device)
    exp3_ckpt_obj = torch.load(EXP3_BASELINE_CKPT, map_location=device, weights_only=True)
    exp3_model_lat.load_state_dict(exp3_ckpt_obj["state_dict"])
    exp3_lat = measure_inference_latency_and_memory(exp3_model_lat, device)
    del exp3_model_lat
    print(
        f"  Exp 3 Baseline (Haar N={len(haar_dev_items)}): Acc={exp3_baseline_metrics['accuracy']:.4f}, "
        f"MacroF1={exp3_baseline_metrics['macro_f1']:.4f}, WeightedF1={exp3_baseline_metrics['weighted_f1']:.4f}, "
        f"Latency={exp3_lat['latency_ms_per_crop']:.2f} ms"
    )

    # 2. Initialize MTCNN (CUDA) and run benchmark
    mtcnn = MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device)
    bench_path = EXP4_DIR / "exp4_detector_benchmark.json"
    if args.skip_benchmark and bench_path.is_file():
        detector_benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    else:
        detector_benchmark = run_detector_benchmark(dev_manifest, mtcnn)
        bench_path.write_text(json.dumps(detector_benchmark, indent=2), encoding="utf-8")

    # 3. Build / load full MTCNN train & dev caches (unaligned + aligned)
    dev_cache_info = extract_or_load_mtcnn_cache("dev", dev_manifest, mtcnn)
    train_cache_info = extract_or_load_mtcnn_cache("train", train_manifest, mtcnn)

    # Release MTCNN VRAM cleanly before ResNet18 training
    del mtcnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_npz = np.load(train_cache_info["npz_path"])
    dev_npz = np.load(dev_cache_info["npz_path"])

    train_labels = train_npz["labels"]
    dev_labels = dev_npz["labels"]
    dev_ids = [str(s) for s in dev_npz["sample_ids"]]

    # Matched intersection between Haar (1,080) and MTCNN dev caches for paired cohort comparison
    haar_id_set = set(haar_dev_ids)
    haar_pos = {sid: idx for idx, sid in enumerate(haar_dev_ids)}
    mtcnn_pos = {sid: idx for idx, sid in enumerate(dev_ids)}
    common_ids = [sid for sid in dev_ids if sid in haar_id_set]
    common_haar_idx = [haar_pos[sid] for sid in common_ids]
    common_mtcnn_idx = [mtcnn_pos[sid] for sid in common_ids]
    print(f"\nCohort counts -> Haar dev={len(haar_dev_ids)}, MTCNN dev={len(dev_ids)}, Matched intersection={len(common_ids)}")

    # 4. Zero-shot evaluation of Exp 3 ResNet18 on MTCNN Unaligned and MTCNN Aligned
    zs_unaligned_full = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, dev_npz["unaligned"], dev_labels, device)
    zs_aligned_full = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, dev_npz["aligned"], dev_labels, device)
    zs_haar_matched = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, haar_dev_crops[common_haar_idx], haar_dev_labels[common_haar_idx], device)
    zs_unaligned_matched = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, dev_npz["unaligned"][common_mtcnn_idx], dev_labels[common_mtcnn_idx], device)
    zs_aligned_matched = evaluate_checkpoint_on_arrays(EXP3_BASELINE_CKPT, dev_npz["aligned"][common_mtcnn_idx], dev_labels[common_mtcnn_idx], device)
    print(
        f"  Exp 3 Zero-Shot -> Haar(N={len(haar_dev_ids)})={exp3_baseline_metrics['macro_f1']:.4f} | "
        f"MTCNN-Unaligned(N={len(dev_ids)})={zs_unaligned_full['macro_f1']:.4f} | "
        f"MTCNN-Aligned(N={len(dev_ids)})={zs_aligned_full['macro_f1']:.4f}"
    )

    # 5. Controlled Retraining (using exact Phase 17A Exp 3 weighting family: smoothed inverse-sqrt class weights)
    class_weights = compute_smoothed_class_weights(train_labels, power=0.5, normalize_mean=True)
    weights_dict = {EMOTION_CLASSES[i]: float(class_weights[i]) for i in range(len(EMOTION_CLASSES))}

    res_4a = train_backbone_experiment(
        experiment_id="exp4a_mtcnn_unaligned",
        experiment_name="Exp 4A: ResNet18 + MTCNN Square Unaligned",
        arch="resnet18",
        out_dir=EXP4_DIR,
        ckpt_filename="exp4a_resnet18_mtcnn_unaligned_best.pt",
        train_crops=train_npz["unaligned"],
        train_labels=train_labels,
        dev_crops=dev_npz["unaligned"],
        dev_labels=dev_labels,
        criterion_fn=lambda dev: nn.CrossEntropyLoss(weight=class_weights.to(dev)),
        preprocessing_config=get_preprocessing_config("imagenet", detector="mtcnn_640_fallback_native", alignment="square_unaligned_margin_0.08"),
        batch_size=32,
        head_epochs=3,
        finetune_epochs=5,
        head_lr=1e-3,
        backbone_lr=1e-4,
        finetune_head_lr=5e-4,
        patience=4,
        augment=False,
        seed=SEED,
        extra_config={"loss": "CrossEntropyLoss(smoothed_sqrt_weights)", "class_weights": weights_dict},
    )

    res_4b = train_backbone_experiment(
        experiment_id="exp4b_mtcnn_aligned",
        experiment_name="Exp 4B: ResNet18 + MTCNN 5-Pt Landmark Aligned",
        arch="resnet18",
        out_dir=EXP4_DIR,
        ckpt_filename="exp4b_resnet18_mtcnn_aligned_best.pt",
        train_crops=train_npz["aligned"],
        train_labels=train_labels,
        dev_crops=dev_npz["aligned"],
        dev_labels=dev_labels,
        criterion_fn=lambda dev: nn.CrossEntropyLoss(weight=class_weights.to(dev)),
        preprocessing_config=get_preprocessing_config("imagenet", detector="mtcnn_640_fallback_native", alignment="5pt_landmark_similarity_yaw_safe"),
        batch_size=32,
        head_epochs=3,
        finetune_epochs=5,
        head_lr=1e-3,
        backbone_lr=1e-4,
        finetune_head_lr=5e-4,
        patience=4,
        augment=False,
        seed=SEED,
        extra_config={"loss": "CrossEntropyLoss(smoothed_sqrt_weights)", "class_weights": weights_dict},
    )

    # Matched intersection cohort evaluation for 4A and 4B
    m4a_matched = evaluate_checkpoint_on_arrays(Path(res_4a["checkpoint_path"]), dev_npz["unaligned"][common_mtcnn_idx], dev_labels[common_mtcnn_idx], device)
    m4b_matched = evaluate_checkpoint_on_arrays(Path(res_4b["checkpoint_path"]), dev_npz["aligned"][common_mtcnn_idx], dev_labels[common_mtcnn_idx], device)

    # Selection gate: compare Exp 4A & 4B vs Exp 3 Haar baseline (Macro F1 primary, matched cohort consistency)
    candidates = [
        ("exp3_haar_baseline", exp3_baseline_metrics["macro_f1"], zs_haar_matched["macro_f1"], str(EXP3_BASELINE_CKPT), "haar", "none"),
        ("exp4a_mtcnn_unaligned", res_4a["final_dev_metrics"]["macro_f1"], m4a_matched["macro_f1"], res_4a["checkpoint_path"], "mtcnn", "unaligned"),
        ("exp4b_mtcnn_aligned", res_4b["final_dev_metrics"]["macro_f1"], m4b_matched["macro_f1"], res_4b["checkpoint_path"], "mtcnn", "aligned"),
    ]
    winner = max(candidates, key=lambda item: (item[1], item[2]))
    selected_variant = winner[0]
    if selected_variant == "exp3_haar_baseline":
        gate_reason = "Neither MTCNN unaligned nor landmark-aligned improved Dev Macro F1 over Exp 3 Haar baseline; retaining Haar baseline per validation gate."
    elif selected_variant == "exp4a_mtcnn_unaligned":
        gate_reason = (
            f"MTCNN Square Unaligned (Dev Macro F1={res_4a['final_dev_metrics']['macro_f1']:.4f}, Matched={m4a_matched['macro_f1']:.4f}) "
            f"outperformed Exp 3 Haar ({exp3_baseline_metrics['macro_f1']:.4f}) and MTCNN Aligned ({res_4b['final_dev_metrics']['macro_f1']:.4f})."
        )
    else:
        gate_reason = (
            f"MTCNN 5-Pt Landmark Aligned (Dev Macro F1={res_4b['final_dev_metrics']['macro_f1']:.4f}, Matched={m4b_matched['macro_f1']:.4f}) "
            f"outperformed Exp 3 Haar ({exp3_baseline_metrics['macro_f1']:.4f}) and MTCNN Unaligned ({res_4a['final_dev_metrics']['macro_f1']:.4f})."
        )

    protected_after = verify_protected_assets()
    summary = {
        "experiment": "Phase 17B Experiment 4: Improved Face Detection and Landmarks",
        "elapsed_seconds": float(time.perf_counter() - t_exp_start),
        "environment": collect_environment_info(),
        "protected_assets_verified": {"before": protected_before["all_ok"], "after": protected_after["all_ok"]},
        "detector_benchmark": detector_benchmark,
        "cache_extraction": {
            "train": {k: v for k, v in train_cache_info["meta"].items() if k != "records"},
            "dev": {k: v for k, v in dev_cache_info["meta"].items() if k != "records"},
            "matched_haar_mtcnn_dev_count": len(common_ids),
        },
        "zero_shot_exp3_comparison": {
            "haar_full_1080": exp3_baseline_metrics,
            "haar_baseline_latency": exp3_lat,
            "mtcnn_unaligned_full": zs_unaligned_full,
            "mtcnn_aligned_full": zs_aligned_full,
            "matched_cohort": {
                "n": len(common_ids),
                "haar": zs_haar_matched,
                "mtcnn_unaligned": zs_unaligned_matched,
                "mtcnn_aligned": zs_aligned_matched,
            },
        },
        "trained_models": {
            "exp4a_mtcnn_unaligned": res_4a,
            "exp4b_mtcnn_aligned": res_4b,
        },
        "matched_cohort_trained_comparison": {
            "n": len(common_ids),
            "exp3_haar": zs_haar_matched,
            "exp4a_mtcnn_unaligned": m4a_matched,
            "exp4b_mtcnn_aligned": m4b_matched,
        },
        "validation_gate": {
            "passed": True,
            "selected_variant": selected_variant,
            "selected_detector": winner[4],
            "selected_alignment": winner[5],
            "selected_checkpoint": winner[3],
            "selected_checkpoint_sha256": sha256_file(winner[3]),
            "reason": gate_reason,
        },
    }
    out_json = EXP4_DIR / "exp4_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExperiment 4 DONE in {summary['elapsed_seconds']:.1f}s -> selected: {selected_variant}")
    print("Reason:", gate_reason)


if __name__ == "__main__":
    main()
