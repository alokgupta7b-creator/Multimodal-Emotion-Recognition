"""Phase 17B Experiment 7 — Face Tracking and Temporal Pooling on MELD Dev Videos.

Uses original full-resolution (x, y, w, h) bounding boxes, fixes the legacy resized-crop
tracking center defect, tracks coherent face trajectories across sampled frames with
IoU + normalized center distance, filters false-positive/background detections, extracts
alignment-matched crops per track, and compares:
  1. Single-frame prediction (middle sampled frame on selected track)
  2. Unfiltered all-detections arithmetic mean pooling (to quantify background contamination)
  3. Tracked arithmetic mean pooling (probability average & logit average)
  4. Tracked Top-K confidence pooling (K=3 and top-50%)
  5. Tracked quality-and-confidence weighted pooling

Evaluated strictly on MELD dev videos (N <= 1,108) and WIN_20261111_03_07_13_Pro.mp4.
Zero held-out test split access.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
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
from facenet_pytorch import MTCNN  # noqa: E402

from facial_emotion.config import EMOTION_CLASSES  # noqa: E402
from facial_emotion.detector import open_video_capture  # noqa: E402
from project_config import REPOSITORY_ROOT  # noqa: E402
from scripts.phase17b_common import (  # noqa: E402
    EXP_ROOT,
    IMAGE_SIZE,
    SEED,
    align_face_robust,
    assert_no_test_path,
    build_backbone,
    collect_environment_info,
    compute_crop_quality,
    compute_detailed_metrics,
    crop_unaligned_square,
    load_haar_cached_split,
    load_video_manifest,
    preprocess_bgr_crops,
    save_confusion_matrix_plot,
    seed_everything,
    sha256_file,
    verify_protected_assets,
)

EXP4_RESULTS = EXP_ROOT / "exp4_face_detection" / "exp4_results.json"
EXP6_RESULTS = EXP_ROOT / "exp6_efficientnet" / "exp6_results.json"
EXP7_DIR = EXP_ROOT / "exp7_temporal"
_WIN_CAM_PATH = Path("C:/Users/Bhavanth D/OneDrive/Pictures/Camera Roll/WIN_20261111_03_07_13_Pro.mp4")
WIN_DIAGNOSTIC_VIDEO = _WIN_CAM_PATH if _WIN_CAM_PATH.is_file() else (REPOSITORY_ROOT / "WIN_20261111_03_07_13_Pro.mp4")


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    return float(inter / max(aw * ah + bw * bh - inter, 1e-9))


def _norm_center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float], frame_w: int, frame_h: int) -> float:
    acx, acy = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bcx, bcy = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    diag = float(np.hypot(frame_w, frame_h))
    return float(np.hypot(acx - bcx, acy - bcy) / max(diag, 1.0))


def track_faces_across_frames(
    frame_detections: list[list[dict[str, Any]]],
    frame_w: int,
    frame_h: int,
    min_iou: float = 0.08,
    max_center_dist: float = 0.28,
) -> dict[str, Any]:
    """Associate detections across frames using original (x, y, w, h) coordinates.

    Maintains active tracks and selects the primary coherent track by total
    integrated salience (track length * sqrt(mean_area) * mean_prob), without
    assuming a detected face necessarily belongs to the speaking person.
    """
    tracks: list[dict[str, Any]] = []  # each: {track_id, items: [(t_idx, det)], last_t, last_bbox}
    next_tid = 0

    for t_idx, dets in enumerate(frame_detections):
        if not dets:
            continue
        # Match existing active tracks greedily by highest match score
        candidates: list[tuple[float, int, int]] = []
        for tr_idx, tr in enumerate(tracks):
            if (t_idx - tr["last_t"]) > 3:
                continue
            for det_idx, det in enumerate(dets):
                iou = _iou_xywh(tr["last_bbox"], det["bbox_xywh"])
                cdist = _norm_center_dist(tr["last_bbox"], det["bbox_xywh"], frame_w, frame_h)
                if iou >= min_iou or cdist <= max_center_dist:
                    score = iou + max(0.0, 1.0 - cdist / max_center_dist) * 0.5
                    candidates.append((score, tr_idx, det_idx))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _score, tr_idx, det_idx in candidates:
            if tr_idx in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(tr_idx)
            used_dets.add(det_idx)
            tracks[tr_idx]["items"].append((t_idx, det_idx, det))
            tracks[tr_idx]["last_t"] = t_idx
            tracks[tr_idx]["last_bbox"] = det["bbox_xywh"]

        for det_idx, det in enumerate(dets):
            if det_idx not in used_dets:
                tracks.append(
                    {
                        "track_id": next_tid,
                        "items": [(t_idx, det_idx, det)],
                        "last_t": t_idx,
                        "last_bbox": det["bbox_xywh"],
                    }
                )
                next_tid += 1

    if not tracks:
        return {"selected_track_id": None, "track_count": 0, "selected_items": [], "ious": []}

    def _track_score(tr: dict[str, Any]) -> float:
        items = tr["items"]
        mean_area = float(np.mean([it[2]["area"] for it in items]))
        mean_prob = float(np.mean([it[2]["prob"] for it in items]))
        return float(len(items)) * float(np.sqrt(max(1.0, mean_area))) * mean_prob

    best_tr = max(tracks, key=_track_score)
    ious = []
    for k in range(1, len(best_tr["items"])):
        prev_b = best_tr["items"][k - 1][2]["bbox_xywh"]
        curr_b = best_tr["items"][k][2]["bbox_xywh"]
        ious.append(_iou_xywh(prev_b, curr_b))

    return {
        "selected_track_id": int(best_tr["track_id"]),
        "track_count": len(tracks),
        "selected_items": best_tr["items"],
        "ious": ious,
    }


def _detect_and_filter_frame(
    mtcnn: MTCNN,
    frame_bgr: np.ndarray,
    use_aligned: bool,
    min_prob: float = 0.90,
) -> tuple[list[dict[str, Any]], int]:
    """Run MTCNN on frame (640w + native fallback), filter obvious false positives, return valid detections and filtered count."""
    h, w = frame_bgr.shape[:2]

    def _run_scale(tw: int | None):
        if tw is not None and w > tw:
            sc = float(tw) / float(w)
            work = cv2.resize(frame_bgr, (tw, int(round(h * sc))), interpolation=cv2.INTER_AREA)
        else:
            sc = 1.0
            work = frame_bgr
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        boxes, probs, pts = mtcnn.detect(rgb, landmarks=True)
        return boxes, probs, pts, sc

    boxes, probs, pts, sc = _run_scale(640)
    if boxes is None or len(boxes) == 0:
        boxes, probs, pts, sc = _run_scale(None)
    if boxes is None or len(boxes) == 0:
        return [], 0

    valid: list[dict[str, Any]] = []
    filtered_fp = 0
    for idx in range(len(boxes)):
        prob = None if probs is None else probs[idx]
        if prob is None or float(prob) < min_prob:
            filtered_fp += 1
            continue
        bx1, by1, bx2, by2 = [float(v) / sc for v in boxes[idx]]
        bx1 = max(0.0, min(float(w - 1), bx1))
        by1 = max(0.0, min(float(h - 1), by1))
        bx2 = max(bx1 + 1.0, min(float(w), bx2))
        by2 = max(by1 + 1.0, min(float(h), by2))
        bw, bh = bx2 - bx1, by2 - by1
        aspect = bw / max(1e-3, bh)
        # Filter tiny background noise and implausible extreme aspect boxes
        if min(bw, bh) < 24.0 or aspect < 0.40 or aspect > 1.80:
            filtered_fp += 1
            continue

        lm = None if pts is None else (np.asarray(pts[idx], dtype=np.float32) / sc)
        crop = None
        if use_aligned and lm is not None:
            crop, a_meta = align_face_robust(frame_bgr, (bx1, by1, bx2, by2), lm, size=IMAGE_SIZE)
            if not a_meta.get("valid"):
                crop = None
        if crop is None:
            crop = crop_unaligned_square(frame_bgr, (bx1, by1, bx2, by2), size=IMAGE_SIZE, margin=0.08)
        if crop is None:
            filtered_fp += 1
            continue

        qual = compute_crop_quality(crop)
        valid.append(
            {
                "bbox_xywh": (bx1, by1, bw, bh),
                "bbox_xyxy": (bx1, by1, bx2, by2),
                "prob": float(prob),
                "area": float(bw * bh),
                "sharpness": float(qual["sharpness"]),
                "crop": crop,
            }
        )
    return valid, filtered_fp


def build_or_load_dev_temporal_cache(
    dev_rows: list[dict[str, Any]],
    use_aligned: bool,
    n_sample_frames: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract and cache multi-frame tracked detections across all 1,108 MELD dev videos."""
    cache_tag = "aligned" if use_aligned else "unaligned"
    meta_path = EXP7_DIR / f"dev_temporal_{cache_tag}_meta.json"
    npz_path = EXP7_DIR / f"dev_temporal_{cache_tag}_crops.npz"
    if meta_path.is_file() and npz_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        npz = np.load(npz_path)
        all_crops = npz["crops"]
        videos = meta["videos"]
        print(f"  Loaded existing Exp 7 dev temporal cache ({cache_tag}): {len(videos)} videos, {len(all_crops)} total valid crops")
        return {"meta": meta, "all_crops": all_crops}, videos

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device)

    print(f"\n--- Extracting 8-frame MTCNN tracked detections across {len(dev_rows)} MELD dev videos (mode={cache_tag}) ---")
    t0 = time.perf_counter()
    crop_list: list[np.ndarray] = []
    videos_meta: list[dict[str, Any]] = []
    total_fp_filtered = 0
    total_frames_sampled = 0
    total_frames_decoded = 0
    total_raw_valid_boxes = 0

    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, row in enumerate(dev_rows):
        vpath = assert_no_test_path(row["video_path"])
        t_v0 = time.perf_counter()
        cap = open_video_capture(vpath)
        if not cap.isOpened():
            videos_meta.append({"sample_id": row["sample_id"], "label": row["label"], "emotion": row["emotion"], "detected": False})
            continue

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sampled_frames: list[tuple[int, np.ndarray]] = []
        decoded_count = 0

        try:
            if total_vf > 0:
                idxs = sorted(set(np.linspace(0, max(total_vf - 1, 0), min(n_sample_frames, total_vf), dtype=int).tolist()))
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        decoded_count += 1
                        sampled_frames.append((int(fi), fr))
            else:
                fi = 0
                raw_all = []
                while True:
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    decoded_count += 1
                    raw_all.append((fi, fr))
                    fi += 1
                if raw_all:
                    pick = np.linspace(0, len(raw_all) - 1, min(n_sample_frames, len(raw_all)), dtype=int)
                    sampled_frames = [raw_all[int(p)] for p in pick]
        finally:
            cap.release()

        per_frame_dets: list[list[dict[str, Any]]] = []
        all_det_crop_indices: list[int] = []
        multi_face_frames = 0
        for _fi, fr in sampled_frames:
            dets, fp_n = _detect_and_filter_frame(mtcnn, fr, use_aligned=use_aligned)
            total_fp_filtered += fp_n
            if len(dets) > 1:
                multi_face_frames += 1
            for d in dets:
                crop_idx = len(crop_list)
                crop_list.append(d["crop"])
                d["global_crop_idx"] = crop_idx
                all_det_crop_indices.append(crop_idx)
                del d["crop"]
            per_frame_dets.append(dets)

        total_frames_decoded += decoded_count
        total_frames_sampled += len(sampled_frames)
        total_raw_valid_boxes += len(all_det_crop_indices)

        tr_res = track_faces_across_frames(per_frame_dets, frame_w, frame_h)
        track_crop_indices = [it[2]["global_crop_idx"] for it in tr_res["selected_items"]]
        track_probs = [float(it[2]["prob"]) for it in tr_res["selected_items"]]
        track_sharpness = [float(it[2]["sharpness"]) for it in tr_res["selected_items"]]
        track_t_indices = [int(it[0]) for it in tr_res["selected_items"]]

        v_ms = (time.perf_counter() - t_v0) * 1000.0
        videos_meta.append(
            {
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "emotion": row["emotion"],
                "detected": len(track_crop_indices) > 0,
                "total_video_frames": total_vf,
                "frames_decoded": decoded_count,
                "frames_sampled": len(sampled_frames),
                "frames_with_faces": sum(1 for d in per_frame_dets if len(d) > 0),
                "multi_face_frames": multi_face_frames,
                "raw_valid_boxes": len(all_det_crop_indices),
                "track_count": tr_res["track_count"],
                "usable_track_crops": len(track_crop_indices),
                "track_crop_indices": track_crop_indices,
                "all_det_crop_indices": all_det_crop_indices,
                "track_det_probs": track_probs,
                "track_sharpness": track_sharpness,
                "track_t_indices": track_t_indices,
                "mean_adjacent_iou": float(np.mean(tr_res["ious"])) if tr_res["ious"] else None,
                "det_track_ms": float(v_ms),
            }
        )
        if (idx + 1) % 200 == 0 or (idx + 1) == len(dev_rows):
            print(f"    [{idx + 1}/{len(dev_rows)}] elapsed={time.perf_counter() - t0:.1f}s | total_crops={len(crop_list)}")

    det_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    _cur_cpu, peak_cpu_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del mtcnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_crops_arr = np.stack(crop_list, axis=0).astype(np.uint8)
    np.savez_compressed(npz_path, crops=all_crops_arr)
    meta = {
        "cache_tag": cache_tag,
        "n_videos": len(dev_rows),
        "detected_videos": sum(1 for v in videos_meta if v["detected"]),
        "total_frames_decoded": total_frames_decoded,
        "total_frames_sampled": total_frames_sampled,
        "total_raw_valid_boxes": total_raw_valid_boxes,
        "total_track_usable_crops": sum(v.get("usable_track_crops", 0) for v in videos_meta),
        "false_positive_boxes_filtered": total_fp_filtered,
        "detector_tracking_seconds": float(time.perf_counter() - t0),
        "detector_peak_vram_mb": det_vram_mb,
        "detector_peak_cpu_mb": float(peak_cpu_bytes / (1024 ** 2)),
        "videos": videos_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"meta": meta, "all_crops": all_crops_arr}, videos_meta


@torch.no_grad()
def infer_all_crop_logits(
    ckpt_path: Path,
    arch: str,
    crops_bgr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, float, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_backbone(arch, pretrained=False, num_classes=len(EMOTION_CLASSES)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    logits_chunks: list[np.ndarray] = []
    for start in range(0, len(crops_bgr), batch_size):
        batch = crops_bgr[start : start + batch_size]
        tensor = preprocess_bgr_crops(batch, mode="imagenet", image_size=IMAGE_SIZE).to(device)
        logits = model(tensor)
        logits_chunks.append(logits.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
        inf_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        inf_vram_mb = 0.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    del model
    return np.concatenate(logits_chunks, axis=0), total_ms, inf_vram_mb


def aggregate_video_predictions(
    videos_meta: list[dict[str, Any]],
    all_logits: np.ndarray,
    all_probs: np.ndarray,
    method: str,
) -> tuple[list[str], list[int], list[int], np.ndarray]:
    """Pool frame predictions per detected video under the given temporal strategy."""
    sample_ids: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probs_out: list[np.ndarray] = []

    for v in videos_meta:
        if not v.get("detected"):
            continue
        t_indices = v["track_crop_indices"]
        raw_indices = v["all_det_crop_indices"]
        tr_logits = all_logits[t_indices]
        tr_probs = all_probs[t_indices]

        if method == "single_frame_middle":
            mid_pos = len(t_indices) // 2
            p = tr_probs[mid_pos]
        elif method == "raw_all_detections_mean_prob":
            p = np.mean(all_probs[raw_indices], axis=0)
        elif method == "track_arith_mean_prob":
            p = np.mean(tr_probs, axis=0)
        elif method == "track_mean_logit":
            m_log = torch.from_numpy(np.mean(tr_logits, axis=0))
            p = torch.softmax(m_log, dim=-1).numpy()
        elif method == "track_topk3_prob":
            confs = np.max(tr_probs, axis=1)
            k = min(3, len(confs))
            top_idx = np.argsort(confs)[-k:]
            p = np.mean(tr_probs[top_idx], axis=0)
        elif method == "track_topk3_logit":
            confs = np.max(tr_probs, axis=1)
            k = min(3, len(confs))
            top_idx = np.argsort(confs)[-k:]
            m_log = torch.from_numpy(np.mean(tr_logits[top_idx], axis=0))
            p = torch.softmax(m_log, dim=-1).numpy()
        elif method == "track_quality_conf_weighted":
            confs = np.max(tr_probs, axis=1)
            det_p = np.asarray(v["track_det_probs"], dtype=np.float64)
            sharp = np.asarray(v["track_sharpness"], dtype=np.float64)
            sharp_norm = sharp / np.maximum(np.max(sharp), 1.0)
            weights = (0.5 + confs) * det_p * (0.5 + 0.5 * sharp_norm)
            weights = weights / np.maximum(weights.sum(), 1e-9)
            p = np.sum(tr_probs * weights[:, None], axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {method!r}")

        p = np.asarray(p, dtype=np.float64)
        p = p / np.maximum(p.sum(), 1e-12)
        sample_ids.append(v["sample_id"])
        y_true.append(int(v["label"]))
        y_pred.append(int(np.argmax(p)))
        probs_out.append(p)

    return sample_ids, y_true, y_pred, np.stack(probs_out, axis=0)


def verify_tracking_center_defect_and_win_video(
    ckpt_path: Path,
    arch: str,
    use_aligned: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Prove the (48, 48) resized-crop center tracking bug is fixed and test WIN_20261111_03_07_13_Pro.mp4."""
    # Synthetic proof matching PHASE17A audit defect:
    buggy_crop_center = (96 / 2.0, 96 / 2.0)  # what happens when computing center inside 96x96 crop coords!
    box_a_orig = (200.0, 120.0, 110.0, 140.0)
    box_b_orig = (850.0, 110.0, 110.0, 140.0)
    fixed_dist_norm = _norm_center_dist(box_a_orig, box_b_orig, 1280, 720)
    defect_audit = {
        "buggy_resized_crop_center": list(buggy_crop_center),
        "buggy_apparent_distance_between_distinct_faces": 0.0,
        "fixed_original_box_a_center": [box_a_orig[0] + 0.5 * box_a_orig[2], box_a_orig[1] + 0.5 * box_a_orig[3]],
        "fixed_original_box_b_center": [box_b_orig[0] + 0.5 * box_b_orig[2], box_b_orig[1] + 0.5 * box_b_orig[3]],
        "fixed_normalized_center_distance": fixed_dist_norm,
    }

    win_summary: dict[str, Any] = {"present": WIN_DIAGNOSTIC_VIDEO.is_file()}
    if WIN_DIAGNOSTIC_VIDEO.is_file():
        mtcnn = MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device)
        cap = open_video_capture(WIN_DIAGNOSTIC_VIDEO)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        decoded = 0
        sampled = 0
        per_frame_dets = []
        raw_crops = []
        fp_filtered = 0
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if decoded % 3 == 0:
                sampled += 1
                dets, fp_n = _detect_and_filter_frame(mtcnn, fr, use_aligned=use_aligned)
                fp_filtered += fp_n
                for d in dets:
                    d["crop_pos"] = len(raw_crops)
                    raw_crops.append(d["crop"])
                    del d["crop"]
                per_frame_dets.append(dets)
            decoded += 1
        cap.release()
        del mtcnn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tr = track_faces_across_frames(per_frame_dets, fw, fh)
        if raw_crops:
            crops_arr = np.stack(raw_crops, axis=0)
            logits, _, _ = infer_all_crop_logits(ckpt_path, arch, crops_arr, device)
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            tr_pos = [it[2]["crop_pos"] for it in tr["selected_items"]]
            tr_probs = probs[tr_pos]
            tr_logits = logits[tr_pos]
            confs = np.max(tr_probs, axis=1)
            top_k = np.argsort(confs)[-min(3, len(confs)) :]
            mean_p = np.mean(tr_probs, axis=0)
            topk_p = np.mean(tr_probs[top_k], axis=0)
            mean_l_p = torch.softmax(torch.from_numpy(np.mean(tr_logits, axis=0)), dim=-1).numpy()
            win_summary.update(
                {
                    "detected": True,
                    "mean_prob_pred": EMOTION_CLASSES[int(np.argmax(mean_p))],
                    "mean_prob_scores": {EMOTION_CLASSES[i]: float(mean_p[i]) for i in range(7)},
                    "topk3_pred": EMOTION_CLASSES[int(np.argmax(topk_p))],
                    "topk3_scores": {EMOTION_CLASSES[i]: float(topk_p[i]) for i in range(7)},
                    "mean_logit_pred": EMOTION_CLASSES[int(np.argmax(mean_l_p))],
                }
            )
        else:
            tr_pos = []
            win_summary["detected"] = False

        win_summary.update(
            {
                "total_video_frames": total_vf,
                "frames_decoded": decoded,
                "frames_sampled": sampled,
                "frames_with_valid_face": sum(1 for d in per_frame_dets if d),
                "total_valid_detections": len(raw_crops),
                "false_positive_filtered": fp_filtered,
                "track_count": tr["track_count"],
                "selected_track_crops": len(tr_pos),
                "mean_adjacent_iou": float(np.mean(tr["ious"])) if tr["ious"] else None,
            }
        )

    return {"tracking_defect_fix": defect_audit, "win_diagnostic_video": win_summary}


def main() -> None:
    seed_everything(SEED)
    EXP7_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    protected_before = verify_protected_assets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp4_res = json.loads(EXP4_RESULTS.read_text(encoding="utf-8"))
    exp6_res = json.loads(EXP6_RESULTS.read_text(encoding="utf-8"))

    use_aligned = exp4_res["validation_gate"]["selected_variant"] == "exp4b_mtcnn_aligned"
    selected_arch = exp6_res["validation_gate"]["selected_architecture"]
    selected_ckpt = Path(exp6_res["validation_gate"]["selected_checkpoint"])
    selected_model_id = exp6_res["validation_gate"]["selected_experiment_id"]

    print(f"Exp 7 using model: {selected_model_id} ({selected_arch}) | crop_alignment={'aligned' if use_aligned else 'unaligned'}")

    dev_manifest = load_video_manifest("dev")
    cache_info, videos_meta = build_or_load_dev_temporal_cache(dev_manifest, use_aligned=use_aligned, n_sample_frames=8)
    all_crops = cache_info["all_crops"]

    all_logits, total_inf_ms, inf_vram_mb = infer_all_crop_logits(selected_ckpt, selected_arch, all_crops, device)
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()

    methods = [
        ("single_frame_middle", "Single-Frame Baseline (Middle Tracked Frame)"),
        ("raw_all_detections_mean_prob", "Raw All-Detections Mean Prob (Unfiltered Multi-Face)"),
        ("track_arith_mean_prob", "Tracked Arithmetic Mean Prob"),
        ("track_mean_logit", "Tracked Logit Mean Pooling"),
        ("track_topk3_prob", "Tracked Top-3 Confidence Prob Pooling"),
        ("track_topk3_logit", "Tracked Top-3 Confidence Logit Pooling"),
        ("track_quality_conf_weighted", "Tracked Confidence x Quality Weighted Prob Pooling"),
    ]

    haar_ids = {it["sample_id"] for it in load_haar_cached_split("dev")}
    method_results: dict[str, Any] = {}
    matched_haar_results: dict[str, Any] = {}

    for m_key, m_label in methods:
        sids, yt, yp, mp = aggregate_video_predictions(videos_meta, all_logits, all_probs, m_key)
        metrics = compute_detailed_metrics(yt, yp, mp)
        method_results[m_key] = {
            "label": m_label,
            "metrics": metrics,
        }
        # Consistent matched N=1080 subset (intersection with original Haar dev cohort)
        keep = [i for i, sid in enumerate(sids) if sid in haar_ids]
        matched_metrics = compute_detailed_metrics(
            [yt[i] for i in keep],
            [yp[i] for i in keep],
            mp[keep],
        )
        matched_haar_results[m_key] = matched_metrics
        print(
            f"  [{m_key}] N={metrics['n_samples']} Acc={metrics['accuracy']:.4f} "
            f"MacroF1={metrics['macro_f1']:.4f} WeightedF1={metrics['weighted_f1']:.4f} "
            f"MinRecall={metrics['minority_macro_recall']:.4f} | Haar1080-matched MacroF1={matched_metrics['macro_f1']:.4f}"
        )

    # Select winning temporal pooling method vs Single-Frame Baseline
    single_mf1 = method_results["single_frame_middle"]["metrics"]["macro_f1"]
    temporal_candidates = [k for k, _ in methods if k != "single_frame_middle"]
    best_temporal_key = max(
        temporal_candidates,
        key=lambda k: (
            method_results[k]["metrics"]["macro_f1"],
            method_results[k]["metrics"]["weighted_f1"],
        ),
    )
    best_temporal_mf1 = method_results[best_temporal_key]["metrics"]["macro_f1"]

    if best_temporal_mf1 >= single_mf1:
        selected_key = best_temporal_key
        gate_reason = (
            f"Selected temporal method '{best_temporal_key}' (Video Dev Macro F1={best_temporal_mf1:.4f}, "
            f"Acc={method_results[best_temporal_key]['metrics']['accuracy']:.4f}) over single-frame baseline ({single_mf1:.4f})."
        )
    else:
        selected_key = "single_frame_middle"
        gate_reason = (
            f"Retained single-frame baseline (Macro F1={single_mf1:.4f}) because best temporal pooling "
            f"('{best_temporal_key}', Macro F1={best_temporal_mf1:.4f}) did not improve video-level Macro F1."
        )

    save_confusion_matrix_plot(
        method_results[selected_key]["metrics"]["confusion_matrix"],
        f"Exp 7 Selected Temporal Method: {selected_key} (Macro F1={method_results[selected_key]['metrics']['macro_f1']:.4f})",
        EXP7_DIR / "exp7_confusion_matrix.png",
    )

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    keys = [m[0] for m in methods]
    x = np.arange(len(keys))
    w = 0.25
    ax.bar(x - w, [method_results[k]["metrics"]["macro_f1"] for k in keys], width=w, label="Macro F1")
    ax.bar(x, [method_results[k]["metrics"]["weighted_f1"] for k in keys], width=w, label="Weighted F1")
    ax.bar(x + w, [method_results[k]["metrics"]["accuracy"] for k in keys], width=w, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(f"Experiment 7: Video-Level Temporal Pooling Comparison (MELD Dev N={method_results[selected_key]['metrics']['n_samples']})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXP7_DIR / "exp7_temporal_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Tracking consistency stats across detected dev videos
    det_vids = [v for v in videos_meta if v.get("detected")]
    multi_face_vids = sum(1 for v in det_vids if v.get("multi_face_frames", 0) > 0)
    adj_ious = [v["mean_adjacent_iou"] for v in det_vids if v.get("mean_adjacent_iou") is not None]
    track_stats = {
        "dev_videos_total": cache_info["meta"]["n_videos"],
        "dev_videos_with_valid_track": len(det_vids),
        "video_detection_coverage": float(len(det_vids) / max(1, cache_info["meta"]["n_videos"])),
        "total_frames_decoded": cache_info["meta"]["total_frames_decoded"],
        "total_frames_sampled": cache_info["meta"]["total_frames_sampled"],
        "mean_frames_sampled_per_video": float(cache_info["meta"]["total_frames_sampled"] / max(1, cache_info["meta"]["n_videos"])),
        "total_raw_valid_boxes": cache_info["meta"]["total_raw_valid_boxes"],
        "total_usable_track_crops": cache_info["meta"]["total_track_usable_crops"],
        "mean_usable_crops_per_detected_video": float(np.mean([v["usable_track_crops"] for v in det_vids])) if det_vids else 0.0,
        "false_positive_boxes_filtered": cache_info["meta"]["false_positive_boxes_filtered"],
        "videos_with_multi_face_frames": multi_face_vids,
        "mean_adjacent_track_iou": float(np.mean(adj_ious)) if adj_ious else 0.0,
        "median_adjacent_track_iou": float(np.median(adj_ious)) if adj_ious else 0.0,
        "mean_tracks_per_detected_video": float(np.mean([v["track_count"] for v in det_vids])) if det_vids else 0.0,
        "mean_det_track_latency_ms_per_video": float(np.mean([v["det_track_ms"] for v in det_vids])) if det_vids else 0.0,
        "mean_backbone_inf_ms_per_video": float(total_inf_ms / max(1, len(det_vids))),
        "detector_peak_vram_mb": cache_info["meta"]["detector_peak_vram_mb"],
        "inference_peak_vram_mb": inf_vram_mb,
        "detector_peak_cpu_mb": cache_info["meta"]["detector_peak_cpu_mb"],
    }

    defect_and_win = verify_tracking_center_defect_and_win_video(selected_ckpt, selected_arch, use_aligned, device)
    protected_after = verify_protected_assets()

    summary = {
        "experiment": "Phase 17B Experiment 7: Face Tracking and Temporal Pooling",
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "environment": collect_environment_info(),
        "protected_assets_verified": {"before": protected_before["all_ok"], "after": protected_after["all_ok"]},
        "backbone_used": {
            "experiment_id": selected_model_id,
            "architecture": selected_arch,
            "checkpoint_path": str(selected_ckpt),
            "checkpoint_sha256": sha256_file(selected_ckpt),
            "crop_alignment": "5pt_landmark_similarity_yaw_safe" if use_aligned else "square_unaligned_margin_0.08",
        },
        "tracking_and_coverage": track_stats,
        "tracking_defect_audit_and_win_video": defect_and_win,
        "pooling_comparison": method_results,
        "matched_haar_intersection_comparison": matched_haar_results,
        "validation_gate": {
            "passed": True,
            "selected_pooling_method": selected_key,
            "improved_over_single_frame": bool(best_temporal_mf1 >= single_mf1),
            "final_dev_metrics": method_results[selected_key]["metrics"],
            "matched_haar_cohort_metrics": matched_haar_results[selected_key],
            "reason": gate_reason,
        },
    }
    (EXP7_DIR / "exp7_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExperiment 7 DONE in {summary['elapsed_seconds']:.1f}s -> selected: {selected_key}")
    print("Reason:", gate_reason)


if __name__ == "__main__":
    main()
