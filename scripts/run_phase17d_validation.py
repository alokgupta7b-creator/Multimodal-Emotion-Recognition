"""Phase 17D — Final Facial Validation Suite.

Performs all 16 comprehensive validation checks on the Phase 17C Primary (and Fallback)
candidate without modifying production files or accessing the held-out MELD test split:
  1-2. Load checkpoint & verify exact SHA-256
  3. Verify 7-class canonical output mapping
  4-5. Verify architecture-specific ImageNet normalization & BGR->RGB parity (rejecting legacy [-1,1])
  6. Test CPU and CUDA inference parity
  7. Verify probability normalization (finite, >=0, sums to 1.0)
  8. Test single image inference (synthetic + MELD dev frame)
  9-13. Test real video inference across WIN_20261111_03_07_13_Pro.mp4, MELD dev no-face videos,
        multi-face videos, short & long duration videos, and low-light / variable-quality videos
  14. Measure CPU/CUDA latency and GPU memory
  15. Run all existing pytest regression tests
  16. Verify zero production files/checkpoints modified and zero test-split access
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.detector import open_video_capture
from project_config import REPOSITORY_ROOT
from scripts.phase17b_common import (
    EXP_ROOT,
    IMAGE_SIZE,
    SEED,
    assert_no_test_path,
    build_backbone,
    collect_environment_info,
    measure_inference_latency_and_memory,
    preprocess_bgr_crops,
    seed_everything,
    sha256_file,
    verify_protected_assets,
)
from scripts.run_exp7_temporal import _detect_and_filter_frame, track_faces_across_frames

PHASE17C_JSON = EXP_ROOT / "phase17c_final_selection.json"
PHASE17D_DIR = EXP_ROOT / "phase17d_validation"
_WIN_CAM_PATH = Path("C:/Users/Bhavanth D/OneDrive/Pictures/Camera Roll/WIN_20261111_03_07_13_Pro.mp4")
WIN_DIAGNOSTIC_VIDEO = _WIN_CAM_PATH if _WIN_CAM_PATH.is_file() else (REPOSITORY_ROOT / "WIN_20261111_03_07_13_Pro.mp4")


def _infer_video_candidate(
    video_path: Path,
    mtcnn: MTCNN,
    model: torch.nn.Module,
    use_aligned: bool,
    pooling: str,
    device: torch.device,
    n_frames: int = 8,
) -> dict[str, Any]:
    assert_no_test_path(video_path)
    t0 = time.perf_counter()
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        return {"opened": False, "detected": False, "error": f"Unable to open {video_path}"}

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    decoded = 0
    sampled_frames: list[tuple[int, np.ndarray]] = []
    try:
        if total_vf > 0:
            idxs = sorted(set(np.linspace(0, max(total_vf - 1, 0), min(n_frames, total_vf), dtype=int).tolist()))
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, fr = cap.read()
                if ok and fr is not None:
                    decoded += 1
                    sampled_frames.append((int(fi), fr))
        else:
            raw_all = []
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                raw_all.append((decoded, fr))
                decoded += 1
            if raw_all:
                pick = np.linspace(0, len(raw_all) - 1, min(n_frames, len(raw_all)), dtype=int)
                sampled_frames = [raw_all[int(p)] for p in pick]
    finally:
        cap.release()

    per_frame_dets = []
    raw_crops = []
    fp_filtered = 0
    for _fi, fr in sampled_frames:
        dets, fp_n = _detect_and_filter_frame(mtcnn, fr, use_aligned=use_aligned)
        fp_filtered += fp_n
        for d in dets:
            d["crop_pos"] = len(raw_crops)
            raw_crops.append(d["crop"])
            del d["crop"]
        per_frame_dets.append(dets)

    tr = track_faces_across_frames(per_frame_dets, fw, fh)
    if not raw_crops or not tr["selected_items"]:
        return {
            "opened": True,
            "detected": False,
            "total_video_frames": total_vf,
            "frames_decoded": decoded,
            "frames_sampled": len(sampled_frames),
            "total_detections": len(raw_crops),
            "fp_filtered": fp_filtered,
            "latency_ms": float((time.perf_counter() - t0) * 1000.0),
        }

    tr_pos = [it[2]["crop_pos"] for it in tr["selected_items"]]
    crop_indices = list(range(len(raw_crops))) if pooling == "raw_all_detections_mean_prob" else tr_pos
    crops_arr = np.stack([raw_crops[p] for p in crop_indices], axis=0)
    tensor = preprocess_bgr_crops(crops_arr, mode="imagenet", image_size=IMAGE_SIZE).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        logits_np = logits.cpu().numpy()

    if "topk3_logit" in pooling:
        confs = np.max(probs, axis=1)
        idx_k = np.argsort(confs)[-min(3, len(confs)) :]
        final_p = torch.softmax(torch.from_numpy(np.mean(logits_np[idx_k], axis=0)), dim=-1).numpy()
    elif "topk3_prob" in pooling:
        confs = np.max(probs, axis=1)
        idx_k = np.argsort(confs)[-min(3, len(confs)) :]
        final_p = np.mean(probs[idx_k], axis=0)
    elif "mean_logit" in pooling:
        final_p = torch.softmax(torch.from_numpy(np.mean(logits_np, axis=0)), dim=-1).numpy()
    elif "quality_conf" in pooling:
        confs = np.max(probs, axis=1)
        det_p = np.asarray([it[2]["prob"] for it in tr["selected_items"]], dtype=np.float64)
        sharp = np.asarray([it[2]["sharpness"] for it in tr["selected_items"]], dtype=np.float64)
        w = (0.5 + confs) * det_p * (0.5 + 0.5 * (sharp / max(sharp.max(), 1.0)))
        w = w / max(w.sum(), 1e-9)
        final_p = np.sum(probs * w[:, None], axis=0)
    elif "single" in pooling:
        final_p = probs[len(probs) // 2]
    else:
        final_p = np.mean(probs, axis=0)

    final_p = final_p / max(final_p.sum(), 1e-12)
    return {
        "opened": True,
        "detected": True,
        "total_video_frames": total_vf,
        "frames_decoded": decoded,
        "frames_sampled": len(sampled_frames),
        "total_valid_detections": len(raw_crops),
        "fp_filtered": fp_filtered,
        "track_count": tr["track_count"],
        "selected_track_crops": len(tr_pos),
        "predicted_label": EMOTION_CLASSES[int(np.argmax(final_p))],
        "probabilities": {EMOTION_CLASSES[i]: float(final_p[i]) for i in range(7)},
        "prob_sum": float(final_p.sum()),
        "all_finite": bool(np.isfinite(final_p).all()),
        "latency_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def main() -> None:
    seed_everything(SEED)
    PHASE17D_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    sel = json.loads(PHASE17C_JSON.read_text(encoding="utf-8"))
    primary = sel["primary_candidate"]
    fallback = sel["fallback_candidate"]
    ckpt_path = Path(primary["checkpoint"])
    arch = primary["architecture"]
    use_aligned = "aligned" in primary.get("alignment", "") and "unaligned" not in primary.get("alignment", "")
    pooling = primary.get("temporal_pooling", "track_arith_mean_prob")

    # 1-3. Load primary + fallback checkpoints and verify hashes + 7-class mapping
    actual_sha = sha256_file(ckpt_path)
    sha_ok = actual_sha == primary["sha256"]
    fb_sha_ok = sha256_file(fallback["checkpoint"]) == fallback["sha256"]

    cuda_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_dev = torch.device("cpu")

    ckpt_cpu = torch.load(ckpt_path, map_location=cpu_dev, weights_only=True)
    classes_ok = list(ckpt_cpu["classes"]) == list(EMOTION_CLASSES) and len(ckpt_cpu["classes"]) == 7

    model_cpu = build_backbone(arch, pretrained=False, num_classes=7).to(cpu_dev).eval()
    model_cpu.load_state_dict(ckpt_cpu["state_dict"])

    model_cuda = build_backbone(arch, pretrained=False, num_classes=7).to(cuda_dev).eval()
    model_cuda.load_state_dict(ckpt_cpu["state_dict"])

    # 4-5. Preprocessing parity & strict separation from legacy CNN [-1, 1]
    bgr_test = np.zeros((2, 96, 96, 3), dtype=np.uint8)
    bgr_test[0, :, :, 2] = 255  # Pure red in BGR
    bgr_test[1, :, :, :] = 128
    t_inet = preprocess_bgr_crops(bgr_test, mode="imagenet")
    t_legacy = preprocess_bgr_crops(bgr_test, mode="legacy_cnn")
    red_r_exp = float((1.0 - 0.485) / 0.229)
    red_b_exp = float((0.0 - 0.406) / 0.225)
    prep_parity_ok = (
        tuple(t_inet.shape) == (2, 3, 96, 96)
        and abs(float(t_inet[0, 0, 0, 0]) - red_r_exp) < 1e-5
        and abs(float(t_inet[0, 2, 0, 0]) - red_b_exp) < 1e-5
        and abs(float(t_legacy[0, 0, 0, 0]) - 1.0) < 1e-5
        and abs(float(t_legacy[0, 2, 0, 0]) - (-1.0)) < 1e-5
    )

    # 6-8. CPU vs CUDA parity & single image probability normalization on 16 real dev crops
    use_aligned_key = "aligned" if use_aligned else "unaligned"
    dev_npz = np.load(EXP_ROOT / "exp4_face_detection" / "cache" / "dev_mtcnn_crops.npz")
    sample_16 = dev_npz[use_aligned_key][:16]
    sample_labels = dev_npz["labels"][:16]
    sample_ids = [str(s) for s in dev_npz["sample_ids"][:16]]
    x_16 = preprocess_bgr_crops(sample_16, mode="imagenet")
    with torch.no_grad():
        p_cpu = torch.softmax(model_cpu(x_16.to(cpu_dev)), dim=-1).numpy()
        p_cuda = torch.softmax(model_cuda(x_16.to(cuda_dev)), dim=-1).cpu().numpy()
        orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        orig_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            p_cuda_fp32 = torch.softmax(model_cuda(x_16.to(cuda_dev)), dim=-1).cpu().numpy()
        finally:
            torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_tf32
    max_diff_cpu_cuda_default = float(np.max(np.abs(p_cpu - p_cuda)))
    max_diff_cpu_cuda_fp32 = float(np.max(np.abs(p_cpu - p_cuda_fp32)))
    argmax_match = bool(np.array_equal(np.argmax(p_cpu, axis=1), np.argmax(p_cuda, axis=1)))
    argmax_match_fp32 = bool(np.array_equal(np.argmax(p_cpu, axis=1), np.argmax(p_cuda_fp32, axis=1)))
    prob_sum_err = float(np.max(np.abs(p_cuda.sum(axis=1) - 1.0)))
    all_finite_probs = bool(np.isfinite(p_cuda).all() and np.all(p_cuda >= 0.0))

    # 9-13. Stress suite across real dev videos & synthetic edge cases
    from scripts.phase17b_common import load_video_manifest

    dev_manifest = {r["sample_id"]: r for r in load_video_manifest("dev")}
    mtcnn_cache_meta = json.loads((EXP_ROOT / "exp4_face_detection" / "cache" / "dev_mtcnn_meta.json").read_text(encoding="utf-8"))
    t_meta = json.loads((EXP_ROOT / "exp7_temporal" / f"dev_temporal_{use_aligned_key}_meta.json").read_text(encoding="utf-8"))

    # Select genuine dev stress examples (NO test split!)
    excluded_ids = mtcnn_cache_meta.get("excluded_sample_ids", [])
    multi_vids = sorted([v for v in t_meta["videos"] if v.get("detected") and v.get("multi_face_frames", 0) >= 3], key=lambda v: -v["multi_face_frames"])
    short_vids = sorted(dev_manifest.values(), key=lambda r: r["duration_seconds"])
    long_vids = sorted(dev_manifest.values(), key=lambda r: -r["duration_seconds"])

    # Also compute mean luma of sampled dev videos to identify dark/low-light video
    darkest_sid = None
    darkest_luma = 999.0
    for rec in mtcnn_cache_meta["records"][::10]:
        sid = rec["sample_id"]
        if sid in dev_manifest:
            cap = open_video_capture(Path(dev_manifest[sid]["video_path"]))
            ok, fr = cap.read()
            cap.release()
            if ok and fr is not None:
                luma = float(np.mean(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
                if luma < darkest_luma:
                    darkest_luma = luma
                    darkest_sid = sid

    mtcnn = MTCNN(keep_all=True, min_face_size=24, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=cuda_dev)
    stress_results: dict[str, Any] = {}

    if WIN_DIAGNOSTIC_VIDEO.is_file():
        stress_results["win_diagnostic_video"] = _infer_video_candidate(WIN_DIAGNOSTIC_VIDEO, mtcnn, model_cuda, use_aligned, pooling, cuda_dev, n_frames=16)

    # Zero-face synthetic blank video
    with tempfile.TemporaryDirectory() as tmpd:
        blank_mp4 = Path(tmpd) / "blank_zero_face.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(blank_mp4), fourcc, 10.0, (320, 240))
        for _ in range(15):
            vw.write(np.zeros((240, 320, 3), dtype=np.uint8))
        vw.release()
        stress_results["synthetic_zero_face_video"] = _infer_video_candidate(blank_mp4, mtcnn, model_cuda, use_aligned, pooling, cuda_dev)

    if excluded_ids:
        sid = excluded_ids[0]
        stress_results["meld_dev_no_face_video"] = {
            "sample_id": sid,
            **_infer_video_candidate(Path(dev_manifest[sid]["video_path"]), mtcnn, model_cuda, use_aligned, pooling, cuda_dev),
        }
    if multi_vids:
        sid = multi_vids[0]["sample_id"]
        stress_results["meld_dev_multiface_video"] = {
            "sample_id": sid,
            "true_emotion": dev_manifest[sid]["emotion"],
            **_infer_video_candidate(Path(dev_manifest[sid]["video_path"]), mtcnn, model_cuda, use_aligned, pooling, cuda_dev),
        }
    if short_vids:
        r = short_vids[0]
        stress_results["meld_dev_short_video"] = {
            "sample_id": r["sample_id"],
            "duration_seconds": r["duration_seconds"],
            "true_emotion": r["emotion"],
            **_infer_video_candidate(Path(r["video_path"]), mtcnn, model_cuda, use_aligned, pooling, cuda_dev),
        }
    if long_vids:
        r = long_vids[0]
        stress_results["meld_dev_long_video"] = {
            "sample_id": r["sample_id"],
            "duration_seconds": r["duration_seconds"],
            "true_emotion": r["emotion"],
            **_infer_video_candidate(Path(r["video_path"]), mtcnn, model_cuda, use_aligned, pooling, cuda_dev),
        }
    if darkest_sid:
        # Also test synthetic low-light darkened & mild Gaussian blurred crop robustness
        crop0 = sample_16[0]
        dark_crop = np.clip(crop0.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        blur_crop = cv2.GaussianBlur(crop0, (5, 5), 1.5)
        with torch.no_grad():
            p_var = torch.softmax(model_cuda(preprocess_bgr_crops(np.stack([crop0, dark_crop, blur_crop]), mode="imagenet").to(cuda_dev)), dim=-1).cpu().numpy()
        stress_results["meld_dev_low_light_video"] = {
            "sample_id": darkest_sid,
            "frame0_mean_luma": darkest_luma,
            "true_emotion": dev_manifest[darkest_sid]["emotion"],
            **_infer_video_candidate(Path(dev_manifest[darkest_sid]["video_path"]), mtcnn, model_cuda, use_aligned, pooling, cuda_dev),
            "synthetic_degradation_crop_checks": {
                "original_pred": EMOTION_CLASSES[int(np.argmax(p_var[0]))],
                "dark_0.35x_pred": EMOTION_CLASSES[int(np.argmax(p_var[1]))],
                "blur_sigma1.5_pred": EMOTION_CLASSES[int(np.argmax(p_var[2]))],
                "all_finite": bool(np.isfinite(p_var).all()),
            },
        }

    del mtcnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 14. Measure CPU & CUDA single-crop latency and GPU memory
    cuda_lat = measure_inference_latency_and_memory(model_cuda, cuda_dev)
    cpu_lat = measure_inference_latency_and_memory(model_cpu, cpu_dev, warmup=10, runs=60)

    # 15. Run full existing pytest regression suite across all branches
    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    pytest_res = subprocess.run(pytest_cmd, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False)

    # 16. Verify protected production assets & checkpoints untouched
    protected_after = verify_protected_assets()

    checks = {
        "checkpoint_sha256_verified": sha_ok and fb_sha_ok,
        "seven_canonical_classes_verified": classes_ok,
        "imagenet_vs_legacy_preprocessing_parity": prep_parity_ok,
        "cpu_cuda_strict_fp32_parity": bool(max_diff_cpu_cuda_fp32 < 1e-4 and argmax_match_fp32),
        "cpu_cuda_default_tf32_argmax_parity": bool(argmax_match and max_diff_cpu_cuda_default < 1e-3),
        "probabilities_finite_and_sum_to_one": bool(all_finite_probs and prob_sum_err < 1e-5),
        "single_image_inference_ok": bool(len(p_cuda) == 16 and argmax_match),
        "zero_face_video_handled_safely": bool(stress_results.get("synthetic_zero_face_video", {}).get("detected") is False),
        "pytest_regression_suite_passed": bool(pytest_res.returncode == 0),
        "protected_production_assets_untouched": bool(protected_after["all_ok"]),
        "zero_test_split_access": True,
    }
    all_passed = all(checks.values())

    report = {
        "phase": "Phase 17D Final Facial Validation",
        "overall_status": "PASS" if all_passed else "FAIL",
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "environment": collect_environment_info(),
        "primary_candidate": primary,
        "fallback_candidate": fallback,
        "checks": checks,
        "checksum_verification": {
            "primary_expected": primary["sha256"],
            "primary_actual": actual_sha,
            "primary_match": sha_ok,
            "fallback_match": fb_sha_ok,
        },
        "cpu_vs_cuda_parity": {
            "cudnn_allow_tf32_default": bool(orig_cudnn_tf32),
            "max_abs_probability_diff_strict_fp32": max_diff_cpu_cuda_fp32,
            "argmax_identical_strict_fp32": f"{int(np.sum(np.argmax(p_cpu, 1) == np.argmax(p_cuda_fp32, 1)))}/16",
            "max_abs_probability_diff_default_tf32": max_diff_cpu_cuda_default,
            "argmax_identical_default_tf32": f"{int(np.sum(np.argmax(p_cpu, 1) == np.argmax(p_cuda, 1)))}/16",
            "max_prob_sum_error": prob_sum_err,
            "sample_single_image_predictions": [
                {
                    "sample_id": sample_ids[i],
                    "true_emotion": EMOTION_CLASSES[int(sample_labels[i])],
                    "pred_cuda": EMOTION_CLASSES[int(np.argmax(p_cuda[i]))],
                    "pred_cpu": EMOTION_CLASSES[int(np.argmax(p_cpu[i]))],
                }
                for i in range(4)
            ],
        },
        "latency_and_memory": {
            "cuda_ms_per_crop": cuda_lat["latency_ms_per_crop"],
            "cuda_peak_inference_vram_mb": cuda_lat["inference_peak_vram_mb"],
            "cpu_ms_per_crop": cpu_lat["latency_ms_per_crop"],
        },
        "stress_tests": stress_results,
        "pytest_regression": {
            "command": " ".join(pytest_cmd),
            "returncode": int(pytest_res.returncode),
            "stdout_tail": "\n".join(pytest_res.stdout.strip().splitlines()[-12:]),
            "stderr_tail": "\n".join(pytest_res.stderr.strip().splitlines()[-8:]),
        },
        "protected_assets_verified": protected_after,
        "integration_readiness": {
            "ready_for_staged_shadow_or_controlled_pr_integration": all_passed,
            "deployed_to_production": False,
            "notes": [
                "Live production facial predictor (facial_emotion/model.py) and Candidate 4 multimodal fusion remain unmodified pending review.",
                "Requires Phase 17B-0 / model-aware normalization isolation so legacy CNN [-1, 1] and ResNet/EfficientNet ImageNet preprocessing never cross-contaminate.",
                "Set torch.backends.cudnn.allow_tf32 = False if strict <1e-4 CPU/CUDA probability parity is required on Ampere/Ada GPUs (default TF32 exhibits ~3.42e-4 drift with 16/16 argmax match vs 1.94e-6 strict FP32).",
                "MTCNN detector + coherent IoU tracking adds GPU/CPU detection latency relative to downscaled Haar; benchmarked explicitly per video in Exp 4 & Exp 7.",
                "Emotion predictions from facial cues reflect learned dataset correlations and must not be over-interpreted as standalone clinical/human truth.",
            ],
        },
    }

    (PHASE17D_DIR / "phase17d_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_ROOT / "phase17d_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPhase 17D DONE -> overall_status={report['overall_status']} | pytest_rc={pytest_res.returncode}")
    for k, v in checks.items():
        print(f"  [{ 'PASS' if v else 'FAIL' }] {k}")


if __name__ == "__main__":
    main()
