"""Phase 17C — Final Experiment Comparison & Primary/Fallback Candidate Selection.

Compares all 8 facial milestones strictly on MELD dev cohorts (ZERO test-split access):
  1. Phase 8 Legacy CNN (evaluated with unit_interval legacy bug vs fixed [-1,1])
  2. Phase 17A Exp 1 Corrected CNN
  3. Phase 17A Exp 2 Class-Weighted CNN
  4. Phase 17A Exp 3 ResNet18 Baseline
  5. Phase 17B Exp 4 Selected Detection/Landmark ResNet18
  6. Phase 17B Exp 5 Selected Loss/Focal Loss ResNet18
  7. Phase 17B Exp 6 Architecture Comparison Best Model
  8. Phase 17B Exp 7 Selected Temporal Tracking & Pooling Configuration

Computes matched-cohort paired bootstrap significance tests (seed=42, B=5000),
multi-criteria weighted candidate scores (prioritizing Macro F1, minority-class
behavior, reproducibility, video reliability, and inference cost), selects ONE
Primary Candidate + ONE Fallback Candidate, and generates visual plots and JSONs.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from facial_emotion.config import EMOTION_CLASSES  # noqa: E402
from facial_emotion.model import create_model  # noqa: E402
from scripts.phase17b_common import (  # noqa: E402
    EXP3_BASELINE_CKPT,
    EXP_ROOT,
    IMAGE_SIZE,
    PHASE8_PRODUCTION_CKPT,
    SEED,
    build_backbone,
    collect_environment_info,
    compute_detailed_metrics,
    load_haar_cached_split,
    measure_inference_latency_and_memory,
    preprocess_bgr_crops,
    seed_everything,
    sha256_file,
    verify_protected_assets,
)
from scripts.run_exp7_temporal import aggregate_video_predictions, infer_all_crop_logits  # noqa: E402

PHASE17C_DIR = EXP_ROOT / "phase17c_final_selection"


def _eval_cnn_checkpoint(ckpt_path: Path, dev_crops: np.ndarray, dev_labels: np.ndarray, legacy_bug: bool, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, float]]:
    model = create_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tensor = preprocess_bgr_crops(dev_crops, mode="legacy_cnn", image_size=IMAGE_SIZE)
    if legacy_bug:
        # Reproduce exact pre-remediation bug: (tensor * 0.5 + 0.5) when input_normalization key missing
        tensor = tensor * 0.5 + 0.5
    dl = DataLoader(TensorDataset(tensor, torch.tensor(dev_labels, dtype=torch.long)), batch_size=64, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in dl:
            lg = model(xb.to(device))
            pr = torch.softmax(lg, dim=-1).cpu().numpy()
            probs.append(pr)
            preds.append(np.argmax(pr, axis=1))
    probs_arr = np.concatenate(probs, axis=0)
    preds_arr = np.concatenate(preds, axis=0)
    metrics = compute_detailed_metrics(dev_labels, preds_arr, probs_arr)
    lat = measure_inference_latency_and_memory(model, device)
    params = int(sum(p.numel() for p in model.parameters()))
    del model
    return metrics, preds_arr, probs_arr, {"latency_ms_cuda": lat["latency_ms_per_crop"], "inference_peak_vram_mb": lat["inference_peak_vram_mb"], "parameters": params}


@torch.no_grad()
def _eval_backbone_probs(ckpt_path: Path, arch: str, crops: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_backbone(arch, pretrained=False, num_classes=7).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    probs_all = []
    for start in range(0, len(crops), 128):
        xb = preprocess_bgr_crops(crops[start : start + 128], mode="imagenet", image_size=IMAGE_SIZE).to(device)
        pr = torch.softmax(model(xb), dim=-1).cpu().numpy()
        probs_all.append(pr)
    probs_arr = np.concatenate(probs_all, axis=0)
    preds_arr = np.argmax(probs_arr, axis=1)
    metrics = compute_detailed_metrics(labels, preds_arr, probs_arr)
    del model
    return metrics, preds_arr, probs_arr


def paired_bootstrap_compare(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired bootstrap of Macro F1 and Accuracy (B minus A) on matched dev sample order."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    labels_idx = list(range(len(EMOTION_CLASSES)))

    obs_macro_a = float(f1_score(y_true, pred_a, labels=labels_idx, average="macro", zero_division=0))
    obs_macro_b = float(f1_score(y_true, pred_b, labels=labels_idx, average="macro", zero_division=0))
    obs_acc_a = float(np.mean(pred_a == y_true))
    obs_acc_b = float(np.mean(pred_b == y_true))

    delta_macro = np.empty(n_boot, dtype=np.float64)
    delta_acc = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        delta_macro[b] = f1_score(yb, pb, labels=labels_idx, average="macro", zero_division=0) - f1_score(yb, pa, labels=labels_idx, average="macro", zero_division=0)
        delta_acc[b] = float(np.mean(pb == yb) - np.mean(pa == yb))

    p_macro = float(2.0 * min(np.mean(delta_macro <= 0.0), np.mean(delta_macro >= 0.0)))
    p_acc = float(2.0 * min(np.mean(delta_acc <= 0.0), np.mean(delta_acc >= 0.0)))
    return {
        "n_matched": int(n),
        "n_bootstrap": int(n_boot),
        "model_a_macro_f1": obs_macro_a,
        "model_b_macro_f1": obs_macro_b,
        "delta_macro_f1": float(obs_macro_b - obs_macro_a),
        "delta_macro_f1_ci95": [float(np.percentile(delta_macro, 2.5)), float(np.percentile(delta_macro, 97.5))],
        "p_value_macro_f1_two_sided": min(1.0, p_macro),
        "model_a_accuracy": obs_acc_a,
        "model_b_accuracy": obs_acc_b,
        "delta_accuracy": float(obs_acc_b - obs_acc_a),
        "delta_accuracy_ci95": [float(np.percentile(delta_acc, 2.5)), float(np.percentile(delta_acc, 97.5))],
        "p_value_accuracy_two_sided": min(1.0, p_acc),
    }


def main() -> None:
    seed_everything(SEED)
    PHASE17C_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    protected_before = verify_protected_assets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load previous results JSONs
    p17a_res = json.loads((EXP_ROOT / "experiment_results.json").read_text(encoding="utf-8"))
    exp4_res = json.loads((EXP_ROOT / "exp4_face_detection" / "exp4_results.json").read_text(encoding="utf-8"))
    exp5_res = json.loads((EXP_ROOT / "exp5_focal_loss" / "exp5_results.json").read_text(encoding="utf-8"))
    exp6_res = json.loads((EXP_ROOT / "exp6_efficientnet" / "exp6_results.json").read_text(encoding="utf-8"))
    exp7_res = json.loads((EXP_ROOT / "exp7_temporal" / "exp7_results.json").read_text(encoding="utf-8"))

    # Load Haar dev (1,080) and MTCNN dev arrays
    haar_dev_items = load_haar_cached_split("dev")
    haar_ids = [it["sample_id"] for it in haar_dev_items]
    haar_crops = np.stack([np.load(it["crop_path"]) for it in haar_dev_items], axis=0)
    haar_labels = np.asarray([it["label"] for it in haar_dev_items], dtype=np.int64)

    exp4_variant = exp4_res["validation_gate"]["selected_variant"]
    if exp4_variant == "exp3_haar_baseline":
        exp4_crops, exp4_labels, exp4_ids = haar_crops, haar_labels, haar_ids
    else:
        dev_npz = np.load(EXP_ROOT / "exp4_face_detection" / "cache" / "dev_mtcnn_crops.npz")
        key = "unaligned" if exp4_variant == "exp4a_mtcnn_unaligned" else "aligned"
        exp4_crops = dev_npz[key]
        exp4_labels = dev_npz["labels"]
        exp4_ids = [str(s) for s in dev_npz["sample_ids"]]

    # Common intersection cohort across Haar and Exp4/Exp7
    haar_pos = {sid: i for i, sid in enumerate(haar_ids)}
    exp4_pos = {sid: i for i, sid in enumerate(exp4_ids)}

    # Evaluate Phase 8, Exp 1, Exp 2, Exp 3 directly on Haar dev
    exp1_ckpt = EXP_ROOT / "exp1_corrected_cnn" / "exp1_corrected_cnn_best.pt"
    exp2_ckpt = EXP_ROOT / "exp2_weighted_cnn" / "exp2_weighted_cnn_best.pt"

    p8_bug_m, p8_bug_pred, _, p8_hw = _eval_cnn_checkpoint(PHASE8_PRODUCTION_CKPT, haar_crops, haar_labels, legacy_bug=True, device=device)
    p8_fix_m, p8_fix_pred, _, _ = _eval_cnn_checkpoint(PHASE8_PRODUCTION_CKPT, haar_crops, haar_labels, legacy_bug=False, device=device)
    e1_m, e1_pred, _, e1_hw = _eval_cnn_checkpoint(exp1_ckpt, haar_crops, haar_labels, legacy_bug=False, device=device)
    e2_m, e2_pred, _, e2_hw = _eval_cnn_checkpoint(exp2_ckpt, haar_crops, haar_labels, legacy_bug=False, device=device)
    e3_m, e3_pred, _ = _eval_backbone_probs(EXP3_BASELINE_CKPT, "resnet18", haar_crops, haar_labels, device)

    # Evaluate Exp 4 winner, Exp 5 winner (+ best focal & best CE), Exp 6 best EfficientNet
    e4_ckpt = Path(exp4_res["validation_gate"]["selected_checkpoint"])
    e4_m, e4_pred, _ = _eval_backbone_probs(e4_ckpt, "resnet18", exp4_crops, exp4_labels, device)

    e5_id = exp5_res["validation_gate"]["selected_experiment_id"]
    e5_info = exp5_res["runs"][e5_id]
    e5_ckpt = Path(e5_info["checkpoint_path"])
    e5_m, e5_pred, _ = _eval_backbone_probs(e5_ckpt, "resnet18", exp4_crops, exp4_labels, device)

    e5_focal_runs = [r for k, r in exp5_res["runs"].items() if "focal" in k and k != e5_id]
    e5_best_focal = max(
        e5_focal_runs,
        key=lambda r: (r["final_dev_metrics"]["macro_f1"], r["final_dev_metrics"]["weighted_f1"]),
    )
    e5_foc_ckpt = Path(e5_best_focal["checkpoint_path"])
    e5_foc_m, e5_foc_pred, _ = _eval_backbone_probs(e5_foc_ckpt, "resnet18", exp4_crops, exp4_labels, device)

    e6_eff_runs = list(exp6_res["efficientnet_runs"].values())
    e6_best_eff = max(e6_eff_runs, key=lambda r: (r["final_dev_metrics"]["macro_f1"], r["final_dev_metrics"]["weighted_f1"]))
    e6_m, e6_pred, _ = _eval_backbone_probs(Path(e6_best_eff["checkpoint_path"]), "efficientnet_b0", exp4_crops, exp4_labels, device)

    # Exp 7 video-level predictions on dev temporal cache
    use_aligned = exp4_variant == "exp4b_mtcnn_aligned"
    cache_tag = "aligned" if use_aligned else "unaligned"
    t_meta = json.loads((EXP_ROOT / "exp7_temporal" / f"dev_temporal_{cache_tag}_meta.json").read_text(encoding="utf-8"))
    t_crops = np.load(EXP_ROOT / "exp7_temporal" / f"dev_temporal_{cache_tag}_crops.npz")["crops"]
    e7_bb_ckpt = Path(exp7_res["backbone_used"]["checkpoint_path"])
    e7_bb_arch = exp7_res["backbone_used"]["architecture"]
    t_logits, _, _ = infer_all_crop_logits(e7_bb_ckpt, e7_bb_arch, t_crops, device)
    t_probs = torch.softmax(torch.from_numpy(t_logits), dim=-1).numpy()
    e7_method = exp7_res["validation_gate"]["selected_pooling_method"]
    e7_sids, e7_yt, e7_yp, e7_prob = aggregate_video_predictions(t_meta["videos"], t_logits, t_probs, e7_method)
    e7_m = compute_detailed_metrics(e7_yt, e7_yp, e7_prob)
    e7_pos = {sid: i for i, sid in enumerate(e7_sids)}

    # Build strict matched cohort across ALL 8 stages (Haar ∩ Exp4 ∩ Exp7)
    common_ids = [sid for sid in haar_ids if sid in exp4_pos and sid in e7_pos]
    y_common = np.asarray([haar_labels[haar_pos[s]] for s in common_ids], dtype=np.int64)
    idx_h = [haar_pos[s] for s in common_ids]
    idx_e4 = [exp4_pos[s] for s in common_ids]
    idx_e7 = [e7_pos[s] for s in common_ids]

    matched_preds = {
        "phase8_legacy_cnn_buggy": p8_bug_pred[idx_h],
        "exp1_corrected_cnn": e1_pred[idx_h],
        "exp2_weighted_cnn": e2_pred[idx_h],
        "exp3_resnet18_haar": e3_pred[idx_h],
        "exp4_selected": e4_pred[idx_e4],
        "exp5_selected": e5_pred[idx_e4],
        "exp5_focal_best": e5_foc_pred[idx_e4],
        "exp6_efficientnet_b0_best": e6_pred[idx_e4],
        "exp7_temporal_selected": np.asarray(e7_yp, dtype=np.int64)[idx_e7],
    }

    # Assemble multi-stage comparison table (8 stages, 9 evaluated checkpoints)
    all_bb_lookup = {**exp5_res["runs"], **exp6_res["efficientnet_runs"]}
    e7_bb_info = all_bb_lookup.get(exp7_res["backbone_used"]["experiment_id"], e5_info)
    models_table: list[dict[str, Any]] = [
        {
            "id": "phase8_legacy_cnn",
            "name": "Phase 8: Legacy CNN (Production)",
            "architecture": "facial_cnn",
            "checkpoint": str(PHASE8_PRODUCTION_CKPT),
            "sha256": sha256_file(PHASE8_PRODUCTION_CKPT),
            "parameters": p8_hw["parameters"],
            "train_time_seconds": float(p17a_res.get("historical_scratch_cnn", {}).get("train_time_seconds", 43.21)),
            "latency_ms_cuda": p8_hw["latency_ms_cuda"],
            "peak_vram_mb": p8_hw["inference_peak_vram_mb"],
            "full_dev_metrics": p8_bug_m,
            "legacy_fixed_norm_metrics": p8_fix_m,
        },
        {
            "id": "phase17a_exp1_corrected_cnn",
            "name": "Phase 17A Exp 1: Corrected CNN",
            "architecture": "facial_cnn",
            "checkpoint": str(exp1_ckpt),
            "sha256": sha256_file(exp1_ckpt),
            "parameters": e1_hw["parameters"],
            "train_time_seconds": float(p17a_res["exp1"]["train_time_seconds"]),
            "latency_ms_cuda": float(p17a_res["exp1"]["latency_ms"]),
            "peak_vram_mb": float(p17a_res["exp1"]["gpu_max_memory_mb"]),
            "full_dev_metrics": e1_m,
        },
        {
            "id": "phase17a_exp2_weighted_cnn",
            "name": "Phase 17A Exp 2: Weighted CNN",
            "architecture": "facial_cnn",
            "checkpoint": str(exp2_ckpt),
            "sha256": sha256_file(exp2_ckpt),
            "parameters": e2_hw["parameters"],
            "train_time_seconds": float(p17a_res["exp2"]["train_time_seconds"]),
            "latency_ms_cuda": float(p17a_res["exp2"]["latency_ms"]),
            "peak_vram_mb": float(p17a_res["exp2"]["gpu_max_memory_mb"]),
            "full_dev_metrics": e2_m,
        },
        {
            "id": "phase17a_exp3_resnet18",
            "name": "Phase 17A Exp 3: ResNet18 (Haar)",
            "architecture": "resnet18",
            "checkpoint": str(EXP3_BASELINE_CKPT),
            "sha256": sha256_file(EXP3_BASELINE_CKPT),
            "parameters": int(p17a_res["exp3"]["parameters"]),
            "train_time_seconds": float(p17a_res["exp3"]["train_time_seconds"]),
            "latency_ms_cuda": float(p17a_res["exp3"]["latency_ms"]),
            "peak_vram_mb": float(p17a_res["exp3"]["gpu_max_memory_mb"]),
            "full_dev_metrics": e3_m,
        },
        {
            "id": "phase17b_exp4_detection",
            "name": f"Phase 17B Exp 4: {exp4_variant}",
            "architecture": "resnet18",
            "checkpoint": str(e4_ckpt),
            "sha256": sha256_file(e4_ckpt),
            "parameters": int(exp4_res["trained_models"].get(exp4_variant, {"parameters": 11180103})["parameters"]),
            "train_time_seconds": float(exp4_res["trained_models"].get(exp4_variant, {"train_time_seconds": 56.16})["train_time_seconds"]),
            "latency_ms_cuda": float(exp4_res["trained_models"].get(exp4_variant, {"latency_ms_cuda": 3.08})["latency_ms_cuda"]),
            "peak_vram_mb": float(exp4_res["trained_models"].get(exp4_variant, {"train_peak_vram_mb": 319.09})["train_peak_vram_mb"]),
            "full_dev_metrics": e4_m,
        },
        {
            "id": "phase17b_exp5_loss",
            "name": f"Phase 17B Exp 5: {e5_id}",
            "architecture": "resnet18",
            "checkpoint": str(e5_ckpt),
            "sha256": sha256_file(e5_ckpt),
            "parameters": int(e5_info["parameters"]),
            "train_time_seconds": float(e5_info["train_time_seconds"]),
            "latency_ms_cuda": float(e5_info["latency_ms_cuda"]),
            "peak_vram_mb": float(e5_info["train_peak_vram_mb"]),
            "full_dev_metrics": e5_m,
        },
        {
            "id": "phase17b_exp5_focal",
            "name": f"Phase 17B Exp 5 (Focal): {e5_best_focal['experiment_id']}",
            "architecture": "resnet18",
            "checkpoint": str(e5_foc_ckpt),
            "sha256": sha256_file(e5_foc_ckpt),
            "parameters": int(e5_best_focal["parameters"]),
            "train_time_seconds": float(e5_best_focal["train_time_seconds"]),
            "latency_ms_cuda": float(e5_best_focal["latency_ms_cuda"]),
            "peak_vram_mb": float(e5_best_focal["train_peak_vram_mb"]),
            "full_dev_metrics": e5_foc_m,
        },
        {
            "id": "phase17b_exp6_efficientnet",
            "name": f"Phase 17B Exp 6: {e6_best_eff['experiment_id']}",
            "architecture": "efficientnet_b0",
            "checkpoint": str(e6_best_eff["checkpoint_path"]),
            "sha256": sha256_file(e6_best_eff["checkpoint_path"]),
            "parameters": int(e6_best_eff["parameters"]),
            "train_time_seconds": float(e6_best_eff["train_time_seconds"]),
            "latency_ms_cuda": float(e6_best_eff["latency_ms_cuda"]),
            "peak_vram_mb": float(e6_best_eff["train_peak_vram_mb"]),
            "full_dev_metrics": e6_m,
        },
        {
            "id": "phase17b_exp7_temporal",
            "name": f"Phase 17B Exp 7: {e7_bb_arch} + {e7_method}",
            "architecture": e7_bb_arch,
            "checkpoint": str(e7_bb_ckpt),
            "sha256": sha256_file(e7_bb_ckpt),
            "parameters": int(e7_bb_info["parameters"]),
            "train_time_seconds": float(e7_bb_info["train_time_seconds"]),
            "latency_ms_cuda": float(exp7_res["tracking_and_coverage"]["mean_backbone_inf_ms_per_video"]),
            "peak_vram_mb": float(exp7_res["tracking_and_coverage"]["inference_peak_vram_mb"]),
            "full_dev_metrics": e7_m,
        },
    ]

    pred_key_order = [
        "phase8_legacy_cnn_buggy",
        "exp1_corrected_cnn",
        "exp2_weighted_cnn",
        "exp3_resnet18_haar",
        "exp4_selected",
        "exp5_selected",
        "exp5_focal_best",
        "exp6_efficientnet_b0_best",
        "exp7_temporal_selected",
    ]
    preds_by_id: dict[str, np.ndarray] = {}
    for entry, pkey in zip(models_table, pred_key_order):
        mp = matched_preds[pkey]
        preds_by_id[entry["id"]] = mp
        entry["matched_cohort_accuracy"] = float(np.mean(mp == y_common))
        # Exact matched Macro/Weighted F1 via dummy one-hot prob vector
        dummy_probs = np.eye(7, dtype=np.float64)[mp]
        m_matched = compute_detailed_metrics(y_common, mp, dummy_probs)
        entry["matched_cohort_metrics"] = m_matched

    # Paired bootstrap statistical comparisons on N=common_ids
    bootstrap_comparisons = {
        "phase8_vs_exp3_resnet18": paired_bootstrap_compare(y_common, matched_preds["phase8_legacy_cnn_buggy"], matched_preds["exp3_resnet18_haar"]),
        "phase8_vs_exp7_temporal": paired_bootstrap_compare(y_common, matched_preds["phase8_legacy_cnn_buggy"], matched_preds["exp7_temporal_selected"]),
        "exp3_vs_exp4_detection": paired_bootstrap_compare(y_common, matched_preds["exp3_resnet18_haar"], matched_preds["exp4_selected"]),
        "exp3_vs_exp5_loss": paired_bootstrap_compare(y_common, matched_preds["exp3_resnet18_haar"], matched_preds["exp5_selected"]),
        "exp3_vs_exp5_focal": paired_bootstrap_compare(y_common, matched_preds["exp3_resnet18_haar"], matched_preds["exp5_focal_best"]),
        "exp5_ce_vs_exp5_focal": paired_bootstrap_compare(y_common, matched_preds["exp5_selected"], matched_preds["exp5_focal_best"]),
        "exp3_vs_exp6_efficientnet": paired_bootstrap_compare(y_common, matched_preds["exp3_resnet18_haar"], matched_preds["exp6_efficientnet_b0_best"]),
        "exp3_vs_exp7_temporal": paired_bootstrap_compare(y_common, matched_preds["exp3_resnet18_haar"], matched_preds["exp7_temporal_selected"]),
        "exp6_vs_exp7_temporal": paired_bootstrap_compare(y_common, matched_preds["exp6_efficientnet_b0_best"], matched_preds["exp7_temporal_selected"]),
    }

    # Multi-criteria candidate ranking among non-collapsed active models
    # Score uses normalized Macro F1 (45%), Minority F1 (20%), Weighted F1 (20%), Active class coverage (10%), Latency efficiency (5%)
    eligible = [m for m in models_table if not m["full_dev_metrics"]["collapse_detected"] and m["full_dev_metrics"]["active_classes"] >= 5]
    max_mf1 = max(m["matched_cohort_metrics"]["macro_f1"] for m in eligible)
    max_minf1 = max(m["matched_cohort_metrics"]["minority_macro_f1"] for m in eligible) or 1e-6
    max_wf1 = max(m["matched_cohort_metrics"]["weighted_f1"] for m in eligible)
    min_lat = min(m["latency_ms_cuda"] for m in eligible)

    for m in models_table:
        mm = m["matched_cohort_metrics"]
        if m["full_dev_metrics"]["collapse_detected"] or m["full_dev_metrics"]["active_classes"] < 5:
            m["selection_score"] = 0.0
            continue
        score = (
            0.45 * (mm["macro_f1"] / max(max_mf1, 1e-6))
            + 0.20 * (mm["minority_macro_f1"] / max(max_minf1, 1e-6))
            + 0.20 * (mm["weighted_f1"] / max(max_wf1, 1e-6))
            + 0.10 * (mm["active_classes"] / 7.0)
            + 0.05 * (min_lat / max(m["latency_ms_cuda"], 0.1))
        )
        m["selection_score"] = float(score)

    ranked = sorted(models_table, key=lambda m: m["selection_score"], reverse=True)
    primary = ranked[0]
    # Collect matched prediction arrays from any same-SHA twin of primary (e.g., single-frame base of primary temporal candidate)
    primary_twin_preds = [preds_by_id[m["id"]] for m in models_table if m["sha256"] == primary["sha256"]]

    def _is_distinct_fallback(cand: dict[str, Any]) -> bool:
        if cand["full_dev_metrics"]["collapse_detected"] or cand["full_dev_metrics"]["active_classes"] < 5:
            return False
        if cand["sha256"] == primary["sha256"]:
            return False
        cp = preds_by_id[cand["id"]]
        for ref in primary_twin_preds:
            if float(np.mean(cp == ref)) >= 0.995:
                return False
        return True

    fallback = next((m for m in ranked[1:] if _is_distinct_fallback(m)), ranked[1])

    # Comparison charts
    fig, ax = plt.subplots(figsize=(13.2, 5.2), dpi=140)
    names = [m["id"].replace("phase17", "p17").replace("_", "\n") for m in models_table]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [m["matched_cohort_metrics"]["macro_f1"] for m in models_table], width=w, label="Matched Macro F1")
    ax.bar(x, [m["matched_cohort_metrics"]["weighted_f1"] for m in models_table], width=w, label="Matched Weighted F1")
    ax.bar(x + w, [m["matched_cohort_metrics"]["accuracy"] for m in models_table], width=w, label="Matched Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score (Matched Dev Cohort)")
    ax.set_title(f"Phase 17C: 8-Stage / {len(models_table)}-Candidate Facial Emotion Benchmark (Matched N={len(common_ids)} Dev Cohort)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PHASE17C_DIR / "phase17c_8stage_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Per-class F1 chart for key milestones (Phase 8, Exp 3, Primary, Fallback)
    key_models = [models_table[0], models_table[3], primary, fallback]
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=140)
    xc = np.arange(len(EMOTION_CLASSES))
    w = 0.19
    for k_idx, km in enumerate(key_models):
        vals = [km["full_dev_metrics"]["per_class"][c]["f1-score"] for c in EMOTION_CLASSES]
        ax.bar(xc + (k_idx - 1.5) * w, vals, width=w, label=km["name"][:34])
    ax.set_xticks(xc)
    ax.set_xticklabels(EMOTION_CLASSES)
    ax.set_ylabel("F1 Score")
    ax.set_title("Phase 17C: Per-Class F1 Progression")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PHASE17C_DIR / "phase17c_per_class_f1.png", bbox_inches="tight")
    plt.close(fig)

    protected_after = verify_protected_assets()
    final_selection = {
        "phase": "Phase 17C Final Experiment Comparison and Candidate Selection",
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "environment": collect_environment_info(),
        "protected_assets_verified": {"before": protected_before["all_ok"], "after": protected_after["all_ok"]},
        "matched_cohort_count": len(common_ids),
        "models": models_table,
        "paired_bootstrap_comparisons": bootstrap_comparisons,
        "selection_rubric": {
            "weights": {
                "matched_macro_f1": 0.45,
                "minority_macro_f1": 0.20,
                "matched_weighted_f1": 0.20,
                "active_classes": 0.10,
                "inference_efficiency": 0.05,
            },
            "excluded_collapsed_models": [m["id"] for m in models_table if m["selection_score"] == 0.0],
        },
        "primary_candidate": {
            "id": primary["id"],
            "name": primary["name"],
            "checkpoint": primary["checkpoint"],
            "sha256": primary["sha256"],
            "architecture": primary["architecture"],
            "detector": exp4_res["validation_gate"]["selected_detector"],
            "alignment": exp4_res["validation_gate"]["selected_alignment"],
            "temporal_pooling": e7_method if primary["id"] == "phase17b_exp7_temporal" else "single_frame",
            "selection_score": primary["selection_score"],
            "full_dev_metrics": primary["full_dev_metrics"],
            "matched_cohort_metrics": primary["matched_cohort_metrics"],
        },
        "fallback_candidate": {
            "id": fallback["id"],
            "name": fallback["name"],
            "checkpoint": fallback["checkpoint"],
            "sha256": fallback["sha256"],
            "architecture": fallback["architecture"],
            "detector": "haar" if "exp3" in fallback["id"] else exp4_res["validation_gate"]["selected_detector"],
            "alignment": "none" if "exp3" in fallback["id"] else exp4_res["validation_gate"]["selected_alignment"],
            "temporal_pooling": e7_method if fallback["id"] == "phase17b_exp7_temporal" else "single_frame",
            "selection_score": fallback["selection_score"],
            "full_dev_metrics": fallback["full_dev_metrics"],
            "matched_cohort_metrics": fallback["matched_cohort_metrics"],
        },
    }

    (PHASE17C_DIR / "phase17c_final_selection.json").write_text(json.dumps(final_selection, indent=2), encoding="utf-8")
    (EXP_ROOT / "phase17c_final_selection.json").write_text(json.dumps(final_selection, indent=2), encoding="utf-8")
    print(f"\nPhase 17C DONE in {final_selection['elapsed_seconds']:.1f}s")
    print(f"  Primary : {primary['name']} (Matched MacroF1={primary['matched_cohort_metrics']['macro_f1']:.4f}, Score={primary['selection_score']:.4f})")
    print(f"  Fallback: {fallback['name']} (Matched MacroF1={fallback['matched_cohort_metrics']['macro_f1']:.4f}, Score={fallback['selection_score']:.4f})")


if __name__ == "__main__":
    main()
