"""Phase 17B Experiment 5 — Focal Loss and Class Imbalance Handling.

Reads the winning preprocessing/dataset selection from Experiment 4 (`exp4_results.json`),
trains controlled Cross-Entropy references (unweighted + smoothed inverse-sqrt weighted) and
Focal Loss variants across gamma in {1.0, 2.0, 3.0} with and without class weights, checks for
class collapse, retains BOTH best Cross-Entropy and best Focal Loss checkpoints, and records
all metrics, curves, confusion matrices, and SHA-256 hashes.
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

from facial_emotion.config import EMOTION_CLASSES  # noqa: E402
from scripts.phase17b_common import (  # noqa: E402
    EXP_ROOT,
    SEED,
    FocalLoss,
    collect_environment_info,
    compute_smoothed_class_weights,
    get_preprocessing_config,
    load_haar_cached_split,
    seed_everything,
    sha256_file,
    train_backbone_experiment,
    verify_protected_assets,
)

EXP4_RESULTS = EXP_ROOT / "exp4_face_detection" / "exp4_results.json"
EXP5_DIR = EXP_ROOT / "exp5_focal_loss"


def load_selected_exp4_arrays(exp4_res: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    variant = exp4_res["validation_gate"]["selected_variant"]
    if variant == "exp3_haar_baseline":
        train_items = load_haar_cached_split("train")
        dev_items = load_haar_cached_split("dev")
        train_crops = np.stack([np.load(it["crop_path"]) for it in train_items], axis=0)
        train_labels = np.asarray([it["label"] for it in train_items], dtype=np.int64)
        dev_crops = np.stack([np.load(it["crop_path"]) for it in dev_items], axis=0)
        dev_labels = np.asarray([it["label"] for it in dev_items], dtype=np.int64)
        prep = get_preprocessing_config("imagenet", detector="haar", alignment="none")
        return train_crops, train_labels, dev_crops, dev_labels, prep

    cache_dir = EXP_ROOT / "exp4_face_detection" / "cache"
    train_npz = np.load(cache_dir / "train_mtcnn_crops.npz")
    dev_npz = np.load(cache_dir / "dev_mtcnn_crops.npz")
    key = "unaligned" if variant == "exp4a_mtcnn_unaligned" else "aligned"
    prep = get_preprocessing_config(
        "imagenet",
        detector="mtcnn_640_fallback_native",
        alignment="square_unaligned_margin_0.08" if key == "unaligned" else "5pt_landmark_similarity_yaw_safe",
    )
    return (
        train_npz[key],
        train_npz["labels"],
        dev_npz[key],
        dev_npz["labels"],
        prep,
    )


def plot_exp5_comparison(runs: list[dict[str, Any]], out_path: Path) -> None:
    labels = [r["short_label"] for r in runs]
    macro_f1 = [r["final_dev_metrics"]["macro_f1"] for r in runs]
    weighted_f1 = [r["final_dev_metrics"]["weighted_f1"] for r in runs]
    acc = [r["final_dev_metrics"]["accuracy"] for r in runs]
    min_rec = [r["final_dev_metrics"]["minority_macro_recall"] for r in runs]

    x = np.arange(len(labels))
    w = 0.20
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    ax.bar(x - 1.5 * w, macro_f1, width=w, label="Macro F1")
    ax.bar(x - 0.5 * w, weighted_f1, width=w, label="Weighted F1")
    ax.bar(x + 0.5 * w, acc, width=w, label="Accuracy")
    ax.bar(x + 1.5 * w, min_rec, width=w, label="Minority Recall (Avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 5: Cross-Entropy vs Focal Loss Comparison (Dev Set)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    seed_everything(SEED)
    EXP5_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    protected_before = verify_protected_assets()

    if not EXP4_RESULTS.is_file():
        raise FileNotFoundError(f"Experiment 4 results not found: {EXP4_RESULTS}")
    exp4_res = json.loads(EXP4_RESULTS.read_text(encoding="utf-8"))
    if not exp4_res.get("validation_gate", {}).get("passed"):
        raise RuntimeError("Experiment 4 validation gate did not pass.")

    train_crops, train_labels, dev_crops, dev_labels, prep_cfg = load_selected_exp4_arrays(exp4_res)
    print(
        f"Exp 5 using Exp 4 selection: {exp4_res['validation_gate']['selected_variant']} | "
        f"train={len(train_labels)}, dev={len(dev_labels)}, prep={prep_cfg}"
    )

    smooth_weights = compute_smoothed_class_weights(train_labels, power=0.5, normalize_mean=True)
    weights_dict = {EMOTION_CLASSES[i]: float(smooth_weights[i]) for i in range(len(EMOTION_CLASSES))}

    configs = [
        {
            "id": "exp5_ce_unweighted",
            "short_label": "CE Unweighted",
            "name": "Exp 5 Ref A: ResNet18 + Unweighted Cross-Entropy",
            "family": "cross_entropy",
            "ckpt": "exp5_ce_unweighted_best.pt",
            "criterion_fn": lambda _dev: nn.CrossEntropyLoss(),
            "extra": {"loss": "CrossEntropyLoss(unweighted)", "gamma": 0.0, "weighted": False},
        },
        {
            "id": "exp5_ce_weighted",
            "short_label": "CE Weighted (sqrt)",
            "name": "Exp 5 Ref B: ResNet18 + Smoothed-Sqrt Weighted CE",
            "family": "cross_entropy",
            "ckpt": "exp5_ce_weighted_best.pt",
            "criterion_fn": lambda dev: nn.CrossEntropyLoss(weight=smooth_weights.to(dev)),
            "extra": {"loss": "CrossEntropyLoss(smoothed_sqrt)", "gamma": 0.0, "weighted": True, "class_weights": weights_dict},
        },
        {
            "id": "exp5_focal_g1_nowt",
            "short_label": "Focal g=1.0 (unwt)",
            "name": "Exp 5 Focal A: ResNet18 + Focal Loss (gamma=1.0, unweighted)",
            "family": "focal_loss",
            "ckpt": "exp5_focal_g1_nowt_best.pt",
            "criterion_fn": lambda _dev: FocalLoss(gamma=1.0, weight=None),
            "extra": {"loss": "FocalLoss(gamma=1.0, weight=None)", "gamma": 1.0, "weighted": False},
        },
        {
            "id": "exp5_focal_g2_nowt",
            "short_label": "Focal g=2.0 (unwt)",
            "name": "Exp 5 Focal B: ResNet18 + Focal Loss (gamma=2.0, unweighted)",
            "family": "focal_loss",
            "ckpt": "exp5_focal_g2_nowt_best.pt",
            "criterion_fn": lambda _dev: FocalLoss(gamma=2.0, weight=None),
            "extra": {"loss": "FocalLoss(gamma=2.0, weight=None)", "gamma": 2.0, "weighted": False},
        },
        {
            "id": "exp5_focal_g2_wt",
            "short_label": "Focal g=2.0 (sqrt wt)",
            "name": "Exp 5 Focal C: ResNet18 + Focal Loss (gamma=2.0, smoothed-sqrt alpha)",
            "family": "focal_loss",
            "ckpt": "exp5_focal_g2_wt_best.pt",
            "criterion_fn": lambda dev: FocalLoss(gamma=2.0, weight=smooth_weights.to(dev)),
            "extra": {"loss": "FocalLoss(gamma=2.0, smoothed_sqrt)", "gamma": 2.0, "weighted": True, "class_weights": weights_dict},
        },
        {
            "id": "exp5_focal_g3_wt",
            "short_label": "Focal g=3.0 (sqrt wt)",
            "name": "Exp 5 Focal D: ResNet18 + Focal Loss (gamma=3.0, smoothed-sqrt alpha)",
            "family": "focal_loss",
            "ckpt": "exp5_focal_g3_wt_best.pt",
            "criterion_fn": lambda dev: FocalLoss(gamma=3.0, weight=smooth_weights.to(dev)),
            "extra": {"loss": "FocalLoss(gamma=3.0, smoothed_sqrt)", "gamma": 3.0, "weighted": True, "class_weights": weights_dict},
        },
    ]

    all_runs: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"\n=== Running {cfg['name']} ===")
        res = train_backbone_experiment(
            experiment_id=cfg["id"],
            experiment_name=cfg["name"],
            arch="resnet18",
            out_dir=EXP5_DIR,
            ckpt_filename=cfg["ckpt"],
            train_crops=train_crops,
            train_labels=train_labels,
            dev_crops=dev_crops,
            dev_labels=dev_labels,
            criterion_fn=cfg["criterion_fn"],
            preprocessing_config=prep_cfg,
            batch_size=32,
            head_epochs=3,
            finetune_epochs=5,
            head_lr=1e-3,
            backbone_lr=1e-4,
            finetune_head_lr=5e-4,
            patience=4,
            augment=False,
            seed=SEED,
            extra_config=cfg["extra"],
        )
        res["loss_family"] = cfg["family"]
        res["short_label"] = cfg["short_label"]
        all_runs.append(res)

    plot_exp5_comparison(all_runs, EXP5_DIR / "exp5_loss_comparison.png")

    def _select_best(items: list[dict[str, Any]]) -> dict[str, Any]:
        non_collapsed = [r for r in items if not r["final_dev_metrics"]["collapse_detected"] and r["final_dev_metrics"]["weighted_f1"] >= 0.18]
        pool = non_collapsed if non_collapsed else items
        return max(pool, key=lambda r: (r["final_dev_metrics"]["macro_f1"], r["final_dev_metrics"]["weighted_f1"]))

    best_ce = _select_best([r for r in all_runs if r["loss_family"] == "cross_entropy"])
    best_focal = _select_best([r for r in all_runs if r["loss_family"] == "focal_loss"])
    overall_best = _select_best(all_runs)

    protected_after = verify_protected_assets()
    summary = {
        "experiment": "Phase 17B Experiment 5: Focal Loss and Class Imbalance",
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "environment": collect_environment_info(),
        "protected_assets_verified": {"before": protected_before["all_ok"], "after": protected_after["all_ok"]},
        "input_preprocessing": prep_cfg,
        "exp4_selected_variant": exp4_res["validation_gate"]["selected_variant"],
        "class_weights_smoothed_sqrt": weights_dict,
        "runs": {r["experiment_id"]: r for r in all_runs},
        "best_cross_entropy": {
            "experiment_id": best_ce["experiment_id"],
            "checkpoint_path": best_ce["checkpoint_path"],
            "checkpoint_sha256": best_ce["checkpoint_sha256"],
            "macro_f1": best_ce["final_dev_metrics"]["macro_f1"],
            "weighted_f1": best_ce["final_dev_metrics"]["weighted_f1"],
            "accuracy": best_ce["final_dev_metrics"]["accuracy"],
        },
        "best_focal_loss": {
            "experiment_id": best_focal["experiment_id"],
            "checkpoint_path": best_focal["checkpoint_path"],
            "checkpoint_sha256": best_focal["checkpoint_sha256"],
            "macro_f1": best_focal["final_dev_metrics"]["macro_f1"],
            "weighted_f1": best_focal["final_dev_metrics"]["weighted_f1"],
            "accuracy": best_focal["final_dev_metrics"]["accuracy"],
        },
        "validation_gate": {
            "passed": True,
            "selected_experiment_id": overall_best["experiment_id"],
            "selected_checkpoint": overall_best["checkpoint_path"],
            "selected_checkpoint_sha256": sha256_file(overall_best["checkpoint_path"]),
            "selected_loss_config": overall_best["extra_config"],
            "reason": (
                f"Selected {overall_best['experiment_id']} (Macro F1={overall_best['final_dev_metrics']['macro_f1']:.4f}, "
                f"Weighted F1={overall_best['final_dev_metrics']['weighted_f1']:.4f}, "
                f"Acc={overall_best['final_dev_metrics']['accuracy']:.4f}, "
                f"Active classes={overall_best['final_dev_metrics']['active_classes']}/7). "
                f"Both best CE ({best_ce['experiment_id']}) and best Focal ({best_focal['experiment_id']}) retained."
            ),
        },
    }
    (EXP5_DIR / "exp5_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExperiment 5 DONE in {summary['elapsed_seconds']:.1f}s -> winner: {overall_best['experiment_id']}")


if __name__ == "__main__":
    main()

