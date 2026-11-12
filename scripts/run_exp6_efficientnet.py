"""Phase 17B Experiment 6 — EfficientNet-B0 Architecture Comparison.

Compares ImageNet-pretrained EfficientNet-B0 against the best ResNet18 configuration using
identical data splits, preprocessing from Exp 4, and matched loss objectives (Exp 5 winning
loss + smoothed weighted CE reference). Evaluates whether EfficientNet-B0 justifies its
runtime/complexity/resource footprint.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from typing import Any

import numpy as np
import torch.nn as nn

from facial_emotion.config import EMOTION_CLASSES
from scripts.phase17b_common import (
    EXP_ROOT,
    SEED,
    FocalLoss,
    collect_environment_info,
    compute_smoothed_class_weights,
    seed_everything,
    sha256_file,
    train_backbone_experiment,
    verify_protected_assets,
)
from scripts.run_exp5_focal_loss import load_selected_exp4_arrays

EXP4_RESULTS = EXP_ROOT / "exp4_face_detection" / "exp4_results.json"
EXP5_RESULTS = EXP_ROOT / "exp5_focal_loss" / "exp5_results.json"
EXP6_DIR = EXP_ROOT / "exp6_efficientnet"


def _make_criterion_from_extra(extra_cfg: dict[str, Any], smooth_weights):
    gamma = float(extra_cfg.get("gamma", 0.0))
    weighted = bool(extra_cfg.get("weighted", False))
    if gamma > 0.0:
        return lambda dev: FocalLoss(gamma=gamma, weight=smooth_weights.to(dev) if weighted else None)
    return lambda dev: nn.CrossEntropyLoss(weight=smooth_weights.to(dev) if weighted else None)


def main() -> None:
    seed_everything(SEED)
    EXP6_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    protected_before = verify_protected_assets()

    exp4_res = json.loads(EXP4_RESULTS.read_text(encoding="utf-8"))
    exp5_res = json.loads(EXP5_RESULTS.read_text(encoding="utf-8"))

    train_crops, train_labels, dev_crops, dev_labels, prep_cfg = load_selected_exp4_arrays(exp4_res)
    smooth_weights = compute_smoothed_class_weights(train_labels, power=0.5, normalize_mean=True)
    weights_dict = {EMOTION_CLASSES[i]: float(smooth_weights[i]) for i in range(len(EMOTION_CLASSES))}

    exp5_winner_id = exp5_res["validation_gate"]["selected_experiment_id"]
    exp5_winner = exp5_res["runs"][exp5_winner_id]
    winner_extra = exp5_res["validation_gate"]["selected_loss_config"]

    runs_to_train = [
        {
            "id": "exp6a_efficientnet_b0_best_loss",
            "name": f"Exp 6A: EfficientNet-B0 + {winner_extra.get('loss', 'Selected Loss')}",
            "ckpt": "exp6a_efficientnet_b0_best_loss.pt",
            "criterion_fn": _make_criterion_from_extra(winner_extra, smooth_weights),
            "extra": {**winner_extra, "matched_resnet_id": exp5_winner_id},
        }
    ]
    # Also run Smoothed Weighted CE reference if Exp 5 winner used Focal Loss or Unweighted CE
    if exp5_winner_id != "exp5_ce_weighted":
        runs_to_train.append(
            {
                "id": "exp6b_efficientnet_b0_ce_weighted",
                "name": "Exp 6B: EfficientNet-B0 + Smoothed-Sqrt Weighted CE",
                "ckpt": "exp6b_efficientnet_b0_ce_weighted.pt",
                "criterion_fn": lambda dev: nn.CrossEntropyLoss(weight=smooth_weights.to(dev)),
                "extra": {"loss": "CrossEntropyLoss(smoothed_sqrt)", "gamma": 0.0, "weighted": True, "class_weights": weights_dict, "matched_resnet_id": "exp5_ce_weighted"},
            }
        )
    else:
        # Also test Focal g=2.0 unweighted on EfficientNet-B0 for comparison fairness
        best_focal_id = exp5_res["best_focal_loss"]["experiment_id"]
        focal_extra = exp5_res["runs"][best_focal_id]["extra_config"]
        runs_to_train.append(
            {
                "id": "exp6b_efficientnet_b0_best_focal",
                "name": f"Exp 6B: EfficientNet-B0 + {focal_extra.get('loss', 'Focal Loss')}",
                "ckpt": "exp6b_efficientnet_b0_best_focal.pt",
                "criterion_fn": _make_criterion_from_extra(focal_extra, smooth_weights),
                "extra": {**focal_extra, "matched_resnet_id": best_focal_id},
            }
        )

    eff_runs: dict[str, Any] = {}
    for cfg in runs_to_train:
        print(f"\n=== Running {cfg['name']} ===")
        res = train_backbone_experiment(
            experiment_id=cfg["id"],
            experiment_name=cfg["name"],
            arch="efficientnet_b0",
            out_dir=EXP6_DIR,
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
        eff_runs[cfg["id"]] = res

    valid_eff = [r for r in eff_runs.values() if not r["final_dev_metrics"]["collapse_detected"] and r["final_dev_metrics"]["weighted_f1"] >= 0.18]
    best_eff = max(valid_eff if valid_eff else list(eff_runs.values()), key=lambda r: (r["final_dev_metrics"]["macro_f1"], r["final_dev_metrics"]["weighted_f1"]))

    eff_macro = best_eff["final_dev_metrics"]["macro_f1"]
    res_macro = exp5_winner["final_dev_metrics"]["macro_f1"]
    eff_wf1 = best_eff["final_dev_metrics"]["weighted_f1"]
    res_wf1 = exp5_winner["final_dev_metrics"]["weighted_f1"]
    eff_lat = best_eff["latency_ms_cuda"]
    res_lat = exp5_winner["latency_ms_cuda"]

    # Gate: select EfficientNet-B0 only if it improves Macro F1 cleanly or clearly beats efficiency+F1 tradeoff without losing Weighted F1 badly
    select_eff = (eff_macro > res_macro + 0.002) and (eff_wf1 >= res_wf1 - 0.03)
    if select_eff:
        selected_arch = "efficientnet_b0"
        selected_id = best_eff["experiment_id"]
        selected_ckpt = best_eff["checkpoint_path"]
        reason = (
            f"Selected {selected_id} (EfficientNet-B0, params={best_eff['parameters']:,}): "
            f"Macro F1={eff_macro:.4f} vs ResNet18={res_macro:.4f}, "
            f"Weighted F1={eff_wf1:.4f} vs {res_wf1:.4f}, CUDA latency={eff_lat:.2f}ms vs {res_lat:.2f}ms."
        )
    else:
        selected_arch = "resnet18"
        selected_id = exp5_winner_id
        selected_ckpt = exp5_winner["checkpoint_path"]
        reason = (
            f"Retained {exp5_winner_id} (ResNet18, params={exp5_winner['parameters']:,}): "
            f"Best EfficientNet-B0 ({best_eff['experiment_id']}) achieved Macro F1={eff_macro:.4f} "
            f"(Weighted F1={eff_wf1:.4f}, CUDA latency={eff_lat:.2f}ms) vs ResNet18 Macro F1={res_macro:.4f} "
            f"(Weighted F1={res_wf1:.4f}, CUDA latency={res_lat:.2f}ms) — does not justify switching backbone."
        )

    protected_after = verify_protected_assets()
    summary = {
        "experiment": "Phase 17B Experiment 6: EfficientNet-B0 Architecture Comparison",
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "environment": collect_environment_info(),
        "protected_assets_verified": {"before": protected_before["all_ok"], "after": protected_after["all_ok"]},
        "input_preprocessing": prep_cfg,
        "resnet18_reference": {
            "experiment_id": exp5_winner_id,
            "checkpoint_path": exp5_winner["checkpoint_path"],
            "checkpoint_sha256": exp5_winner["checkpoint_sha256"],
            "parameters": exp5_winner["parameters"],
            "train_time_seconds": exp5_winner["train_time_seconds"],
            "train_peak_vram_mb": exp5_winner["train_peak_vram_mb"],
            "latency_ms_cuda": res_lat,
            "latency_ms_cpu": exp5_winner["latency_ms_cpu"],
            "final_dev_metrics": exp5_winner["final_dev_metrics"],
        },
        "efficientnet_runs": eff_runs,
        "validation_gate": {
            "passed": True,
            "selected_architecture": selected_arch,
            "selected_experiment_id": selected_id,
            "selected_checkpoint": selected_ckpt,
            "selected_checkpoint_sha256": sha256_file(selected_ckpt),
            "reason": reason,
        },
    }
    (EXP6_DIR / "exp6_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExperiment 6 DONE in {summary['elapsed_seconds']:.1f}s -> winner: {selected_id} ({selected_arch})")
    print("Reason:", reason)


if __name__ == "__main__":
    main()

