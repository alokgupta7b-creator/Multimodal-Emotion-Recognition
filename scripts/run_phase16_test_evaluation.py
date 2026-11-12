"""Phase 16: Final Scientific Evaluation on the Untouched MELD Test Split.

Strict Constraints:
- Evaluates DistilBERT text baseline, Wav2Vec2 speech baseline, Equal-Weight
  late fusion, and Candidate 4 adaptive fusion on the 2,610 held-out test samples.
- All checkpoints, thresholds (tau_a=0.20, lambda_max=0.40, kappa_a=1.50, lambda_v=0.00),
  and formulas (M_a = p_a[y_hat_a] - p_a[neutral]) remain 100% frozen.
- 10,000 paired bootstrap resamples (seed=42) for 95% CIs and p-values.
- Saves:
  - models/phase16_final_evaluation/final_results.json
  - models/phase16_final_evaluation/test_predictions.jsonl
  - models/phase16_final_evaluation/confusion_matrix_text.png
  - models/phase16_final_evaluation/confusion_matrix_candidate4.png
  - models/phase16_final_evaluation/confusion_matrix_audio.png
  - models/phase16_final_evaluation/confusion_matrix_equal_fusion.png
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from api.config import ApiSettings
from api.fusion import Candidate4FusionEngine, CLASS_NAMES, NUM_CLASSES
from api.model_registry import ModelRegistry
from project_config import MELD_ANNOTATIONS, MELD_PREPARED_ROOT, REPOSITORY_ROOT
from speech_emotion.audio_loader import load_audio
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def compute_brier_score(y_true_indices: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between probabilities and one-hot true labels."""
    N, K = probs.shape
    one_hot = np.zeros((N, K), dtype=np.float64)
    one_hot[np.arange(N), y_true_indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_ece(y_true_indices: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true_indices)

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def paired_bootstrap_test(
    y_true: List[str],
    y_pred_baseline: List[str],
    y_pred_candidate: List[str],
    num_iterations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap significance test for macro F1, accuracy, and weighted F1."""
    rng = np.random.RandomState(seed)
    N = len(y_true)
    indices_matrix = rng.randint(0, N, size=(num_iterations, N))

    y_t_arr = np.array(y_true)
    y_b_arr = np.array(y_pred_baseline)
    y_c_arr = np.array(y_pred_candidate)

    diff_macro_f1 = np.empty(num_iterations, dtype=np.float64)
    diff_accuracy = np.empty(num_iterations, dtype=np.float64)
    diff_weighted_f1 = np.empty(num_iterations, dtype=np.float64)

    per_class_diffs = {cls: np.empty(num_iterations, dtype=np.float64) for cls in CLASS_NAMES}

    for b in range(num_iterations):
        idx = indices_matrix[b]
        sample_true = y_t_arr[idx]
        sample_b = y_b_arr[idx]
        sample_c = y_c_arr[idx]

        f1_b = f1_score(sample_true, sample_b, labels=CLASS_NAMES, average="macro", zero_division=0)
        f1_c = f1_score(sample_true, sample_c, labels=CLASS_NAMES, average="macro", zero_division=0)
        diff_macro_f1[b] = f1_c - f1_b

        acc_b = accuracy_score(sample_true, sample_b)
        acc_c = accuracy_score(sample_true, sample_c)
        diff_accuracy[b] = acc_c - acc_b

        wf1_b = f1_score(sample_true, sample_b, labels=CLASS_NAMES, average="weighted", zero_division=0)
        wf1_c = f1_score(sample_true, sample_c, labels=CLASS_NAMES, average="weighted", zero_division=0)
        diff_weighted_f1[b] = wf1_c - wf1_b

        # Per class F1
        pc_b = f1_score(sample_true, sample_b, labels=CLASS_NAMES, average=None, zero_division=0)
        pc_c = f1_score(sample_true, sample_c, labels=CLASS_NAMES, average=None, zero_division=0)
        for i, cls in enumerate(CLASS_NAMES):
            per_class_diffs[cls][b] = pc_c[i] - pc_b[i]

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    def p_val(arr):
        p_le = np.mean(arr <= 0)
        p_ge = np.mean(arr >= 0)
        return float(2 * min(p_le, p_ge))

    return {
        "iterations": num_iterations,
        "seed": seed,
        "macro_f1": {
            "mean_difference": float(np.mean(diff_macro_f1)),
            "ci_95": ci(diff_macro_f1),
            "p_value": p_val(diff_macro_f1),
            "significant_at_05": bool(ci(diff_macro_f1)[0] > 0 or ci(diff_macro_f1)[1] < 0),
        },
        "accuracy": {
            "mean_difference": float(np.mean(diff_accuracy)),
            "ci_95": ci(diff_accuracy),
            "p_value": p_val(diff_accuracy),
            "significant_at_05": bool(ci(diff_accuracy)[0] > 0 or ci(diff_accuracy)[1] < 0),
        },
        "weighted_f1": {
            "mean_difference": float(np.mean(diff_weighted_f1)),
            "ci_95": ci(diff_weighted_f1),
            "p_value": p_val(diff_weighted_f1),
            "significant_at_05": bool(ci(diff_weighted_f1)[0] > 0 or ci(diff_weighted_f1)[1] < 0),
        },
        "per_class_f1": {
            cls: {
                "mean_difference": float(np.mean(per_class_diffs[cls])),
                "ci_95": ci(per_class_diffs[cls]),
                "p_value": p_val(per_class_diffs[cls]),
                "significant_at_05": bool(ci(per_class_diffs[cls])[0] > 0 or ci(per_class_diffs[cls])[1] < 0),
            }
            for cls in CLASS_NAMES
        },
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path):
    """Render and save a 7x7 confusion matrix heatmap with counts and row normalized percentages."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True Emotion Label",
        xlabel="Predicted Emotion Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 16: SCIENTIFIC HELD-OUT TEST SPLIT EVALUATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = REPOSITORY_ROOT / "models" / "phase16_final_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Registry & Models
    print("\nLoading Model Registry...")
    settings = ApiSettings(device=device)
    registry = ModelRegistry(settings)
    t0_load = time.perf_counter()
    registry.load()
    load_ms = (time.perf_counter() - t0_load) * 1000
    print(f"Registry loaded in {load_ms:.2f} ms")

    text_predictor = registry.require("text")
    audio_predictor = registry.require("audio")
    fusion_engine = Candidate4FusionEngine()

    # 2. Load Test Split
    test_csv = MELD_ANNOTATIONS / "test_sent_emo.csv"
    with test_csv.open(encoding="utf-8", newline="") as f:
        annotations = list(csv.DictReader(f))

    audio_manifest_path = MELD_PREPARED_ROOT / "reports" / "test" / "audio_manifest.csv"
    with audio_manifest_path.open(encoding="utf-8", newline="") as f:
        audio_manifest = list(csv.DictReader(f))

    N = len(annotations)
    print(f"\nEvaluating {N} samples on the MELD test split...")

    predictions_file = out_dir / "test_predictions.jsonl"
    pred_handle = open(predictions_file, "w", encoding="utf-8")

    # Storage arrays
    y_true_labels: List[str] = []
    y_pred_text: List[str] = []
    y_pred_audio: List[str] = []
    y_pred_equal: List[str] = []
    y_pred_cand4: List[str] = []

    probs_text_list: List[np.ndarray] = []
    probs_audio_list: List[np.ndarray] = []
    probs_equal_list: List[np.ndarray] = []
    probs_cand4_list: List[np.ndarray] = []

    text_latencies_ms: List[float] = []
    audio_latencies_ms: List[float] = []
    fusion_latencies_ms: List[float] = []

    # Failure / Availability tracking
    audio_availability = {"available": 0, "silent": 0, "corrupt": 0, "missing": 0}
    rules_applied = Counter()

    matched_valid_audio_indices: List[int] = []

    t_start_all = time.perf_counter()

    for i in range(N):
        anno = annotations[i]
        audio_row = audio_manifest[i]

        dia = int(anno["Dialogue_ID"])
        utt = int(anno["Utterance_ID"])
        true_emotion = anno["Emotion"].strip().lower()
        text_utterance = anno["Utterance"].strip()
        y_true_labels.append(true_emotion)

        # -----------------------------
        # 1. DistilBERT Text Inference
        # -----------------------------
        t_t0 = time.perf_counter()
        # Fallback for empty text if any
        eval_text = text_utterance if text_utterance else "..."
        try:
            t_res = text_predictor.predict(eval_text)
            p_text = np.array([float(t_res["probabilities"][c]) for c in CLASS_NAMES], dtype=np.float64)
            pred_t_label = t_res["predicted_label"]
        except Exception as exc:
            # Fallback uniform
            p_text = np.ones(NUM_CLASSES, dtype=np.float64) / NUM_CLASSES
            pred_t_label = "neutral"
        text_latencies_ms.append((time.perf_counter() - t_t0) * 1000)

        y_pred_text.append(pred_t_label)
        probs_text_list.append(p_text)

        # -----------------------------
        # 2. Wav2Vec2 Audio Inference
        # -----------------------------
        audio_path_str = audio_row.get("output_path", "")
        audio_path = Path(audio_path_str) if audio_path_str else None

        p_audio: np.ndarray | None = None
        audio_avail = False
        exclusion_reason: str | None = None
        pred_a_label: str | None = None

        if audio_path and audio_path.is_file():
            waveform, meta = load_audio(audio_path, target_sr=16000, mono=True, min_duration=0.01)
            if not meta.is_valid:
                if meta.status == "silent_audio":
                    audio_availability["silent"] += 1
                else:
                    audio_availability["corrupt"] += 1
                exclusion_reason = f"Invalid audio: {meta.status} ({meta.reason})"
            else:
                t_a0 = time.perf_counter()
                try:
                    a_res = audio_predictor.predict_file(audio_path)
                    p_audio = np.array([float(a_res["probabilities"][c]) for c in CLASS_NAMES], dtype=np.float64)
                    pred_a_label = a_res["predicted_label"]
                    audio_avail = True
                    audio_availability["available"] += 1
                    matched_valid_audio_indices.append(i)
                except Exception as exc:
                    exclusion_reason = f"Inference failure: {exc}"
                    audio_availability["corrupt"] += 1
                audio_latencies_ms.append((time.perf_counter() - t_a0) * 1000)
        else:
            audio_availability["missing"] += 1
            exclusion_reason = "Audio file missing"

        if p_audio is not None:
            probs_audio_list.append(p_audio)
            y_pred_audio.append(pred_a_label)
        else:
            probs_audio_list.append(p_text)  # Standalone audio fallback for array alignment
            y_pred_audio.append("neutral")

        # -----------------------------
        # 3. Naive Equal-Weight Fusion
        # -----------------------------
        if audio_avail and p_audio is not None:
            p_equal = (p_text + p_audio) / 2.0
            p_equal /= p_equal.sum()
        else:
            p_equal = np.copy(p_text)

        probs_equal_list.append(p_equal)
        y_pred_equal.append(CLASS_NAMES[int(p_equal.argmax())])

        # -----------------------------
        # 4. Candidate 4 Adaptive Fusion
        # -----------------------------
        t_f0 = time.perf_counter()
        f_res = fusion_engine.fuse(
            text_probabilities=p_text,
            audio_probabilities=p_audio if audio_avail else None,
            audio_exclusion_reason=exclusion_reason,
        )
        fusion_latencies_ms.append((time.perf_counter() - t_f0) * 1000)

        p_cand4 = np.array([f_res.probabilities[c] for c in CLASS_NAMES], dtype=np.float64)
        probs_cand4_list.append(p_cand4)
        y_pred_cand4.append(f_res.predicted_emotion)
        rules_applied[f_res.rule_applied] += 1

        # Write record
        rec = {
            "index": i,
            "dialogue_id": dia,
            "utterance_id": utt,
            "true_emotion": true_emotion,
            "text": text_utterance,
            "audio_available": audio_avail,
            "exclusion_reason": exclusion_reason,
            "distilbert": {
                "pred": pred_t_label,
                "confidence": float(p_text.max()),
                "probabilities": {c: float(p_text[ci]) for ci, c in enumerate(CLASS_NAMES)},
            },
            "wav2vec2": {
                "pred": pred_a_label,
                "confidence": float(p_audio.max()) if p_audio is not None else None,
                "probabilities": {c: float(p_audio[ci]) for ci, c in enumerate(CLASS_NAMES)} if p_audio is not None else None,
            },
            "equal_fusion": {
                "pred": y_pred_equal[-1],
                "confidence": float(p_equal.max()),
            },
            "candidate4": {
                "pred": f_res.predicted_emotion,
                "confidence": float(f_res.confidence),
                "rule_applied": f_res.rule_applied,
                "modality_weights": f_res.modality_weights,
                "margin_audio": f_res.margin_audio,
                "probabilities": f_res.probabilities,
            },
        }
        pred_handle.write(json.dumps(rec) + "\n")

        if (i + 1) % 250 == 0 or (i + 1) == N:
            elapsed = time.perf_counter() - t_start_all
            ms_per_sample = (elapsed / (i + 1)) * 1000
            print(f"  Processed {i+1:>4}/{N} samples ({ms_per_sample:5.2f} ms/sample) | Elapsed: {elapsed:5.1f}s")

    pred_handle.close()
    total_eval_time_sec = time.perf_counter() - t_start_all
    print(f"\nFinished all {N} evaluations in {total_eval_time_sec:.2f} seconds.")

    # Convert probability lists to numpy matrices
    y_true_indices = np.array([CLASS_NAMES.index(lbl) for lbl in y_true_labels])
    P_text = np.array(probs_text_list)
    P_equal = np.array(probs_equal_list)
    P_cand4 = np.array(probs_cand4_list)

    # -------------------------------------------------------------
    # Compute Comprehensive Metrics for Each Strategy
    # -------------------------------------------------------------
    print("\nComputing classification metrics...")

    def evaluate_strategy(y_preds: List[str], probs: np.ndarray, name: str) -> Dict[str, Any]:
        acc = accuracy_score(y_true_labels, y_preds)
        macro_f1 = f1_score(y_true_labels, y_preds, labels=CLASS_NAMES, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_labels, y_preds, labels=CLASS_NAMES, average="weighted", zero_division=0)
        macro_prec = precision_score(y_true_labels, y_preds, labels=CLASS_NAMES, average="macro", zero_division=0)
        macro_rec = recall_score(y_true_labels, y_preds, labels=CLASS_NAMES, average="macro", zero_division=0)

        brier = compute_brier_score(y_true_indices, probs)
        ece = compute_ece(y_true_indices, probs, num_bins=10)

        report = classification_report(
            y_true_labels,
            y_preds,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true_labels, y_preds, labels=CLASS_NAMES)

        per_class = {}
        for cls in CLASS_NAMES:
            per_class[cls] = {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }

        return {
            "name": name,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "brier_score": float(brier),
            "ece_10bin": float(ece),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    results_text = evaluate_strategy(y_pred_text, P_text, "DistilBERT Text Baseline")
    results_equal = evaluate_strategy(y_pred_equal, P_equal, "Naive Equal-Weight Late Fusion")
    results_cand4 = evaluate_strategy(y_pred_cand4, P_cand4, "Candidate 4 Adaptive Multimodal Fusion")

    # Audio standalone on matched valid audio cohort
    valid_true = [y_true_labels[idx] for idx in matched_valid_audio_indices]
    valid_audio_preds = [y_pred_audio[idx] for idx in matched_valid_audio_indices]
    valid_audio_probs = np.array([probs_audio_list[idx] for idx in matched_valid_audio_indices])
    valid_true_indices = np.array([CLASS_NAMES.index(lbl) for lbl in valid_true])

    cm_audio = confusion_matrix(valid_true, valid_audio_preds, labels=CLASS_NAMES)
    rep_audio = classification_report(valid_true, valid_audio_preds, labels=CLASS_NAMES, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    results_audio = {
        "name": "Wav2Vec2 Speech Baseline (Matched Valid Audio Cohort)",
        "cohort_size": len(matched_valid_audio_indices),
        "accuracy": float(accuracy_score(valid_true, valid_audio_preds)),
        "macro_f1": float(f1_score(valid_true, valid_audio_preds, labels=CLASS_NAMES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(valid_true, valid_audio_preds, labels=CLASS_NAMES, average="weighted", zero_division=0)),
        "brier_score": float(compute_brier_score(valid_true_indices, valid_audio_probs)),
        "ece_10bin": float(compute_ece(valid_true_indices, valid_audio_probs, num_bins=10)),
        "per_class": {
            cls: {
                "precision": rep_audio[cls]["precision"],
                "recall": rep_audio[cls]["recall"],
                "f1": rep_audio[cls]["f1-score"],
                "support": rep_audio[cls]["support"],
            }
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm_audio.tolist(),
    }

    # -------------------------------------------------------------
    # Paired Bootstrap Significance Testing (10,000 resamples)
    # -------------------------------------------------------------
    print("\nRunning 10,000 paired bootstrap resamples (seed=42)...")
    t0_boot = time.perf_counter()
    bootstrap_cand4_vs_text = paired_bootstrap_test(
        y_true_labels,
        y_pred_text,
        y_pred_cand4,
        num_iterations=10000,
        seed=42,
    )

    bootstrap_equal_vs_text = paired_bootstrap_test(
        y_true_labels,
        y_pred_text,
        y_pred_equal,
        num_iterations=10000,
        seed=42,
    )
    boot_sec = time.perf_counter() - t0_boot
    print(f"Bootstrap completed in {boot_sec:.2f} seconds.")

    # -------------------------------------------------------------
    # Generate Confusion Matrix Visualizations
    # -------------------------------------------------------------
    print("\nGenerating confusion matrix images...")
    plot_confusion_matrix(
        np.array(results_text["confusion_matrix"]),
        "DistilBERT Text Baseline - Test Split Confusion Matrix",
        out_dir / "confusion_matrix_text.png",
    )
    plot_confusion_matrix(
        np.array(results_cand4["confusion_matrix"]),
        "Candidate 4 Adaptive Fusion - Test Split Confusion Matrix",
        out_dir / "confusion_matrix_candidate4.png",
    )
    plot_confusion_matrix(
        cm_audio,
        "Wav2Vec2 Speech Baseline - Test Split Confusion Matrix",
        out_dir / "confusion_matrix_audio.png",
    )
    plot_confusion_matrix(
        np.array(results_equal["confusion_matrix"]),
        "Equal-Weight Late Fusion - Test Split Confusion Matrix",
        out_dir / "confusion_matrix_equal_fusion.png",
    )
    print("✔ Confusion matrix plots saved.")

    # -------------------------------------------------------------
    # Latency Stats
    # -------------------------------------------------------------
    def latency_stats(vals: List[float]):
        arr = np.array(vals)
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    latency_report = {
        "text_latency": latency_stats(text_latencies_ms),
        "audio_latency": latency_stats(audio_latencies_ms),
        "fusion_latency": latency_stats(fusion_latencies_ms),
        "total_inference_time_sec": total_eval_time_sec,
    }

    # -------------------------------------------------------------
    # Compile Master Results JSON
    # -------------------------------------------------------------
    final_output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "dataset": {
            "split": "test",
            "total_samples": N,
            "class_names": list(CLASS_NAMES),
            "label_distribution": {cls: int(np.sum(y_true_indices == i)) for i, cls in enumerate(CLASS_NAMES)},
        },
        "failure_and_availability": {
            "audio_summary": audio_availability,
            "effective_multimodal_cohort_size": audio_availability["available"],
            "multimodal_coverage_pct": float(audio_availability["available"] / N * 100),
            "rules_applied_breakdown": dict(rules_applied),
        },
        "models": {
            "text_baseline": results_text,
            "audio_baseline": results_audio,
            "equal_weight_fusion": results_equal,
            "candidate4_adaptive_fusion": results_cand4,
        },
        "statistical_significance": {
            "candidate4_vs_text_baseline": bootstrap_cand4_vs_text,
            "equal_fusion_vs_text_baseline": bootstrap_equal_vs_text,
        },
        "latencies": latency_report,
    }

    results_json_path = out_dir / "final_results.json"
    results_json_path.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\n✔ Master results JSON saved to {results_json_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 70)
    print("FINAL TEST SPLIT EVALUATION METRICS COMPARISON (N = 2,610)")
    print("=" * 70)
    print(f"{'Strategy':<35} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Brier':<7} | {'ECE':<7}")
    print("-" * 88)
    for res in [results_text, results_audio, results_equal, results_cand4]:
        name = res["name"][:35]
        acc = f"{res['accuracy']*100:.2f}%"
        mf1 = f"{res['macro_f1']:.4f}"
        wf1 = f"{res['weighted_f1']:.4f}"
        brier = f"{res['brier_score']:.4f}"
        ece = f"{res['ece_10bin']:.4f}"
        print(f"{name:<35} | {acc:<9} | {mf1:<9} | {wf1:<11} | {brier:<7} | {ece:<7}")
    print("-" * 88)

    print("\nPER-CLASS F1 SCORE BREAKDOWN:")
    print(f"{'Emotion':<10} | {'DistilBERT':<12} | {'Equal Fusion':<14} | {'Candidate 4':<13} | {'Δ(C4 - Text)':<14} | {'95% Bootstrap CI':<20}")
    print("-" * 92)
    for cls in CLASS_NAMES:
        f1_t = results_text["per_class"][cls]["f1"]
        f1_eq = results_equal["per_class"][cls]["f1"]
        f1_c4 = results_cand4["per_class"][cls]["f1"]
        diff_info = bootstrap_cand4_vs_text["per_class_f1"][cls]
        diff_val = f1_c4 - f1_t
        ci_str = f"[{diff_info['ci_95'][0]:+.4f}, {diff_info['ci_95'][1]:+.4f}]"
        sig_marker = "*" if diff_info["significant_at_05"] else " "
        print(f"{cls:<10} | {f1_t:<12.4f} | {f1_eq:<14.4f} | {f1_c4:<13.4f} | {diff_val:>+10.4f} {sig_marker} | {ci_str:<20}")
    print("-" * 92)

    diff_mf1 = bootstrap_cand4_vs_text["macro_f1"]
    diff_acc = bootstrap_cand4_vs_text["accuracy"]
    print(f"\nCandidate 4 vs Text Baseline Statistical Summary (10,000 bootstrap iterations):")
    print(f"  Δ Macro F1: {diff_mf1['mean_difference']:+.4f} (95% CI: [{diff_mf1['ci_95'][0]:+.4f}, {diff_mf1['ci_95'][1]:+.4f}], p = {diff_mf1['p_value']:.4f})")
    print(f"  Δ Accuracy: {diff_acc['mean_difference']:+.4f} (95% CI: [{diff_acc['ci_95'][0]:+.4f}, {diff_acc['ci_95'][1]:+.4f}], p = {diff_acc['p_value']:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

