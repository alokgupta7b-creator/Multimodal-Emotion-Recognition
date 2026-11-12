"""Phase 17A: Approved Facial Emotion Experiments 1-3.

Executes sequentially with strict reproducibility, checkpoint preservation, and test-set isolation:
- Experiment 1: Corrected CNN (matching [-1, 1] normalization, AdamW, standard CE)
- Experiment 2: Class-Weighted CNN (matching [-1, 1] normalization, class-weighted CE)
- Experiment 3: ResNet18 Transfer Learning (ImageNet pretrained, head replacement, 2-stage fine-tuning)

Outputs isolated in models/facial_experiments/
Zero access to MELD test split!
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.model import create_model, preprocess_face_crops
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

EXP_BASE = REPOSITORY_ROOT / "models" / "facial_experiments"
EXP1_DIR = EXP_BASE / "exp1_corrected_cnn"
EXP2_DIR = EXP_BASE / "exp2_weighted_cnn"
EXP3_DIR = EXP_BASE / "exp3_resnet18"

for d in [EXP1_DIR, EXP2_DIR, EXP3_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class FaceCropDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform_type: str = "cnn", is_training: bool = False):
        self.samples = samples
        self.transform_type = transform_type
        self.is_training = is_training
        # ImageNet constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.samples[index]
        crop = np.load(row["path"])  # (96, 96, 3) BGR uint8

        if self.is_training and random.random() < 0.5:
            crop = np.fliplr(crop).copy()

        # Convert BGR to RGB float in [0, 1]
        rgb = np.ascontiguousarray(crop[..., ::-1]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, 96, 96)

        if self.transform_type == "cnn":
            # Exact [-1.0, 1.0] scaling: (tensor - 0.5) / 0.5
            norm_tensor = (tensor - 0.5) / 0.5
        elif self.transform_type == "imagenet":
            # Standard ImageNet normalization
            norm_tensor = (tensor - self.mean) / self.std
        else:
            norm_tensor = tensor

        return norm_tensor, row["label"]


def load_split_data(split: str) -> list[dict[str, Any]]:
    assert split in {"train", "dev"}, f"Only train and dev splits allowed! Got {split}"
    manifest_path = MELD_PREPARED_ROOT / "reports" / split / "video_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    cache_dir = REPOSITORY_ROOT / "models" / "facial_emotion" / "face_cache" / split
    class_to_idx = {name: i for i, name in enumerate(EMOTION_CLASSES)}

    samples = []
    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        crop_path = cache_dir / f"{sid}.npy"
        if crop_path.is_file():
            label_idx = class_to_idx[row["emotion"].lower()]
            samples.append({
                "sample_id": sid,
                "path": crop_path,
                "label": label_idx,
                "emotion": row["emotion"].lower(),
            })
    return samples


def compute_metrics(y_true: list[int], y_pred: list[int], probs: list[np.ndarray]) -> dict[str, Any]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    probs_arr = np.array(probs)

    rep = classification_report(y_true_arr, y_pred_arr, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(7)))

    mean_probs = probs_arr.mean(axis=0)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, labels=list(range(7)), average="weighted", zero_division=0)),
        "per_class": rep,
        "confusion_matrix": cm.tolist(),
        "mean_predicted_probabilities": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, mean_probs)},
        "predicted_counts": {cls: int((y_pred_arr == idx).sum()) for idx, cls in enumerate(EMOTION_CLASSES)},
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, probs = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * len(targets)

            p = torch.softmax(logits, dim=-1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            probs.extend(p.cpu().numpy())

    avg_loss = total_loss / max(1, len(y_true))
    metrics = compute_metrics(y_true, y_pred, probs)
    return avg_loss, metrics


def measure_inference_latency(model: nn.Module, sample_tensor: torch.Tensor, n_runs: int = 50) -> float:
    model.eval()
    sample = sample_tensor.unsqueeze(0).to(DEVICE)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n_runs


def run_experiment_1(train_samples: list[dict], dev_samples: list[dict]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 1: CORRECTED CNN BASELINE")
    print("=" * 70)
    seed_everything(SEED)

    train_ds = FaceCropDataset(train_samples, transform_type="cnn", is_training=True)
    dev_ds = FaceCropDataset(dev_samples, transform_type="cnn", is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience = 2
    stale = 0
    history = []

    t_start = time.perf_counter()
    epochs = 6
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)

        train_loss /= len(train_samples)
        dev_loss, dev_metrics = evaluate(model, dev_loader, criterion)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(rec)
        print(f"  Exp 1 Epoch {epoch:2d}/{epochs}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    train_duration = time.perf_counter() - t_start

    # Load best checkpoint and evaluate
    model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, dev_loader, criterion)
    latency_ms = measure_inference_latency(model, dev_ds[0][0])
    total_params = sum(p.numel() for p in model.parameters())

    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE.type == "cuda" else None

    ckpt_path = EXP1_DIR / "exp1_corrected_cnn_best.pt"
    torch.save({
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": SEED,
        "input_normalization": "[-1.0, 1.0]",
        "architecture": "3-layer-cnn",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }, ckpt_path)
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    result = {
        "experiment": "Exp 1: Corrected CNN",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "parameters": total_params,
        "train_time_seconds": train_duration,
        "latency_ms": latency_ms,
        "gpu_max_memory_mb": gpu_mem_mb,
        "best_epoch": best_epoch,
        "history": history,
        "final_dev_metrics": final_metrics,
    }
    with (EXP1_DIR / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def run_experiment_2(train_samples: list[dict], dev_samples: list[dict]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 2: CLASS-WEIGHTED CNN")
    print("=" * 70)
    seed_everything(SEED)

    train_ds = FaceCropDataset(train_samples, transform_type="cnn", is_training=True)
    dev_ds = FaceCropDataset(dev_samples, transform_type="cnn", is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    # Compute class weights strictly from training set labels
    label_counts = np.bincount([s["label"] for s in train_samples], minlength=7).astype(float)
    N_total = len(train_samples)
    # Balanced class weights: N / (7 * N_c)
    weights = N_total / (7.0 * np.maximum(label_counts, 1.0))
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Computed Training Class Weights:")
    for cls, cnt, w in zip(EMOTION_CLASSES, label_counts, weights):
        print(f"  {cls:10s}: {int(cnt):5d} samples -> Weight: {w:.4f}")

    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience = 2
    stale = 0
    history = []

    t_start = time.perf_counter()
    epochs = 6
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)

        train_loss /= len(train_samples)
        dev_loss, dev_metrics = evaluate(model, dev_loader, eval_criterion)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(rec)
        print(f"  Exp 2 Epoch {epoch:2d}/{epochs}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    train_duration = time.perf_counter() - t_start

    model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, dev_loader, eval_criterion)
    latency_ms = measure_inference_latency(model, dev_ds[0][0])
    total_params = sum(p.numel() for p in model.parameters())

    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE.type == "cuda" else None

    ckpt_path = EXP2_DIR / "exp2_weighted_cnn_best.pt"
    torch.save({
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": SEED,
        "input_normalization": "[-1.0, 1.0]",
        "class_weights": weights.tolist(),
        "architecture": "3-layer-cnn-weighted",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }, ckpt_path)
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    result = {
        "experiment": "Exp 2: Class-Weighted CNN",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "parameters": total_params,
        "class_weights": {cls: float(w) for cls, w in zip(EMOTION_CLASSES, weights)},
        "train_time_seconds": train_duration,
        "latency_ms": latency_ms,
        "gpu_max_memory_mb": gpu_mem_mb,
        "best_epoch": best_epoch,
        "history": history,
        "final_dev_metrics": final_metrics,
    }
    with (EXP2_DIR / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def run_experiment_3(train_samples: list[dict], dev_samples: list[dict]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 3: RESNET18 TRANSFER LEARNING")
    print("=" * 70)
    seed_everything(SEED)

    train_ds = FaceCropDataset(train_samples, transform_type="imagenet", is_training=True)
    dev_ds = FaceCropDataset(dev_samples, transform_type="imagenet", is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    # Class weights for loss
    label_counts = np.bincount([s["label"] for s in train_samples], minlength=7).astype(float)
    N_total = len(train_samples)
    weights = np.sqrt(N_total / (7.0 * np.maximum(label_counts, 1.0)))  # Smoothed sqrt weights for deep network
    weights = weights / weights.mean()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # 1. Load ImageNet-pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(EMOTION_CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    eval_criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    history = []
    t_start = time.perf_counter()

    # Stage 1: Freeze backbone, train classification head for 3 epochs
    print("\nStage 1: Training classification head (backbone frozen)...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer_head.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_head.step()
            train_loss += float(loss.item()) * len(targets)

        train_loss /= len(train_samples)
        dev_loss, dev_metrics = evaluate(model, dev_loader, eval_criterion)
        rec = {
            "epoch": epoch,
            "stage": "head_only",
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(rec)
        print(f"  Stage 1 Epoch {epoch}/3: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: Unfreeze entire network for fine-tuning
    print("\nStage 2: End-to-end fine-tuning with differential learning rates...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_all = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)

    patience = 2
    stale = 0
    fine_tune_epochs = 5
    for epoch in range(4, 4 + fine_tune_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer_all.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_all.step()
            train_loss += float(loss.item()) * len(targets)

        train_loss /= len(train_samples)
        dev_loss, dev_metrics = evaluate(model, dev_loader, eval_criterion)
        rec = {
            "epoch": epoch,
            "stage": "fine_tune",
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(rec)
        print(f"  Stage 2 Epoch {epoch}/{3+fine_tune_epochs}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    train_duration = time.perf_counter() - t_start

    model.load_state_dict(best_state)
    final_loss, final_metrics = evaluate(model, dev_loader, eval_criterion)
    latency_ms = measure_inference_latency(model, dev_ds[0][0])
    total_params = sum(p.numel() for p in model.parameters())

    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE.type == "cuda" else None

    ckpt_path = EXP3_DIR / "exp3_resnet18_best.pt"
    torch.save({
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": SEED,
        "input_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "architecture": "resnet18-imagenet-transfer",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }, ckpt_path)
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    result = {
        "experiment": "Exp 3: ResNet18 Transfer Learning",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "parameters": total_params,
        "train_time_seconds": train_duration,
        "latency_ms": latency_ms,
        "gpu_max_memory_mb": gpu_mem_mb,
        "best_epoch": best_epoch,
        "history": history,
        "final_dev_metrics": final_metrics,
    }
    with (EXP3_DIR / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_comparisons(all_results: dict[str, Any]):
    print("\nGenerating comparison plots...")
    # 1. Confusion Matrices Comparison (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    models_order = [
        ("Historical Reference (Scratch CNN)", all_results["historical"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 1: Corrected CNN", all_results["exp1"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 2: Class-Weighted CNN", all_results["exp2"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 3: ResNet18 (ImageNet)", all_results["exp3"]["final_dev_metrics"]["confusion_matrix"]),
    ]

    for idx, (title, cm_list) in enumerate(models_order):
        ax = axes[idx]
        cm = np.array(cm_list)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12, fontweight="bold")
        tick_marks = np.arange(len(EMOTION_CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(EMOTION_CLASSES, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(EMOTION_CLASSES, fontsize=9)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    cm_path = EXP_BASE / "confusion_matrices_comparison.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix comparison to: {cm_path}")

    # 2. Training Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for exp_key, color, label in [
        ("exp1", "#2563eb", "Exp 1: Corrected CNN"),
        ("exp2", "#d97706", "Exp 2: Weighted CNN"),
        ("exp3", "#059669", "Exp 3: ResNet18"),
    ]:
        hist = all_results[exp_key]["history"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        dev_f1 = [h["dev_macro_f1"] for h in hist]

        ax1.plot(epochs, train_loss, marker="o", color=color, label=label, linewidth=2)
        ax2.plot(epochs, dev_f1, marker="s", color=color, label=label, linewidth=2)

    ax1.axhline(y=np.log(7), color="red", linestyle="--", label="Theoretical ln(7) = 1.946")
    ax1.set_title("Training Loss Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Development Macro F1 Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = EXP_BASE / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {curves_path}")


def main():
    print("=" * 80)
    print("PHASE 17A: FACIAL EMOTION EXPERIMENTS 1-3 EXECUTION RUNNER")
    print("=" * 80)
    print(f"Compute Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Preflight Verification
    print("\nRunning Preflight Checks...")
    historical_meta = REPOSITORY_ROOT / "docs" / "facial_diagnostics" / "model_audit.json"
    assert historical_meta.is_file(), "Diagnostic model audit file missing!"
    with historical_meta.open() as f:
        hist_data = json.load(f)

    train_samples = load_split_data("train")
    dev_samples = load_split_data("dev")
    print(f"Loaded {len(train_samples)} training samples and {len(dev_samples)} development samples.")
    assert len(train_samples) > 0 and len(dev_samples) > 0, "Empty splits detected!"

    # Verify zero overlap
    train_ids = {s["sample_id"] for s in train_samples}
    dev_ids = {s["sample_id"] for s in dev_samples}
    assert len(train_ids.intersection(dev_ids)) == 0, "DATA LEAKAGE: Overlapping sample IDs!"
    print("Preflight Check Passed: Strict split separation and label alignment verified.")

    # 2. Run Experiment 1
    res1 = run_experiment_1(train_samples, dev_samples)

    # 3. Run Experiment 2
    res2 = run_experiment_2(train_samples, dev_samples)

    # 4. Run Experiment 3
    res3 = run_experiment_3(train_samples, dev_samples)

    # 5. Format Historical Baseline
    historical_summary = {
        "experiment": "Historical Reference (Un-normalized Scratch CNN)",
        "parameters": 94151,
        "final_dev_metrics": {
            "accuracy": hist_data["dev_accuracy"],
            "macro_f1": hist_data["dev_macro_f1"],
            "weighted_f1": 0.2525,  # From training_metadata.json
            "confusion_matrix": hist_data["dev_confusion_matrix"],
            "per_class": hist_data["dev_per_class"],
            "mean_predicted_probabilities": hist_data["mean_predicted_probabilities"],
        }
    }

    all_results = {
        "historical": historical_summary,
        "exp1": res1,
        "exp2": res2,
        "exp3": res3,
    }

    # Save comprehensive results JSON
    master_results_path = EXP_BASE / "experiment_results.json"
    with master_results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMaster experimental results saved to: {master_results_path}")

    # 6. Generate comparison plots
    plot_comparisons(all_results)

    print("\n" + "=" * 80)
    print("PHASE 17A EXPERIMENTS 1-3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

