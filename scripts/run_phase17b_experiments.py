"""Phase 17B: Facial Emotion Experiments 4-7.

Executes sequentially with strict reproducibility, checkpoint preservation, and test-set isolation:
- Experiment 4: EfficientNet-B0 Transfer Learning (ImageNet weights, 2-stage fine-tuning)
- Experiment 5: ResNet-18 with Focal Loss (gamma=2.0 + alpha weighting)
- Experiment 6: Comprehensive Comparative Audit & Selection of Champion Backbone
- Experiment 7: Video Pipeline Upgrade (Bounding box spatial tracking, false positive filtering, top-K pooling)

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

import cv2
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

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.detector import FaceDetector, FaceDetection
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

EXP_BASE = REPOSITORY_ROOT / "models" / "facial_experiments"
EXP4_DIR = EXP_BASE / "exp4_efficientnet_b0"
EXP5_DIR = EXP_BASE / "exp5_resnet18_focal"
EXP7_DIR = EXP_BASE / "exp7_video_pipeline"

for d in [EXP4_DIR, EXP5_DIR, EXP7_DIR]:
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
    def __init__(self, samples: list[dict[str, Any]], is_training: bool = False):
        self.samples = samples
        self.is_training = is_training
        # Standard ImageNet normalization constants
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
        norm_tensor = (tensor - self.mean) / self.std

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


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by downweighting easy examples."""
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def run_experiment_4(train_samples: list[dict], dev_samples: list[dict]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 4: EFFICIENTNET-B0 TRANSFER LEARNING")
    print("=" * 70)
    seed_everything(SEED)

    train_ds = FaceCropDataset(train_samples, is_training=True)
    dev_ds = FaceCropDataset(dev_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    # Smoothed class weights
    label_counts = np.bincount([s["label"] for s in train_samples], minlength=7).astype(float)
    N_total = len(train_samples)
    weights = np.sqrt(N_total / (7.0 * np.maximum(label_counts, 1.0)))
    weights = weights / weights.mean()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # Load EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(EMOTION_CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    eval_criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    history = []
    t_start = time.perf_counter()

    # Stage 1: Freeze backbone, train head
    print("\nStage 1: Training classifier head (features frozen)...")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

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

    # Stage 2: Fine-tune backbone
    print("\nStage 2: Fine-tuning entire EfficientNet-B0 network...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_all = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
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

    ckpt_path = EXP4_DIR / "exp4_efficientnet_b0_best.pt"
    torch.save({
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": SEED,
        "input_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "architecture": "efficientnet_b0-imagenet-transfer",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }, ckpt_path)
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    result = {
        "experiment": "Exp 4: EfficientNet-B0 Transfer Learning",
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
    with (EXP4_DIR / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def run_experiment_5(train_samples: list[dict], dev_samples: list[dict]) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 5: RESNET-18 WITH FOCAL LOSS (GAMMA=2.0)")
    print("=" * 70)
    seed_everything(SEED)

    train_ds = FaceCropDataset(train_samples, is_training=True)
    dev_ds = FaceCropDataset(dev_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    # Class weights for focal loss
    label_counts = np.bincount([s["label"] for s in train_samples], minlength=7).astype(float)
    N_total = len(train_samples)
    weights = np.sqrt(N_total / (7.0 * np.maximum(label_counts, 1.0)))
    weights = weights / weights.mean()
    alpha_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(EMOTION_CLASSES))
    model = model.to(DEVICE)

    # Focal loss with gamma=2.0
    focal_criterion = FocalLoss(alpha=alpha_weights_tensor, gamma=2.0)
    eval_criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = -1
    history = []
    t_start = time.perf_counter()

    # Stage 1: Freeze backbone, train head
    print("\nStage 1: Training head with Focal Loss (backbone frozen)...")
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name

    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer_head.zero_grad()
            logits = model(inputs)
            loss = focal_criterion(logits, targets)
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
        print(f"  Stage 1 Epoch {epoch}/3: Train Focal Loss={train_loss:.4f} | Dev CE Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: Fine-tune backbone
    print("\nStage 2: Fine-tuning with Focal Loss...")
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
            loss = focal_criterion(logits, targets)
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
        print(f"  Stage 2 Epoch {epoch}/{3+fine_tune_epochs}: Train Focal Loss={train_loss:.4f} | Dev CE Loss={dev_loss:.4f} | Dev Acc={dev_metrics['accuracy']*100:.2f}% | Dev Macro F1={dev_metrics['macro_f1']:.4f}")

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

    ckpt_path = EXP5_DIR / "exp5_resnet18_focal_best.pt"
    torch.save({
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": SEED,
        "input_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "loss": "focal_loss_gamma_2",
        "architecture": "resnet18-focal-transfer",
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
    }, ckpt_path)
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    result = {
        "experiment": "Exp 5: ResNet-18 with Focal Loss",
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
    with (EXP5_DIR / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def run_experiment_7_video_pipeline(champion_model: nn.Module) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 7: UPGRADED VIDEO PIPELINE AUDIT")
    print("=" * 70)
    champion_model.eval()

    video_path = REPOSITORY_ROOT / "datasets" / "MELD_raw_acquisition" / "media" / "dev_splits_complete" / "dia4_utt1.mp4"
    assert video_path.is_file(), f"Video file missing: {video_path}"

    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = FaceDetector(min_face_size=36)  # Filter out tiny noise <36px

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    sample_every = 3
    frame_idx = 0
    all_raw_detections = []
    tracked_face_crops = []
    tracked_metadata = []
    previous_box = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            raw_detections = detector.detect(frame, size=96)
            all_raw_detections.extend(raw_detections)

            # Filter out anomalous background detections:
            # - Bounding box area >= 40x40
            # - Position y <= 0.8 * height (exclude floor/table noise)
            valid_detections = [
                d for d in raw_detections
                if d.bbox[2] >= 40 and d.bbox[3] >= 40 and (d.bbox[1] + d.bbox[3]) <= 0.85 * height
            ]

            if valid_detections:
                if previous_box is None:
                    # Select largest face in frame (the primary subject)
                    chosen = max(valid_detections, key=lambda d: d.bbox[2] * d.bbox[3])
                else:
                    # Select face maximizing IoU with previous frame, fallback to distance
                    chosen = max(
                        valid_detections,
                        key=lambda d: compute_iou(previous_box, d.bbox) * 10.0 - np.linalg.norm(np.array(d.center) - np.array(previous_center)) / 100.0
                    )

                previous_box = chosen.bbox
                previous_center = chosen.center
                tracked_face_crops.append(chosen.crop)
                tracked_metadata.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "bbox": chosen.bbox,
                    "center": chosen.center,
                    "area": chosen.bbox[2] * chosen.bbox[3],
                })

        frame_idx += 1
    capture.release()

    print(f"Video: {video_path.name} ({total_frames} frames, {width}x{height})")
    print(f"Raw Haar Cascade detections (unfiltered): {len(all_raw_detections)} faces")
    print(f"Upgraded Pipeline tracked primary face:     {len(tracked_face_crops)} coherent frames")

    # Evaluate predictions with Champion Model
    frame_logits = []
    frame_probs = []
    with torch.no_grad():
        for crop in tracked_face_crops:
            rgb = np.ascontiguousarray(crop[..., ::-1]).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            norm_t = (t - mean) / std
            logit = champion_model(norm_t)[0]
            prob = torch.softmax(logit, dim=-1)
            frame_logits.append(logit.cpu().numpy())
            frame_probs.append(prob.cpu().numpy())

    frame_logits = np.array(frame_logits)
    frame_probs = np.array(frame_probs)

    # 1. Unweighted arithmetic mean
    mean_probs = frame_probs.mean(axis=0)
    mean_pred_idx = int(mean_probs.argmax())

    # 2. Top-K pooled prediction (Top-K highest activations per class)
    K = min(5, len(frame_logits))
    top_k_logits = np.sort(frame_logits, axis=0)[-K:, :].mean(axis=0)
    top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    top_k_pred_idx = int(top_k_probs.argmax())

    print("\n--- Pipeline Upgrade Results on Case-Study Video (dia4_utt1.mp4, GT: neutral) ---")
    print("1. Historical Baseline (Buggy Haar Cascade + Un-normalized CNN):")
    print("   Prediction: neutral | Confidence: 15.2% | Faces counted: 52 | Entropy: 1.940 (Random Guessing)")

    print("\n2. Upgraded ResNet-18 with Tracking (Arithmetic Mean):")
    print(f"   Prediction: {EMOTION_CLASSES[mean_pred_idx]} | Confidence: {mean_probs[mean_pred_idx]*100:.2f}% | Coherent frames: {len(tracked_face_crops)}")
    for cls, p in zip(EMOTION_CLASSES, mean_probs):
        print(f"     {cls:10s}: {p*100:5.2f}%")

    print(f"\n3. Upgraded ResNet-18 with Top-{K} Emotion Pooling:")
    print(f"   Prediction: {EMOTION_CLASSES[top_k_pred_idx]} | Confidence: {top_k_probs[top_k_pred_idx]*100:.2f}%")
    for cls, p in zip(EMOTION_CLASSES, top_k_probs):
        print(f"     {cls:10s}: {p*100:5.2f}%")

    entropy_upgraded = float(-(top_k_probs * np.log(np.maximum(top_k_probs, 1e-12))).sum())

    video_results = {
        "video": video_path.name,
        "ground_truth": "neutral",
        "raw_detections_count": len(all_raw_detections),
        "tracked_frames_count": len(tracked_face_crops),
        "tracked_metadata": tracked_metadata,
        "historical_confidence": 0.152,
        "upgraded_arithmetic_confidence": float(mean_probs.max()),
        "upgraded_arithmetic_pred": EMOTION_CLASSES[mean_pred_idx],
        "upgraded_arithmetic_probs": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, mean_probs)},
        "upgraded_top_k_confidence": float(top_k_probs.max()),
        "upgraded_top_k_pred": EMOTION_CLASSES[top_k_pred_idx],
        "upgraded_top_k_probs": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, top_k_probs)},
        "upgraded_entropy": entropy_upgraded,
    }
    with (EXP7_DIR / "video_pipeline_upgrade_results.json").open("w") as f:
        json.dump(video_results, f, indent=2)

    return video_results


def plot_all_experiments_comparison(all_results: dict[str, Any]):
    print("\nGenerating comprehensive 6-model comparison plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    models_order = [
        ("Historical Reference (Scratch CNN)", all_results["historical"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 1: Corrected CNN Baseline", all_results["exp1"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 2: Class-Weighted CNN", all_results["exp2"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 3: ResNet-18 (Weighted CE)", all_results["exp3"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 4: EfficientNet-B0 (Weighted CE)", all_results["exp4"]["final_dev_metrics"]["confusion_matrix"]),
        ("Exp 5: ResNet-18 (Focal Loss)", all_results["exp5"]["final_dev_metrics"]["confusion_matrix"]),
    ]

    for idx, (title, cm_list) in enumerate(models_order):
        ax = axes[idx]
        cm = np.array(cm_list)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=11, fontweight="bold")
        tick_marks = np.arange(len(EMOTION_CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(EMOTION_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(EMOTION_CLASSES, fontsize=8)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    cm_path = EXP_BASE / "all_experiments_confusion_matrices.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved 6-model confusion matrix comparison to: {cm_path}")

    # Training curves comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    palette = {
        "exp3": ("#059669", "Exp 3: ResNet-18 (Weighted CE)"),
        "exp4": ("#7c3aed", "Exp 4: EfficientNet-B0"),
        "exp5": ("#dc2626", "Exp 5: ResNet-18 (Focal Loss)"),
    }

    for key, (color, label) in palette.items():
        hist = all_results[key]["history"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        dev_f1 = [h["dev_macro_f1"] for h in hist]
        ax1.plot(epochs, train_loss, marker="o", color=color, label=label, linewidth=2)
        ax2.plot(epochs, dev_f1, marker="s", color=color, label=label, linewidth=2)

    ax1.set_title("Training Loss Progression (Pretrained Models)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Development Macro F1 Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = EXP_BASE / "pretrained_training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved pretrained curves comparison to: {curves_path}")


def main():
    print("=" * 80)
    print("PHASE 17B: EXPERIMENTS 4-7 EXECUTION RUNNER")
    print("=" * 80)
    print(f"Compute Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Preflight
    train_samples = load_split_data("train")
    dev_samples = load_split_data("dev")
    print(f"Loaded {len(train_samples)} training samples and {len(dev_samples)} development samples.")

    # 2. Run Experiment 4: EfficientNet-B0
    res4 = run_experiment_4(train_samples, dev_samples)

    # 3. Run Experiment 5: ResNet-18 with Focal Loss
    res5 = run_experiment_5(train_samples, dev_samples)

    # 4. Load Previous Results for Experiment 6 Comparison
    exp_master_path = EXP_BASE / "experiment_results.json"
    assert exp_master_path.is_file(), "Previous results.json missing!"
    with exp_master_path.open() as f:
        prev_results = json.load(f)

    # Combine all results
    all_results = {
        **prev_results,
        "exp4": res4,
        "exp5": res5,
    }

    # 5. Experiment 6: Select Champion Model
    candidates = [
        ("Exp 3: ResNet-18 (Weighted CE)", all_results["exp3"]["final_dev_metrics"]["macro_f1"], all_results["exp3"]["checkpoint_path"], "resnet18"),
        ("Exp 4: EfficientNet-B0 (Weighted CE)", all_results["exp4"]["final_dev_metrics"]["macro_f1"], all_results["exp4"]["checkpoint_path"], "efficientnet_b0"),
        ("Exp 5: ResNet-18 (Focal Loss)", all_results["exp5"]["final_dev_metrics"]["macro_f1"], all_results["exp5"]["checkpoint_path"], "resnet18"),
    ]
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    champion_name, champion_f1, champion_ckpt_path, champion_arch = candidates_sorted[0]

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: CHAMPION BACKBONE SELECTION")
    print("=" * 70)
    for name, f1, _, arch in candidates_sorted:
        print(f"  {name:38s} -> Dev Macro F1: {f1:.4f} ({arch})")
    print(f"\nSELECTED CHAMPION: {champion_name} (Macro F1 = {champion_f1:.4f})")

    # Load champion model weights
    if champion_arch == "resnet18":
        champion_model = models.resnet18(weights=None)
        champion_model.fc = nn.Linear(champion_model.fc.in_features, len(EMOTION_CLASSES))
    else:
        champion_model = models.efficientnet_b0(weights=None)
        champion_model.classifier[1] = nn.Linear(champion_model.classifier[1].in_features, len(EMOTION_CLASSES))

    champion_ckpt = torch.load(champion_ckpt_path, map_location=DEVICE, weights_only=True)
    champion_model.load_state_dict(champion_ckpt["state_dict"])
    champion_model.to(DEVICE).eval()

    # 6. Run Experiment 7: Video Pipeline Upgrade
    res7 = run_experiment_7_video_pipeline(champion_model)
    all_results["exp7_video_pipeline"] = res7
    all_results["champion_model"] = {
        "name": champion_name,
        "macro_f1": champion_f1,
        "checkpoint_path": champion_ckpt_path,
        "architecture": champion_arch,
    }

    # Save updated master results
    with exp_master_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated all experimental results in: {exp_master_path}")

    # 7. Generate 6-model comparison plots
    plot_all_experiments_comparison(all_results)

    print("\n" + "=" * 80)
    print("PHASE 17B EXPERIMENTS 4-7 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

