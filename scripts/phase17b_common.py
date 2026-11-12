"""Shared utilities, safety guards, models, losses, preprocessing, and training engine for Phase 17B/C/D."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from facial_emotion.config import CLASS_TO_INDEX, EMOTION_CLASSES  # noqa: E402
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT  # noqa: E402

SEED = 42
IMAGE_SIZE = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MINORITY_CLASSES = ("disgust", "fear", "sadness")

EXP_ROOT = REPOSITORY_ROOT / "models" / "facial_experiments"
EXP3_BASELINE_CKPT = EXP_ROOT / "exp3_resnet18" / "exp3_resnet18_best.pt"
PHASE8_PRODUCTION_CKPT = REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt"

# Expected hashes recorded before Phase 17B Exps 4-7 began
PROTECTED_ASSET_HASHES: dict[str, str] = {
    "models/facial_emotion/facial_cnn_best.pt": "85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760",
    "models/facial_experiments/exp1_corrected_cnn/exp1_corrected_cnn_best.pt": "a2be816a67bb094026bdf43fb31884ee11812882de5f2c8108f26262b0f89613",
    "models/facial_experiments/exp2_weighted_cnn/exp2_weighted_cnn_best.pt": "422a5934c5f1337733eeb9c5274f4bb52f19d1ad81b7d1656b0eb2d25cc8aa8a",
    "models/facial_experiments/exp3_resnet18/exp3_resnet18_best.pt": "106b524e837417373dd904518da3ad389287c843d93c7f92596c5b1daf2b1446",
    "facial_emotion/model.py": "08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b",
    "facial_emotion/detector.py": "24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d",
    "facial_emotion/remediation.py": "3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07",
    "api/fusion.py": "ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d",
}

_FORBIDDEN_TEST_TOKENS = (
    "output_repeated_splits_test",
    "test_manifest.csv",
    "test_sent_emo.csv",
    "face_cache/test",
    "face_cache\\test",
    "reports/test",
    "reports\\test",
    "facial_emotion_phase9_5_tracking/test",
    "facial_emotion_phase9_5_tracking\\test",
)


def assert_valid_split(split: str) -> str:
    """Block any access to the held-out MELD test split."""
    if split not in {"train", "dev"}:
        raise PermissionError(f"SAFETY GATE BLOCK: Forbidden split {split!r}. Only 'train' and 'dev' are permitted.")
    return split


def assert_no_test_path(path: Path | str) -> Path:
    """Reject any path pointing into held-out MELD test assets."""
    text = str(path)
    lower = text.lower()
    for token in _FORBIDDEN_TEST_TOKENS:
        if token in lower:
            raise PermissionError(f"SAFETY GATE BLOCK: Held-out test path access attempted: {path}")
    return Path(path)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protected_assets() -> dict[str, Any]:
    """Confirm no production or Phase 17A baseline file/checkpoint was modified."""
    checks: dict[str, Any] = {}
    all_ok = True
    for rel, expected in PROTECTED_ASSET_HASHES.items():
        p = REPOSITORY_ROOT / rel
        if not p.is_file():
            checks[rel] = {"ok": False, "expected": expected, "actual": "MISSING"}
            all_ok = False
            continue
        actual = sha256_file(p)
        ok = actual == expected
        if not ok:
            all_ok = False
        checks[rel] = {"ok": ok, "expected": expected, "actual": actual}
    if not all_ok:
        raise RuntimeError(f"INTEGRITY FAILURE: Protected asset checksum mismatch: {checks}")
    return {"all_ok": True, "files": checks}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_environment_info() -> dict[str, Any]:
    import facenet_pytorch
    import sklearn
    import torchvision

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cv2_version": cv2.__version__,
        "facenet_pytorch_version": getattr(facenet_pytorch, "__version__", "installed"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": SEED,
        "canonical_classes": list(EMOTION_CLASSES),
    }


def load_video_manifest(split: str) -> list[dict[str, Any]]:
    """Load MELD video manifest strictly for train or dev."""
    assert_valid_split(split)
    manifest_path = assert_no_test_path(MELD_PREPARED_ROOT / "reports" / split / "video_manifest.csv")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            emotion = row.get("emotion", "").strip().lower()
            if emotion not in CLASS_TO_INDEX:
                continue
            vpath = assert_no_test_path(row["resolved_video_path"])
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dialogue_id": int(row["dialogue_id"]) if row.get("dialogue_id", "").isdigit() else -1,
                    "utterance_id": int(row["utterance_id"]) if row.get("utterance_id", "").isdigit() else -1,
                    "emotion": emotion,
                    "label": int(CLASS_TO_INDEX[emotion]),
                    "video_path": str(vpath),
                    "duration_seconds": float(row["duration_seconds"]) if row.get("duration_seconds") else 0.0,
                    "frame_count": int(float(row["frame_count"])) if row.get("frame_count") else 0,
                    "width": int(float(row["width"])) if row.get("width") else 0,
                    "height": int(float(row["height"])) if row.get("height") else 0,
                }
            )
    return rows


def load_haar_cached_split(split: str) -> list[dict[str, Any]]:
    """Load original Phase 8 Haar face_cache metadata & paths for train or dev."""
    assert_valid_split(split)
    manifest_rows = {r["sample_id"]: r for r in load_video_manifest(split)}
    cache_dir = assert_no_test_path(REPOSITORY_ROOT / "models" / "facial_emotion" / "face_cache" / split)
    items: list[dict[str, Any]] = []
    for sample_id, row in manifest_rows.items():
        npy_path = cache_dir / f"{sample_id}.npy"
        if npy_path.is_file():
            items.append({**row, "crop_path": str(npy_path)})
    return items


# ---------------------------------------------------------------------------
# Face crop & landmark alignment helpers
# ---------------------------------------------------------------------------


def crop_unaligned_square(
    frame_bgr: np.ndarray,
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    size: int = IMAGE_SIZE,
    margin: float = 0.08,
) -> np.ndarray | None:
    """Extract a clean square-padded BGR crop around an (x1, y1, x2, y2) detection box."""
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    nx1 = int(max(0, np.floor(cx - side / 2.0)))
    ny1 = int(max(0, np.floor(cy - side / 2.0)))
    nx2 = int(min(w_img, np.ceil(cx + side / 2.0)))
    ny2 = int(min(h_img, np.ceil(cy + side / 2.0)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    interp = cv2.INTER_CUBIC if crop.shape[0] < size or crop.shape[1] < size else cv2.INTER_AREA
    return cv2.resize(crop, (size, size), interpolation=interp)


def align_face_robust(
    frame_bgr: np.ndarray,
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    landmarks_5pt: np.ndarray,
    size: int = IMAGE_SIZE,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Similarity-align a face crop using 5-point landmarks with yaw-safe scale clamping."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pts = np.asarray(landmarks_5pt, dtype=np.float32)
    if pts.shape != (5, 2):
        return None, {"valid": False}
    le, re, _nose, lm, rm = pts
    eye_dx = float(re[0] - le[0])
    eye_dy = float(re[1] - le[1])
    eye_dist = float(np.hypot(eye_dx, eye_dy))
    eyes_center = (float(0.5 * (le[0] + re[0])), float(0.5 * (le[1] + re[1])))
    mouth_center = (float(0.5 * (lm[0] + rm[0])), float(0.5 * (lm[1] + rm[1])))
    v_face = float(np.hypot(mouth_center[0] - eyes_center[0], mouth_center[1] - eyes_center[1]))

    if eye_dx > 0.18 * bw:
        angle_deg = float(np.clip(np.degrees(np.arctan2(eye_dy, eye_dx)), -35.0, 35.0))
    else:
        angle_deg = 0.0

    scale_candidates = [0.80 * size / max(bw, bh)]
    if eye_dist > 0.22 * bw:
        scale_candidates.append(0.35 * size / eye_dist)
    if v_face > 0.20 * bh:
        scale_candidates.append(0.36 * size / v_face)
    scale = float(min(scale_candidates))

    M = cv2.getRotationMatrix2D(eyes_center, angle_deg, scale)
    M[0, 2] += 0.50 * size - eyes_center[0]
    M[1, 2] += 0.36 * size - eyes_center[1]
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (size, size),
        flags=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )
    pts_h = np.hstack([pts, np.ones((5, 1), dtype=np.float32)])
    pts_out = pts_h @ M.T
    meta = {
        "valid": True,
        "roll_deg": angle_deg,
        "scale": scale,
        "eye_dist_orig": eye_dist,
        "v_face_orig": v_face,
        "mouth_y_aligned": float(0.5 * (pts_out[3, 1] + pts_out[4, 1])),
        "eyes_y_aligned": float(0.5 * (pts_out[0, 1] + pts_out[1, 1])),
    }
    return aligned, meta


def compute_crop_quality(crop_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return {"sharpness": sharpness, "brightness": brightness, "contrast": contrast}


def preprocess_bgr_crops(
    crops_bgr: np.ndarray,
    mode: str = "imagenet",
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Strictly separated preprocessing by architecture family.

    - mode='imagenet': BGR->RGB, [0, 1], ImageNet mean/std (ResNet18, EfficientNet-B0)
    - mode='legacy_cnn': BGR->RGB, (x/255.0 - 0.5) / 0.5 in [-1, 1] (Legacy CNN ONLY)
    """
    arr = np.asarray(crops_bgr)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3) BGR image batch, got {arr.shape}")
    if arr.shape[1] != image_size or arr.shape[2] != image_size:
        resized = [cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA) for img in arr]
        arr = np.stack(resized, axis=0)

    rgb = np.ascontiguousarray(arr[..., ::-1], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
    if mode == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        return (tensor - mean) / std
    if mode == "legacy_cnn":
        return (tensor - 0.5) / 0.5
    raise ValueError(f"Unsupported preprocessing mode: {mode!r}")


def get_preprocessing_config(mode: str, detector: str = "mtcnn", alignment: str = "none") -> dict[str, Any]:
    if mode == "imagenet":
        return {
            "mode": "imagenet",
            "color_order": "BGR_to_RGB",
            "image_size": IMAGE_SIZE,
            "scale": "1/255.0",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "detector": detector,
            "alignment": alignment,
        }
    if mode == "legacy_cnn":
        return {
            "mode": "legacy_cnn",
            "color_order": "BGR_to_RGB",
            "image_size": IMAGE_SIZE,
            "scale": "(x / 255.0 - 0.5) / 0.5",
            "range": [-1.0, 1.0],
            "detector": detector,
            "alignment": alignment,
        }
    raise ValueError(f"Unknown preprocessing mode: {mode!r}")


# ---------------------------------------------------------------------------
# Dataset, Backbones, and Loss Functions
# ---------------------------------------------------------------------------


class ArrayFaceDataset(Dataset):
    """Fast in-memory uint8 BGR array dataset with explicit preprocessing & safe augmentation."""

    def __init__(
        self,
        crops_bgr: np.ndarray,
        labels: np.ndarray,
        preprocessing_mode: str = "imagenet",
        is_training: bool = False,
        augment: bool = False,
    ) -> None:
        if preprocessing_mode not in {"imagenet", "legacy_cnn"}:
            raise ValueError(f"Invalid preprocessing_mode={preprocessing_mode!r}")
        self.crops_bgr = crops_bgr
        self.labels = np.asarray(labels, dtype=np.int64)
        self.preprocessing_mode = preprocessing_mode
        self.is_training = is_training
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_bgr = self.crops_bgr[idx]
        if self.is_training and self.augment:
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)
            # Mild brightness/contrast jitter (never heavy distortion)
            if random.random() < 0.5:
                alpha = random.uniform(0.90, 1.10)
                beta = random.uniform(-10.0, 10.0)
                img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        elif self.is_training:
            # Match Phase 17A Exp 3 horizontal flip default when augment=False
            if random.random() < 0.5:
                img_bgr = cv2.flip(img_bgr, 1)

        rgb = np.ascontiguousarray(img_bgr[..., ::-1], dtype=np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        if self.preprocessing_mode == "imagenet":
            t = (t - self.mean) / self.std
        else:
            t = (t - 0.5) / 0.5
        return t, int(self.labels[idx])


def build_backbone(arch: str, pretrained: bool = True, num_classes: int = 7) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone architecture: {arch!r}")


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights (alpha)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        targets_unsq = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets_unsq).squeeze(1)
        pt = probs.gather(1, targets_unsq).squeeze(1).clamp(1e-6, 1.0)
        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt
        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_smoothed_class_weights(labels: np.ndarray, power: float = 0.5, normalize_mean: bool = True) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(EMOTION_CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw = (counts.sum() / counts) ** power
    if normalize_mean:
        raw = raw / raw.mean()
    return torch.tensor(raw, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics, Evaluation, Latency, and Visualization
# ---------------------------------------------------------------------------


def compute_detailed_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    probs: list[np.ndarray] | np.ndarray,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=list(EMOTION_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(EMOTION_CLASSES)))).tolist()
    pred_counts = np.bincount(y_pred_arr, minlength=len(EMOTION_CLASSES))
    active_classes = int((pred_counts > 0).sum())
    max_pred_share = float(pred_counts.max() / max(1, len(y_pred_arr)))
    collapse_detected = bool(active_classes <= 2 or max_pred_share >= 0.85)

    per_class = {
        cls_name: {
            "precision": float(report[cls_name]["precision"]),
            "recall": float(report[cls_name]["recall"]),
            "f1-score": float(report[cls_name]["f1-score"]),
            "support": int(report[cls_name]["support"]),
        }
        for cls_name in EMOTION_CLASSES
    }
    minority_f1 = float(np.mean([per_class[c]["f1-score"] for c in MINORITY_CLASSES]))
    minority_recall = float(np.mean([per_class[c]["recall"] for c in MINORITY_CLASSES]))

    clipped_probs = np.clip(probs_arr, 1e-12, 1.0)
    entropies = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)

    return {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "minority_macro_f1": minority_f1,
        "minority_macro_recall": minority_recall,
        "active_classes": active_classes,
        "max_predicted_class_share": max_pred_share,
        "collapse_detected": collapse_detected,
        "mean_entropy": float(np.mean(entropies)),
        "per_class": per_class,
        "confusion_matrix": cm,
        "mean_predicted_probabilities": {
            cls_name: float(probs_arr[:, i].mean()) for i, cls_name in enumerate(EMOTION_CLASSES)
        },
        "predicted_counts": {cls_name: int(pred_counts[i]) for i, cls_name in enumerate(EMOTION_CLASSES)},
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_count += bs
        all_targets.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    probs_arr = np.concatenate(all_probs)
    logits_arr = np.concatenate(all_logits)
    metrics = compute_detailed_metrics(y_true, y_pred, probs_arr)
    return total_loss / max(1, total_count), metrics, probs_arr, logits_arr


def measure_inference_latency_and_memory(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    warmup: int = 20,
    runs: int = 200,
) -> dict[str, float]:
    model = model.to(device).eval()
    x = torch.randn((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    return {
        "latency_ms_per_crop": float(elapsed_ms / (runs * batch_size)),
        "inference_peak_vram_mb": peak_mb,
    }


def save_confusion_matrix_plot(cm: list[list[int]] | np.ndarray, title: str, out_path: Path) -> None:
    arr = np.asarray(cm, dtype=np.float64)
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, np.maximum(row_sums, 1.0))
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=140)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-Normalized Recall")
    ax.set_xticks(range(len(EMOTION_CLASSES)))
    ax.set_yticks(range(len(EMOTION_CLASSES)))
    ax.set_xticklabels(EMOTION_CLASSES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(EMOTION_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    ax.set_title(title, fontsize=11, pad=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j])
            txt_color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=txt_color, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_training_curves_plot(history: list[dict[str, Any]], title: str, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    macro_f1 = [h["dev_macro_f1"] for h in history]
    weighted_f1 = [h["dev_weighted_f1"] for h in history]
    acc = [h["dev_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    ax1.plot(epochs, train_loss, marker="o", label="Train Loss")
    ax1.plot(epochs, dev_loss, marker="s", label="Dev Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, macro_f1, marker="o", linewidth=2.0, label="Dev Macro F1")
    ax2.plot(epochs, weighted_f1, marker="s", label="Dev Weighted F1")
    ax2.plot(epochs, acc, marker="^", linestyle="--", label="Dev Accuracy")
    ax2.set_title("Dev Evaluation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unified Two-Stage Training Runner
# ---------------------------------------------------------------------------


def train_backbone_experiment(
    *,
    experiment_id: str,
    experiment_name: str,
    arch: str,
    out_dir: Path,
    ckpt_filename: str,
    train_crops: np.ndarray,
    train_labels: np.ndarray,
    dev_crops: np.ndarray,
    dev_labels: np.ndarray,
    criterion_fn,
    preprocessing_config: dict[str, Any],
    batch_size: int = 32,
    head_epochs: int = 3,
    finetune_epochs: int = 5,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    finetune_head_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 4,
    augment: bool = False,
    seed: int = SEED,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a 7-class image backbone with 2-stage transfer learning, Macro-F1 early stopping, and full audit logs."""
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = preprocessing_config["mode"]
    if mode != "imagenet" and arch in {"resnet18", "efficientnet_b0"}:
        raise ValueError(f"SAFETY GATE: {arch} requires 'imagenet' normalization, got {mode!r}")

    train_ds = ArrayFaceDataset(train_crops, train_labels, preprocessing_mode=mode, is_training=True, augment=augment)
    dev_ds = ArrayFaceDataset(dev_crops, dev_labels, preprocessing_mode=mode, is_training=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_backbone(arch, pretrained=True, num_classes=len(EMOTION_CLASSES)).to(device)
    param_count = int(sum(p.numel() for p in model.parameters()))
    criterion = criterion_fn(device)

    def _set_head_only(is_head_only: bool) -> None:
        if arch == "resnet18":
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith("fc.") if is_head_only else True
        elif arch == "efficientnet_b0":
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith("classifier.") if is_head_only else True

    def _make_optimizer(stage: str) -> torch.optim.Optimizer:
        if stage == "head_only":
            head_params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
        if arch == "resnet18":
            back_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
            head_params = list(model.fc.parameters())
        else:
            back_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
            head_params = list(model.classifier.parameters())
        return torch.optim.AdamW(
            [
                {"params": back_params, "lr": backbone_lr},
                {"params": head_params, "lr": finetune_head_lr},
            ],
            weight_decay=weight_decay,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_finetune = 0
    history: list[dict[str, Any]] = []
    ckpt_path = out_dir / ckpt_filename

    total_epochs = head_epochs + finetune_epochs
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    current_stage = None

    t_start = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        stage = "head_only" if epoch <= head_epochs else "fine_tune"
        if stage != current_stage:
            current_stage = stage
            _set_head_only(stage == "head_only")
            optimizer = _make_optimizer(stage)
            if stage == "fine_tune" and finetune_epochs > 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)
            else:
                scheduler = None

        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            bs = int(y.size(0))
            running_loss += float(loss.item()) * bs
            seen += bs

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, seen)
        dev_loss, dev_metrics, _, _ = evaluate_model(model, dev_loader, criterion, device)
        ep_sec = time.perf_counter() - ep_t0

        ep_record = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_minority_macro_f1": dev_metrics["minority_macro_f1"],
            "dev_minority_macro_recall": dev_metrics["minority_macro_recall"],
            "active_classes": dev_metrics["active_classes"],
            "collapse_detected": dev_metrics["collapse_detected"],
            "predicted_counts": dev_metrics["predicted_counts"],
            "epoch_seconds": ep_sec,
        }
        history.append(ep_record)
        print(
            f"  [{experiment_id}] Ep {epoch:02d}/{total_epochs} ({stage}) "
            f"tr_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            f"acc={dev_metrics['accuracy']:.4f} macroF1={dev_metrics['macro_f1']:.4f} "
            f"wF1={dev_metrics['weighted_f1']:.4f} minRec={dev_metrics['minority_macro_recall']:.4f} "
            f"active={dev_metrics['active_classes']}/7 ({ep_sec:.1f}s)"
        )

        # Model selection monitors dev Macro F1 (penalize single-class collapse only as tie-breaker if non-collapsed exists)
        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_finetune = 0
        elif stage == "fine_tune":
            stale_finetune += 1
            if stale_finetune >= patience:
                print(f"  [{experiment_id}] Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_macro_f1={best_macro_f1:.4f})")
                break

    train_time_seconds = time.perf_counter() - t_start
    train_peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a checkpoint for {experiment_id}")

    checkpoint_payload = {
        "state_dict": best_state,
        "classes": list(EMOTION_CLASSES),
        "seed": seed,
        "architecture": arch,
        "input_normalization": preprocessing_config,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "extra_config": extra_config or {},
    }
    torch.save(checkpoint_payload, ckpt_path)
    ckpt_sha256 = sha256_file(ckpt_path)

    # Restore best checkpoint & verify reproducibility
    reloaded = build_backbone(arch, pretrained=False, num_classes=len(EMOTION_CLASSES)).to(device)
    loaded_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if list(loaded_ckpt["classes"]) != list(EMOTION_CLASSES):
        raise AssertionError(f"Class mapping drift in {ckpt_path}: {loaded_ckpt['classes']}")
    reloaded.load_state_dict(loaded_ckpt["state_dict"])
    reloaded.eval()

    final_dev_loss, final_metrics, dev_probs, dev_logits = evaluate_model(reloaded, dev_loader, criterion, device)
    if abs(final_metrics["macro_f1"] - best_macro_f1) > 1e-6:
        raise AssertionError(
            f"Reproducibility check failed for {experiment_id}: "
            f"saved={best_macro_f1:.6f} vs reloaded={final_metrics['macro_f1']:.6f}"
        )

    cuda_latency = measure_inference_latency_and_memory(reloaded, device)
    cpu_model = build_backbone(arch, pretrained=False, num_classes=len(EMOTION_CLASSES))
    cpu_model.load_state_dict(loaded_ckpt["state_dict"])
    cpu_latency = measure_inference_latency_and_memory(cpu_model, torch.device("cpu"), warmup=10, runs=60)

    cm_path = out_dir / f"{experiment_id}_confusion_matrix.png"
    curve_path = out_dir / f"{experiment_id}_training_curves.png"
    cfg_path = out_dir / f"{experiment_id}_config.json"
    res_path = out_dir / f"{experiment_id}_results.json"

    save_confusion_matrix_plot(final_metrics["confusion_matrix"], f"{experiment_name} (Dev Macro F1={final_metrics['macro_f1']:.4f})", cm_path)
    save_training_curves_plot(history, f"{experiment_name}", curve_path)

    config_record = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "architecture": arch,
        "parameters": param_count,
        "seed": seed,
        "batch_size": batch_size,
        "head_epochs": head_epochs,
        "finetune_epochs": finetune_epochs,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "finetune_head_lr": finetune_head_lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "augment": augment,
        "preprocessing": preprocessing_config,
        "extra_config": extra_config or {},
        "environment": collect_environment_info(),
    }
    cfg_path.write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    result = {
        "experiment_id": experiment_id,
        "experiment": experiment_name,
        "architecture": arch,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "parameters": param_count,
        "train_samples": int(len(train_labels)),
        "dev_samples": int(len(dev_labels)),
        "train_time_seconds": float(train_time_seconds),
        "train_peak_vram_mb": train_peak_vram_mb,
        "latency_ms_cuda": cuda_latency["latency_ms_per_crop"],
        "inference_peak_vram_mb": cuda_latency["inference_peak_vram_mb"],
        "latency_ms_cpu": cpu_latency["latency_ms_per_crop"],
        "best_epoch": best_epoch,
        "final_dev_loss": final_dev_loss,
        "preprocessing": preprocessing_config,
        "extra_config": extra_config or {},
        "history": history,
        "final_dev_metrics": final_metrics,
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "config": str(cfg_path),
            "confusion_matrix": str(cm_path),
            "training_curves": str(curve_path),
            "results": str(res_path),
        },
    }
    res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Release GPU memory cleanly between runs
    del model, reloaded, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

