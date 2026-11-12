"""Phase P2 Facial Emotion Retraining: ResNet18 v2 with quality filtering,
realistic webcam/facial augmentations, and anti-collapse imbalance losses.

Safety invariants:
- Trains strictly on MELD `train` crops (never touches `dev_val` or held-out MELD `test`).
- Monitors epoch selection & calibration strictly on MELD `dev_cal` video-level detections.
- Evaluates final comparison out-of-sample on disjoint MELD `dev_val` videos.
- Writes artifacts strictly to `models/facial_emotion/resnet18_v2/`, never overwriting v1.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from facial_emotion.config import DEFAULT_IMAGE_SIZE, EMOTION_CLASSES, FACIAL_MODEL_ROOT
from facial_emotion.resnet_predictor import (
    DEFAULT_FALLBACK_CHECKPOINT,
    DEFAULT_PRIMARY_CHECKPOINT,
    FALLBACK_CHECKPOINT_SHA256,
    PRIMARY_CHECKPOINT_SHA256,
    aggregate_crop_outputs,
    build_resnet18_classifier,
    compute_sha256,
)
from project_config import MELD_EXTRACTED_ROOT, MELD_PREPARED_ROOT, REPOSITORY_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET18_V2_DIR = FACIAL_MODEL_ROOT / "resnet18_v2"
SPLIT_PATH = REPOSITORY_ROOT / "models" / "remediation_splits" / "meld_dev_cal_val_split.json"
EXP4_CACHE_DIR = REPOSITORY_ROOT / "models" / "facial_experiments" / "exp4_face_detection" / "cache"
EXP7_TEMPORAL_DIR = REPOSITORY_ROOT / "models" / "facial_experiments" / "exp7_temporal"

FORBIDDEN_TEST_TOKENS = (
    "output_repeated_splits_test",
    "test_manifest.csv",
    "test_sent_emo.csv",
    "reports/test",
    "reports\\test",
)


def _assert_no_test_path(path: Path) -> Path:
    lower = str(path).lower()
    for token in FORBIDDEN_TEST_TOKENS:
        if token in lower:
            raise PermissionError(f"SAFETY GATE BLOCK: Forbidden test path {path}")
    return path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def verify_v1_baselines_intact() -> dict[str, str]:
    """Ensure protected v1 facial checkpoints remain untouched before training."""
    p_sha = compute_sha256(DEFAULT_PRIMARY_CHECKPOINT)
    f_sha = compute_sha256(DEFAULT_FALLBACK_CHECKPOINT)
    if p_sha != PRIMARY_CHECKPOINT_SHA256:
        raise RuntimeError(f"Protected primary v1 checkpoint hash mismatch: {p_sha}")
    if f_sha != FALLBACK_CHECKPOINT_SHA256:
        raise RuntimeError(f"Protected fallback v1 checkpoint hash mismatch: {f_sha}")
    return {"primary_v1": p_sha, "fallback_v1": f_sha}


def augment_face_crop_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Apply realistic webcam-oriented augmentations without destroying expression semantics."""
    out = img_bgr.copy()
    h, w = out.shape[:2]

    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    if random.random() < 0.65:
        angle = random.uniform(-12.0, 12.0)
        scale = random.uniform(0.90, 1.08)
        tx = random.uniform(-0.05 * w, 0.05 * w)
        ty = random.uniform(-0.05 * h, 0.05 * h)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.7:
        alpha = random.uniform(0.80, 1.22)
        beta = random.uniform(-18.0, 18.0)
        gamma = random.uniform(0.85, 1.18)
        arr = np.clip(out.astype(np.float32) * alpha + beta, 0.0, 255.0)
        if abs(gamma - 1.0) > 0.02:
            arr = 255.0 * np.power(np.clip(arr / 255.0, 0.0, 1.0), gamma)
        out = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if random.random() < 0.25:
        # Simulate mild spectacles/webcam glare over upper-cheek/eye band OR small occlusion patch
        if random.random() < 0.5:
            overlay = out.astype(np.float32)
            cx = random.randint(int(0.25 * w), int(0.75 * w))
            cy = random.randint(int(0.22 * h), int(0.50 * h))
            rx = random.randint(8, 18)
            ry = random.randint(4, 10)
            yy, xx = np.ogrid[:h, :w]
            mask = (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2) <= 1.0
            strength = random.uniform(0.18, 0.38)
            overlay[mask] = overlay[mask] * (1.0 - strength) + 245.0 * strength
            out = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
        else:
            pw = random.randint(10, 20)
            ph = random.randint(8, 16)
            x0 = random.randint(0, max(0, w - pw))
            y0 = random.randint(0, max(0, h - ph))
            fill = np.mean(out, axis=(0, 1), keepdims=True).astype(np.uint8)
            out[y0 : y0 + ph, x0 : x0 + pw] = fill

    if random.random() < 0.15:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=random.uniform(0.4, 0.9))

    return out


class FacialV2Dataset(Dataset):
    """BGR uint8 face-crop dataset with strict ImageNet RGB normalization."""

    def __init__(self, crops_bgr: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        self.crops_bgr = crops_bgr
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.crops_bgr[idx]
        if self.augment:
            img = augment_face_crop_bgr(img)
        rgb = np.ascontiguousarray(img[..., ::-1], dtype=np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return (t - self.mean) / self.std, int(self.labels[idx])


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Menon et al. Logit-Adjusted Cross-Entropy preventing rare-class collapse without flat entropy."""

    def __init__(self, class_priors: np.ndarray, tau: float = 0.5, label_smoothing: float = 0.04) -> None:
        super().__init__()
        log_prior = torch.log(torch.tensor(np.clip(class_priors, 1e-8, 1.0), dtype=torch.float32))
        self.register_buffer("adjustment", float(tau) * log_prior)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.adjustment.to(logits.device)
        return F.cross_entropy(adjusted, targets, label_smoothing=self.label_smoothing)


def load_clean_train_crops() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load strictly MELD train MTCNN crops with conservative quality filtering."""
    npz_path = _assert_no_test_path(EXP4_CACHE_DIR / "train_mtcnn_crops.npz")
    meta_path = _assert_no_test_path(EXP4_CACHE_DIR / "train_mtcnn_meta.json")
    split_cfg = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    forbidden_dev_sids = set(split_cfg["cal_sample_ids"]) | set(split_cfg["val_sample_ids"])

    train_npz = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recs = meta["records"]
    sids = [str(s) for s in train_npz["sample_ids"]]
    if set(sids) & forbidden_dev_sids:
        raise RuntimeError("Train/dev contamination detected!")

    probs = np.array([float(r.get("prob", 1.0)) for r in recs], dtype=np.float32)
    yaws = np.array([abs(float(r.get("yaw_asym", 0.0))) for r in recs], dtype=np.float32)
    widths = np.array([float(r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) for r in recs], dtype=np.float32)

    keep = (probs >= 0.94) & (widths >= 28.0) & (yaws <= 1.2)
    crops = train_npz["unaligned"][keep]
    labels = train_npz["labels"][keep].astype(np.int64)
    extra_count = 0
    multiframe_path = RESNET18_V2_DIR / "train_multiframe_crops.npz"
    if multiframe_path.is_file():
        extra_npz = np.load(multiframe_path)
        extra_sids = [str(s) for s in extra_npz["sample_ids"]]
        if set(extra_sids) & forbidden_dev_sids:
            raise RuntimeError("Train/dev contamination detected in multiframe cache!")
        extra_crops = extra_npz["crops"]
        extra_labels = extra_npz["labels"].astype(np.int64)
        extra_count = int(len(extra_labels))
        crops = np.concatenate([crops, extra_crops], axis=0)
        labels = np.concatenate([labels, extra_labels], axis=0)

    stats = {
        "total_cached_train_crops": int(len(sids)),
        "filtered_base_train_crops": int(keep.sum()),
        "multiframe_extra_crops": extra_count,
        "filtered_train_crops": int(len(labels)),
        "dropped_low_quality": int((~keep).sum()),
        "filter_rule": "prob>=0.94 and bbox_w>=28 and |yaw_asym|<=1.2",
        "class_counts": {EMOTION_CLASSES[i]: int((labels == i).sum()) for i in range(7)},
    }
    return crops, labels, stats


def load_dev_video_cache() -> dict[str, Any]:
    """Load pre-extracted MELD dev multi-frame detections split cleanly into dev_cal and dev_val."""
    split_cfg = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    cal_sids = set(split_cfg["cal_sample_ids"])
    val_sids = set(split_cfg["val_sample_ids"])

    meta = json.loads(_assert_no_test_path(EXP7_TEMPORAL_DIR / "dev_temporal_unaligned_meta.json").read_text(encoding="utf-8"))
    crops = np.load(_assert_no_test_path(EXP7_TEMPORAL_DIR / "dev_temporal_unaligned_crops.npz"))["crops"]

    cal_videos: list[dict[str, Any]] = []
    val_videos: list[dict[str, Any]] = []
    for v in meta["videos"]:
        if not v["detected"]:
            continue
        entry = {
            "sample_id": v["sample_id"],
            "label": int(v["label"]),
            "indices": np.asarray(v["all_det_crop_indices"], dtype=np.int64),
        }
        if v["sample_id"] in cal_sids:
            cal_videos.append(entry)
        elif v["sample_id"] in val_sids:
            val_videos.append(entry)

    return {"all_crops": crops, "cal_videos": cal_videos, "val_videos": val_videos}


@torch.no_grad()
def evaluate_video_subset(
    model: nn.Module,
    crop_loader: DataLoader,
    videos: list[dict[str, Any]],
    device: torch.device,
    pooling: str = "raw_all_detections_mean_prob",
) -> dict[str, Any]:
    """Evaluate ResNet18 strictly in IEEE FP32 across all MELD dev detections per video."""
    model.eval()
    logits_chunks: list[np.ndarray] = []
    with torch.amp.autocast("cuda", enabled=False):
        for x, _y in crop_loader:
            out = model(x.to(device, dtype=torch.float32))
            logits_chunks.append(out.detach().cpu().numpy().astype(np.float64))
    all_logits = np.concatenate(logits_chunks, axis=0)

    n_vids = len(videos)
    probs_v = np.zeros((n_vids, 7), dtype=np.float64)
    mean_logits_v = np.zeros((n_vids, 7), dtype=np.float64)
    y_true = np.zeros(n_vids, dtype=np.int64)
    for idx, v in enumerate(videos):
        v_logits = all_logits[v["indices"]]
        probs_v[idx] = aggregate_crop_outputs(v_logits, method=pooling)
        mean_logits_v[idx] = np.mean(v_logits, axis=0)
        y_true[idx] = v["label"]

    y_pred = probs_v.argmax(axis=1)
    conf = probs_v.max(axis=1)
    correct = (y_pred == y_true).astype(np.float64)
    clipped = np.clip(probs_v, 1e-12, 1.0)
    nll = float(-np.mean(np.log(clipped[np.arange(n_vids), y_true])))
    one_hot = np.eye(7, dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs_v - one_hot) ** 2, axis=1)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for bi in range(15):
        m = (conf >= edges[bi]) & (conf < edges[bi + 1] if bi < 14 else conf <= edges[bi + 1])
        if np.any(m):
            ece += (float(m.sum()) / n_vids) * abs(float(correct[m].mean()) - float(conf[m].mean()))

    per_cls = classification_report(
        y_true, y_pred, labels=list(range(7)), target_names=list(EMOTION_CLASSES), output_dict=True, zero_division=0
    )
    active_classes = int(sum(1 for c in EMOTION_CLASSES if np.sum(y_pred == EMOTION_CLASSES.index(c)) > 0))
    entropy = float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))
    return {
        "n_videos": n_vids,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(7)), average="weighted", zero_division=0)),
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf)),
        "low_conf_rate_lt45": float(np.mean(conf < 0.45)),
        "mean_entropy": entropy,
        "active_classes": active_classes,
        "per_class_f1": {c: float(per_cls[c]["f1-score"]) for c in EMOTION_CLASSES},
        "pred_histogram": {EMOTION_CLASSES[i]: int(np.sum(y_pred == i)) for i in range(7)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist(),
        "probs": probs_v,
        "mean_logits": mean_logits_v,
        "y_true": y_true,
    }


def _strip_dropout_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    clean: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k.replace("fc.1.", "fc.")
        clean[new_k] = v.detach().cpu().clone()
    return clean


def train_variant(
    *,
    variant_id: str,
    criterion: nn.Module,
    train_crops: np.ndarray,
    train_labels: np.ndarray,
    dev_cache: dict[str, Any],
    dev_crop_loader: DataLoader,
    batch_size: int = 64,
    head_epochs: int = 2,
    finetune_epochs: int = 10,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    finetune_head_lr: float = 4e-4,
    dropout_p: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one candidate ResNet18_v2 variant strictly selecting best epoch via dev_cal Macro-F1."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feats, len(EMOTION_CLASSES)))
    model = backbone.to(device)
    criterion = criterion.to(device)

    train_loader = DataLoader(
        FacialV2Dataset(train_crops, train_labels, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_score = -1.0
    best_epoch = 0
    best_clean_state: dict[str, torch.Tensor] | None = None
    best_cal_summary: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    total_epochs = head_epochs + finetune_epochs
    current_stage = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        stage = "head_only" if epoch <= head_epochs else "fine_tune"
        if stage != current_stage:
            current_stage = stage
            for n, p in model.named_parameters():
                p.requires_grad = n.startswith("fc.") if stage == "head_only" else True
            if stage == "head_only":
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)
                scheduler = None
            else:
                back_p = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
                head_p = list(model.fc.parameters())
                optimizer = torch.optim.AdamW(
                    [{"params": back_p, "lr": backbone_lr}, {"params": head_p, "lr": finetune_head_lr}],
                    weight_decay=3e-4,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

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

        # Monitor strictly on dev_cal multi-frame video predictions
        cal_res = evaluate_video_subset(model, dev_crop_loader, dev_cache["cal_videos"], device)
        ep_row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": running_loss / max(1, seen),
            "dev_cal_acc": cal_res["accuracy"],
            "dev_cal_macro_f1": cal_res["macro_f1"],
            "dev_cal_weighted_f1": cal_res["weighted_f1"],
            "dev_cal_ece": cal_res["ece"],
            "dev_cal_nll": cal_res["nll"],
            "dev_cal_active_classes": cal_res["active_classes"],
            "dev_cal_disgust_f1": cal_res["per_class_f1"]["disgust"],
            "dev_cal_fear_f1": cal_res["per_class_f1"]["fear"],
        }
        history.append(ep_row)
        print(
            f"  [{variant_id}] Ep {epoch:02d}/{total_epochs} ({stage:9s}) loss={ep_row['train_loss']:.4f} | "
            f"dev_cal Acc={cal_res['accuracy']*100:.2f}% MF1={cal_res['macro_f1']*100:.2f}% "
            f"WF1={cal_res['weighted_f1']*100:.2f}% Disg={cal_res['per_class_f1']['disgust']*100:.1f}% "
            f"Fear={cal_res['per_class_f1']['fear']*100:.1f}% act={cal_res['active_classes']}/7",
            flush=True,
        )

        # Selection score strictly on dev_cal (Macro-F1 + weighted-F1 stability + anti-collapse requirement)
        base_score = 0.75 * cal_res["macro_f1"] + 0.25 * cal_res["weighted_f1"]
        selection_score = base_score if cal_res["active_classes"] >= 5 else base_score * 0.5
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_clean_state = _strip_dropout_state_dict(model.state_dict())
            best_cal_summary = {k: v for k, v in cal_res.items() if k not in ("probs", "mean_logits", "y_true")}

    train_s = time.perf_counter() - started
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    assert best_clean_state is not None and best_cal_summary is not None

    return {
        "variant_id": variant_id,
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "train_seconds": float(train_s),
        "peak_vram_mb": peak_vram_mb,
        "state_dict": best_clean_state,
        "dev_cal_metrics": best_cal_summary,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2 ResNet18 v2 facial emotion recognition.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    v1_hashes = verify_v1_baselines_intact()
    RESNET18_V2_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train_crops, train_labels, train_stats = load_clean_train_crops()
    dev_cache = load_dev_video_cache()
    dev_crop_loader = DataLoader(
        FacialV2Dataset(dev_cache["all_crops"], np.zeros(len(dev_cache["all_crops"]), dtype=np.int64), augment=False),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    counts = np.bincount(train_labels, minlength=7).astype(np.float64)
    priors = counts / counts.sum()

    # Moderated class weights (power=0.35, capped at 3.0 to prevent extreme flat-entropy distortion)
    raw_wt = np.power(counts.sum() / np.maximum(counts, 1.0), 0.35)
    raw_wt = np.minimum(raw_wt / raw_wt.mean(), 3.0)
    mod_weights = torch.tensor(raw_wt, dtype=torch.float32)

    print(
        f"Loaded {len(train_labels)} clean train crops ({train_stats['class_counts']}); "
        f"dev_cal={len(dev_cache['cal_videos'])} vids, dev_val={len(dev_cache['val_videos'])} vids.",
        flush=True,
    )
    candidates = [
        ("la_ce_tau040", LogitAdjustedCrossEntropyLoss(priors, tau=0.40, label_smoothing=0.04)),
        ("la_ce_tau065", LogitAdjustedCrossEntropyLoss(priors, tau=0.65, label_smoothing=0.04)),
        ("mod_wt_ce_p035", nn.CrossEntropyLoss(weight=mod_weights, label_smoothing=0.03)),
    ]

    results: list[dict[str, Any]] = []
    for vid, crit in candidates:
        print(f"\n=== Training Facial V2 Variant: {vid} ===", flush=True)
        res = train_variant(
            variant_id=vid,
            criterion=crit,
            train_crops=train_crops,
            train_labels=train_labels,
            dev_cache=dev_cache,
            dev_crop_loader=dev_crop_loader,
            batch_size=args.batch_size,
            head_epochs=args.head_epochs,
            finetune_epochs=args.finetune_epochs,
            seed=args.seed,
        )
        results.append(res)

    # Select winner strictly on dev_cal metrics (never peeking at dev_val!)
    winner = max(
        results,
        key=lambda r: (
            r["dev_cal_metrics"]["active_classes"] >= 5,
            0.75 * r["dev_cal_metrics"]["macro_f1"] + 0.25 * r["dev_cal_metrics"]["weighted_f1"],
        ),
    )
    print(f"\n=== Selected Winning Variant Strictly From dev_cal: {winner['variant_id']} (ep={winner['best_epoch']}) ===", flush=True)

    ckpt_path = RESNET18_V2_DIR / "resnet18_v2_best.pt"
    ckpt_payload = {
        "state_dict": winner["state_dict"],
        "classes": list(EMOTION_CLASSES),
        "architecture": "resnet18",
        "model_identifier": f"facial-resnet18-v2-{winner['variant_id']}",
        "seed": args.seed,
        "best_epoch": winner["best_epoch"],
        "input_normalization": {
            "mode": "imagenet",
            "color_order": "BGR_to_RGB",
            "image_size": DEFAULT_IMAGE_SIZE,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "train_stats": train_stats,
    }
    torch.save(ckpt_payload, ckpt_path)
    ckpt_sha = compute_sha256(ckpt_path)

    # Reload into exact inference backbone + verify IEEE FP32 CPU vs CUDA parity & untouched dev_val
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model_gpu = build_resnet18_classifier(num_classes=7).to(device_cuda)
    loaded = torch.load(ckpt_path, map_location=device_cuda, weights_only=True)
    eval_model_gpu.load_state_dict(loaded["state_dict"], strict=True)
    eval_model_gpu.eval()

    eval_model_cpu = build_resnet18_classifier(num_classes=7)
    eval_model_cpu.load_state_dict(loaded["state_dict"], strict=True)
    eval_model_cpu.eval()

    with torch.no_grad():
        sample_x = next(iter(dev_crop_loader))[0][:32]
        probs_cpu = torch.softmax(eval_model_cpu(sample_x.float()), dim=-1).numpy()
        with torch.amp.autocast("cuda", enabled=False):
            probs_gpu = torch.softmax(eval_model_gpu(sample_x.to(device_cuda, dtype=torch.float32)), dim=-1).cpu().numpy()
        fp32_parity_diff = float(np.max(np.abs(probs_cpu - probs_gpu)))

    # Final untouched out-of-sample evaluation on dev_val (both pooling modes)
    t_inf0 = time.perf_counter()
    dev_val_raw = evaluate_video_subset(eval_model_gpu, dev_crop_loader, dev_cache["val_videos"], device_cuda, pooling="raw_all_detections_mean_prob")
    val_eval_ms = (time.perf_counter() - t_inf0) * 1000.0
    dev_val_logit = evaluate_video_subset(eval_model_gpu, dev_crop_loader, dev_cache["val_videos"], device_cuda, pooling="raw_mean_logit")

    # Save stripped summary JSON (excluding giant arrays)
    metadata = {
        "model": "ResNet18_v2",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "protected_v1_hashes_verified": v1_hashes,
        "selected_variant": winner["variant_id"],
        "best_epoch": winner["best_epoch"],
        "classes": list(EMOTION_CLASSES),
        "train_stats": train_stats,
        "candidates_dev_cal_summary": [
            {
                "variant_id": r["variant_id"],
                "best_epoch": r["best_epoch"],
                "train_seconds": r["train_seconds"],
                "peak_vram_mb": r["peak_vram_mb"],
                "dev_cal_metrics": r["dev_cal_metrics"],
                "history": r["history"],
            }
            for r in results
        ],
        "dev_val_uncalibrated_raw_prob": {k: v for k, v in dev_val_raw.items() if k not in ("probs", "mean_logits", "y_true")},
        "dev_val_uncalibrated_mean_logit": {k: v for k, v in dev_val_logit.items() if k not in ("probs", "mean_logits", "y_true")},
        "fp32_cpu_cuda_max_prob_diff": fp32_parity_diff,
        "dev_all_crops_forward_ms": val_eval_ms,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path = RESNET18_V2_DIR / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save dev_cal + dev_val video probs for clean downstream P1/MM comparison
    dev_cal_final = evaluate_video_subset(eval_model_gpu, dev_crop_loader, dev_cache["cal_videos"], device_cuda, pooling="raw_all_detections_mean_prob")
    np.savez_compressed(
        RESNET18_V2_DIR / "dev_video_outputs.npz",
        cal_probs=dev_cal_final["probs"],
        cal_mean_logits=dev_cal_final["mean_logits"],
        cal_y=dev_cal_final["y_true"],
        val_probs=dev_val_raw["probs"],
        val_mean_logits=dev_val_logit["mean_logits"],
        val_y=dev_val_raw["y_true"],
    )

    print(
        f"\n[Facial V2 dev_val Untouched] raw_all_detections_mean_prob: "
        f"Acc={dev_val_raw['accuracy']*100:.2f}% MF1={dev_val_raw['macro_f1']*100:.2f}% "
        f"WF1={dev_val_raw['weighted_f1']*100:.2f}% ECE={dev_val_raw['ece']:.4f} NLL={dev_val_raw['nll']:.4f} "
        f"conf={dev_val_raw['mean_confidence']*100:.1f}% <45%={dev_val_raw['low_conf_rate_lt45']*100:.1f}% "
        f"active={dev_val_raw['active_classes']}/7 FP32_diff={fp32_parity_diff:.2e}"
    )
    print(f"  per_class_f1={ {k: round(v*100, 2) for k, v in dev_val_raw['per_class_f1'].items()} }")
    print(f"  pred_hist={dev_val_raw['pred_histogram']}")


if __name__ == "__main__":
    main()

