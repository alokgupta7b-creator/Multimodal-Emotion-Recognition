"""Train and evaluate the facial CNN on validated MELD video manifests."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from facial_emotion.config import EMOTION_CLASSES, FACIAL_MODEL_ROOT
from facial_emotion.detector import FaceDetector
from facial_emotion.model import create_model, preprocess_face_crops, temporal_logits
from project_config import MELD_PREPARED_ROOT


def _torch() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except (ImportError, OSError) as exc:
        raise RuntimeError("Facial training requires a working PyTorch installation.") from exc
    return torch, DataLoader, Dataset


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _faces_from_video(path: Path, detector: FaceDetector, count: int = 4, size: int = 96) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return []
        indices = sorted({round(i * (frame_count - 1) / max(1, count - 1)) for i in range(count)})
        selected: list[np.ndarray] = []
        previous_center: tuple[float, float] | None = None
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            faces = detector.detect_and_align(frame, size=size)
            if not faces:
                continue
            if previous_center is None:
                face = faces[len(faces) // 2]
            else:
                center = np.asarray(previous_center)
                face = min(
                    faces,
                    key=lambda crop: float(np.linalg.norm(np.asarray([crop.shape[1] / 2, crop.shape[0] / 2]) - center)),
                )
            selected.append(face)
            previous_center = (face.shape[1] / 2, face.shape[0] / 2)
        return selected
    finally:
        capture.release()


def _load_rows(split: str, prepared_root: Path) -> pd.DataFrame:
    path = prepared_root / "reports" / split / "video_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Validated video manifest not found: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame["status"].eq("ok")].copy()
    frame["label"] = frame["emotion"].astype(str).str.lower().map({name: i for i, name in enumerate(EMOTION_CLASSES)})
    if frame["label"].isna().any():
        raise ValueError(f"Unknown facial labels in {split} manifest")
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def _build_cache(rows: pd.DataFrame, split: str, cache_root: Path, detector: FaceDetector, frame_count: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    cache_dir = cache_root / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.to_dict("records"):
        sample_id = str(row["sample_id"])
        crop_path = cache_dir / f"{sample_id}_{frame_count}f.npy"
        if crop_path.is_file():
            try:
                crop = np.load(crop_path, allow_pickle=False)
            except (OSError, ValueError):
                crop = None
            if crop is not None and crop.shape == (frame_count, 96, 96, 3) and crop.dtype == np.uint8:
                valid.append({**row, "crop_path": crop_path, "face_status": "reused", "detected_frames": frame_count})
                continue
        video_path = Path(str(row["resolved_video_path"]))
        faces = _faces_from_video(video_path, detector, count=frame_count)
        if not faces:
            excluded.append({**row, "face_status": "no_detectable_face", "reason": "No face detected in sampled frames", "sampled_frames": frame_count})
            continue
        coverage = len(faces) / frame_count
        while len(faces) < frame_count:
            faces.append(faces[-1])
        temporary_path = crop_path.with_name(f"{crop_path.name}.tmp")
        with temporary_path.open("wb") as handle:
            np.save(handle, np.stack(faces))
        temporary_path.replace(crop_path)
        valid.append({**row, "crop_path": crop_path, "face_status": "extracted", "detected_frames": int(round(coverage * frame_count)), "sampled_frames": frame_count, "face_detection_coverage": coverage})
    return valid, excluded


def _dataset(rows: list[dict[str, Any]], torch: Any, Dataset: Any) -> Any:
    class FaceDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            crop = np.load(rows[index]["crop_path"])
            if getattr(self, "training", False) and random.random() < 0.5:
                crop = crop[:, :, ::-1, :].copy()
            tensor = preprocess_face_crops(crop, torch)
            return tensor, int(rows[index]["label"])
    return FaceDataset()


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="weighted", zero_division=0)),
        "per_class": classification_report(labels, predictions, labels=list(range(7)), target_names=list(EMOTION_CLASSES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(7))).tolist(),
    }


def _evaluate(model: Any, loader: Any, torch: Any, device: Any, criterion: Any | None = None) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    criterion = criterion or torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            logits = temporal_logits(model, inputs)
            total_loss += float(criterion(logits, target).item()) * len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return total_loss / max(1, len(labels)), _metrics(labels, predictions)


def train_facial_model(
    prepared_root: Path = MELD_PREPARED_ROOT,
    output_dir: Path = Path("models/facial_emotion_phase9"),
    epochs: int = 8,
    batch_size: int = 32,
    patience: int = 2,
    seed: int = 42,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    torch, DataLoader, Dataset = _torch()
    _seed_everything(seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = FaceDetector()
    cache_root = Path(output_dir) / "face_cache"
    train_rows, train_excluded = _build_cache(_load_rows("train", prepared_root), "train", cache_root, detector)
    dev_rows, dev_excluded = _build_cache(_load_rows("dev", prepared_root), "dev", cache_root, detector)
    if evaluate_test:
        test_rows, test_excluded = _build_cache(_load_rows("test", prepared_root), "test", cache_root, detector)
    else:
        test_rows, test_excluded = [], []
    if not train_rows or not dev_rows or (evaluate_test and not test_rows):
        raise RuntimeError("Face preparation produced an empty required split.")
    train_dataset = _dataset(train_rows, torch, Dataset)
    counts = np.bincount([row["label"] for row in train_rows], minlength=7)
    sample_weights = [1.0 / max(1, counts[row["label"]]) for row in train_rows]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_rows), replacement=True)
    train_dataset.training = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(_dataset(dev_rows, torch, Dataset), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = (
        DataLoader(_dataset(test_rows, torch, Dataset), batch_size=batch_size, shuffle=False, num_workers=0)
        if evaluate_test
        else None
    )
    model = create_model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1 = -1.0
    stale = 0
    best_path = Path(output_dir) / "facial_cnn_best.pt"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(temporal_logits(model, inputs), target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(target)
        dev_loss, dev_metrics = _evaluate(model, dev_loader, torch, device, criterion=torch.nn.CrossEntropyLoss())
        record = {"epoch": epoch, "train_loss": train_loss / len(train_rows), "dev_loss": dev_loss, "dev_metrics": dev_metrics}
        history.append(record)
        if dev_metrics["macro_f1"] > best_f1:
            best_f1, stale = dev_metrics["macro_f1"], 0
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": list(EMOTION_CLASSES), "seed": seed, "input_normalization": True}, best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    started_eval = time.perf_counter()
    test_metrics = None
    if evaluate_test:
        _, test_metrics = _evaluate(model, test_loader, torch, device)
        latency_ms = (time.perf_counter() - started_eval) * 1000 / len(test_rows)
    else:
        latency_ms = None
    metadata = {
        "model": "facial-cnn-phase9-temporal-v1",
        "classes": list(EMOTION_CLASSES),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": {"epochs": epochs, "batch_size": batch_size, "patience": patience, "seed": seed, "image_size": 96, "frame_count": 4, "frame_sampling": "uniform; deterministic middle detected crop; missing detections padded with last crop", "balanced_sampler": True, "class_weighted_loss": False, "augmentation": "horizontal flip", "evaluate_test": evaluate_test},
        "counts": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows), "train_excluded_no_face": len(train_excluded), "dev_excluded_no_face": len(dev_excluded), "test_excluded_no_face": len(test_excluded)},
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "test_metrics": ({**test_metrics, "inference_latency_ms_per_sample": latency_ms} if test_metrics else None),
        "gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None,
        "excluded": {"train": train_excluded, "dev": dev_excluded, "test": test_excluded},
    }
    (Path(output_dir) / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=MELD_PREPARED_ROOT)
    parser.add_argument("--output", type=Path, default=Path("models/facial_emotion_phase9"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    train_facial_model(args.prepared_root, args.output, args.epochs, args.batch_size, evaluate_test=not args.skip_test)


if __name__ == "__main__":
    main()
