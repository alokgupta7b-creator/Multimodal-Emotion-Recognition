"""DistilBERT text emotion training and inference.

The TF-IDF baseline remains independent in ``text_emotion.train``. This
module owns only the transformer model and stores it under
``models/text_emotion/distilbert``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from project_config import MELD_SPLITS, TEXT_MODEL_PATH, meld_annotation_path
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES
from text_emotion.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

DISTILBERT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "text_emotion" / "distilbert"


@dataclass(frozen=True)
class DistilBertTrainingConfig:
    """Conservative defaults for a six-gigabyte GPU."""

    checkpoint: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 4
    patience: int = 2
    seed: int = 42
    use_fp16: bool = True
    gradient_checkpointing: bool = False


def _dependencies() -> tuple[Any, Any, Any, Any]:
    """Import optional transformer dependencies with an actionable error."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from torch.utils.data import DataLoader, Dataset
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "DistilBERT requires working PyTorch and Transformers installations. "
            "PyTorch initialization failed; check the PyTorch DLL/OpenMP installation."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, (DataLoader, Dataset)


def load_meld_frame(split: str) -> pd.DataFrame:
    """Load and normalize one MELD split without touching other splits."""
    frame = pd.read_csv(meld_annotation_path(split)).dropna(subset=["Utterance", "Emotion"]).copy()
    frame["text"] = frame["Utterance"].map(preprocess_text)
    frame["label_name"] = frame["Emotion"].astype(str).str.strip().str.lower()
    frame = frame[frame["text"] != ""].copy()
    unknown = sorted(set(frame["label_name"]) - set(MELD_CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown MELD labels in {split}: {unknown}")
    frame["label"] = frame["label_name"].map({label: i for i, label in enumerate(MELD_CLASS_NAMES)})
    return frame.reset_index(drop=True)


def _metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Calculate the required metrics in canonical class order."""
    target_names = list(MELD_CLASS_NAMES)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="weighted", zero_division=0)),
        "per_class": classification_report(labels, predictions, labels=list(range(7)), target_names=target_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(7)).copy()).tolist(),
    }


class _TextDataset:
    """Small adapter kept local so importing this module does not import torch."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: Any, max_length: int, torch: Any) -> None:
        self.items = [
            (tokenizer(text, truncation=True, max_length=max_length), int(label))
            for text, label in zip(texts, labels)
        ]
        self.torch = torch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded, label = self.items[index]
        return {key: value for key, value in encoded.items()} | {"labels": label}


def _collator(tokenizer: Any, torch: Any) -> Any:
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.tensor([item.pop("labels") for item in items], dtype=torch.long)
        batch = tokenizer.pad(items, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch
    return collate


def _evaluate(model: Any, loader: Any, torch: Any, device: Any) -> tuple[dict[str, Any], list[int], list[int]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return _metrics(labels, predictions), labels, predictions


def train_distilbert(config: DistilBertTrainingConfig = DistilBertTrainingConfig(), output_dir: Path = DISTILBERT_MODEL_DIR) -> dict[str, Any]:
    """Train on MELD train, select on dev, and evaluate once on test."""
    torch, AutoModelForSequenceClassification, AutoTokenizer, loaders = _dependencies()
    DataLoader, _ = loaders
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(config.use_fp16 and device.type == "cuda")

    train_frame = load_meld_frame("train")
    dev_frame = load_meld_frame("dev")
    test_frame = load_meld_frame("test")
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    label2id = {label: i for i, label in enumerate(MELD_CLASS_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=7,
        id2label={i: label for label, i in label2id.items()},
        label2id=label2id,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    train_loader = DataLoader(
        _TextDataset(train_frame["text"], train_frame["label"], tokenizer, config.max_length, torch),
        batch_size=config.batch_size, shuffle=True, collate_fn=_collator(tokenizer, torch),
    )
    dev_loader = DataLoader(
        _TextDataset(dev_frame["text"], dev_frame["label"], tokenizer, config.max_length, torch),
        batch_size=config.eval_batch_size, shuffle=False, collate_fn=_collator(tokenizer, torch),
    )
    test_loader = DataLoader(
        _TextDataset(test_frame["text"], test_frame["label"], tokenizer, config.max_length, torch),
        batch_size=config.eval_batch_size, shuffle=False, collate_fn=_collator(tokenizer, torch),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    best_f1 = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    best_dir = output_dir / "best"
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss = model(**batch).loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, _, _ = _evaluate(model, dev_loader, torch, device)
        history.append({"epoch": epoch, **dev_metrics})
        logger.info("DistilBERT epoch %s dev macro F1 %.4f", epoch, dev_metrics["macro_f1"])
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_metrics, test_labels, test_predictions = _evaluate(model, test_loader, torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "DistilBERT",
        "checkpoint": config.checkpoint,
        "classes": list(MELD_CLASS_NAMES),
        "config": asdict(config),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_samples": len(train_frame),
        "dev_samples": len(dev_frame),
        "test_samples": len(test_frame),
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - start,
        "dev_history": history,
        "test_metrics": test_metrics,
        "test_labels": test_labels,
        "test_predictions": test_predictions,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    return metadata


class DistilBertTextPredictor:
    """Reusable seven-class DistilBERT inference interface."""

    def __init__(self, model_dir: Path = DISTILBERT_MODEL_DIR / "best", device: str | None = None) -> None:
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self.torch, AutoModelForSequenceClassification, AutoTokenizer, _ = _dependencies()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"DistilBERT model directory not found: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        requested = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.device = self.torch.device(requested)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input must be a non-empty string.")
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Text input is empty after preprocessing.")
        with self.torch.no_grad():
            encoded = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)[0].cpu().numpy()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError("DistilBERT produced invalid probabilities.")
        probabilities /= probabilities.sum()
        label = str(MELD_CLASS_NAMES[int(probabilities.argmax())])
        return {
            "predicted_label": label,
            "confidence": float(probabilities.max()),
            "probabilities": {name: float(value) for name, value in zip(MELD_CLASS_NAMES, probabilities)},
            "model_identifier": "distilbert-base-uncased",
        }
