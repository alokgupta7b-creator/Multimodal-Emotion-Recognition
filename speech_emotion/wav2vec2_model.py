"""Wav2Vec2 speech emotion recognition.

This module is separate from the MFCC baseline. It uses MELD manifests for
split membership, the existing audio loader for deterministic 16 kHz mono
waveforms, and the shared seven-class label mapping.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from calibration_utils import (
    apply_logit_calibration,
    load_logit_calibration,
    resolve_calibration_path,
)
from project_config import MELD_MANIFESTS
from speech_emotion.audio_loader import load_audio
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES

WAV2VEC2_ROOT = Path(__file__).resolve().parent.parent / "models" / "speech_emotion" / "wav2vec2"


@dataclass(frozen=True)
class Wav2Vec2TrainingConfig:
    checkpoint: str = "facebook/wav2vec2-base"
    sample_rate: int = 16000
    max_duration_seconds: float = 8.0
    batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    epochs: int = 4
    patience: int = 2
    freeze_feature_encoder: bool = True
    use_fp16: bool = True
    seed: int = 42


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Wav2Vec2 requires working PyTorch and Transformers installations."
        ) from exc
    return torch, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, (DataLoader, Dataset)


def load_meld_audio_records(split: str, audio_root: Path) -> list[dict[str, Any]]:
    """Resolve existing MELD manifest audio paths without downloading data."""
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Unknown split: {split}")
    manifest = MELD_MANIFESTS / f"{split}_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"MELD manifest not found: {manifest}")
    frame = pd.read_csv(manifest)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        emotion = str(row.emotion).strip().lower()
        if emotion not in MELD_CLASS_NAMES:
            raise ValueError(f"Unknown MELD emotion: {emotion}")
        path = Path(audio_root) / str(row.audio_path)
        if path.is_file():
            records.append({"path": path, "label": MELD_CLASS_NAMES.index(emotion), "sample_id": row.sample_id})
    if not records:
        raise FileNotFoundError(
            f"No MELD audio files found under {audio_root}. "
            "Download/place the licensed MELD audio before training."
        )
    return records


def trim_leading_trailing_silence(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    rel_db: float = 32.0,
    pad_ms: float = 200.0,
    min_kept_sec: float = 0.4,
    min_trimmed_sec: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Trim leading and trailing low-energy dead air using a gain-invariant relative threshold."""
    n = int(len(waveform))
    orig_sec = float(n / max(sample_rate, 1))
    if orig_sec < 0.8:
        return waveform, {"trimmed": False, "orig_sec": orig_sec, "kept_sec": orig_sec}

    frame_len = int(0.025 * sample_rate)
    hop_len = int(0.010 * sample_rate)
    if frame_len <= 0 or hop_len <= 0 or n < frame_len:
        return waveform, {"trimmed": False, "orig_sec": orig_sec, "kept_sec": orig_sec}

    n_frames = 1 + (n - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + (np.arange(n_frames) * hop_len)[:, None]
    frames = waveform[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-18)
    frame_db = 20.0 * np.log10(rms)

    peak_db = float(np.percentile(frame_db, 95))
    thresh_db = peak_db - float(rel_db)
    active = np.where(frame_db >= thresh_db)[0]
    if len(active) == 0:
        return waveform, {"trimmed": False, "orig_sec": orig_sec, "kept_sec": orig_sec}

    pad_samples = int((pad_ms / 1000.0) * sample_rate)
    start = max(0, int(active[0]) * hop_len - pad_samples)
    end = min(n, int(active[-1]) * hop_len + frame_len + pad_samples)
    kept_sec = float((end - start) / sample_rate)
    if kept_sec < min_kept_sec or (orig_sec - kept_sec) < min_trimmed_sec:
        return waveform, {"trimmed": False, "orig_sec": orig_sec, "kept_sec": orig_sec}

    return waveform[start:end], {
        "trimmed": True,
        "orig_sec": orig_sec,
        "kept_sec": kept_sec,
    }


def preprocess_speech_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    max_seconds: float = 8.0,
    trim_silence: bool = True,
    low_volume_db: float = -40.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Validate, optionally edge-trim dead air, and align waveform length for Wav2Vec2."""
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected non-empty 1D mono waveform, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Speech waveform contains non-finite values (NaN or Inf).")

    orig_sec = float(arr.shape[0] / max(sample_rate, 1))
    rms = float(np.sqrt(max(float(np.mean(arr ** 2)), 1e-16)))
    rms_db = float(20.0 * np.log10(rms))

    if trim_silence:
        arr, trim_info = trim_leading_trailing_silence(arr, sample_rate=sample_rate)
        after_trim_sec = float(trim_info["kept_sec"])
    else:
        after_trim_sec = orig_sec

    trimmed_silence_s = max(0.0, orig_sec - after_trim_sec)
    max_samples = int(round(max_seconds * sample_rate))
    if max_samples > 0 and arr.shape[0] > max_samples:
        arr = arr[:max_samples]

    processed_sec = float(arr.shape[0] / max(sample_rate, 1))
    return arr.astype(np.float32, copy=False), {
        "audio_duration_s": orig_sec,
        "processed_duration_s": processed_sec,
        "trimmed_silence_s": float(trimmed_silence_s),
        "audio_rms_db": rms_db,
        "low_volume_warning": 1.0 if rms_db < low_volume_db else 0.0,
    }


class _AudioDataset:
    def __init__(self, records: list[dict[str, Any]], config: Wav2Vec2TrainingConfig, torch: Any) -> None:
        self.records = records
        self.config = config
        self.torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, metadata = load_audio(
            record["path"], target_sr=self.config.sample_rate, mono=True,
            min_duration=0.01, max_duration=1000.0, silence_threshold_db=-200.0,
        )
        if not metadata.is_valid:
            raise ValueError(f"Invalid audio {record['path']}: {metadata.status} ({metadata.reason})")
        processed, _meta = preprocess_speech_waveform(
            waveform,
            sample_rate=self.config.sample_rate,
            max_seconds=self.config.max_duration_seconds,
            trim_silence=True,
        )
        return {"input_values": processed, "labels": record["label"]}


def _collator(feature_extractor: Any, torch: Any, sample_rate: int) -> Any:
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = [item["input_values"] for item in items]
        batch = feature_extractor(
            values,
            sampling_rate=sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([item["labels"] for item in items], dtype=torch.long)
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
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(7)), average="weighted", zero_division=0)),
        "per_class": classification_report(labels, predictions, labels=list(range(7)), target_names=list(MELD_CLASS_NAMES), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(7))).tolist(),
    }
    return metrics, labels, predictions


def train_wav2vec2(audio_root: Path, config: Wav2Vec2TrainingConfig = Wav2Vec2TrainingConfig(), output_dir: Path = WAV2VEC2_ROOT) -> dict[str, Any]:
    """Train Wav2Vec2 using train/dev and evaluate once on test."""
    torch, AutoFeatureExtractor, Model, loaders = _dependencies()
    DataLoader, _ = loaders
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(config.checkpoint)
    label2id = {label: i for i, label in enumerate(MELD_CLASS_NAMES)}
    model = Model.from_pretrained(config.checkpoint, num_labels=7, label2id=label2id, id2label={i: label for label, i in label2id.items()})
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()
    model.to(device)
    train_records = load_meld_audio_records("train", audio_root)
    dev_records = load_meld_audio_records("dev", audio_root)
    test_records = load_meld_audio_records("test", audio_root)
    collate = _collator(extractor, torch, config.sample_rate)
    train_loader = DataLoader(_AudioDataset(train_records, config, torch), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(_AudioDataset(dev_records, config, torch), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(_AudioDataset(test_records, config, torch), batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp = bool(config.use_fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_f1 = -1.0
    stale = 0
    best_dir = Path(output_dir) / "best"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=amp):
                loss = model(**batch).loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, _, _ = _evaluate(model, dev_loader, torch, device)
        history.append({"epoch": epoch, **dev_metrics})
        if dev_metrics["macro_f1"] > best_f1:
            best_f1, stale = dev_metrics["macro_f1"], 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            extractor.save_pretrained(best_dir)
        else:
            stale += 1
            if stale >= config.patience:
                break
    model = Model.from_pretrained(best_dir).to(device)
    test_metrics, labels, predictions = _evaluate(model, test_loader, torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "Wav2Vec2",
        "checkpoint": config.checkpoint,
        "classes": list(MELD_CLASS_NAMES),
        "config": asdict(config),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_samples": len(train_records),
        "dev_samples": len(dev_records),
        "test_samples": len(test_records),
        "training_seconds": time.perf_counter() - started,
        "dev_history": history,
        "test_metrics": test_metrics,
        "test_labels": labels,
        "test_predictions": predictions,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    return metadata


def evaluate_saved_wav2vec2(
    audio_root: Path,
    model_dir: Path = WAV2VEC2_ROOT / "best",
    output_dir: Path = WAV2VEC2_ROOT,
) -> dict[str, Any]:
    """Evaluate an existing best checkpoint once on the held-out test split."""
    torch, AutoFeatureExtractor, Model, loaders = _dependencies()
    DataLoader, _ = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = Model.from_pretrained(model_dir).to(device)
    records = load_meld_audio_records("test", audio_root)
    loader = DataLoader(
        _AudioDataset(records, Wav2Vec2TrainingConfig(), torch),
        batch_size=Wav2Vec2TrainingConfig.eval_batch_size,
        shuffle=False,
        collate_fn=_collator(extractor, torch, Wav2Vec2TrainingConfig.sample_rate),
    )
    started = time.perf_counter()
    metrics, labels, predictions = _evaluate(model, loader, torch, device)
    metrics["inference_latency_ms_per_sample"] = (time.perf_counter() - started) * 1000 / len(records)
    metadata = {
        "model": "Wav2Vec2",
        "checkpoint": "facebook/wav2vec2-base",
        "model_dir": str(model_dir),
        "classes": list(MELD_CLASS_NAMES),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "test_samples": len(records),
        "test_metrics": metrics,
        "test_labels": labels,
        "test_predictions": predictions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_evaluation.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    return metadata


class Wav2Vec2Predictor:
    def __init__(
        self,
        model_dir: Path = WAV2VEC2_ROOT / "best",
        device: str | None = None,
        max_seconds: float = 8.0,
        trim_silence: bool = True,
        use_attention_mask: bool = True,
        calibration_path: Path | str | None = None,
        use_calibration: bool = True,
    ) -> None:
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self.torch, AutoFeatureExtractor, Model, _ = _dependencies()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Wav2Vec2 model directory not found: {model_dir}")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        self.model = Model.from_pretrained(model_dir, local_files_only=True)
        requested = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        self.device = self.torch.device(requested)
        self.model.to(self.device).eval()
        self.max_seconds = float(max_seconds)
        self.trim_silence = bool(trim_silence)
        self.model_dir = model_dir
        self.model_identifier = "speech-wav2vec2-v2" if "wav2vec2_v2" in model_dir.parts else "facebook/wav2vec2-base"
        self.use_attention_mask = bool(use_attention_mask)
        resolved_cal = resolve_calibration_path(model_dir, explicit_path=calibration_path, use_calibration=use_calibration)
        self.calibration = load_logit_calibration(resolved_cal, MELD_CLASS_NAMES) if resolved_cal is not None else None

    def predict_file(self, audio_path: Path) -> dict[str, Any]:
        waveform, metadata = load_audio(audio_path, target_sr=16000, mono=True, min_duration=0.05)
        if not metadata.is_valid:
            raise ValueError(f"Invalid audio: {metadata.status} ({metadata.reason})")
        processed, prep_meta = preprocess_speech_waveform(
            waveform,
            sample_rate=16000,
            max_seconds=self.max_seconds,
            trim_silence=self.trim_silence,
        )
        inputs = self.extractor(
            processed,
            sampling_rate=16000,
            return_attention_mask=self.use_attention_mask,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.perf_counter()
        with self.torch.no_grad():
            raw_logits = self.model(**inputs).logits[0].detach().cpu().numpy()
        calibrated_logits = apply_logit_calibration(raw_logits, self.calibration)
        probabilities = self.torch.softmax(self.torch.from_numpy(calibrated_logits), dim=-1).numpy().astype(np.float64)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
            raise RuntimeError("Wav2Vec2 produced invalid probability values.")
        probabilities /= probabilities.sum()
        entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
        return {
            "predicted_label": MELD_CLASS_NAMES[int(probabilities.argmax())],
            "confidence": float(probabilities.max()),
            "probabilities": {label: float(value) for label, value in zip(MELD_CLASS_NAMES, probabilities)},
            "model_identifier": self.model_identifier,
            "inference_latency_ms": (time.perf_counter() - started) * 1000,
            "prediction_entropy": entropy,
            "calibrated": self.calibration is not None,
            "calibration_method": self.calibration.method if self.calibration is not None else None,
            **prep_meta,
        }
