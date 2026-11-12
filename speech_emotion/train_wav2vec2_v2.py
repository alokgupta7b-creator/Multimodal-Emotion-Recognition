"""Phase P2 Speech Emotion Retraining: Wav2Vec2 v2 with P0 silence/attention-mask
consistency, acoustic augmentations, length-bucket batching, and Logit-Adjusted CE.

Safety invariants:
- Trains strictly on MELD `train` (never accesses `dev_val` or held-out MELD `test`).
- Monitors epoch checkpoint selection strictly via `dev_cal` (`N=559` valid utterances).
- Evaluates final comparison strictly once out-of-sample on `dev_val` (`N=548`).
- Writes strictly to `models/speech_emotion/wav2vec2_v2/`, preserving v1 `wav2vec2/best/`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT
from speech_emotion.audio_loader import load_audio
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES
from speech_emotion.wav2vec2_model import (
    WAV2VEC2_ROOT,
    load_meld_audio_records,
    preprocess_speech_waveform,
)

WAV2VEC2_V1_WEIGHTS = WAV2VEC2_ROOT / "best" / "model.safetensors"
WAV2VEC2_V2_ROOT = REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2_v2"
SPLIT_PATH = REPOSITORY_ROOT / "models" / "remediation_splits" / "meld_dev_cal_val_split.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_speech_waveform(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply label-preserving acoustic perturbations (gain jitter, SNR noise, edge jitter)."""
    wav = np.asarray(waveform, dtype=np.float32).copy()

    # 1. Microphone gain perturbation (-6 dB to +4 dB)
    if random.random() < 0.7:
        gain_db = random.uniform(-6.0, 4.0)
        wav *= float(10.0 ** (gain_db / 20.0))

    # 2. Additive natural colored/white room noise at 18-32 dB SNR
    if random.random() < 0.55:
        sig_rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
        snr_db = random.uniform(18.0, 32.0)
        target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
        white = np.random.normal(0.0, 1.0, size=wav.shape[0]).astype(np.float32)
        if random.random() < 0.5 and wav.shape[0] > 4:
            # Soft low-frequency drift simulating ambient room rumble
            low = np.cumsum(white).astype(np.float32)
            low -= np.mean(low)
            low /= float(np.std(low) + 1e-8)
            noise = 0.65 * white + 0.35 * low
        else:
            noise = white
        noise *= target_noise_rms / float(np.sqrt(np.mean(noise ** 2) + 1e-12))
        wav = wav + noise

    # 3. Mild boundary edge shift / short pad jitter (<= 120 ms, capped at 8.0 s)
    if random.random() < 0.35:
        max_len = 8 * sample_rate
        shift = random.randint(0, min(int(0.12 * sample_rate), max(0, wav.shape[0] // 8)))
        if shift > 0:
            if random.random() < 0.5 and wav.shape[0] + shift <= max_len:
                wav = np.concatenate([np.zeros(shift, dtype=np.float32), wav])
            elif wav.shape[0] - shift >= int(0.4 * sample_rate):
                wav = wav[shift:]

    return np.clip(wav, -0.99, 0.99).astype(np.float32, copy=False)


class PreloadedSpeechDataset(Dataset):
    """In-memory P0-preprocessed waveform dataset supporting fast bucketed batching."""

    def __init__(
        self,
        waveforms: list[np.ndarray],
        labels: list[int],
        augment: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self.waveforms = waveforms
        self.labels = labels
        self.lengths = [int(len(w)) for w in waveforms]
        self.augment = augment
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wav = self.waveforms[idx]
        if self.augment:
            wav = augment_speech_waveform(wav, sample_rate=self.sample_rate)
        return {"input_values": wav, "labels": self.labels[idx]}


class LengthBucketBatchSampler(BatchSampler):
    """Group similar-duration utterances per batch to eliminate wasted attention padding."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 42) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        # Add small multiplicative jitter so buckets vary across epochs
        noisy_len = self.lengths * rng.uniform(0.85, 1.15, size=len(self.lengths))
        order = np.argsort(noisy_len)
        batches = [order[i : i + self.batch_size].tolist() for i in range(0, len(order), self.batch_size)]
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


class LogitAdjustedCELoss(nn.Module):
    def __init__(self, class_priors: np.ndarray, tau: float = 0.45, label_smoothing: float = 0.03) -> None:
        super().__init__()
        log_prior = torch.log(torch.tensor(np.clip(class_priors, 1e-8, 1.0), dtype=torch.float32))
        self.register_buffer("adjustment", float(tau) * log_prior)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment.to(logits.device), targets, label_smoothing=self.label_smoothing)


def preload_preprocessed_split(
    records: list[dict[str, Any]],
    *,
    max_seconds: float = 8.0,
    silence_threshold_db: float = -60.0,
) -> tuple[list[np.ndarray], list[int], list[str], int]:
    """Load and apply exact P0 waveform preprocessing once into RAM."""
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    sids: list[str] = []
    skipped = 0
    for rec in records:
        wav, meta = load_audio(
            rec["path"],
            target_sr=16000,
            mono=True,
            min_duration=0.05,
            max_duration=1000.0,
            silence_threshold_db=silence_threshold_db,
        )
        if not meta.is_valid:
            skipped += 1
            continue
        proc, _ = preprocess_speech_waveform(wav, sample_rate=16000, max_seconds=max_seconds, trim_silence=True)
        waveforms.append(proc)
        labels.append(int(rec["label"]))
        sids.append(str(rec["sample_id"]))
    return waveforms, labels, sids, skipped


@torch.no_grad()
def evaluate_speech_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Evaluate Wav2Vec2 strictly in IEEE FP32 (autocast disabled)."""
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[int] = []
    with torch.amp.autocast("cuda", enabled=False):
        for batch in loader:
            inp = {
                "input_values": batch["input_values"].to(device, dtype=torch.float32),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inp).logits.detach().cpu().numpy().astype(np.float64)
            logits_list.append(logits)
            labels_list.extend(batch["labels"].cpu().tolist())

    all_logits = np.concatenate(logits_list, axis=0)
    shifted = all_logits - np.max(all_logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    y_true = np.asarray(labels_list, dtype=np.int64)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(np.float64)
    clipped = np.clip(probs, 1e-12, 1.0)
    nll = float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))
    one_hot = np.eye(7, dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for bi in range(15):
        m = (conf >= edges[bi]) & (conf < edges[bi + 1] if bi < 14 else conf <= edges[bi + 1])
        if np.any(m):
            ece += (float(m.sum()) / len(y_true)) * abs(float(correct[m].mean()) - float(conf[m].mean()))

    per_cls = classification_report(
        y_true, y_pred, labels=list(range(7)), target_names=list(MELD_CLASS_NAMES), output_dict=True, zero_division=0
    )
    active_classes = int(sum(1 for i in range(7) if np.sum(y_pred == i) > 0))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(7)), average="weighted", zero_division=0)),
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf)),
        "low_conf_rate_lt45": float(np.mean(conf < 0.45)),
        "mean_entropy": float(np.mean(-np.sum(clipped * np.log(clipped), axis=1))),
        "active_classes": active_classes,
        "per_class_f1": {c: float(per_cls[c]["f1-score"]) for c in MELD_CLASS_NAMES},
        "pred_histogram": {MELD_CLASS_NAMES[i]: int(np.sum(y_pred == i)) for i in range(7)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist(),
        "logits": all_logits,
        "probs": probs,
        "y_true": y_true,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2 Wav2Vec2 v2 speech emotion model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.45)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    args = parser.parse_args()

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    v1_sha = _sha256_file(WAV2VEC2_V1_WEIGHTS)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_cfg = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    cal_sids = set(split_cfg["cal_sample_ids"])
    val_sids = set(split_cfg["val_sample_ids"])

    # Never access test split!
    print("Pre-loading MELD train and dev audio with P0 preprocessing...")
    t_load0 = time.perf_counter()
    train_recs = load_meld_audio_records("train", MELD_PREPARED_ROOT)
    dev_recs = load_meld_audio_records("dev", MELD_PREPARED_ROOT)
    cal_recs = [r for r in dev_recs if r["sample_id"] in cal_sids]
    val_recs = [r for r in dev_recs if r["sample_id"] in val_sids]

    tr_wavs, tr_lbls, tr_sids, tr_skipped = preload_preprocessed_split(train_recs)
    cal_wavs, cal_lbls, cal_sids_ok, cal_skipped = preload_preprocessed_split(cal_recs)
    val_wavs, val_lbls, val_sids_ok, val_skipped = preload_preprocessed_split(val_recs)
    if set(tr_sids) & (cal_sids | val_sids):
        raise RuntimeError("Train/dev contamination detected!")
    print(
        f"Loaded in {time.perf_counter()-t_load0:.1f}s | train={len(tr_wavs)} (skipped={tr_skipped}) "
        f"| dev_cal={len(cal_wavs)} (skipped={cal_skipped}) | dev_val={len(val_wavs)} (skipped={val_skipped})"
    )

    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)
    label2id = {lbl: i for i, lbl in enumerate(MELD_CLASS_NAMES)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=7,
        label2id=label2id,
        id2label=id2label,
        local_files_only=True,
    )
    model.config.mask_time_prob = 0.04
    model.config.mask_time_length = 3
    model.config.mask_feature_prob = 0.0
    model.config.hidden_dropout = 0.10
    model.config.attention_dropout = 0.10
    model.config.feat_proj_dropout = 0.10
    model.config.final_dropout = 0.10
    model.freeze_feature_encoder()
    model.to(device)

    def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [it["input_values"] for it in items]
        max_samples = max(len(v) for v in vals)
        if max_samples < 4800:
            batch = extractor(
                vals,
                sampling_rate=16000,
                padding="max_length",
                max_length=4800,
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            batch = extractor(
                vals,
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        batch["labels"] = torch.tensor([it["labels"] for it in items], dtype=torch.long)
        return batch

    train_ds = PreloadedSpeechDataset(tr_wavs, tr_lbls, augment=True)
    batch_sampler = LengthBucketBatchSampler(train_ds.lengths, batch_size=args.batch_size, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0)
    cal_loader = DataLoader(PreloadedSpeechDataset(cal_wavs, cal_lbls, augment=False), batch_size=4, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(PreloadedSpeechDataset(val_wavs, val_lbls, augment=False), batch_size=4, shuffle=False, collate_fn=collate_fn)

    counts = np.bincount(np.asarray(tr_lbls, dtype=np.int64), minlength=7).astype(np.float64)
    priors = counts / counts.sum()
    criterion = LogitAdjustedCELoss(priors, tau=args.tau, label_smoothing=0.03).to(device)

    encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("projector.", "classifier."))]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("projector.", "classifier."))]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(0.10 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_dir = WAV2VEC2_V2_ROOT / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    best_cal_summary: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    t_train0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, 1):
            inp = {
                "input_values": batch["input_values"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            targets = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(**inp).logits
                raw_loss = criterion(logits, targets)
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()
            bs = int(targets.size(0))
            running_loss += float(raw_loss.item()) * bs
            seen += bs

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        cal_res = evaluate_speech_loader(model, cal_loader, device)
        ep_sec = time.perf_counter() - ep_t0
        ep_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "dev_cal_acc": cal_res["accuracy"],
            "dev_cal_macro_f1": cal_res["macro_f1"],
            "dev_cal_weighted_f1": cal_res["weighted_f1"],
            "dev_cal_ece": cal_res["ece"],
            "dev_cal_nll": cal_res["nll"],
            "dev_cal_active_classes": cal_res["active_classes"],
            "dev_cal_disgust_f1": cal_res["per_class_f1"]["disgust"],
            "dev_cal_fear_f1": cal_res["per_class_f1"]["fear"],
            "epoch_seconds": ep_sec,
        }
        history.append(ep_row)
        print(
            f"  [Wav2Vec2_v2] Ep {epoch}/{args.epochs} loss={ep_row['train_loss']:.4f} ({ep_sec:.1f}s) | "
            f"dev_cal Acc={cal_res['accuracy']*100:.2f}% MF1={cal_res['macro_f1']*100:.2f}% "
            f"WF1={cal_res['weighted_f1']*100:.2f}% Disg={cal_res['per_class_f1']['disgust']*100:.1f}% "
            f"Fear={cal_res['per_class_f1']['fear']*100:.1f}% act={cal_res['active_classes']}/7",
            flush=True,
        )

        base_score = 0.75 * cal_res["macro_f1"] + 0.25 * cal_res["weighted_f1"]
        score = base_score if cal_res["active_classes"] >= 5 else base_score * 0.5
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_cal_summary = {k: v for k, v in cal_res.items() if k not in ("logits", "probs", "y_true")}
            model.save_pretrained(best_dir)
            extractor.save_pretrained(best_dir)

    train_seconds = time.perf_counter() - t_train0
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0

    # Reload best checkpoint & evaluate once untouched on dev_val + check CPU vs CUDA FP32 parity
    best_model_gpu = Wav2Vec2ForSequenceClassification.from_pretrained(best_dir, local_files_only=True).to(device)
    t_eval0 = time.perf_counter()
    val_res = evaluate_speech_loader(best_model_gpu, val_loader, device)
    val_ms_per_clip = (time.perf_counter() - t_eval0) * 1000.0 / max(1, len(val_wavs))
    cal_final = evaluate_speech_loader(best_model_gpu, cal_loader, device)

    # Parity on 8 dev_val clips CPU vs CUDA FP32
    best_model_cpu = Wav2Vec2ForSequenceClassification.from_pretrained(best_dir, local_files_only=True).to("cpu").eval()
    sample_inputs = extractor(val_wavs[:8], sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        p_cpu = torch.softmax(best_model_cpu(**sample_inputs).logits.float(), dim=-1).numpy()
        with torch.amp.autocast("cuda", enabled=False):
            p_gpu = torch.softmax(best_model_gpu(**{k: v.to(device) for k, v in sample_inputs.items()}).logits.float(), dim=-1).cpu().numpy()
        fp32_parity_diff = float(np.max(np.abs(p_cpu - p_gpu)))

    if _sha256_file(WAV2VEC2_V1_WEIGHTS) != v1_sha:
        raise RuntimeError("Protected v1 Wav2Vec2 weights modified unexpectedly!")

    np.savez_compressed(
        WAV2VEC2_V2_ROOT / "dev_outputs.npz",
        cal_logits=cal_final["logits"],
        cal_probs=cal_final["probs"],
        cal_y=cal_final["y_true"],
        cal_sample_ids=np.asarray(cal_sids_ok),
        val_logits=val_res["logits"],
        val_probs=val_res["probs"],
        val_y=val_res["y_true"],
        val_sample_ids=np.asarray(val_sids_ok),
    )

    metadata = {
        "model": "Wav2Vec2_v2",
        "base_checkpoint": "facebook/wav2vec2-base",
        "saved_best_dir": str(best_dir),
        "v1_weights_sha256_verified": v1_sha,
        "classes": list(MELD_CLASS_NAMES),
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "loss": f"LogitAdjustedCE(tau={args.tau}, label_smoothing=0.03)",
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "p0_preprocessing": True,
            "return_attention_mask": True,
            "bucket_batching": True,
            "augmentations": ["gain_-6_to_+4_dB", "snr_18_32_dB_noise", "edge_jitter_120ms", "mask_time_0.05"],
        },
        "train_samples": len(tr_wavs),
        "train_skipped": tr_skipped,
        "dev_cal_samples": len(cal_wavs),
        "dev_val_samples": len(val_wavs),
        "best_epoch": best_epoch,
        "train_seconds": float(train_seconds),
        "peak_vram_mb": peak_vram_mb,
        "fp32_cpu_cuda_max_prob_diff": fp32_parity_diff,
        "dev_val_cuda_ms_per_clip": val_ms_per_clip,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "history": history,
        "dev_cal_metrics": best_cal_summary,
        "dev_val_uncalibrated_metrics": {k: v for k, v in val_res.items() if k not in ("logits", "probs", "y_true")},
    }
    (WAV2VEC2_V2_ROOT / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"\n[Speech V2 dev_val Untouched] Acc={val_res['accuracy']*100:.2f}% MF1={val_res['macro_f1']*100:.2f}% "
        f"WF1={val_res['weighted_f1']*100:.2f}% ECE={val_res['ece']:.4f} NLL={val_res['nll']:.4f} "
        f"conf={val_res['mean_confidence']*100:.1f}% <45%={val_res['low_conf_rate_lt45']*100:.1f}% "
        f"active={val_res['active_classes']}/7 FP32_diff={fp32_parity_diff:.2e}"
    )
    print(f"  per_class_f1={ {k: round(v*100, 2) for k, v in val_res['per_class_f1'].items()} }")
    print(f"  pred_hist={val_res['pred_histogram']}")


if __name__ == "__main__":
    main()

