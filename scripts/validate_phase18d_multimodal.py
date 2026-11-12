"""Phase 18D — Empirical Multimodal Fusion Validation strictly on MELD dev split (N=1,108).

Evaluates all modality combinations and facial-extension candidates strictly on MELD `dev`
(zero access to `test` split):
  1. Text only (DistilBERT)
  2. Audio only (Wav2Vec2)
  3. Video only (ResNet18 + MTCNN temporal)
  4. Candidate 4 Text + Audio (preserved production baseline)
  5. Advisory Text + Video & Text + Audio + Video (lambda_video=0.0, facial reported & tracked)
  6. Fixed quality-gated Text + Audio + Video (lambda_video in {0.03, 0.05, 0.08, 0.12})
  7. Confidence/agreement-gated Video synergy rules (preserving protected text classes & Candidate 4)
  8. Audio + Video (no-text multimodal fallback blend comparison)

Measures Accuracy, Macro F1, Weighted F1, Minority Macro F1, NLL, Brier Score, and ECE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import csv
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from api.fusion import Candidate4FusionEngine, FusionConfig
from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.resnet_predictor import (
    ResNetFacialEmotionPredictor,
    aggregate_crop_outputs,
)
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT, meld_annotation_path
from speech_emotion.wav2vec2_model import Wav2Vec2Predictor
from text_emotion.distilbert_model import DistilBertTextPredictor

EXP7_DIR = REPOSITORY_ROOT / "models" / "facial_experiments" / "exp7_temporal"
OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase18d_dev_fusion_benchmark.json"
CACHE_NPZ = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase18d_dev_modality_probs.npz"
MINORITY_IDX = [0, 1, 2, 5, 6]


def _compute_calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct = (preds == y_true).astype(np.float64)

    n = len(y_true)
    true_probs = probs[np.arange(n), y_true]
    nll = float(-np.mean(np.log(np.clip(true_probs, 1e-12, 1.0))))

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (confs >= lo) & (confs <= hi if idx == n_bins - 1 else confs < hi)
        if np.any(mask):
            ece += (float(np.sum(mask)) / float(n)) * abs(float(np.mean(correct[mask])) - float(np.mean(confs[mask])))

    return {"nll": round(nll, 5), "brier": round(brier, 5), "ece": round(float(ece), 5)}


def _score_predictions(name: str, y_true: np.ndarray, probs: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    if valid_mask is not None:
        yt = y_true[valid_mask]
        pr = probs[valid_mask]
    else:
        yt = y_true
        pr = probs
    yp = np.argmax(pr, axis=1)
    per_class = f1_score(yt, yp, labels=list(range(7)), average=None, zero_division=0)
    cal = _compute_calibration_metrics(yt, pr)
    return {
        "strategy": name,
        "n_samples": int(len(yt)),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "macro_f1": round(float(f1_score(yt, yp, labels=list(range(7)), average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(yt, yp, labels=list(range(7)), average="weighted", zero_division=0)), 4),
        "minority_macro_f1": round(float(np.mean(per_class[MINORITY_IDX])), 4),
        "active_classes": int(np.sum(per_class > 0.0)),
        **cal,
    }


def _collect_or_load_dev_predictions() -> dict[str, Any]:
    if CACHE_NPZ.is_file():
        data = np.load(CACHE_NPZ, allow_pickle=False)
        return {k: data[k] for k in data.files}

    dev_ann = pd.read_csv(meld_annotation_path("dev")).dropna(subset=["Utterance", "Emotion"]).copy()
    text_by_sid: dict[str, str] = {}
    label_by_sid: dict[str, int] = {}
    for row in dev_ann.itertuples(index=False):
        sid = f"dev_dia{int(row.Dialogue_ID)}_utt{int(row.Utterance_ID)}"
        emo = str(row.Emotion).strip().lower()
        text_by_sid[sid] = str(row.Utterance)
        label_by_sid[sid] = EMOTION_CLASSES.index(emo)

    audio_map: dict[str, Path] = {}
    audio_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    with audio_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("audio_status") == "ok":
                p = Path(row.get("output_path", ""))
                if p.is_file():
                    audio_map[str(row["sample_id"]).strip()] = p

    exp7_meta = json.loads((EXP7_DIR / "dev_temporal_unaligned_meta.json").read_text(encoding="utf-8"))
    exp7_crops = np.load(EXP7_DIR / "dev_temporal_unaligned_crops.npz")["crops"]
    video_by_sid = {v["sample_id"]: v for v in exp7_meta["videos"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_pred = DistilBertTextPredictor(device=device)
    audio_pred = Wav2Vec2Predictor(device=device)
    video_pred = ResNetFacialEmotionPredictor(device=device, strict_fp32=True)

    all_logits, _ = video_pred.infer_crop_logits(exp7_crops, batch_size=128)

    sample_ids = sorted(s for s in label_by_sid if s in audio_map and s in video_by_sid)
    n = len(sample_ids)
    y_true = np.zeros(n, dtype=np.int64)
    p_text = np.zeros((n, 7), dtype=np.float64)
    p_audio = np.zeros((n, 7), dtype=np.float64)
    p_video = np.full((n, 7), np.nan, dtype=np.float64)
    audio_valid = np.zeros(n, dtype=bool)
    video_valid = np.zeros(n, dtype=bool)
    video_cov = np.zeros(n, dtype=np.float64)
    video_ent = np.full(n, np.nan, dtype=np.float64)

    for idx, sid in enumerate(sample_ids):
        y_true[idx] = label_by_sid[sid]
        t_res = text_pred.predict(text_by_sid[sid])
        p_text[idx] = [t_res["probabilities"][c] for c in EMOTION_CLASSES]

        try:
            a_res = audio_pred.predict_file(audio_map[sid])
            p_audio[idx] = [a_res["probabilities"][c] for c in EMOTION_CLASSES]
            audio_valid[idx] = True
        except Exception:
            audio_valid[idx] = False

        v_info = video_by_sid[sid]
        if v_info.get("detected") and v_info.get("all_det_crop_indices"):
            vp = aggregate_crop_outputs(all_logits[v_info["all_det_crop_indices"]], method="raw_all_detections_mean_prob")
            p_video[idx] = vp
            video_valid[idx] = True
            sampled = max(1, int(v_info.get("frames_sampled", 1)))
            video_cov[idx] = float(v_info.get("frames_with_faces", 0)) / float(sampled)
            video_ent[idx] = float(-(vp * np.log(np.maximum(vp, 1e-12))).sum())

        if (idx + 1) % 250 == 0 or (idx + 1) == n:
            print(f"  [Phase 18D dev cache] processed {idx + 1}/{n}")

    np.savez_compressed(
        CACHE_NPZ,
        y_true=y_true,
        p_text=p_text,
        p_audio=p_audio,
        p_video=p_video,
        audio_valid=audio_valid,
        video_valid=video_valid,
        video_cov=video_cov,
        video_ent=video_ent,
    )
    return {
        "y_true": y_true,
        "p_text": p_text,
        "p_audio": p_audio,
        "p_video": p_video,
        "audio_valid": audio_valid,
        "video_valid": video_valid,
        "video_cov": video_cov,
        "video_ent": video_ent,
    }


def _fuse_candidate4_batch(
    p_text: np.ndarray,
    p_audio: np.ndarray,
    audio_valid: np.ndarray,
    p_video: np.ndarray,
    video_valid: np.ndarray,
    video_cov: np.ndarray,
    video_ent: np.ndarray,
    lambda_video: float = 0.0,
) -> np.ndarray:
    engine = Candidate4FusionEngine(FusionConfig(lambda_video=lambda_video))
    out = np.zeros_like(p_text, dtype=np.float64)
    for idx in range(len(p_text)):
        t_dict = {c: float(p_text[idx, i]) for i, c in enumerate(EMOTION_CLASSES)}
        a_dict = {c: float(p_audio[idx, i]) for i, c in enumerate(EMOTION_CLASSES)} if audio_valid[idx] else None
        v_dict = {c: float(p_video[idx, i]) for i, c in enumerate(EMOTION_CLASSES)} if video_valid[idx] else None
        vq = {"face_detection_coverage": float(video_cov[idx]), "prediction_entropy": float(video_ent[idx])} if video_valid[idx] else None
        res = engine.fuse(text_probabilities=t_dict, audio_probabilities=a_dict, video_probabilities=v_dict, video_quality=vq)
        out[idx] = [res.probabilities[c] for c in EMOTION_CLASSES]
    return out


def _fuse_conditional_video_synergy(
    p_text: np.ndarray,
    p_audio: np.ndarray,
    audio_valid: np.ndarray,
    p_video: np.ndarray,
    video_valid: np.ndarray,
    video_cov: np.ndarray,
    video_ent: np.ndarray,
    max_w_v: float,
    tau_v: float = 0.15,
    include_protected: bool = False,
    agreement_only: bool = False,
) -> np.ndarray:
    """Test selective confidence/quality-gated video extension without changing Candidate 4 audio rules."""
    base_engine = Candidate4FusionEngine(FusionConfig(lambda_video=0.0))
    out = np.zeros_like(p_text, dtype=np.float64)
    neutral_idx = EMOTION_CLASSES.index("neutral")
    protected_idx = {EMOTION_CLASSES.index(c) for c in ("fear", "disgust", "sadness", "joy")}

    for idx in range(len(p_text)):
        t_dict = {c: float(p_text[idx, i]) for i, c in enumerate(EMOTION_CLASSES)}
        a_dict = {c: float(p_audio[idx, i]) for i, c in enumerate(EMOTION_CLASSES)} if audio_valid[idx] else None
        c4 = base_engine.fuse(text_probabilities=t_dict, audio_probabilities=a_dict)
        p_c4 = np.array([c4.probabilities[c] for c in EMOTION_CLASSES], dtype=np.float64)

        if not video_valid[idx]:
            out[idx] = p_c4
            continue

        t_pred = int(np.argmax(p_text[idx]))
        if not include_protected and t_pred in protected_idx:
            out[idx] = p_c4
            continue

        if video_cov[idx] < 0.50 or video_ent[idx] > 1.70:
            out[idx] = p_c4
            continue

        pv = p_video[idx]
        v_pred = int(np.argmax(pv))
        v_margin = float(pv[v_pred] - pv[neutral_idx])
        if v_pred == neutral_idx or v_margin <= tau_v:
            out[idx] = p_c4
            continue

        if agreement_only and audio_valid[idx] and int(np.argmax(p_audio[idx])) != v_pred:
            out[idx] = p_c4
            continue

        w_v = min(max_w_v, 0.25 * v_margin)
        blended = (1.0 - w_v) * p_c4 + w_v * pv
        out[idx] = blended / np.sum(blended)

    return out


def main() -> None:
    t0 = time.perf_counter()
    data = _collect_or_load_dev_predictions()
    y_true = data["y_true"]
    p_text = data["p_text"]
    p_audio = data["p_audio"]
    p_video = data["p_video"]
    audio_valid = data["audio_valid"]
    video_valid = data["video_valid"]
    video_cov = data["video_cov"]
    video_ent = data["video_ent"]

    # Fill uniform fallback strictly for audio-only invalid cases (mask handles fair comparisons)
    safe_video = np.where(video_valid[:, None], p_video, 1.0 / 7.0)
    both_av = audio_valid & video_valid

    table: list[dict[str, Any]] = []
    table.append(_score_predictions("text_only_distilbert", y_true, p_text))
    table.append(_score_predictions("audio_only_wav2vec2", y_true, p_audio, valid_mask=audio_valid))
    table.append(_score_predictions("video_only_resnet18_exp7", y_true, safe_video, valid_mask=video_valid))

    c4_ta = _fuse_candidate4_batch(p_text, p_audio, audio_valid, p_video, video_valid, video_cov, video_ent, lambda_video=0.0)
    table.append(_score_predictions("candidate4_text_audio_baseline", y_true, c4_ta))

    # Text + Video (no audio)
    no_audio = np.zeros_like(audio_valid, dtype=bool)
    for lam in (0.03, 0.05, 0.08, 0.12):
        tv = _fuse_candidate4_batch(p_text, p_audio, no_audio, p_video, video_valid, video_cov, video_ent, lambda_video=lam)
        table.append(_score_predictions(f"text_video_lambda_{lam:.2f}", y_true, tv))
        tav = _fuse_candidate4_batch(p_text, p_audio, audio_valid, p_video, video_valid, video_cov, video_ent, lambda_video=lam)
        table.append(_score_predictions(f"text_audio_video_lambda_{lam:.2f}", y_true, tav))

    # Conditional video synergy rules
    for max_wv, agree in ((0.06, False), (0.10, False), (0.15, False), (0.10, True), (0.15, True)):
        p_cond = _fuse_conditional_video_synergy(
            p_text, p_audio, audio_valid, p_video, video_valid, video_cov, video_ent, max_w_v=max_wv, agreement_only=agree
        )
        label = f"cond_video_wv{max_wv:.2f}_agree{int(agree)}"
        table.append(_score_predictions(label, y_true, p_cond))

    # Audio + Video (when Text is unavailable) on valid A+V cohort
    table.append(_score_predictions("audio_only_on_av_cohort", y_true, p_audio, valid_mask=both_av))
    table.append(_score_predictions("video_only_on_av_cohort", y_true, safe_video, valid_mask=both_av))
    for wv in (0.10, 0.20, 0.30, 0.40, 0.50):
        av_blend = (1.0 - wv) * p_audio + wv * safe_video
        table.append(_score_predictions(f"audio_video_blend_wv_{wv:.2f}", y_true, av_blend, valid_mask=both_av))

    report = {
        "n_dev_total": int(len(y_true)),
        "n_audio_valid": int(np.sum(audio_valid)),
        "n_video_valid": int(np.sum(video_valid)),
        "n_audio_and_video_valid": int(np.sum(both_av)),
        "zero_test_split_access": True,
        "elapsed_seconds": round(float(time.perf_counter() - t0), 2),
        "benchmarks": table,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

