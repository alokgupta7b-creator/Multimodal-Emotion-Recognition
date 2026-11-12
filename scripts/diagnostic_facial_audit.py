"""Diagnostic facial audit script.
Audits:
1. Dataset distribution, image quality, duplicate crops, and train/dev split integrity.
2. Model architecture, parameter count, loss convergence, per-class metrics, and confusion matrix on dev set.
3. Real-world video inference breakdown (frame-by-frame distributions, face counts, temporal effects).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.detector import FaceDetector, iter_video_faces
from facial_emotion.model import (
    FacialEmotionPredictor,
    create_model,
    preprocess_face_crops,
    temporal_logits,
    temporal_probabilities,
)
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

RESULTS_DIR = REPOSITORY_ROOT / "docs" / "facial_diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_dataset():
    print("=" * 60)
    print("1. DATASET AUDIT")
    print("=" * 60)

    train_manifest = pd.read_csv(MELD_PREPARED_ROOT / "reports" / "train" / "video_manifest.csv")
    dev_manifest = pd.read_csv(MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv")

    train_cache = REPOSITORY_ROOT / "models" / "facial_emotion" / "face_cache" / "train"
    dev_cache = REPOSITORY_ROOT / "models" / "facial_emotion" / "face_cache" / "dev"

    train_files = sorted(list(train_cache.glob("*.npy")))
    dev_files = sorted(list(dev_cache.glob("*.npy")))

    print(f"Train manifest videos: {len(train_manifest)} | Cached face crops: {len(train_files)} ({(len(train_files)/len(train_manifest))*100:.2f}%)")
    print(f"Dev manifest videos:   {len(dev_manifest)} | Cached face crops: {len(dev_files)} ({(len(dev_files)/len(dev_manifest))*100:.2f}%)")

    # Class distribution
    train_dist = train_manifest["emotion"].value_counts().to_dict()
    dev_dist = dev_manifest["emotion"].value_counts().to_dict()

    train_cache_samples = {p.stem.replace(".npy", "") for p in train_files}
    dev_cache_samples = {p.stem.replace(".npy", "") for p in dev_files}

    train_cached_df = train_manifest[train_manifest["sample_id"].isin(train_cache_samples)]
    dev_cached_df = dev_manifest[dev_manifest["sample_id"].isin(dev_cache_samples)]

    train_cached_dist = train_cached_df["emotion"].value_counts().to_dict()
    dev_cached_dist = dev_cached_df["emotion"].value_counts().to_dict()

    print("\nClass Distribution:")
    for cls in EMOTION_CLASSES:
        t_raw = train_dist.get(cls, 0)
        t_cached = train_cached_dist.get(cls, 0)
        d_raw = dev_dist.get(cls, 0)
        d_cached = dev_cached_dist.get(cls, 0)
        print(f"  {cls:10s} -> Train Raw: {t_raw:5d} ({t_raw/len(train_manifest)*100:5.2f}%) | Train Cached: {t_cached:5d} ({t_cached/len(train_cached_df)*100:5.2f}%) | Dev Cached: {d_cached:4d} ({d_cached/len(dev_cached_df)*100:5.2f}%)")

    # Image Quality Audit
    print("\nAuditing image quality and duplicates across cached crops...")
    train_hashes = {}
    train_duplicates = 0
    black_count = 0
    low_contrast_count = 0
    intensities = []
    stds = []

    for p in train_files:
        arr = np.load(p)
        intensities.append(float(arr.mean()))
        stds.append(float(arr.std()))
        if arr.max() < 10:
            black_count += 1
        if arr.std() < 15:
            low_contrast_count += 1
        h = hashlib.md5(arr.tobytes()).hexdigest()
        if h in train_hashes:
            train_duplicates += 1
        else:
            train_hashes[h] = p.name

    train_dev_duplicates = 0
    for p in dev_files:
        arr = np.load(p)
        h = hashlib.md5(arr.tobytes()).hexdigest()
        if h in train_hashes:
            train_dev_duplicates += 1

    print(f"Mean pixel intensity (train): {np.mean(intensities):.2f} +/- {np.std(intensities):.2f}")
    print(f"Mean contrast std (train):   {np.mean(stds):.2f} +/- {np.std(stds):.2f}")
    print(f"Black images (max < 10):     {black_count}")
    print(f"Low-contrast images (std<15):{low_contrast_count}")
    print(f"Intra-train duplicates:      {train_duplicates} (identical crops)")
    print(f"Train-to-dev duplicate crops: {train_dev_duplicates} (leakage between splits)")

    dataset_summary = {
        "train_manifest_count": len(train_manifest),
        "dev_manifest_count": len(dev_manifest),
        "train_cached_count": len(train_files),
        "dev_cached_count": len(dev_files),
        "train_excluded_faces": len(train_manifest) - len(train_files),
        "dev_excluded_faces": len(dev_manifest) - len(dev_files),
        "train_distribution": train_cached_dist,
        "dev_distribution": dev_cached_dist,
        "train_intra_duplicates": train_duplicates,
        "train_dev_duplicates": train_dev_duplicates,
        "black_images": black_count,
        "low_contrast_images": low_contrast_count,
        "mean_intensity": float(np.mean(intensities)),
        "mean_contrast": float(np.mean(stds)),
    }
    with (RESULTS_DIR / "dataset_audit.json").open("w") as f:
        json.dump(dataset_summary, f, indent=2)

    return dataset_summary


def audit_model():
    print("\n" + "=" * 60)
    print("2. MODEL ARCHITECTURE & CONVERGENCE AUDIT")
    print("=" * 60)

    # 1. Architecture inspection
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Architecture layers:")
    for idx, layer in enumerate(model):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {idx}: {layer} (parameters: {layer_params})")
    print(f"Total model parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 2. Checkpoint inspection
    ckpt_path = REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    meta_path = REPOSITORY_ROOT / "models" / "facial_emotion" / "training_metadata.json"
    with meta_path.open() as f:
        metadata = json.load(f)

    print("\nTraining configuration from metadata:")
    print(json.dumps(metadata["config"], indent=2))

    print("\nEpoch Loss & Metrics Progression:")
    for ep in metadata["history"]:
        epoch = ep["epoch"]
        t_loss = ep["train_loss"]
        d_loss = ep["dev_loss"]
        d_acc = ep["dev_metrics"]["accuracy"]
        d_f1 = ep["dev_metrics"]["macro_f1"]
        print(f"  Epoch {epoch}: Train Loss = {t_loss:.4f} | Dev Loss = {d_loss:.4f} | Dev Acc = {d_acc*100:.2f}% | Dev Macro F1 = {d_f1:.4f}")

    theoretical_random_loss = float(np.log(7))
    print(f"\nTheoretical uniform random loss for 7 classes: ln(7) = {theoretical_random_loss:.4f}")
    loss_delta = abs(metadata["history"][-1]["train_loss"] - theoretical_random_loss)
    print(f"Delta between final train loss ({metadata['history'][-1]['train_loss']:.4f}) and ln(7): {loss_delta:.4f}")
    if loss_delta < 0.01:
        print("  -> CRITICAL FINDING: Model train loss never diverged from random baseline! Zero effective feature learning.")

    # 3. Evaluate frozen checkpoint on DEV split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    dev_manifest = pd.read_csv(MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv")
    dev_cache = REPOSITORY_ROOT / "models" / "facial_emotion" / "face_cache" / "dev"
    class_to_idx = {name: i for i, name in enumerate(EMOTION_CLASSES)}

    dev_y_true = []
    dev_y_pred = []
    dev_probs = []

    print("\nEvaluating frozen checkpoint on development set crops...")
    with torch.no_grad():
        for _, row in dev_manifest.iterrows():
            sid = row["sample_id"]
            p = dev_cache / f"{sid}.npy"
            if not p.is_file():
                continue
            crop = np.load(p)  # (96, 96, 3)
            # Match preprocessing in model.py
            batch = np.expand_dims(crop, axis=0)  # (1, 96, 96, 3)
            tensor = preprocess_face_crops(batch, torch).to(device)  # (1, 3, 96, 96)
            logits = model(tensor)
            prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(prob.argmax())

            dev_y_true.append(class_to_idx[row["emotion"].lower()])
            dev_y_pred.append(pred_idx)
            dev_probs.append(prob)

    dev_y_true = np.array(dev_y_true)
    dev_y_pred = np.array(dev_y_pred)
    dev_probs = np.array(dev_probs)

    report_dict = classification_report(dev_y_true, dev_y_pred, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(dev_y_true, dev_y_pred, labels=list(range(7)))

    print("\n--- DEVELOPMENT SET CLASSIFICATION REPORT ---")
    print(f"{'Class':10s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    print("-" * 55)
    for cls in EMOTION_CLASSES:
        p = report_dict[cls]["precision"]
        r = report_dict[cls]["recall"]
        f1 = report_dict[cls]["f1-score"]
        s = int(report_dict[cls]["support"])
        print(f"{cls:10s} | {p:10.4f} | {r:10.4f} | {f1:10.4f} | {s:8d}")
    print("-" * 55)
    print(f"{'Macro Avg':10s} | {report_dict['macro avg']['precision']:10.4f} | {report_dict['macro avg']['recall']:10.4f} | {report_dict['macro avg']['f1-score']:10.4f} | {len(dev_y_true):8d}")
    print(f"{'Accuracy':10s} | {report_dict['accuracy']*100:.2f}%")

    # Probability Distribution Analysis
    mean_probs = dev_probs.mean(axis=0)
    print("\nMean Predicted Probability across Dev Set:")
    for cls, prob in zip(EMOTION_CLASSES, mean_probs):
        print(f"  {cls:10s}: {prob*100:5.2f}%")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Development Set Facial CNN Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(EMOTION_CLASSES))
    plt.xticks(tick_marks, EMOTION_CLASSES, rotation=45)
    plt.yticks(tick_marks, EMOTION_CLASSES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "dev_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    model_summary = {
        "total_parameters": total_params,
        "layers": len(list(model)),
        "dev_accuracy": report_dict["accuracy"],
        "dev_macro_f1": report_dict["macro avg"]["f1-score"],
        "dev_per_class": report_dict,
        "dev_confusion_matrix": cm.tolist(),
        "mean_predicted_probabilities": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, mean_probs)},
    }
    with (RESULTS_DIR / "model_audit.json").open("w") as f:
        json.dump(model_summary, f, indent=2)

    return model_summary


def audit_video_inference():
    print("\n" + "=" * 60)
    print("3. VIDEO INFERENCE AUDIT")
    print("=" * 60)

    video_path = REPOSITORY_ROOT / "datasets" / "MELD_raw_acquisition" / "media" / "dev_splits_complete" / "dia4_utt1.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    print(f"Video: {video_path.name}")
    print(f"Dimensions: {width}x{height} | Total Frames: {total_frames} | FPS: {fps:.2f} | Duration: {duration:.2f}s")

    detector = FaceDetector()
    predictor = FacialEmotionPredictor(REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt", device="cuda")

    # 1. Step-by-step extraction of sampled frames (sample_every=3)
    sample_every = 3
    capture = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    sampled_frame_records = []
    all_face_crops = []
    face_metadata = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            detections = detector.detect(frame, size=96)
            rec = {
                "frame_index": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "faces_detected": len(detections),
                "bboxes": [d.bbox for d in detections],
            }
            sampled_frame_records.append(rec)
            for d_idx, d in enumerate(detections):
                all_face_crops.append(d.crop)
                face_metadata.append({
                    "frame_index": frame_idx,
                    "face_index_in_frame": d_idx,
                    "bbox": d.bbox,
                    "center": d.center,
                })
        frame_idx += 1
    capture.release()

    print(f"Sampled frames (every {sample_every} frames): {len(sampled_frame_records)}")
    print(f"Total face detections across sampled frames: {len(all_face_crops)}")

    # Frame-by-frame face count table
    print("\nFrame-by-Frame Detection Breakdown:")
    print(f"{'Frame #':8s} | {'Time (s)':8s} | {'Faces Detected':15s} | {'Bounding Boxes (x, y, w, h)'}")
    print("-" * 65)
    for rec in sampled_frame_records:
        bboxes_str = "; ".join(f"({x},{y},{w},{h})" for x, y, w, h in rec["bboxes"])
        print(f"{rec['frame_index']:8d} | {rec['timestamp_sec']:8.2f} | {rec['faces_detected']:15d} | {bboxes_str}")

    # 2. Frame-level / Face-level Emotion Probability Predictions
    print("\nEvaluating individual face crop predictions...")
    individual_preds = []
    with torch.no_grad():
        for f_meta, crop in zip(face_metadata, all_face_crops):
            batch = np.expand_dims(crop, axis=0)
            tensor = preprocess_face_crops(batch, predictor.torch).to(predictor.device)
            logits = predictor.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_label = EMOTION_CLASSES[int(probs.argmax())]
            individual_preds.append({
                **f_meta,
                "predicted_label": pred_label,
                "confidence": float(probs.max()),
                "probabilities": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, probs)},
            })

    # Save face-level distributions to CSV
    df_faces = pd.DataFrame([
        {
            "frame": p["frame_index"],
            "face_in_frame": p["face_index_in_frame"],
            "bbox": str(p["bbox"]),
            "pred": p["predicted_label"],
            "conf": p["confidence"],
            **{f"prob_{cls}": p["probabilities"][cls] for cls in EMOTION_CLASSES}
        }
        for p in individual_preds
    ])
    df_faces.to_csv(RESULTS_DIR / "frame_level_face_predictions.csv", index=False)
    print(f"Saved {len(df_faces)} individual face predictions to: {RESULTS_DIR / 'frame_level_face_predictions.csv'}")

    # 3. Temporal Segment Analysis: Early (0-33%), Middle (33-66%), Late (66-100%)
    n_faces = len(individual_preds)
    t1 = n_faces // 3
    t2 = 2 * n_faces // 3

    early_preds = individual_preds[:t1]
    mid_preds = individual_preds[t1:t2]
    late_preds = individual_preds[t2:]

    def segment_analysis(segment_name: str, preds: list[dict]):
        if not preds:
            return {}
        probs = np.array([[p["probabilities"][cls] for cls in EMOTION_CLASSES] for p in preds])
        mean_prob = probs.mean(axis=0)
        top_idx = int(mean_prob.argmax())
        return {
            "name": segment_name,
            "face_count": len(preds),
            "predicted_emotion": EMOTION_CLASSES[top_idx],
            "confidence": float(mean_prob[top_idx]),
            "mean_probabilities": {cls: float(p) for cls, p in zip(EMOTION_CLASSES, mean_prob)},
        }

    early_summary = segment_analysis("Early (0-33%)", early_preds)
    mid_summary = segment_analysis("Middle (33-66%)", mid_preds)
    late_summary = segment_analysis("Late (66-100%)", late_preds)

    print("\nTemporal Segment Breakdown:")
    for seg in [early_summary, mid_summary, late_summary]:
        print(f"  {seg['name']}: {seg['face_count']} faces -> Top: {seg['predicted_emotion']} ({seg['confidence']*100:.1f}%)")

    # 4. Standard Aggregated Video Inference
    full_res = predictor.predict_video(video_path, sample_every=3)
    print("\nFull Video Predictor Output (Standard Pipeline):")
    for k, v in full_res.items():
        print(f"  {k}: {v}")

    # Save summary
    video_summary = {
        "video_name": video_path.name,
        "duration_sec": duration,
        "total_frames": total_frames,
        "sampled_frames_count": len(sampled_frame_records),
        "total_detected_faces": len(all_face_crops),
        "standard_pipeline_result": full_res,
        "temporal_segments": {
            "early": early_summary,
            "middle": mid_summary,
            "late": late_summary,
        },
    }
    with (RESULTS_DIR / "video_audit.json").open("w") as f:
        json.dump(video_summary, f, indent=2)

    return video_summary


def main():
    print("STARTING FACIAL EMOTION COMPREHENSIVE DIAGNOSTIC AUDIT")
    t0 = time.time()
    audit_dataset()
    audit_model()
    audit_video_inference()
    print(f"\nDiagnostic audit complete in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
