"""Validate MELD split, label, ID, and audio-video alignment integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project_config import MELD_MANIFESTS, MELD_PREPARED_ROOT, MELD_SPLITS

EMOTIONS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}


def validate(prepared_root: Path) -> dict[str, object]:
    frames = {split: pd.read_csv(MELD_MANIFESTS / f"{split}_manifest.csv") for split in MELD_SPLITS}
    all_ids: dict[str, list[str]] = {}
    errors: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        ids = frame["sample_id"].astype(str)
        all_ids[split] = ids.tolist()
        counts[split] = {"manifest": len(frame)}
        if ids.duplicated().any():
            errors.append({"type": "duplicate_sample_id_within_split", "split": split, "count": int(ids.duplicated().sum())})
        invalid_labels = sorted(set(frame["emotion"].astype(str).str.lower()) - EMOTIONS)
        if invalid_labels:
            errors.append({"type": "invalid_labels", "split": split, "labels": invalid_labels})
        if frame[["sample_id", "dialogue_id", "utterance_id", "emotion", "video_path", "audio_path"]].isnull().any().any():
            errors.append({"type": "missing_required_annotation", "split": split})
        video_manifest = prepared_root / "reports" / split / "video_manifest.csv"
        audio_manifest = prepared_root / "reports" / split / "audio_manifest.csv"
        videos = pd.read_csv(video_manifest) if video_manifest.is_file() else pd.DataFrame()
        audios = pd.read_csv(audio_manifest) if audio_manifest.is_file() else pd.DataFrame()
        counts[split]["valid_video"] = len(videos)
        counts[split]["valid_audio"] = len(audios)
        if len(videos) != len(frame):
            errors.append({"type": "video_coverage_mismatch", "split": split, "manifest": len(frame), "valid": len(videos)})
        if len(audios) != len(frame):
            errors.append({"type": "audio_coverage_mismatch", "split": split, "manifest": len(frame), "valid": len(audios)})
    for left_index, left in enumerate(MELD_SPLITS):
        for right in MELD_SPLITS[left_index + 1:]:
            overlap = sorted(set(all_ids[left]).intersection(all_ids[right]))
            if overlap:
                errors.append({"type": "cross_split_sample_id_overlap", "left": left, "right": right, "count": len(overlap), "sample_ids": overlap[:20]})
    result = {"valid": not errors, "emotion_classes": sorted(EMOTIONS), "counts": counts, "errors": errors}
    report = prepared_root / "reports" / "multimodal_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=MELD_PREPARED_ROOT)
    args = parser.parse_args()
    result = validate(args.prepared_root)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
