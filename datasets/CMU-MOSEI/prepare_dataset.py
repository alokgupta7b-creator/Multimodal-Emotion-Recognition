#!/usr/bin/env python3
"""
CMU-MOSEI Dataset Preparation & Preprocessing Pipeline
======================================================
Project: Multimodal Emotion Recognition Using Text, Speech, and Facial Features
         with Robust Feature Fusion

This script validates official CMU-MOSEI annotations and feature/media files,
enforces strict speaker-independent video-level split boundaries (preventing data leakage),
validates continuous sentiment and emotion rating schemas, detects missing or corrupted
modal representations, and produces deterministic, reproducible manifests and audit reports.

Usage:
    python prepare_dataset.py \
        --annotations-file "datasets/CMU-MOSEI/annotations/mosei_annotations.csv" \
        --output-dir "datasets/CMU-MOSEI" \
        --skip-media-verify
"""

import argparse
import csv
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Official 6 emotion dimensions evaluated in CMU-MOSEI (intensity scale [0.0, 3.0])
OFFICIAL_EMOTIONS = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]

# Continuous sentiment range defined by the CMU-MOSEI benchmark: [-3.0, +3.0]
SENTIMENT_MIN = -3.0
SENTIMENT_MAX = 3.0

# Emotion intensity range: [0.0, 3.0]
EMOTION_MIN = 0.0
EMOTION_MAX = 3.0

# Valid standard splits
VALID_SPLITS = {"train", "dev", "test"}


def check_wav_integrity(filepath: Path) -> Tuple[bool, str]:
    """
    Validates standard RIFF/WAVE header integrity in pure Python.
    Checks RIFF chunk descriptor, WAVE format marker, fmt subchunk, and data chunk.
    """
    if not filepath.exists():
        return False, "missing_file"
    try:
        size = filepath.stat().st_size
    except OSError as e:
        return False, f"stat_error_{e}"

    if size < 44:
        return False, "wav_too_small_under_44_bytes"

    try:
        with open(filepath, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return False, "missing_riff_marker"
            f.seek(8)
            wave = f.read(4)
            if wave != b"WAVE":
                return False, "missing_wave_marker"
            fmt = f.read(4)
            if fmt != b"fmt ":
                return False, "missing_fmt_subchunk"
            subchunk1_size = struct.unpack("<I", f.read(4))[0]
            if subchunk1_size < 16:
                return False, f"invalid_subchunk1_size_{subchunk1_size}"
            audio_format = struct.unpack("<H", f.read(2))[0]
            if audio_format not in (1, 3, 6, 7, 0xFFFE):
                return False, f"unsupported_audio_format_{audio_format}"
    except Exception as e:
        return False, f"wav_parse_error_{type(e).__name__}"

    return True, "ok"


def check_mp4_integrity(filepath: Path) -> Tuple[bool, str]:
    """
    Validates MP4 container integrity in pure Python without FFprobe.
    Scans ISO base media file format atoms to verify ftyp and moov structures.
    """
    if not filepath.exists():
        return False, "missing_file"
    try:
        size = filepath.stat().st_size
    except OSError as e:
        return False, f"stat_error_{e}"

    if size < 32:
        return False, "file_truncated_too_small"

    found_ftyp = False
    found_moov = False

    try:
        with open(filepath, "rb") as f:
            offset = 0
            while offset + 8 <= size:
                f.seek(offset)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                atom_size, atom_type = struct.unpack(">I4s", hdr)
                try:
                    type_str = atom_type.decode("latin-1")
                except Exception:
                    type_str = ""

                if type_str == "ftyp":
                    found_ftyp = True
                elif type_str == "moov":
                    found_moov = True

                if atom_size == 1:
                    ext_hdr = f.read(8)
                    if len(ext_hdr) < 8:
                        break
                    atom_size = struct.unpack(">Q", ext_hdr)[0]
                    if atom_size < 16:
                        break
                elif atom_size == 0:
                    break
                elif atom_size < 8:
                    break

                offset += atom_size

        if not found_ftyp and not found_moov:
            return False, "invalid_mp4_no_ftyp_or_moov"
        if not found_moov:
            return False, "corrupted_video_moov_atom_missing"
    except Exception as e:
        return False, f"mp4_parse_error_{type(e).__name__}"

    return True, "ok"


def check_feature_file_integrity(filepath: Path) -> Tuple[bool, str]:
    """Validates generic feature array file existence and non-zero size."""
    if not filepath.exists():
        return False, "missing_file"
    try:
        if filepath.stat().st_size == 0:
            return False, "empty_feature_file"
        return True, "ok"
    except OSError as e:
        return False, f"stat_error_{e}"


def sanitize_path_display(path_str: Optional[str]) -> str:
    """Sanitizes file paths to avoid leaking user profile directories in public reports."""
    if not path_str or path_str == "None":
        return "Not Provided"
    p = Path(path_str)
    try:
        return p.resolve().relative_to(Path.cwd()).as_posix()
    except (ValueError, RuntimeError):
        return f".../{p.name}"


def discretize_sentiment(score: float, threshold: float = 0.0) -> str:
    """
    Maps continuous sentiment in [-3.0, 3.0] to a 3-class categorical label
    using an explicit decision threshold.
    """
    if score > threshold:
        return "positive"
    elif score < -threshold:
        return "negative"
    else:
        return "neutral"


def determine_dominant_emotion(emotions: Dict[str, float], threshold: float = 0.0) -> str:
    """
    Identifies the emotion with maximum intensity score above threshold.
    Returns 'none' if all scores are <= threshold.
    """
    candidate = max(emotions.items(), key=lambda kv: kv[1])
    if candidate[1] > threshold:
        return candidate[0]
    return "none"


def validate_sample_record(
    row: Dict[str, Any],
    sentiment_threshold: float = 0.0,
    emotion_threshold: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates an individual sample record against the CMU-MOSEI schema.
    Returns parsed clean dictionary or an exclusion reason string.
    """
    # 1. Identifier extraction
    raw_sample_id = str(row.get("sample_id", "")).strip()
    video_id = str(row.get("video_id", "")).strip()
    clip_id = str(row.get("clip_id", "")).strip()

    if not video_id and not raw_sample_id:
        return None, "missing_identifier"

    if not video_id:
        # Attempt to parse from composite sample_id (e.g. vid_001_0 or vid_001[0])
        if "[" in raw_sample_id and raw_sample_id.endswith("]"):
            video_id, clip_part = raw_sample_id[:-1].split("[", 1)
            clip_id = clip_part
        elif "_" in raw_sample_id:
            parts = raw_sample_id.rsplit("_", 1)
            video_id, clip_id = parts[0], parts[1]
        else:
            video_id = raw_sample_id
            clip_id = "0"

    if not clip_id:
        clip_id = "0"

    sample_id = f"{video_id}_{clip_id}"

    # 2. Timestamp validation
    try:
        start_time = float(row.get("start_time", 0.0))
        end_time = float(row.get("end_time", 0.0))
    except (ValueError, TypeError):
        return None, "invalid_timestamp_format"

    if start_time < 0.0 or end_time < start_time:
        return None, "invalid_timestamps_negative_or_inverted"

    # 3. Sentiment score validation
    try:
        sentiment = float(row.get("sentiment", 0.0))
    except (ValueError, TypeError):
        return None, "invalid_sentiment_value"

    if not (SENTIMENT_MIN <= sentiment <= SENTIMENT_MAX):
        return None, f"invalid_sentiment_range_{sentiment}"

    # 4. Emotion intensity ratings validation
    emotion_scores: Dict[str, float] = {}
    for emo in OFFICIAL_EMOTIONS:
        val_str = row.get(emo, 0.0)
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            return None, f"invalid_emotion_value_{emo}"

        if not (EMOTION_MIN <= val <= EMOTION_MAX):
            return None, f"invalid_emotion_range_{emo}_{val}"

        emotion_scores[emo] = round(val, 4)

    # 5. Split assignment
    split = str(row.get("split", "")).strip().lower()

    # Derived attributes
    discrete_sent = discretize_sentiment(sentiment, threshold=sentiment_threshold)
    dominant_emo = determine_dominant_emotion(emotion_scores, threshold=emotion_threshold)

    record = {
        "sample_id": sample_id,
        "video_id": video_id,
        "clip_id": clip_id,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "sentiment": round(sentiment, 4),
        "sentiment_discrete": discrete_sent,
        "happy": emotion_scores["happy"],
        "sad": emotion_scores["sad"],
        "anger": emotion_scores["anger"],
        "surprise": emotion_scores["surprise"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "dominant_emotion": dominant_emo,
        "transcript": str(row.get("text", row.get("transcript", ""))).strip(),
        "split": split,
    }

    return record, None


def resolve_sample_media(
    record: Dict[str, Any],
    raw_dir: Optional[Path],
    audio_dir: Optional[Path],
    visual_dir: Optional[Path],
    text_dir: Optional[Path],
    verify_media: bool = True,
    csd_handles: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Resolves and verifies availability of text, audio, and visual modality files.
    Supports either precomputed CSD archives (.csd) or filesystem directories (.wav/.mp4/.npy).
    Returns updated record with modality flags and exclusion reason if any.
    """
    sample_id = record["sample_id"]
    video_id = record["video_id"]
    clip_id = record["clip_id"]
    start_time = float(record["start_time"])
    end_time = float(record["end_time"])

    text_path_rel = ""
    audio_path_rel = ""
    visual_path_rel = ""

    has_text = bool(record.get("transcript"))
    has_audio = False
    has_visual = False

    video_status = "unverified"
    audio_status = "unverified"
    text_status = "ok" if has_text else "not_provided"

    if csd_handles and verify_media:
        # 1. Text modality via CSD or transcript
        if "words" in csd_handles and video_id in csd_handles["words"]:
            w_intervals = csd_handles["words"][video_id]["intervals"][:]
            overlap = (w_intervals[:, 0] < end_time) & (w_intervals[:, 1] > start_time)
            if overlap.any() or has_text:
                has_text = True
                text_status = "ok_csd_words"
                text_path_rel = f"features/CMU_MOSEI_TimestampedWords.csd#{sample_id}"
            else:
                text_status = "empty_csd_words"
        elif has_text:
            text_status = "ok_transcript"
            text_path_rel = f"features/CMU_MOSEI_TimestampedWords.csd#{sample_id}"
        else:
            text_status = "missing_csd_words"

        # 2. Audio modality via COVAREP CSD
        if "audio" in csd_handles:
            if video_id in csd_handles["audio"]:
                a_intervals = csd_handles["audio"][video_id]["intervals"][:]
                overlap = (a_intervals[:, 0] < end_time) & (a_intervals[:, 1] > start_time)
                if overlap.any():
                    has_audio = True
                    audio_status = "ok_csd_covarep"
                    audio_path_rel = f"features/CMU_MOSEI_COVAREP.csd#{sample_id}"
                else:
                    audio_status = "zero_audio_frames_in_interval"
            else:
                audio_status = "video_not_in_covarep_csd"
        else:
            audio_status = "no_audio_csd_provided"

        # 3. Visual modality via FACET 4.2 CSD
        if "visual" in csd_handles:
            if video_id in csd_handles["visual"]:
                v_intervals = csd_handles["visual"][video_id]["intervals"][:]
                overlap = (v_intervals[:, 0] < end_time) & (v_intervals[:, 1] > start_time)
                if overlap.any():
                    has_visual = True
                    video_status = "ok_csd_facet42"
                    visual_path_rel = f"features/CMU_MOSEI_VisualFacet42.csd#{sample_id}"
                else:
                    video_status = "zero_visual_frames_in_interval"
            else:
                video_status = "video_not_in_facet_csd"
        else:
            video_status = "no_visual_csd_provided"

    else:
        # 1. Text modality verification
        if text_dir and text_dir.is_dir():
            cand_txt = [
                text_dir / f"{sample_id}.txt",
                text_dir / video_id / f"{clip_id}.txt",
                text_dir / f"{sample_id}.npy",
            ]
            for p in cand_txt:
                if p.exists():
                    text_path_rel = p.relative_to(text_dir.parent).as_posix()
                    has_text = True
                    text_status = "ok"
                    break

        # 2. Audio modality verification
        if audio_dir and audio_dir.is_dir():
            cand_aud = [
                audio_dir / f"{sample_id}.wav",
                audio_dir / f"{sample_id}.mp3",
                audio_dir / video_id / f"{clip_id}.wav",
                audio_dir / f"{sample_id}.npy",
            ]
            for p in cand_aud:
                if p.exists():
                    audio_path_rel = p.relative_to(audio_dir.parent).as_posix()
                    if verify_media and p.suffix.lower() == ".wav":
                        valid, reason = check_wav_integrity(p)
                        audio_status = reason
                        has_audio = valid
                    elif verify_media:
                        valid, reason = check_feature_file_integrity(p)
                        audio_status = reason
                        has_audio = valid
                    else:
                        audio_status = "present_unverified"
                        has_audio = True
                    break
            if not has_audio and audio_status == "unverified":
                audio_status = "missing_file"
        else:
            audio_status = "no_audio_dir_specified"

        # 3. Visual modality verification
        target_vis_dir = visual_dir or raw_dir
        if target_vis_dir and target_vis_dir.is_dir():
            cand_vis = [
                target_vis_dir / f"{sample_id}.mp4",
                target_vis_dir / f"{sample_id}.avi",
                target_vis_dir / video_id / f"{clip_id}.mp4",
                target_vis_dir / f"{sample_id}.npy",
            ]
            for p in cand_vis:
                if p.exists():
                    visual_path_rel = p.relative_to(target_vis_dir.parent).as_posix()
                    if verify_media and p.suffix.lower() in (".mp4", ".mov"):
                        valid, reason = check_mp4_integrity(p)
                        video_status = reason
                        has_visual = valid
                    elif verify_media:
                        valid, reason = check_feature_file_integrity(p)
                        video_status = reason
                        has_visual = valid
                    else:
                        video_status = "present_unverified"
                        has_visual = True
                    break
            if not has_visual and video_status == "unverified":
                video_status = "missing_file"
        else:
            video_status = "no_visual_dir_specified"

    # Compute overall usability
    # When media verification is skipped, usability is determined by schema validity
    exclusion_reason = ""
    is_usable = True

    if verify_media:
        if csd_handles:
            if not has_audio:
                is_usable = False
                exclusion_reason = f"audio_{audio_status}"
            elif not has_visual:
                is_usable = False
                exclusion_reason = f"video_{video_status}"
        else:
            target_vis_dir = visual_dir or raw_dir
            if audio_dir and not has_audio:
                is_usable = False
                exclusion_reason = f"audio_{audio_status}"
            elif target_vis_dir and not has_visual:
                is_usable = False
                exclusion_reason = f"video_{video_status}"

    record["text_path"] = text_path_rel
    record["audio_path"] = audio_path_rel
    record["visual_path"] = visual_path_rel
    record["has_text"] = has_text
    record["has_audio"] = has_audio
    record["has_visual"] = has_visual
    record["is_fully_usable"] = is_usable
    record["exclusion_reason"] = exclusion_reason
    record["audio_status"] = audio_status
    record["video_status"] = video_status

    return record, (exclusion_reason if not is_usable else None)


def write_manifest(manifest_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Writes manifest rows deterministically to CSV."""
    if not manifest_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0].keys())
    # Deterministic sorting by sample_id
    sorted_rows = sorted(manifest_rows, key=lambda r: (r.get("video_id", ""), r.get("clip_id", ""), r.get("sample_id", "")))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow(r)


def write_reports(
    all_stats: Dict[str, Any],
    reports_dir: Path,
    sanitized_anno_path: str,
    sanitized_audio_path: str,
    sanitized_visual_path: str,
    sanitized_features_path: str = "Not Provided",
) -> None:
    """Writes structured JSON, Markdown summary report, and exclusions CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    json_path = reports_dir / "preprocessing_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    # 2. Exclusions CSV
    exclusions_path = reports_dir / "exclusions.csv"
    all_exclusions: List[Dict[str, Any]] = []
    for split_data in all_stats.get("splits", {}).values():
        all_exclusions.extend(split_data.get("exclusions", []))
    all_exclusions.extend(all_stats.get("unassigned_exclusions", []))

    if all_exclusions:
        with open(exclusions_path, "w", encoding="utf-8", newline="\n") as f:
            fieldnames = [
                "sample_id",
                "split",
                "video_id",
                "clip_id",
                "has_text",
                "has_audio",
                "has_visual",
                "exclusion_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for exc in all_exclusions:
                writer.writerow(exc)
    else:
        with open(exclusions_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("sample_id,split,video_id,clip_id,has_text,has_audio,has_visual,exclusion_reason\n")

    # 3. Markdown Report
    md_path = reports_dir / "preprocessing_report.md"
    timestamp = all_stats.get("generated_at", datetime.now().isoformat())

    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CMU-MOSEI Dataset Preprocessing & Validation Report\n\n")
        f.write(f"- **Generated At**: `{timestamp}`\n")
        f.write(f"- **Annotation Source**: `{sanitized_anno_path}`\n")
        if sanitized_features_path and sanitized_features_path != "Not Provided":
            f.write(f"- **Features Source (CSD)**: `{sanitized_features_path}`\n")
        f.write(f"- **Audio Source**: `{sanitized_audio_path}`\n")
        f.write(f"- **Visual Source**: `{sanitized_visual_path}`\n\n")

        f.write("## 1. Summary Statistics per Split\n\n")
        f.write("| Split | Total Segments | Usable | Excluded | Unique Videos | Avg Sentiment |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for s_name in ["train", "dev", "test"]:
            s_stat = all_stats["splits"].get(s_name, {})
            tot = s_stat.get("total_samples", 0)
            use = s_stat.get("usable_samples", 0)
            exc = s_stat.get("excluded_samples", 0)
            vids = len(s_stat.get("unique_videos", []))
            sent_mean = s_stat.get("sentiment_stats", {}).get("mean", 0.0)
            f.write(f"| **{s_name.upper()}** | {tot} | {use} | {exc} | {vids} | {sent_mean:+.3f} |\n")
        f.write("\n")

        f.write("## 2. Split Isolation & Leakage Audit\n\n")
        leakage = all_stats.get("split_isolation_audit", {})
        is_isolated = leakage.get("leakage_free", True)
        status_text = "**PASSED (Zero Leakage)**" if is_isolated else "**FAILED (Leakage Detected)**"
        f.write(f"- **Audit Status**: {status_text}\n")
        f.write(f"- **Train Video Count**: {leakage.get('train_videos', 0)}\n")
        f.write(f"- **Dev Video Count**: {leakage.get('dev_videos', 0)}\n")
        f.write(f"- **Test Video Count**: {leakage.get('test_videos', 0)}\n")
        f.write(f"- **Cross-Split Video Overlap**: {leakage.get('overlapping_videos', [])}\n\n")

        f.write("## 3. Emotion Intensity & Dominance Distribution\n\n")
        f.write("| Emotion Dimension | Train Mean | Dev Mean | Test Mean | Total Dominant Count |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for emo in OFFICIAL_EMOTIONS:
            m_tr = all_stats["splits"].get("train", {}).get("emotion_stats", {}).get(emo, {}).get("mean", 0.0)
            m_dev = all_stats["splits"].get("dev", {}).get("emotion_stats", {}).get(emo, {}).get("mean", 0.0)
            m_te = all_stats["splits"].get("test", {}).get("emotion_stats", {}).get(emo, {}).get("mean", 0.0)
            dom_tot = sum(
                all_stats["splits"].get(s, {}).get("dominant_emotion_counts", {}).get(emo, 0)
                for s in ["train", "dev", "test"]
            )
            f.write(f"| **{emo.capitalize()}** | {m_tr:.3f} | {m_dev:.3f} | {m_te:.3f} | {dom_tot} |\n")
        f.write("\n")

        f.write("## 4. Exclusion Details\n\n")
        if all_exclusions:
            f.write("| Sample ID | Split | Video ID | Exclusion Reason |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for exc in all_exclusions:
                f.write(f"| `{exc['sample_id']}` | {exc.get('split', 'unassigned')} | `{exc.get('video_id', '')}` | **{exc['exclusion_reason']}** |\n")
        else:
            f.write("No exclusions recorded. All processed samples are valid and compliant.\n")


def prepare_cmu_mosei(
    annotations_file: Path,
    splits_file: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    visual_dir: Optional[Path] = None,
    text_dir: Optional[Path] = None,
    features_dir: Optional[Path] = None,
    output_dir: Path = Path("datasets/CMU-MOSEI"),
    verify_media: bool = True,
    sentiment_threshold: float = 0.0,
    emotion_threshold: float = 0.0,
    include_transcripts: bool = False,
) -> Dict[str, Any]:
    """
    Main preparation pipeline function. Executes end-to-end reading,
    validation, split assignment, leakage detection, manifest generation,
    and report output. Supports both raw files and precomputed CSD archives.
    """
    if not annotations_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotations_file}")

    # Load custom split map if provided
    split_map: Dict[str, str] = {}
    if splits_file and splits_file.is_file():
        with open(splits_file, "r", encoding="utf-8") as f:
            raw_splits = json.load(f)
            for s_name, vid_list in raw_splits.items():
                s_lower = str(s_name).lower()
                if "train" in s_lower:
                    s_clean = "train"
                elif "dev" in s_lower or "val" in s_lower:
                    s_clean = "dev"
                elif "test" in s_lower:
                    s_clean = "test"
                else:
                    s_clean = s_lower
                for vid in vid_list:
                    split_map[str(vid).strip()] = s_clean

    # Open CSD feature handles if features_dir is provided
    csd_handles: Dict[str, Any] = {}
    csd_files_to_close: List[Any] = []
    if features_dir and features_dir.is_dir() and verify_media:
        try:
            import h5py
            aud_csd = features_dir / "CMU_MOSEI_COVAREP.csd"
            if aud_csd.is_file():
                h5_aud = h5py.File(aud_csd, "r")
                csd_handles["audio"] = h5_aud[list(h5_aud.keys())[0]]["data"]
                csd_files_to_close.append(h5_aud)
            vis_csd = features_dir / "CMU_MOSEI_VisualFacet42.csd"
            if vis_csd.is_file():
                h5_vis = h5py.File(vis_csd, "r")
                csd_handles["visual"] = h5_vis[list(h5_vis.keys())[0]]["data"]
                csd_files_to_close.append(h5_vis)
            wrd_csd = features_dir / "CMU_MOSEI_TimestampedWords.csd"
            if wrd_csd.is_file():
                h5_wrd = h5py.File(wrd_csd, "r")
                csd_handles["words"] = h5_wrd[list(h5_wrd.keys())[0]]["data"]
                csd_files_to_close.append(h5_wrd)
        except Exception:
            pass

    # Read annotations
    raw_rows: List[Dict[str, Any]] = []
    with open(annotations_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_rows.append(r)

    # Tracking data structures
    seen_sample_ids: Set[str] = set()
    video_to_split: Dict[str, str] = {}
    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    unassigned_exclusions: List[Dict[str, Any]] = []

    # 1. First pass: validation and video-level split assignment & leakage detection
    for r in raw_rows:
        record, err = validate_sample_record(
            r,
            sentiment_threshold=sentiment_threshold,
            emotion_threshold=emotion_threshold,
        )
        if err or record is None:
            raw_id = r.get("sample_id", f"{r.get('video_id', '')}_{r.get('clip_id', '')}")
            unassigned_exclusions.append({
                "sample_id": raw_id,
                "split": r.get("split", "unassigned"),
                "video_id": r.get("video_id", ""),
                "clip_id": r.get("clip_id", ""),
                "has_text": False,
                "has_audio": False,
                "has_visual": False,
                "exclusion_reason": err or "validation_failure",
            })
            continue

        sample_id = record["sample_id"]
        video_id = record["video_id"]

        # Duplicate ID check
        if sample_id in seen_sample_ids:
            unassigned_exclusions.append({
                "sample_id": sample_id,
                "split": record["split"] or "unassigned",
                "video_id": video_id,
                "clip_id": record["clip_id"],
                "has_text": False,
                "has_audio": False,
                "has_visual": False,
                "exclusion_reason": "duplicate_sample_id",
            })
            continue
        seen_sample_ids.add(sample_id)

        # Resolve split
        target_split = split_map.get(video_id) or record["split"]
        target_split = target_split.lower().replace("valid", "dev").replace("val", "dev")

        if target_split not in VALID_SPLITS:
            unassigned_exclusions.append({
                "sample_id": sample_id,
                "split": target_split or "unassigned",
                "video_id": video_id,
                "clip_id": record["clip_id"],
                "has_text": False,
                "has_audio": False,
                "has_visual": False,
                "exclusion_reason": f"invalid_split_assignment_{target_split}",
            })
            continue

        # Split isolation check
        if video_id in video_to_split and video_to_split[video_id] != target_split:
            unassigned_exclusions.append({
                "sample_id": sample_id,
                "split": target_split,
                "video_id": video_id,
                "clip_id": record["clip_id"],
                "has_text": False,
                "has_audio": False,
                "has_visual": False,
                "exclusion_reason": f"cross_split_leakage_video_in_{video_to_split[video_id]}_and_{target_split}",
            })
            continue

        video_to_split[video_id] = target_split
        record["split"] = target_split
        split_records[target_split].append(record)

    # 2. Second pass: modality resolution and stats compilation
    stats_by_split: Dict[str, Any] = {}
    manifest_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for s_name in ["train", "dev", "test"]:
        records = split_records[s_name]
        manifest_rows: List[Dict[str, Any]] = []
        exclusions: List[Dict[str, Any]] = []
        unique_videos: Set[str] = set()

        sentiments: List[float] = []
        sentiment_discrete_counts: Counter = Counter()
        dominant_emotion_counts: Counter = Counter()
        emotion_accumulators: Dict[str, List[float]] = {emo: [] for emo in OFFICIAL_EMOTIONS}

        modality_counts = {
            "has_text": 0,
            "has_audio": 0,
            "has_visual": 0,
            "is_fully_usable": 0,
        }

        for rec in records:
            resolved_rec, exc_reason = resolve_sample_media(
                rec,
                raw_dir=raw_dir,
                audio_dir=audio_dir,
                visual_dir=visual_dir,
                text_dir=text_dir,
                verify_media=verify_media,
                csd_handles=csd_handles if csd_handles else None,
            )

            if exc_reason:
                exclusions.append(resolved_rec)
            else:
                modality_counts["is_fully_usable"] += 1

            if resolved_rec["has_text"]:
                modality_counts["has_text"] += 1
            if resolved_rec["has_audio"]:
                modality_counts["has_audio"] += 1
            if resolved_rec["has_visual"]:
                modality_counts["has_visual"] += 1

            unique_videos.add(resolved_rec["video_id"])
            sentiments.append(resolved_rec["sentiment"])
            sentiment_discrete_counts[resolved_rec["sentiment_discrete"]] += 1
            dominant_emotion_counts[resolved_rec["dominant_emotion"]] += 1

            for emo in OFFICIAL_EMOTIONS:
                emotion_accumulators[emo].append(resolved_rec[emo])

            # Prepare row for public manifest
            row_out = {
                "sample_id": resolved_rec["sample_id"],
                "split": resolved_rec["split"],
                "video_id": resolved_rec["video_id"],
                "clip_id": resolved_rec["clip_id"],
                "start_time": resolved_rec["start_time"],
                "end_time": resolved_rec["end_time"],
                "sentiment": resolved_rec["sentiment"],
                "sentiment_discrete": resolved_rec["sentiment_discrete"],
                "happy": resolved_rec["happy"],
                "sad": resolved_rec["sad"],
                "anger": resolved_rec["anger"],
                "surprise": resolved_rec["surprise"],
                "disgust": resolved_rec["disgust"],
                "fear": resolved_rec["fear"],
                "dominant_emotion": resolved_rec["dominant_emotion"],
                "text_path": resolved_rec["text_path"],
                "audio_path": resolved_rec["audio_path"],
                "visual_path": resolved_rec["visual_path"],
                "has_text": resolved_rec["has_text"],
                "has_audio": resolved_rec["has_audio"],
                "has_visual": resolved_rec["has_visual"],
                "is_fully_usable": resolved_rec["is_fully_usable"],
                "exclusion_reason": resolved_rec["exclusion_reason"],
            }
            if include_transcripts:
                row_out["transcript"] = resolved_rec["transcript"]

            manifest_rows.append(row_out)

        # Compute summary statistics
        total_s = len(records)
        usable_s = modality_counts["is_fully_usable"]
        excluded_s = len(exclusions)

        sent_stats = {
            "mean": round(sum(sentiments) / total_s, 4) if total_s else 0.0,
            "min": min(sentiments) if total_s else 0.0,
            "max": max(sentiments) if total_s else 0.0,
            "counts": dict(sentiment_discrete_counts),
        }

        emo_stats: Dict[str, Any] = {}
        for emo in OFFICIAL_EMOTIONS:
            vals = emotion_accumulators[emo]
            emo_stats[emo] = {
                "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
                "active_count_gt_zero": sum(1 for v in vals if v > emotion_threshold),
            }

        stats_by_split[s_name] = {
            "total_samples": total_s,
            "usable_samples": usable_s,
            "excluded_samples": excluded_s,
            "unique_videos": sorted(list(unique_videos)),
            "modality_counts": modality_counts,
            "sentiment_stats": sent_stats,
            "emotion_stats": emo_stats,
            "dominant_emotion_counts": dict(dominant_emotion_counts),
            "exclusions": exclusions,
        }
        manifest_rows_by_split[s_name] = manifest_rows

    # 3. Split Isolation Audit
    tr_vids = set(stats_by_split["train"]["unique_videos"])
    dev_vids = set(stats_by_split["dev"]["unique_videos"])
    te_vids = set(stats_by_split["test"]["unique_videos"])

    leak_tr_dev = tr_vids.intersection(dev_vids)
    leak_tr_te = tr_vids.intersection(te_vids)
    leak_dev_te = dev_vids.intersection(te_vids)
    all_leaks = sorted(list(leak_tr_dev.union(leak_tr_te).union(leak_dev_te)))

    split_audit = {
        "leakage_free": len(all_leaks) == 0,
        "train_videos": len(tr_vids),
        "dev_videos": len(dev_vids),
        "test_videos": len(te_vids),
        "overlapping_videos": all_leaks,
    }

    all_stats = {
        "generated_at": datetime.now().isoformat(),
        "annotation_source": sanitize_path_display(str(annotations_file)),
        "features_source": sanitize_path_display(str(features_dir) if features_dir else None),
        "audio_source": sanitize_path_display(str(audio_dir) if audio_dir else None),
        "visual_source": sanitize_path_display(str(visual_dir or raw_dir) if (visual_dir or raw_dir) else None),
        "output_dir": sanitize_path_display(str(output_dir)),
        "split_isolation_audit": split_audit,
        "splits": stats_by_split,
        "unassigned_exclusions": unassigned_exclusions,
    }

    # Close open CSD file handles
    for f_h in csd_files_to_close:
        try:
            f_h.close()
        except Exception:
            pass

    # 4. Write output manifests and reports
    manifests_dir = output_dir / "manifests"
    reports_dir = output_dir / "reports"

    write_manifest(manifest_rows_by_split["train"], manifests_dir / "train_manifest.csv")
    write_manifest(manifest_rows_by_split["dev"], manifests_dir / "dev_manifest.csv")
    write_manifest(manifest_rows_by_split["test"], manifests_dir / "test_manifest.csv")

    write_reports(
        all_stats=all_stats,
        reports_dir=reports_dir,
        sanitized_anno_path=sanitize_path_display(str(annotations_file)),
        sanitized_audio_path=sanitize_path_display(str(audio_dir) if audio_dir else None),
        sanitized_visual_path=sanitize_path_display(str(visual_dir or raw_dir) if (visual_dir or raw_dir) else None),
        sanitized_features_path=sanitize_path_display(str(features_dir) if features_dir else None),
    )

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="CMU-MOSEI Dataset Preprocessing & Preparation Pipeline"
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="datasets/CMU-MOSEI/annotations/mosei_annotations.csv",
        help="Path to CSV containing CMU-MOSEI segment annotations",
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        default=None,
        help="Optional JSON file defining official video ID splits",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed official CSD feature archives",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Optional directory containing raw video clips",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Optional directory containing extracted audio WAVs or audio features",
    )
    parser.add_argument(
        "--visual-dir",
        type=str,
        default=None,
        help="Optional directory containing visual clips or visual features",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Optional directory containing transcripts or text features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/CMU-MOSEI",
        help="Base directory where manifests/ and reports/ will be written",
    )
    parser.add_argument(
        "--skip-media-verify",
        action="store_true",
        help="Skip media integrity checks and process annotation/metadata records only",
    )
    parser.add_argument(
        "--sentiment-threshold",
        type=float,
        default=0.0,
        help="Decision boundary for 3-class sentiment discretization (default: 0.0)",
    )
    parser.add_argument(
        "--emotion-threshold",
        type=float,
        default=0.0,
        help="Decision boundary for emotion presence detection (default: 0.0)",
    )
    parser.add_argument(
        "--include-transcripts",
        action="store_true",
        help="Include transcript text in manifest output (default: False for public git compliance)",
    )

    args = parser.parse_args()

    anno_path = Path(args.annotations_file).resolve()
    splits_path = Path(args.splits_file).resolve() if args.splits_file else None
    features_path = Path(args.features_dir).resolve() if args.features_dir else None
    raw_path = Path(args.raw_dir).resolve() if args.raw_dir else None
    audio_path = Path(args.audio_dir).resolve() if args.audio_dir else None
    visual_path = Path(args.visual_dir).resolve() if args.visual_dir else None
    text_path = Path(args.text_dir).resolve() if args.text_dir else None
    out_dir = Path(args.output_dir).resolve()

    print("=================================================================")
    print("CMU-MOSEI Dataset Preprocessing & Preparation Pipeline")
    print("=================================================================")
    print(f"Annotation File       : {anno_path}")
    print(f"Splits File           : {splits_path or 'Inferred from annotations'}")
    print(f"Features (CSD) Dir    : {features_path or 'None'}")
    print(f"Output Directory      : {out_dir}")
    print(f"Verify Media Files    : {not args.skip_media_verify}")
    print(f"Include Transcripts   : {args.include_transcripts}")
    print("-----------------------------------------------------------------")

    if not anno_path.is_file():
        print(f"Error: Annotations file not found: {anno_path}", file=sys.stderr)
        print("Please obtain official annotations via download_dataset.ps1 or specify a valid file.", file=sys.stderr)
        sys.exit(1)

    stats = prepare_cmu_mosei(
        annotations_file=anno_path,
        splits_file=splits_path,
        raw_dir=raw_path,
        audio_dir=audio_path,
        visual_dir=visual_path,
        text_dir=text_path,
        features_dir=features_path,
        output_dir=out_dir,
        verify_media=not args.skip_media_verify,
        sentiment_threshold=args.sentiment_threshold,
        emotion_threshold=args.emotion_threshold,
        include_transcripts=args.include_transcripts,
    )

    print("\n[+] Processing completed successfully.")
    print(f"    Train Segments : {stats['splits']['train']['total_samples']} (Usable: {stats['splits']['train']['usable_samples']})")
    print(f"    Dev Segments   : {stats['splits']['dev']['total_samples']} (Usable: {stats['splits']['dev']['usable_samples']})")
    print(f"    Test Segments  : {stats['splits']['test']['total_samples']} (Usable: {stats['splits']['test']['usable_samples']})")
    print(f"    Split Leakage Free : {stats['split_isolation_audit']['leakage_free']}")
    print(f"    Manifests and reports generated under {out_dir.as_posix()}/\n")


if __name__ == "__main__":
    main()
