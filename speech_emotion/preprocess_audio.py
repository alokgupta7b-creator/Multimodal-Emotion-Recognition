"""
speech_emotion.preprocess_audio
===============================
Dataset-level audio preprocessing, batch feature extraction, and audit logging.

Capabilities:
-------------
1. Official Split Integrity:
   - Preserves official MELD train, dev, and test split definitions.
   - Never randomly repartitions data or mixes splits.
   - Preserves dialogue_id, utterance_id, sample_id, and ground-truth emotion labels.
2. Anomaly Detection & Deduplication:
   - Detects duplicate sample IDs and skips duplicates with explicit warning.
   - Distinguishes missing audio from empty, corrupted, silent, or invalid audio.
   - Catches unexpected or missing emotion labels.
3. Feature Table & Audit Generation:
   - Writes feature CSV: `Dialogue_ID`, `Utterance_ID`, `emotion`,
     `mfcc_mean_01` .. `20`, `mfcc_std_01` .. `20`.
   - Writes failure CSV: `split`, `Dialogue_ID`, `Utterance_ID`, `reason`.
   - Writes detailed machine-readable JSON and human-readable Markdown audit reports.
4. CLI Entry Point:
   - `python -m speech_emotion.preprocess_audio [OPTIONS]`
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from speech_emotion.audio_loader import AudioLoadStatus, AudioMetadata
from speech_emotion.config import (
    AudioConfig,
    DEFAULT_SAMPLE_RATE,
    MELD_EMOTIONS,
    MELD_SPLITS,
)
from speech_emotion.feature_extraction import MFCCExtractor


# ---------------------------------------------------------------------------
# Audit Report Dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AuditReport:
    """
    Aggregated summary of audio preprocessing and feature extraction run.
    """

    split: str
    generated_at: str
    audio_dir: str
    manifest_source: str
    total_records: int
    processed_count: int
    valid_count: int
    invalid_count: int
    missing_audio_count: int
    silent_audio_count: int
    duplicate_count: int
    excluded_count: int
    feature_shape: Tuple[int, ...]
    config_summary: Dict[str, Any]
    emotion_distribution: Dict[str, int]
    failure_breakdown: Dict[str, int]
    warnings: List[str]


# ---------------------------------------------------------------------------
# Manifest and Audio Resolver
# ---------------------------------------------------------------------------

def load_manifest_records(
    source_path: Union[str, Path],
    expected_split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load records from a MELD manifest CSV or annotation CSV.

    Normalizes column names to standard keys:
    - `dialogue_id`: int
    - `utterance_id`: int
    - `emotion`: str (lowercase)
    - `split`: str (lowercase)
    - `sample_id`: str
    """
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest source not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Manifest CSV has no header: {path}")

        col_map = {c.lower().strip(): c for c in reader.fieldnames}

        # Resolve dialogue column
        dia_col = None
        for cand in ("dialogue_id", "dialogueid", "dialogue"):
            if cand in col_map:
                dia_col = col_map[cand]
                break

        # Resolve utterance column
        utt_col = None
        for cand in ("utterance_id", "utteranceid", "utterance"):
            if cand in col_map:
                utt_col = col_map[cand]
                break

        # Resolve emotion column
        emo_col = col_map.get("emotion")
        split_col = col_map.get("split")
        sample_id_col = col_map.get("sample_id")

        if dia_col is None or utt_col is None:
            raise ValueError(f"Manifest missing Dialogue_ID or Utterance_ID columns: {path}")

        for row in reader:
            dia_str = str(row[dia_col]).strip()
            utt_str = str(row[utt_col]).strip()
            try:
                dia_id = int(dia_str)
                utt_id = int(utt_str)
            except ValueError:
                continue

            row_split = str(row.get(split_col, "")).strip().lower() if split_col else (expected_split or "")
            if expected_split and row_split and row_split != expected_split.lower():
                continue

            emotion = str(row.get(emo_col, "neutral")).strip().lower() if emo_col else "neutral"
            sample_id = str(row.get(sample_id_col, "")).strip() if sample_id_col else f"{row_split or 'meld'}_dia{dia_id}_utt{utt_id}"

            records.append({
                "dialogue_id": dia_id,
                "utterance_id": utt_id,
                "emotion": emotion,
                "split": row_split or (expected_split or ""),
                "sample_id": sample_id,
                "raw_row": row,
            })

    return records


# ---------------------------------------------------------------------------
# Split Processor
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    audio_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[AudioConfig] = None,
    limit: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], AuditReport]:
    """
    Process an entire split: load audio, validate, extract features, and record failures.

    Parameters
    ----------
    split:
        Split name ("train", "dev", "test").
    audio_dir:
        Directory containing WAV files named `dia{D}_utt{U}.wav`.
    manifest_path:
        Path to MELD manifest or annotation CSV for this split.
    output_dir:
        Directory to write feature table and audit reports.
    config:
        AudioConfig with feature extraction settings.
    limit:
        If > 0, process only up to `limit` records (useful for smoke tests).

    Returns
    -------
    (feature_rows, failure_rows, audit_report)
    """
    cfg = config or AudioConfig()
    extractor = MFCCExtractor(cfg)

    audio_path = Path(audio_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = load_manifest_records(manifest_path, expected_split=split)
    total_records = len(records)

    if limit > 0:
        records = records[:limit]

    feature_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    warnings: List[str] = []
    emotion_dist: Dict[str, int] = {}
    failure_breakdown: Dict[str, int] = {}

    missing_audio_count = 0
    silent_audio_count = 0
    duplicate_count = 0

    feature_names = cfg.get_feature_names()

    for idx, rec in enumerate(records):
        sample_id = rec["sample_id"]
        dia_id = rec["dialogue_id"]
        utt_id = rec["utterance_id"]
        emotion = rec["emotion"]

        # Deduplication check
        if sample_id in seen_ids:
            duplicate_count += 1
            reason = "duplicate_sample_id"
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            failure_rows.append({
                "split": split,
                "Dialogue_ID": dia_id,
                "Utterance_ID": utt_id,
                "sample_id": sample_id,
                "reason": reason,
            })
            warnings.append(f"Row {idx}: Duplicate sample_id {sample_id!r} skipped")
            continue

        seen_ids.add(sample_id)

        # Emotion validation
        if emotion not in MELD_EMOTIONS:
            reason = f"unexpected_emotion:{emotion}"
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            failure_rows.append({
                "split": split,
                "Dialogue_ID": dia_id,
                "Utterance_ID": utt_id,
                "sample_id": sample_id,
                "reason": reason,
            })
            continue

        # Audio file resolution
        wav_file = audio_path / f"dia{dia_id}_utt{utt_id}.wav"
        if not wav_file.is_file():
            missing_audio_count += 1
            reason = AudioLoadStatus.FILE_NOT_FOUND
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            failure_rows.append({
                "split": split,
                "Dialogue_ID": dia_id,
                "Utterance_ID": utt_id,
                "sample_id": sample_id,
                "reason": "missing_audio",
            })
            continue

        # Extract features
        feats, meta = extractor.extract_from_file(wav_file)

        if not meta.is_valid or feats is None:
            reason = meta.status or "invalid_audio"
            if meta.status == AudioLoadStatus.SILENT_AUDIO:
                silent_audio_count += 1
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1
            failure_rows.append({
                "split": split,
                "Dialogue_ID": dia_id,
                "Utterance_ID": utt_id,
                "sample_id": sample_id,
                "reason": f"{reason}:{meta.reason}" if meta.reason else reason,
            })
            continue

        # Valid feature row matching MELD convention
        row_dict: Dict[str, Any] = {
            "Dialogue_ID": dia_id,
            "Utterance_ID": utt_id,
            "emotion": emotion,
        }
        for name, val in zip(feature_names, feats):
            row_dict[name] = float(val)

        feature_rows.append(row_dict)
        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1

    valid_count = len(feature_rows)
    invalid_count = len(failure_rows)
    excluded_count = invalid_count

    report = AuditReport(
        split=split,
        generated_at=datetime.now(timezone.utc).isoformat(),
        audio_dir=str(audio_path),
        manifest_source=str(manifest_path),
        total_records=total_records,
        processed_count=len(records),
        valid_count=valid_count,
        invalid_count=invalid_count,
        missing_audio_count=missing_audio_count,
        silent_audio_count=silent_audio_count,
        duplicate_count=duplicate_count,
        excluded_count=excluded_count,
        feature_shape=(valid_count, len(feature_names)),
        config_summary={
            "sample_rate": cfg.sample_rate,
            "n_mfcc": cfg.n_mfcc,
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
            "fmin": cfg.fmin,
            "fmax": cfg.fmax,
            "min_duration": cfg.min_duration,
            "max_duration": cfg.max_duration,
            "silence_threshold_db": cfg.silence_threshold_db,
            "engine": cfg.engine,
            "total_features": cfg.total_features,
        },
        emotion_distribution=emotion_dist,
        failure_breakdown=failure_breakdown,
        warnings=warnings,
    )

    # Write output files
    _write_outputs(out_path, split, feature_rows, failure_rows, report, feature_names)

    return feature_rows, failure_rows, report


# ---------------------------------------------------------------------------
# Output Serializers
# ---------------------------------------------------------------------------

def _write_outputs(
    out_dir: Path,
    split: str,
    feature_rows: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
    report: AuditReport,
    feature_names: List[str],
) -> None:
    """Write feature CSV, failure CSV, JSON report, and Markdown report."""
    # 1. Feature CSV
    feat_csv_path = out_dir / f"{split}_mfcc.csv"
    feat_fields = ["Dialogue_ID", "Utterance_ID", "emotion"] + feature_names
    with open(feat_csv_path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=feat_fields)
        writer.writeheader()
        for r in feature_rows:
            writer.writerow(r)

    # 2. Failure CSV
    fail_csv_path = out_dir / f"{split}_mfcc_failures.csv"
    fail_fields = ["split", "Dialogue_ID", "Utterance_ID", "reason"]
    with open(fail_csv_path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=fail_fields, extrasaction="ignore")
        writer.writeheader()
        for r in failure_rows:
            writer.writerow(r)

    # 3. JSON Audit Report
    json_path = out_dir / f"{split}_audio_audit.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(report), fh, indent=2)

    # 4. Markdown Audit Report
    md_path = out_dir / f"{split}_audio_audit.md"
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# MELD Speech Preprocessing & Feature Extraction Audit — {split.upper()}\n\n")
        fh.write(f"- **Generated At**: `{report.generated_at}`\n")
        fh.write(f"- **Split**: `{report.split}`\n")
        fh.write(f"- **Audio Source**: `{report.audio_dir}`\n")
        fh.write(f"- **Manifest**: `{report.manifest_source}`\n\n")

        fh.write("## 1. Accounting Summary\n\n")
        fh.write("| Metric | Count |\n")
        fh.write("| :--- | :--- |\n")
        fh.write(f"| Total Manifest Records | {report.total_records} |\n")
        fh.write(f"| Processed Records | {report.processed_count} |\n")
        fh.write(f"| **Valid Features Extracted** | **{report.valid_count}** |\n")
        fh.write(f"| Invalid / Excluded Recordings | {report.invalid_count} |\n")
        fh.write(f"| Missing Audio Files | {report.missing_audio_count} |\n")
        fh.write(f"| Silent Audio Clips | {report.silent_audio_count} |\n")
        fh.write(f"| Duplicate Records Skipped | {report.duplicate_count} |\n")
        fh.write(f"| Feature Matrix Dimensions | {report.feature_shape[0]} × {report.feature_shape[1]} |\n\n")

        if report.emotion_distribution:
            fh.write("## 2. Emotion Class Distribution (Valid Samples)\n\n")
            fh.write("| Emotion | Samples | Percentage |\n")
            fh.write("| :--- | :--- | :--- |\n")
            for emo, count in sorted(report.emotion_distribution.items(), key=lambda x: -x[1]):
                pct = (count / report.valid_count * 100.0) if report.valid_count > 0 else 0.0
                fh.write(f"| `{emo}` | {count} | {pct:.1f}% |\n")
            fh.write("\n")

        if report.failure_breakdown:
            fh.write("## 3. Failure & Exclusion Breakdown\n\n")
            fh.write("| Failure Reason | Count |\n")
            fh.write("| :--- | :--- |\n")
            for reason, count in sorted(report.failure_breakdown.items(), key=lambda x: -x[1]):
                fh.write(f"| `{reason}` | {count} |\n")
            fh.write("\n")

        fh.write("## 4. Extraction Configuration\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(report.config_summary, indent=2))
        fh.write("\n```\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speech_emotion.preprocess_audio",
        description="MELD Speech Preprocessing, Audio Validation, and MFCC Feature Extraction (Task 4)",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "dev", "test"],
        help="Split to process (default: all).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Root directory containing MELD audio (with subfolders train, dev, test) or split WAVs.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("datasets/MELD/manifests"),
        help="Directory containing train_manifest.csv, dev_manifest.csv, test_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/speech_features"),
        help="Directory where output feature CSVs and audit reports will be stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of utterances to process per split (0 means all).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target audio sample rate in Hz (default: 16000).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = AudioConfig(sample_rate=args.sample_rate)
    splits = MELD_SPLITS if args.split == "all" else (args.split,)

    print("=================================================================")
    print("MELD Speech Preprocessing & MFCC Feature Extraction (Task 4)")
    print("=================================================================")
    print(f"Audio directory : {args.audio_dir}")
    print(f"Manifest dir    : {args.manifest_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Splits          : {', '.join(splits)}")
    print(f"Sample limit    : {args.limit if args.limit > 0 else 'Full dataset'}")
    print(f"Configuration   : {config.n_mfcc} MFCCs ({config.total_features} features per utterance)")
    print("-----------------------------------------------------------------")

    total_valid = 0
    total_failures = 0

    for split in splits:
        # Resolve audio split dir
        split_audio = args.audio_dir / split if (args.audio_dir / split).is_dir() else args.audio_dir
        # Resolve manifest file
        manifest_file = args.manifest_dir / f"{split}_manifest.csv"
        if not manifest_file.is_file():
            # Try annotations fallback
            manifest_file = args.manifest_dir / f"{split}_sent_emo.csv"
            if not manifest_file.is_file():
                print(f"ERROR: Cannot find manifest or annotations for split '{split}' in {args.manifest_dir}", file=sys.stderr)
                return 1

        print(f"\n[*] Processing split '{split.upper()}'...")
        feat_rows, fail_rows, report = process_split(
            split=split,
            audio_dir=split_audio,
            manifest_path=manifest_file,
            output_dir=args.output_dir,
            config=config,
            limit=args.limit,
        )

        total_valid += len(feat_rows)
        total_failures += len(fail_rows)

        print(f"  Processed : {report.processed_count}")
        print(f"  Valid     : {report.valid_count}")
        print(f"  Excluded  : {report.invalid_count} (missing audio: {report.missing_audio_count}, silent: {report.silent_audio_count})")
        print(f"  Output    : {args.output_dir / f'{split}_mfcc.csv'}")

    print("\n=================================================================")
    print(f"Summary: {total_valid} valid feature vectors extracted; {total_failures} excluded.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
