#!/usr/bin/env python3
"""
preprocessing/__main__.py
=========================
Command-line entry point for the multimodal alignment and preprocessing
pipeline.

Run with:
    python -m preprocessing [OPTIONS]

All paths must be supplied at runtime; no hardcoded user paths exist.

Examples
--------
MELD (manifest flags only, no media verification):
    python -m preprocessing \\
        --dataset meld \\
        --manifest datasets/MELD/manifests/train_manifest.csv \\
        --output-dir outputs/alignment/meld/train \\
        --skip-media-verify

MELD (with audio + video roots):
    python -m preprocessing \\
        --dataset meld \\
        --manifest datasets/MELD/manifests/train_manifest.csv \\
        --audio-root <path-to-MELD.Raw/audio/train> \\
        --video-root <path-to-MELD.Raw/train_splits> \\
        --output-dir outputs/alignment/meld/train

CMU-MOSEI (manifest flags only):
    python -m preprocessing \\
        --dataset mosei \\
        --manifest datasets/CMU-MOSEI/manifests/train_manifest.csv \\
        --output-dir outputs/alignment/mosei/train \\
        --skip-media-verify

CMU-MOSEI (with CSD archives):
    python -m preprocessing \\
        --dataset mosei \\
        --manifest datasets/CMU-MOSEI/manifests/train_manifest.csv \\
        --covarep-csd <path-to-CMU_MOSEI_COVAREP.csd> \\
        --facet-csd <path-to-CMU_MOSEI_VisualFacet42.csd> \\
        --output-dir outputs/alignment/mosei/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocessing.align_modalities import run_alignment
from preprocessing.config import AlignmentConfig
from preprocessing.validate_alignment import run_validation, write_report


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m preprocessing",
        description=(
            "Cross-modal text/audio/video alignment and preprocessing pipeline "
            "for MELD and CMU-MOSEI multimodal emotion recognition benchmarks."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["meld", "mosei"],
        help="Target dataset: 'meld' or 'mosei'.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the split manifest CSV (output of the dataset preparation pipeline).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for output alignment records and validation reports.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help=(
            "Root directory for audio files (MELD WAV) or COVAREP NPY arrays "
            "(CMU-MOSEI). If omitted, audio availability is read from manifest flags."
        ),
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Root directory for video files (MELD MP4) or VisualFacet NPY arrays "
            "(CMU-MOSEI). If omitted, visual availability is read from manifest flags."
        ),
    )
    parser.add_argument(
        "--covarep-csd",
        type=Path,
        default=None,
        dest="covarep_csd",
        help="Path to CMU_MOSEI_COVAREP.csd (CMU-MOSEI only).",
    )
    parser.add_argument(
        "--facet-csd",
        type=Path,
        default=None,
        dest="facet_csd",
        help="Path to CMU_MOSEI_VisualFacet42.csd (CMU-MOSEI only).",
    )
    parser.add_argument(
        "--expected-split",
        default=None,
        choices=["train", "dev", "test"],
        help=(
            "Assert that every record in the manifest belongs to this split. "
            "Violations are flagged in the validation report."
        ),
    )
    parser.add_argument(
        "--skip-media-verify",
        action="store_true",
        default=False,
        help="Skip file-level integrity checks; rely on manifest availability flags.",
    )
    parser.add_argument(
        "--transcript-source",
        type=Path,
        default=None,
        help=(
            "Optional path to an external transcript CSV file or directory "
            "(e.g. MELD annotations directory or train_sent_emo.csv) to resolve "
            "utterance text when omitted from the manifest."
        ),
    )
    parser.add_argument(
        "--no-normalize-text",
        action="store_true",
        default=False,
        help="Disable Unicode normalisation of utterance text.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    print(f"[preprocessing] dataset    : {args.dataset}")
    print(f"[preprocessing] manifest   : {args.manifest}")
    print(f"[preprocessing] output-dir : {args.output_dir}")
    print(f"[preprocessing] skip-verify: {args.skip_media_verify}")
    if args.transcript_source:
        print(f"[preprocessing] transcript : {args.transcript_source}")

    # Validate manifest exists
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    # Run alignment
    records, warnings = run_alignment(
        manifest_path=args.manifest,
        dataset=args.dataset,
        audio_root=args.audio_root,
        video_root=args.video_root,
        covarep_csd=args.covarep_csd,
        facet_csd=args.facet_csd,
        normalize_text=not args.no_normalize_text,
        verify_media=not args.skip_media_verify,
        expected_split=args.expected_split,
        transcript_source=args.transcript_source,
    )

    if warnings:
        print(f"[preprocessing] {len(warnings)} warning(s) during alignment:")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... ({len(warnings) - 10} more; see report)")

    # Validate and write report
    report = run_validation(
        records=records,
        warnings=warnings,
        expected_split=args.expected_split,
    )
    write_report(report=report, records=records, output_dir=args.output_dir)

    print(f"\n[preprocessing] Results")
    print(f"  Processed records   : {report.processed_count}")
    print(f"  Fully aligned       : {report.fully_aligned_count}")
    print(f"  Partial (some missing): {report.partially_aligned_count}")
    print(f"  Excluded            : {report.excluded_count}")
    print(f"  Missing text        : {report.missing_text_count}")
    print(f"  Missing audio       : {report.missing_audio_count}")
    print(f"  Missing visual      : {report.missing_visual_count}")
    print(f"\n[preprocessing] Reports written to: {args.output_dir.resolve()}")
    print(f"  alignment_report.json")
    print(f"  alignment_report.md")
    print(f"  alignment_exclusions.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
