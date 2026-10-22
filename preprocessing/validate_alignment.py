"""
preprocessing.validate_alignment
==================================
Generates reproducible validation reports from a list of ``AlignmentRecord``
objects.

The report distinguishes:
  - Input records vs. successfully aligned records.
  - Partially available samples (some modalities missing) vs. fully aligned.
  - Exclusion counts grouped by reason.
  - Split isolation checks (no cross-split sample IDs).
  - Duplicate sample IDs.
  - Invalid or missing timestamps.

Reports are written to ``{output_dir}/alignment_report.json`` and
``{output_dir}/alignment_report.md``.  A per-sample exclusions CSV is
also written to ``{output_dir}/alignment_exclusions.csv``.

Public API
----------
    AlignmentReport    — dataclass holding aggregated counts
    run_validation     — compute AlignmentReport from AlignmentRecord list
    write_report       — serialise report to JSON, Markdown, and CSV
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from preprocessing.align_modalities import AlignmentRecord


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AlignmentReport:
    """Aggregated alignment validation counts and checks."""

    dataset: str
    split: str                          # "mixed" if records span multiple splits
    generated_at: str

    # Counts
    input_count: int                    # rows loaded from manifest
    processed_count: int                # rows that reached the adapter
    duplicate_count: int                # sample IDs seen more than once
    split_violation_count: int          # records from unexpected splits

    # Modality availability
    has_text_count: int
    has_audio_count: int
    has_visual_count: int
    missing_text_count: int
    missing_audio_count: int
    missing_visual_count: int

    # Alignment quality
    fully_aligned_count: int            # text + audio + visual all present, no exclusion
    partially_aligned_count: int        # at least one modality present, some missing
    excluded_count: int                 # records with an exclusion reason

    # Timestamp issues (CMU-MOSEI)
    invalid_timestamp_count: int

    # Exclusion breakdown
    exclusion_breakdown: Dict[str, int]

    # Split isolation
    splits_found: List[str]             # unique split values found in the records
    cross_split_violations: List[str]   # sample_ids that violated expected split

    # Warnings from the alignment pass
    warnings: List[str]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def run_validation(
    records: List[AlignmentRecord],
    warnings: Optional[List[str]] = None,
    expected_split: Optional[str] = None,
    input_count: Optional[int] = None,
) -> AlignmentReport:
    """
    Compute an ``AlignmentReport`` from a list of ``AlignmentRecord`` objects.

    Parameters
    ----------
    records:
        Output of ``align_manifest_records`` or ``run_alignment``.
    warnings:
        Warning strings collected during alignment (passed through unchanged).
    expected_split:
        The split that all records should belong to.  ``None`` means mixed.
    input_count:
        Number of rows that were loaded from the manifest (before dedup).
        When ``None``, defaults to ``len(records)``.

    Returns
    -------
    AlignmentReport
    """
    warnings = warnings or []
    dataset = records[0].dataset if records else "unknown"

    # Determine the split label
    splits_seen: set = set()
    for r in records:
        if r.split:
            splits_seen.add(r.split)
    split_label = expected_split or (list(splits_seen)[0] if len(splits_seen) == 1 else "mixed")

    # Counts
    fully_aligned = 0
    partially_aligned = 0
    excluded = 0
    has_text = 0
    has_audio = 0
    has_visual = 0
    missing_text = 0
    missing_audio = 0
    missing_visual = 0
    invalid_ts = 0
    exclusion_counter: Counter = Counter()
    cross_split_violations: List[str] = []
    duplicate_count = 0
    split_violation_count = 0

    for r in records:
        if r.has_text:
            has_text += 1
        else:
            missing_text += 1

        if r.has_audio:
            has_audio += 1
        else:
            missing_audio += 1

        if r.has_visual:
            has_visual += 1
        else:
            missing_visual += 1

        if r.alignment_status == "ok":
            fully_aligned += 1
        elif r.alignment_status == "excluded":
            excluded += 1
            if r.exclusion_reason:
                exclusion_counter[r.exclusion_reason] += 1
        else:  # "partial"
            partially_aligned += 1
            if r.exclusion_reason:
                exclusion_counter[r.exclusion_reason] += 1

        # Timestamp issues
        if r.exclusion_reason and "timestamp" in r.exclusion_reason:
            invalid_ts += 1

        # Cross-split violations
        if expected_split and r.split != expected_split:
            cross_split_violations.append(r.sample_id)
            split_violation_count += 1

    # Count warnings that mention duplicates
    for w in warnings:
        if "duplicate" in w:
            duplicate_count += 1

    return AlignmentReport(
        dataset=dataset,
        split=split_label,
        generated_at=datetime.now().isoformat(),
        input_count=input_count if input_count is not None else len(records),
        processed_count=len(records),
        duplicate_count=duplicate_count,
        split_violation_count=split_violation_count,
        has_text_count=has_text,
        has_audio_count=has_audio,
        has_visual_count=has_visual,
        missing_text_count=missing_text,
        missing_audio_count=missing_audio,
        missing_visual_count=missing_visual,
        fully_aligned_count=fully_aligned,
        partially_aligned_count=partially_aligned,
        excluded_count=excluded,
        invalid_timestamp_count=invalid_ts,
        exclusion_breakdown=dict(exclusion_counter),
        splits_found=sorted(splits_seen),
        cross_split_violations=cross_split_violations,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_report(
    report: AlignmentReport,
    records: List[AlignmentRecord],
    output_dir: Path,
) -> None:
    """
    Serialise an ``AlignmentReport`` to JSON, Markdown, and a per-sample
    exclusions CSV.

    Parameters
    ----------
    report:
        Computed report from ``run_validation``.
    records:
        Full list of ``AlignmentRecord`` objects for the exclusions CSV.
    output_dir:
        Directory where output files are written.  Created if absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON report
    json_path = output_dir / "alignment_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(report), fh, indent=2, default=str)

    # 2. Exclusions CSV — only records that were not fully aligned
    excl_path = output_dir / "alignment_exclusions.csv"
    excl_records = [r for r in records if r.alignment_status != "ok"]
    excl_fields = [
        "sample_id", "dataset", "split", "dialogue_or_video_id",
        "segment_or_utterance_id", "has_text", "has_audio", "has_visual",
        "alignment_status", "exclusion_reason",
    ]
    with open(excl_path, "w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=excl_fields, extrasaction="ignore")
        writer.writeheader()
        for r in excl_records:
            writer.writerow(dataclasses.asdict(r))

    # 3. Markdown report
    md_path = output_dir / "alignment_report.md"
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Cross-Modal Alignment Report — {report.dataset.upper()} / {report.split}\n\n")
        fh.write(f"- **Generated At**: `{report.generated_at}`\n")
        fh.write(f"- **Dataset**: `{report.dataset}`\n")
        fh.write(f"- **Split(s)**: `{report.split}`\n\n")

        fh.write("## 1. Record Counts\n\n")
        fh.write("| Metric | Count |\n")
        fh.write("| :--- | :--- |\n")
        fh.write(f"| Input records (manifest rows loaded) | {report.input_count} |\n")
        fh.write(f"| Processed records | {report.processed_count} |\n")
        fh.write(f"| Duplicate sample IDs skipped | {report.duplicate_count} |\n")
        fh.write(f"| Split isolation violations | {report.split_violation_count} |\n")
        fh.write(f"| Fully aligned (text + audio + visual) | **{report.fully_aligned_count}** |\n")
        fh.write(f"| Partially aligned (some modalities missing) | {report.partially_aligned_count} |\n")
        fh.write(f"| Excluded | {report.excluded_count} |\n\n")

        fh.write("## 2. Modality Availability\n\n")
        fh.write("| Modality | Present | Missing |\n")
        fh.write("| :--- | :--- | :--- |\n")
        fh.write(f"| **Text** | {report.has_text_count} | {report.missing_text_count} |\n")
        fh.write(f"| **Audio** | {report.has_audio_count} | {report.missing_audio_count} |\n")
        fh.write(f"| **Visual** | {report.has_visual_count} | {report.missing_visual_count} |\n\n")

        if report.invalid_timestamp_count:
            fh.write(f"> **Note**: {report.invalid_timestamp_count} record(s) had invalid or "
                     f"inverted timestamps and were excluded.\n\n")

        if report.exclusion_breakdown:
            fh.write("## 3. Exclusion Breakdown\n\n")
            fh.write("| Reason | Count |\n")
            fh.write("| :--- | :--- |\n")
            for reason, cnt in sorted(report.exclusion_breakdown.items(), key=lambda x: -x[1]):
                fh.write(f"| `{reason}` | {cnt} |\n")
            fh.write("\n")

        if report.cross_split_violations:
            fh.write("## 4. Cross-Split Violations\n\n")
            fh.write(
                "The following sample IDs were present in this manifest but "
                f"belong to a different split (expected `{report.split}`):\n\n"
            )
            for sid in report.cross_split_violations[:20]:
                fh.write(f"- `{sid}`\n")
            if len(report.cross_split_violations) > 20:
                fh.write(f"- … and {len(report.cross_split_violations) - 20} more.\n")
            fh.write("\n")

        if report.warnings:
            fh.write("## 5. Warnings\n\n")
            for w in report.warnings[:50]:
                fh.write(f"- {w}\n")
            if len(report.warnings) > 50:
                fh.write(f"- … ({len(report.warnings) - 50} more warnings)\n")
            fh.write("\n")
