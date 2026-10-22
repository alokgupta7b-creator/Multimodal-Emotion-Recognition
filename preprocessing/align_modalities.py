"""
preprocessing.align_modalities
================================
Cross-modal alignment: produces a canonical ``AlignmentRecord`` for each
sample in a dataset manifest.

Alignment rules
---------------
1. Identity-based matching only.
   Samples are matched using their stable manifest identifiers (MELD:
   ``sample_id = {split}_dia{D}_utt{U}``; CMU-MOSEI: ``sample_id =
   {video_id}_{clip_id}``).  Two records are **never** aligned merely
   because they appear on the same CSV row.

2. Timestamp validation.
   For CMU-MOSEI, ``start_time`` must be non-negative and strictly less than
   ``end_time``.  Invalid intervals are flagged and excluded from the
   fully-aligned count.  MELD does not use continuous timestamps for feature
   alignment; the StartTime/EndTime annotation fields are informational only.

3. Split isolation.
   Records from different official splits (train / dev / test) are never
   mixed.  The alignment routine processes one split at a time.

4. Duplicate detection.
   Duplicate ``sample_id`` values within a manifest are detected and recorded
   in the exclusion log; only the first occurrence is retained.

5. Missing modalities.
   A sample may have missing text, audio, or visual data.  Such samples
   receive ``is_fully_aligned = False`` and a descriptive ``exclusion_reason``.
   They are still present in the alignment output so that partial-modality
   experiments remain possible.

6. No label modification.
   Emotion and sentiment target labels are copied verbatim from the manifest.

Public API
----------
    AlignmentRecord          — dataclass holding one aligned sample
    align_manifest_records   — process a list of manifest dicts → AlignmentRecord list
    align_meld_record        — MELD-specific adapter
    align_mosei_record       — CMU-MOSEI-specific adapter
    load_manifest            — load a manifest CSV into a list of dicts
    run_alignment            — high-level pipeline entry point
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from preprocessing.config import (
    MELD_EMOTIONS,
    MELD_SENTIMENTS,
    MOSEI_EMOTIONS,
    MOSEI_SENTIMENT_MIN,
    MOSEI_SENTIMENT_MAX,
    VALID_SPLITS,
)
from preprocessing.csd_reader import CSDArchiveReader
from preprocessing.preprocess_audio import resolve_audio_modality
from preprocessing.preprocess_text import (
    get_text_field_for_dataset,
    load_transcript_source,
    normalize_text as normalize_text_fn,
    preprocess_text_record,
)
from preprocessing.preprocess_video import resolve_visual_modality


# ---------------------------------------------------------------------------
# Alignment record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AlignmentRecord:
    """
    Canonical cross-modal alignment record.

    All fields use neutral, dataset-agnostic names.  Dataset-specific raw
    values are preserved in ``target_label`` and ``target_label_type``
    rather than being discarded or remapped.
    """

    # --- Identity ---
    dataset: str                      # "meld" or "mosei"
    sample_id: str                    # unique manifest sample identifier
    split: str                        # "train", "dev", or "test"
    dialogue_or_video_id: str         # MELD: dialogue_id (int→str); MOSEI: video_id
    segment_or_utterance_id: str      # MELD: utterance_id (int→str); MOSEI: clip_id

    # --- Timestamps (seconds) ---
    start_time: Optional[float]       # None for MELD (not used for alignment)
    end_time: Optional[float]         # None for MELD

    # --- Text ---
    text_raw: str                     # verbatim utterance text from manifest
    text_normalized: str              # cleaned text (empty string if missing)

    # --- Modality references ---
    audio_ref: str                    # relative path or CSD fragment identifier
    visual_ref: str                   # relative path or CSD fragment identifier

    # --- Availability flags ---
    has_text: bool
    has_audio: bool
    has_visual: bool

    # --- Target labels (verbatim from manifest) ---
    target_label: str                 # primary label (emotion class or sentiment score)
    target_label_type: str            # "categorical_7" / "continuous_6" / etc.

    # --- Validation status ---
    is_fully_aligned: bool            # True only when text+audio+visual all present
    alignment_status: str             # "ok", "partial", "excluded"
    exclusion_reason: str             # empty string when alignment_status == "ok"


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Load a manifest CSV file into a list of row dicts.

    Parameters
    ----------
    manifest_path:
        Absolute or relative path to a manifest CSV produced by one of the
        dataset preparation pipelines.

    Returns
    -------
    list of dicts (fieldnames from CSV header as keys).

    Raises
    ------
    FileNotFoundError
        If ``manifest_path`` does not exist.
    ValueError
        If the file is empty or has no header row.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header row: {manifest_path}")
        for row in reader:
            rows.append(dict(row))

    return rows


# ---------------------------------------------------------------------------
# MELD record adapter
# ---------------------------------------------------------------------------

def align_meld_record(
    row: Dict[str, Any],
    audio_root: Optional[Path],
    video_root: Optional[Path],
    normalize_text: bool = True,
    verify_media: bool = True,
    transcript_source: Optional[Dict[Tuple[str, str], str]] = None,
) -> AlignmentRecord:
    """
    Build an ``AlignmentRecord`` from a single MELD manifest row.

    The MELD manifest schema (from ``datasets/MELD/prepare_dataset.py``)
    uses these columns:

    ``sample_id``, ``split``, ``dialogue_id``, ``utterance_id``,
    ``emotion``, ``sentiment``, ``video_path``, ``audio_path``,
    ``has_text``, ``has_video``, ``has_audio``, ``video_status``,
    ``audio_status``, ``is_fully_usable``, ``exclusion_reason``.

    Optionally (when ``include_transcript_text`` was set):
    ``text``, ``speaker``, ``season``, ``episode``, ``start_time``,
    ``end_time``.

    Parameters
    ----------
    row:
        One row dict from the MELD manifest.
    audio_root, video_root:
        Configurable roots for resolving modality paths.  ``None`` means the
        modality root was not provided.
    normalize_text:
        Apply Unicode normalization and whitespace collapsing.
    verify_media:
        Perform file-level integrity checks for audio and video.
    transcript_source:
        Optional mapping of (dialogue_id, utterance_id) -> raw_text loaded
        from an external transcript source (e.g. MELD annotations CSV).

    Returns
    -------
    AlignmentRecord
    """
    sample_id = str(row.get("sample_id", "")).strip()
    split = str(row.get("split", "")).strip().lower()
    dialogue_id = str(row.get("dialogue_id", "")).strip()
    utterance_id = str(row.get("utterance_id", "")).strip()

    # 1. Text from manifest row if present
    text_raw, text_norm, text_status = preprocess_text_record(
        row,
        text_field="text",
        normalize=normalize_text,
    )
    has_text_in_row = text_status == "ok"
    has_text_manifest_flag = str(row.get("has_text", "False")).strip().lower() == "true"
    text_unresolved = False

    if has_text_in_row:
        has_text = True
    elif transcript_source is not None:
        key = (dialogue_id, utterance_id)
        if key in transcript_source:
            raw_val = transcript_source[key]
            if raw_val and str(raw_val).strip() and str(raw_val).lower() not in ("none", "nan", "null"):
                text_raw = str(raw_val).strip()
                text_norm = normalize_text_fn(text_raw) if normalize_text else text_raw
                has_text = True
                text_status = "ok"
            else:
                has_text = False
                text_status = "empty"
        else:
            has_text = False
            text_status = "transcript_not_found_in_source"
    elif has_text_manifest_flag:
        # Upstream text was flagged as available, but omitted from manifest and no transcript source was provided
        has_text = False
        text_unresolved = True
        text_status = "unresolved_manifest_omitted"
    else:
        has_text = False
        text_status = "missing"

    # MELD does not use continuous timestamps for feature alignment
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Audio
    audio_ref, has_audio, _audio_status = resolve_audio_modality(
        dataset="meld",
        record=row,
        audio_root=audio_root,
        verify=verify_media,
    )

    # Visual
    visual_ref, has_visual, _video_status = resolve_visual_modality(
        dataset="meld",
        record=row,
        video_root=video_root,
        verify=verify_media,
    )

    # When audio_root / video_root are not provided, fall back to manifest flags
    if audio_root is None:
        has_audio_flag = str(row.get("has_audio", "False")).strip().lower()
        has_audio = has_audio_flag == "true"
    if video_root is None:
        has_video_flag = str(row.get("has_video", "False")).strip().lower()
        has_visual = has_video_flag == "true"

    # Manifest exclusion reason takes precedence
    manifest_excl = str(row.get("exclusion_reason", "")).strip()
    manifest_usable = str(row.get("is_fully_usable", "True")).strip().lower() == "true"

    # Build target label string — use emotion as primary
    target_label = str(row.get("emotion", "")).strip().lower()
    target_label_type = "categorical_7"

    # Determine alignment status
    exclusion_reason = ""
    if manifest_excl:
        exclusion_reason = manifest_excl
    elif text_unresolved:
        exclusion_reason = "text_manifest_omitted_upstream_available"
    elif not has_text:
        exclusion_reason = "missing_text"
    elif not has_audio:
        exclusion_reason = "missing_audio"
    elif not has_visual:
        exclusion_reason = "missing_visual"

    is_fully_aligned = has_text and has_audio and has_visual and not exclusion_reason
    alignment_status = "ok" if is_fully_aligned else ("excluded" if exclusion_reason else "partial")

    return AlignmentRecord(
        dataset="meld",
        sample_id=sample_id,
        split=split,
        dialogue_or_video_id=dialogue_id,
        segment_or_utterance_id=utterance_id,
        start_time=start_time,
        end_time=end_time,
        text_raw=text_raw,
        text_normalized=text_norm,
        audio_ref=audio_ref,
        visual_ref=visual_ref,
        has_text=has_text,
        has_audio=has_audio,
        has_visual=has_visual,
        target_label=target_label,
        target_label_type=target_label_type,
        is_fully_aligned=is_fully_aligned,
        alignment_status=alignment_status,
        exclusion_reason=exclusion_reason,
    )


# ---------------------------------------------------------------------------
# CMU-MOSEI record adapter
# ---------------------------------------------------------------------------

def _validate_mosei_timestamps(
    start_time: Any, end_time: Any
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse and validate MOSEI segment timestamps."""
    try:
        start = float(start_time)
        end = float(end_time)
    except (ValueError, TypeError):
        return None, None, "invalid_timestamp_format"

    if start < 0.0:
        return start, end, "invalid_timestamp_negative_start"
    if end <= start:
        return start, end, "invalid_timestamps_inverted"

    return start, end, None


def align_mosei_record(
    row: Dict[str, Any],
    audio_root: Optional[Path] = None,
    video_root: Optional[Path] = None,
    covarep_csd: Optional[Any] = None,
    facet_csd: Optional[Any] = None,
    normalize_text: bool = True,
    verify_media: bool = True,
) -> AlignmentRecord:
    """
    Build an ``AlignmentRecord`` from a single CMU-MOSEI manifest row.

    CMU-MOSEI manifest schema (from ``datasets/CMU-MOSEI/prepare_dataset.py``):

    ``sample_id``, ``video_id``, ``clip_id``, ``start_time``, ``end_time``,
    ``sentiment``, ``sentiment_discrete``, ``happy``, ``sad``, ``anger``,
    ``surprise``, ``disgust``, ``fear``, ``dominant_emotion``, ``transcript``,
    ``split``, ``text_path``, ``audio_path``, ``visual_path``, ``has_text``,
    ``has_audio``, ``has_visual``, ``is_fully_usable``, ``exclusion_reason``,
    ``audio_status``, ``video_status``.

    Parameters
    ----------
    row:
        One row dict from the CMU-MOSEI manifest.
    audio_root, video_root:
        Filesystem roots for NPY feature arrays (when CSD is unavailable).
    covarep_csd, facet_csd:
        Path or open h5py.File for the respective CSD archives.
    normalize_text:
        Apply Unicode normalisation.
    verify_media:
        Verify CSD/file availability.

    Returns
    -------
    AlignmentRecord
    """
    sample_id = str(row.get("sample_id", "")).strip()
    video_id = str(row.get("video_id", "")).strip()
    clip_id = str(row.get("clip_id", "")).strip()
    split = str(row.get("split", "")).strip().lower()

    # Timestamps
    start_time, end_time, ts_error = _validate_mosei_timestamps(
        row.get("start_time", 0.0), row.get("end_time", 0.0)
    )

    # Text
    text_raw, text_norm, text_status = preprocess_text_record(
        row,
        text_field="transcript",
        normalize=normalize_text,
    )
    has_text = text_status == "ok"

    # Audio
    audio_ref, has_audio, _audio_status = resolve_audio_modality(
        dataset="mosei",
        record=row,
        audio_root=audio_root,
        covarep_csd=covarep_csd,
        verify=verify_media,
    )

    # Visual
    visual_ref, has_visual, _video_status = resolve_visual_modality(
        dataset="mosei",
        record=row,
        video_root=video_root,
        facet_csd=facet_csd,
        verify=verify_media,
    )

    # Fallback to manifest flags when roots/CSDs unavailable
    if audio_root is None and covarep_csd is None:
        has_audio_flag = str(row.get("has_audio", "False")).strip().lower()
        has_audio = has_audio_flag == "true"
        audio_ref = str(row.get("audio_path", "")).strip()

    if video_root is None and facet_csd is None:
        has_visual_flag = str(row.get("has_visual", "False")).strip().lower()
        has_visual = has_visual_flag == "true"
        visual_ref = str(row.get("visual_path", "")).strip()

    # Build composite target label:
    # Use continuous sentiment score as primary, and emotion intensities as JSON-like string
    sentiment_score = str(row.get("sentiment", "")).strip()
    dominant_emo = str(row.get("dominant_emotion", "")).strip()
    target_label = dominant_emo or sentiment_score
    target_label_type = "continuous_mosei_6dim_plus_sentiment"

    # Exclusion reasons
    manifest_excl = str(row.get("exclusion_reason", "")).strip()
    manifest_usable = str(row.get("is_fully_usable", "True")).strip().lower() == "true"

    exclusion_reason = ""
    if ts_error:
        exclusion_reason = ts_error
    elif manifest_excl:
        exclusion_reason = manifest_excl
    elif not has_text:
        exclusion_reason = "missing_text"
    elif not has_audio:
        exclusion_reason = "missing_audio"
    elif not has_visual:
        exclusion_reason = "missing_visual"

    is_fully_aligned = (
        has_text and has_audio and has_visual
        and not exclusion_reason
        and ts_error is None
    )
    alignment_status = "ok" if is_fully_aligned else ("excluded" if exclusion_reason else "partial")

    return AlignmentRecord(
        dataset="mosei",
        sample_id=sample_id,
        split=split,
        dialogue_or_video_id=video_id,
        segment_or_utterance_id=clip_id,
        start_time=start_time,
        end_time=end_time,
        text_raw=text_raw,
        text_normalized=text_norm,
        audio_ref=audio_ref,
        visual_ref=visual_ref,
        has_text=has_text,
        has_audio=has_audio,
        has_visual=has_visual,
        target_label=target_label,
        target_label_type=target_label_type,
        is_fully_aligned=is_fully_aligned,
        alignment_status=alignment_status,
        exclusion_reason=exclusion_reason,
    )


# ---------------------------------------------------------------------------
# Core alignment pipeline
# ---------------------------------------------------------------------------

def align_manifest_records(
    rows: List[Dict[str, Any]],
    dataset: str,
    audio_root: Optional[Path] = None,
    video_root: Optional[Path] = None,
    covarep_csd: Optional[Any] = None,
    facet_csd: Optional[Any] = None,
    normalize_text: bool = True,
    verify_media: bool = True,
    expected_split: Optional[str] = None,
    transcript_source: Optional[Union[Path, str, Dict[Tuple[str, str], str]]] = None,
) -> Tuple[List[AlignmentRecord], List[str]]:
    """
    Align a list of manifest row dicts and return ``AlignmentRecord`` objects.

    This is the main workhorse of the module.  It:
    - Detects and deduplicates sample IDs (retaining first occurrence).
    - Validates each record's split against ``expected_split`` if provided.
    - Manages CSD archive reader lifecycles (opening once per run, deterministically closing).
    - Joins external transcript sources when provided (MELD).
    - Dispatches to the correct dataset adapter per record.
    - Collects warnings for malformed records without raising.

    Parameters
    ----------
    rows:
        List of manifest row dicts from ``load_manifest``.
    dataset:
        ``"meld"`` or ``"mosei"``.
    audio_root, video_root, covarep_csd, facet_csd:
        Modality resource paths forwarded to per-dataset adapters.
    normalize_text:
        Apply text normalisation.
    verify_media:
        Perform file-level integrity checks.
    expected_split:
        When set, rows whose ``split`` field does not match this value are
        flagged as split-isolation violations and excluded.
    transcript_source:
        Path to a transcript CSV/directory or pre-loaded mapping.

    Returns
    -------
    (alignment_records, warnings)
        alignment_records — list of ``AlignmentRecord`` in input order.
        warnings          — list of human-readable warning strings.
    """
    records: List[AlignmentRecord] = []
    warnings: List[str] = []
    seen_ids: set = set()

    dataset = dataset.lower().strip()

    # Preload transcript source for MELD if provided
    meld_transcripts: Optional[Dict[Tuple[str, str], str]] = None
    if dataset == "meld" and transcript_source is not None:
        if isinstance(transcript_source, (str, Path)):
            try:
                meld_transcripts = load_transcript_source(
                    transcript_source, expected_split=expected_split
                )
            except Exception as exc:
                warnings.append(f"Failed to load transcript source {transcript_source}: {exc}")
        elif isinstance(transcript_source, dict):
            meld_transcripts = transcript_source

    # Manage CSDArchiveReader lifecycle for MOSEI
    covarep_reader: Optional[CSDArchiveReader] = None
    owns_covarep = False
    facet_reader: Optional[CSDArchiveReader] = None
    owns_facet = False

    if dataset == "mosei" and verify_media:
        if covarep_csd is not None:
            if isinstance(covarep_csd, CSDArchiveReader):
                covarep_reader = covarep_csd
            else:
                covarep_reader = CSDArchiveReader(
                    covarep_csd, modality="audio", expected_feature_dim=74
                )
                owns_covarep = True

        if facet_csd is not None:
            if isinstance(facet_csd, CSDArchiveReader):
                facet_reader = facet_csd
            else:
                facet_reader = CSDArchiveReader(
                    facet_csd, modality="visual", expected_feature_dim=35
                )
                owns_facet = True

    try:
        if covarep_reader and not covarep_reader.is_open:
            covarep_reader.open()
        if facet_reader and not facet_reader.is_open:
            facet_reader.open()

        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id", "")).strip()

            # --- Duplicate detection ---
            if sample_id in seen_ids:
                warnings.append(
                    f"Row {idx}: duplicate sample_id {sample_id!r} skipped"
                )
                continue
            if sample_id:
                seen_ids.add(sample_id)

            # --- Split isolation check ---
            row_split = str(row.get("split", "")).strip().lower()
            if expected_split is not None and row_split != expected_split:
                warnings.append(
                    f"Row {idx} ({sample_id!r}): split mismatch — "
                    f"expected {expected_split!r}, got {row_split!r}"
                )
                # Still process but record the violation
                row = dict(row)
                row["exclusion_reason"] = (
                    f"split_isolation_violation_{row_split}_in_{expected_split}_manifest"
                )

            # --- Dispatch to adapter ---
            try:
                if dataset == "meld":
                    rec = align_meld_record(
                        row=row,
                        audio_root=audio_root,
                        video_root=video_root,
                        normalize_text=normalize_text,
                        verify_media=verify_media,
                        transcript_source=meld_transcripts,
                    )
                elif dataset == "mosei":
                    rec = align_mosei_record(
                        row=row,
                        audio_root=audio_root,
                        video_root=video_root,
                        covarep_csd=covarep_reader if covarep_reader is not None else covarep_csd,
                        facet_csd=facet_reader if facet_reader is not None else facet_csd,
                        normalize_text=normalize_text,
                        verify_media=verify_media,
                    )
                else:
                    raise ValueError(f"Unknown dataset: {dataset!r}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"Row {idx} ({sample_id!r}): unhandled error during alignment: {exc}"
                )
                continue

            records.append(rec)

    finally:
        if owns_covarep and covarep_reader is not None:
            covarep_reader.close()
        if owns_facet and facet_reader is not None:
            facet_reader.close()

    return records, warnings


def run_alignment(
    manifest_path: Path,
    dataset: str,
    audio_root: Optional[Path] = None,
    video_root: Optional[Path] = None,
    covarep_csd: Optional[Any] = None,
    facet_csd: Optional[Any] = None,
    normalize_text: bool = True,
    verify_media: bool = True,
    expected_split: Optional[str] = None,
    transcript_source: Optional[Union[Path, str, Dict[Tuple[str, str], str]]] = None,
) -> Tuple[List[AlignmentRecord], List[str]]:
    """
    Load a manifest CSV and produce aligned records.

    This is the top-level convenience function that combines
    ``load_manifest`` and ``align_manifest_records``.

    Parameters
    ----------
    manifest_path:
        Path to a manifest CSV.
    dataset, audio_root, video_root, covarep_csd, facet_csd,
    normalize_text, verify_media, expected_split, transcript_source:
        Forwarded to ``align_manifest_records``.

    Returns
    -------
    (alignment_records, warnings)
    """
    rows = load_manifest(manifest_path)
    return align_manifest_records(
        rows=rows,
        dataset=dataset,
        audio_root=audio_root,
        video_root=video_root,
        covarep_csd=covarep_csd,
        facet_csd=facet_csd,
        normalize_text=normalize_text,
        verify_media=verify_media,
        expected_split=expected_split,
        transcript_source=transcript_source,
    )
