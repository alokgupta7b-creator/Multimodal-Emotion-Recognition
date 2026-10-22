#!/usr/bin/env python3
"""
tests/test_alignment.py
========================
Unit tests for cross-modal alignment logic.

Covers:
- Valid MELD and CMU-MOSEI record alignment.
- Duplicate sample_id detection and suppression.
- Split isolation violations (cross-split records flagged).
- Timestamp boundary conditions (inverted, zero, negative).
- Missing modality handling (text, audio, visual absent).
- Alignment record determinism (identical inputs → identical output).
- Empty manifest handling.

All tests use synthetic data; no real dataset files are required.
"""

from __future__ import annotations

import csv
import io
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the repo root is on sys.path so "preprocessing" resolves as a package
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.align_modalities import (
    AlignmentRecord,
    align_manifest_records,
    align_meld_record,
    align_mosei_record,
    load_manifest,
    run_alignment,
)
from preprocessing.validate_alignment import run_validation, write_report


# ---------------------------------------------------------------------------
# Helpers: synthetic manifest rows
# ---------------------------------------------------------------------------

def _meld_row(
    sample_id="train_dia0_utt0",
    split="train",
    dialogue_id="0",
    utterance_id="0",
    emotion="neutral",
    sentiment="neutral",
    video_path="train_splits/dia0_utt0.mp4",
    audio_path="audio/train/dia0_utt0.wav",
    has_text="True",
    has_video="True",
    has_audio="True",
    video_status="ok",
    audio_status="ok",
    is_fully_usable="True",
    exclusion_reason="",
    text="Hello world",
) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "dialogue_id": dialogue_id,
        "utterance_id": utterance_id,
        "emotion": emotion,
        "sentiment": sentiment,
        "video_path": video_path,
        "audio_path": audio_path,
        "has_text": has_text,
        "has_video": has_video,
        "has_audio": has_audio,
        "video_status": video_status,
        "audio_status": audio_status,
        "is_fully_usable": is_fully_usable,
        "exclusion_reason": exclusion_reason,
        "text": text,
    }


def _mosei_row(
    sample_id="vid001_0",
    video_id="vid001",
    clip_id="0",
    start_time="0.5",
    end_time="4.0",
    sentiment="1.2",
    sentiment_discrete="positive",
    happy="1.5",
    sad="0.0",
    anger="0.0",
    surprise="0.0",
    disgust="0.0",
    fear="0.0",
    dominant_emotion="happy",
    transcript="This is a test transcript.",
    split="train",
    text_path="",
    audio_path="",
    visual_path="",
    has_text="True",
    has_audio="False",
    has_visual="False",
    is_fully_usable="False",
    exclusion_reason="",
    audio_status="no_audio_csd_provided",
    video_status="no_visual_csd_provided",
) -> dict:
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "clip_id": clip_id,
        "start_time": start_time,
        "end_time": end_time,
        "sentiment": sentiment,
        "sentiment_discrete": sentiment_discrete,
        "happy": happy,
        "sad": sad,
        "anger": anger,
        "surprise": surprise,
        "disgust": disgust,
        "fear": fear,
        "dominant_emotion": dominant_emotion,
        "transcript": transcript,
        "split": split,
        "text_path": text_path,
        "audio_path": audio_path,
        "visual_path": visual_path,
        "has_text": has_text,
        "has_audio": has_audio,
        "has_visual": has_visual,
        "is_fully_usable": is_fully_usable,
        "exclusion_reason": exclusion_reason,
        "audio_status": audio_status,
        "video_status": video_status,
    }


class TestMeldAlignment(unittest.TestCase):
    """Alignment tests for the MELD dataset adapter."""

    def _align(self, row, audio_root=None, video_root=None):
        return align_meld_record(
            row=row,
            audio_root=audio_root,
            video_root=video_root,
            normalize_text=True,
            verify_media=False,
        )

    def test_valid_record_produces_alignment_record(self):
        """A standard MELD row without exclusion should produce a record with correct fields."""
        row = _meld_row()
        rec = self._align(row)
        self.assertIsInstance(rec, AlignmentRecord)
        self.assertEqual(rec.dataset, "meld")
        self.assertEqual(rec.sample_id, "train_dia0_utt0")
        self.assertEqual(rec.split, "train")
        self.assertEqual(rec.dialogue_or_video_id, "0")
        self.assertEqual(rec.segment_or_utterance_id, "0")
        self.assertEqual(rec.target_label, "neutral")
        self.assertEqual(rec.target_label_type, "categorical_7")

    def test_text_normalized_correctly(self):
        """Unicode normalization is applied to the text field."""
        row = _meld_row(text="I\u2019m NOT happy!!!")
        rec = self._align(row)
        self.assertIn("NOT", rec.text_normalized)  # negation word preserved
        self.assertNotIn("\u2019", rec.text_normalized)  # curly apostrophe replaced

    def test_empty_text_marked_missing(self):
        """Empty utterance text results in has_text=False and an exclusion reason."""
        row = _meld_row(text="", has_text="False", is_fully_usable="False",
                        exclusion_reason="empty_transcript_text")
        rec = self._align(row)
        self.assertFalse(rec.has_text)
        self.assertFalse(rec.is_fully_aligned)
        self.assertNotEqual(rec.exclusion_reason, "")

    def test_manifest_exclusion_propagated(self):
        """Existing exclusion_reason in manifest is preserved in alignment record."""
        row = _meld_row(
            exclusion_reason="corrupted_video_moov_atom_missing",
            is_fully_usable="False",
            has_video="False",
            video_status="corrupted_moov_atom_missing",
        )
        rec = self._align(row)
        self.assertFalse(rec.is_fully_aligned)
        self.assertIn("corrupted", rec.exclusion_reason)

    def test_no_audio_root_uses_manifest_flags(self):
        """When audio_root is None, has_audio is read from manifest flags."""
        row = _meld_row(has_audio="False", is_fully_usable="False",
                        audio_status="missing_audio", exclusion_reason="missing_audio")
        rec = self._align(row, audio_root=None)
        self.assertFalse(rec.has_audio)

    def test_different_emotions_preserved(self):
        """Emotion labels are copied verbatim from the manifest."""
        for emo in ("anger", "disgust", "fear", "joy", "sadness", "surprise"):
            row = _meld_row(emotion=emo)
            rec = self._align(row)
            self.assertEqual(rec.target_label, emo)

    def test_meld_timestamps_are_none(self):
        """MELD alignment records have None timestamps (not used for segment alignment)."""
        row = _meld_row()
        rec = self._align(row)
        self.assertIsNone(rec.start_time)
        self.assertIsNone(rec.end_time)

    def test_has_text_true_without_text_field_marked_unresolved(self):
        """When text field is absent but has_text=True, mark as unresolved rather than missing."""
        row = _meld_row(has_text="True")
        del row["text"]  # Simulate real MELD manifest where text column was omitted
        rec = align_meld_record(row, audio_root=None, video_root=None, verify_media=False)
        self.assertFalse(rec.has_text)
        self.assertFalse(rec.is_fully_aligned)
        self.assertEqual(rec.exclusion_reason, "text_manifest_omitted_upstream_available")

    def test_has_text_true_with_transcript_source_joined(self):
        """When transcript source is supplied, utterance text is joined and normalized."""
        row = _meld_row(dialogue_id="4", utterance_id="2", has_text="True")
        del row["text"]
        source = {("4", "2"): "I\u2019m speaking now!"}
        rec = align_meld_record(
            row, audio_root=None, video_root=None, verify_media=False, transcript_source=source
        )
        self.assertTrue(rec.has_text)
        self.assertEqual(rec.text_raw, "I\u2019m speaking now!")
        self.assertIn("'", rec.text_normalized)
        self.assertTrue(rec.is_fully_aligned)
        self.assertEqual(rec.exclusion_reason, "")

    def test_transcript_source_missing_entry(self):
        """If transcript source does not contain the key, has_text is False."""
        row = _meld_row(dialogue_id="99", utterance_id="99", has_text="True")
        del row["text"]
        source = {("0", "0"): "Irrelevant"}
        rec = align_meld_record(
            row, audio_root=None, video_root=None, verify_media=False, transcript_source=source
        )
        self.assertFalse(rec.has_text)
        self.assertFalse(rec.is_fully_aligned)



class TestMoseiAlignment(unittest.TestCase):
    """Alignment tests for the CMU-MOSEI dataset adapter."""

    def _align(self, row, audio_root=None, video_root=None):
        return align_mosei_record(
            row=row,
            audio_root=audio_root,
            video_root=video_root,
            normalize_text=True,
            verify_media=False,
        )

    def test_valid_record_produces_alignment_record(self):
        """A standard MOSEI row produces a correct AlignmentRecord."""
        row = _mosei_row()
        rec = self._align(row)
        self.assertIsInstance(rec, AlignmentRecord)
        self.assertEqual(rec.dataset, "mosei")
        self.assertEqual(rec.sample_id, "vid001_0")
        self.assertEqual(rec.dialogue_or_video_id, "vid001")
        self.assertEqual(rec.segment_or_utterance_id, "0")
        self.assertEqual(rec.start_time, 0.5)
        self.assertEqual(rec.end_time, 4.0)

    def test_inverted_timestamps_excluded(self):
        """Segments with end_time <= start_time must be excluded."""
        row = _mosei_row(start_time="5.0", end_time="2.0")
        rec = self._align(row)
        self.assertFalse(rec.is_fully_aligned)
        self.assertIn("inverted", rec.exclusion_reason)

    def test_negative_start_time_excluded(self):
        """Segments with negative start_time must be excluded."""
        row = _mosei_row(start_time="-1.0", end_time="2.0")
        rec = self._align(row)
        self.assertFalse(rec.is_fully_aligned)
        self.assertIn("negative", rec.exclusion_reason)

    def test_zero_duration_excluded(self):
        """Segments with start_time == end_time must be excluded."""
        row = _mosei_row(start_time="3.0", end_time="3.0")
        rec = self._align(row)
        self.assertFalse(rec.is_fully_aligned)

    def test_missing_transcript_flagged(self):
        """Empty transcript results in has_text=False."""
        row = _mosei_row(transcript="", has_text="False")
        rec = self._align(row)
        self.assertFalse(rec.has_text)
        self.assertFalse(rec.is_fully_aligned)

    def test_unicode_text_normalised(self):
        """Unicode text is normalised via NFKC."""
        row = _mosei_row(transcript="\uff49 am happ\uff59")  # full-width latin
        rec = self._align(row)
        # NFKC converts full-width ASCII equivalents
        self.assertEqual(rec.text_normalized, "i am happy")

    def test_manifest_flags_used_when_no_csd(self):
        """When no CSD is provided, has_audio/has_visual are read from manifest."""
        row = _mosei_row(has_audio="True", has_visual="True",
                         audio_path="features/CMU_MOSEI_COVAREP.csd#vid001_0",
                         visual_path="features/CMU_MOSEI_VisualFacet42.csd#vid001_0")
        rec = self._align(row)
        self.assertTrue(rec.has_audio)
        self.assertTrue(rec.has_visual)


class TestAlignManifestRecords(unittest.TestCase):
    """Tests for the batch alignment pipeline."""

    def test_duplicate_sample_ids_suppressed(self):
        """Duplicate sample_ids produce only one record and a warning."""
        rows = [_meld_row(), _meld_row()]  # identical sample_id
        records, warnings = align_manifest_records(rows, dataset="meld", verify_media=False)
        self.assertEqual(len(records), 1)
        self.assertTrue(any("duplicate" in w for w in warnings))

    def test_split_isolation_violation_flagged(self):
        """Records from a different split than expected are flagged."""
        row = _meld_row(split="test")
        records, warnings = align_manifest_records(
            [row], dataset="meld", expected_split="train", verify_media=False
        )
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].is_fully_aligned)
        self.assertTrue(any("split" in w.lower() for w in warnings))

    def test_multiple_records_deterministic_order(self):
        """Identical inputs produce identical output in the same order."""
        rows = [
            _meld_row(sample_id="train_dia0_utt0"),
            _meld_row(sample_id="train_dia0_utt1", utterance_id="1"),
            _meld_row(sample_id="train_dia1_utt0", dialogue_id="1"),
        ]
        records_a, _ = align_manifest_records(rows, dataset="meld", verify_media=False)
        records_b, _ = align_manifest_records(rows, dataset="meld", verify_media=False)
        ids_a = [r.sample_id for r in records_a]
        ids_b = [r.sample_id for r in records_b]
        self.assertEqual(ids_a, ids_b)

    def test_empty_manifest_rows_returns_empty(self):
        """An empty manifest list produces no records and no warnings."""
        records, warnings = align_manifest_records([], dataset="meld", verify_media=False)
        self.assertEqual(records, [])
        self.assertEqual(warnings, [])

    def test_mosei_batch_alignment(self):
        """Batch alignment works for CMU-MOSEI rows."""
        rows = [
            _mosei_row(sample_id="vid001_0", video_id="vid001", clip_id="0"),
            _mosei_row(sample_id="vid001_1", video_id="vid001", clip_id="1",
                       start_time="4.5", end_time="8.0"),
        ]
        records, warnings = align_manifest_records(rows, dataset="mosei", verify_media=False)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(warnings), 0)


class TestLoadManifest(unittest.TestCase):
    """Tests for the manifest CSV loader."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, rows: list, filename="manifest.csv") -> Path:
        path = self.tmpdir / filename
        if not rows:
            path.write_text("sample_id,split\n", encoding="utf-8")
            return path
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return path

    def test_load_valid_csv(self):
        """Loads a well-formed CSV into a list of dicts."""
        path = self._write_csv([_meld_row()])
        rows = load_manifest(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "train_dia0_utt0")

    def test_load_missing_file_raises(self):
        """FileNotFoundError is raised for non-existent manifest."""
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.tmpdir / "does_not_exist.csv")

    def test_load_empty_header_raises(self):
        """Empty manifest raises ValueError."""
        path = self.tmpdir / "empty.csv"
        path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_manifest(path)


class TestRunAlignment(unittest.TestCase):
    """End-to-end tests for run_alignment using a temporary manifest file."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_meld_manifest(self, rows):
        path = self.tmpdir / "manifest.csv"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        return path

    def test_run_alignment_meld(self):
        """run_alignment processes a MELD manifest end-to-end."""
        rows = [
            _meld_row(sample_id="train_dia0_utt0"),
            _meld_row(sample_id="train_dia0_utt1", utterance_id="1"),
        ]
        path = self._write_meld_manifest(rows)
        records, warnings = run_alignment(
            manifest_path=path,
            dataset="meld",
            verify_media=False,
        )
        self.assertEqual(len(records), 2)
        sample_ids = {r.sample_id for r in records}
        self.assertIn("train_dia0_utt0", sample_ids)
        self.assertIn("train_dia0_utt1", sample_ids)

    def test_run_alignment_with_expected_split(self):
        """run_alignment flags cross-split violations."""
        rows = [
            _meld_row(sample_id="train_dia0_utt0", split="train"),
            _meld_row(sample_id="test_dia0_utt0", split="test"),
        ]
        path = self._write_meld_manifest(rows)
        records, warnings = run_alignment(
            manifest_path=path,
            dataset="meld",
            verify_media=False,
            expected_split="train",
        )
        self.assertEqual(len(records), 2)
        violation_records = [r for r in records if "split_isolation" in r.exclusion_reason]
        self.assertEqual(len(violation_records), 1)
