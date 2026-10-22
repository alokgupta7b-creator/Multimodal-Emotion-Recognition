#!/usr/bin/env python3
"""
tests/test_preprocessing.py
=============================
Unit tests for individual preprocessing sub-modules:
  - preprocess_text  : normalize_text, preprocess_text_record
  - preprocess_audio : check_wav_header, validate_audio_array, resolve_audio_modality
  - preprocess_video : check_mp4_container, validate_visual_array, resolve_visual_modality
  - config           : AlignmentConfig validation
  - validate_alignment: run_validation, write_report

All tests are synthetic.  No real dataset downloads or large files are required.
numpy is used only in array validation tests and is available in this environment.
"""

from __future__ import annotations

import csv
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.preprocess_text import normalize_text, preprocess_text_record
from preprocessing.preprocess_audio import (
    AudioStatus,
    check_wav_header,
    resolve_audio_modality,
    validate_audio_array,
)
from preprocessing.preprocess_video import (
    VisualStatus,
    check_mp4_container,
    resolve_visual_modality,
    validate_visual_array,
)
from preprocessing.config import AlignmentConfig
from preprocessing.align_modalities import align_manifest_records
from preprocessing.validate_alignment import run_validation, write_report


# ---------------------------------------------------------------------------
# Text preprocessing tests
# ---------------------------------------------------------------------------

class TestNormalizeText(unittest.TestCase):

    def test_unicode_nfkc(self):
        """Full-width characters are converted to ASCII equivalents."""
        result = normalize_text("\uff48\uff45\uff4c\uff4c\uff4f")  # ｈｅｌｌｏ
        self.assertEqual(result, "hello")

    def test_typographic_quotes_removed(self):
        """Curly quotes are replaced with ASCII apostrophes."""
        result = normalize_text("I\u2019m fine")
        self.assertNotIn("\u2019", result)
        self.assertIn("'", result)

    def test_urls_removed(self):
        """HTTP URLs are stripped."""
        result = normalize_text("Visit https://example.com today!")
        self.assertNotIn("https://", result)
        self.assertNotIn("example.com", result)

    def test_email_removed(self):
        """E-mail addresses are stripped."""
        result = normalize_text("Contact us at admin@example.com")
        self.assertNotIn("@", result)

    def test_negation_preserved(self):
        """'not', 'no', 'never' are not removed."""
        result = normalize_text("I am not happy and never will be")
        self.assertIn("not", result)
        self.assertIn("never", result)

    def test_emotion_punctuation_preserved(self):
        """Exclamation and question marks are preserved by default."""
        result = normalize_text("Are you serious?!")
        self.assertIn("?", result)
        self.assertIn("!", result)

    def test_whitespace_collapsed(self):
        """Multiple spaces and tabs are collapsed to a single space."""
        result = normalize_text("hello   world\t\t!")
        self.assertEqual(result, "hello world !")

    def test_empty_string(self):
        """Empty string returns empty string."""
        result = normalize_text("")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        """Whitespace-only string returns empty string."""
        result = normalize_text("   \t  ")
        self.assertEqual(result, "")

    def test_lowercase_flag(self):
        """Lowercase flag converts to lowercase."""
        result = normalize_text("HELLO WORLD", lowercase=True)
        self.assertEqual(result, "hello world")

    def test_no_lowercase_by_default(self):
        """Case is preserved by default."""
        result = normalize_text("HELLO World")
        self.assertIn("HELLO", result)

    def test_non_string_raises_type_error(self):
        """Non-string input raises TypeError."""
        with self.assertRaises(TypeError):
            normalize_text(None)  # type: ignore
        with self.assertRaises(TypeError):
            normalize_text(42)  # type: ignore

    def test_preserve_punctuation_false_removes_punctuation(self):
        """preserve_punctuation=False removes non-alphanumeric chars."""
        result = normalize_text("Hello! World?", preserve_punctuation=False)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)

    def test_control_characters_removed(self):
        """Control characters (non-printable) are stripped."""
        result = normalize_text("hello\x00\x01world")
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x01", result)


class TestPreprocessTextRecord(unittest.TestCase):

    def test_valid_text_field(self):
        """Standard record with text field returns (raw, processed, 'ok')."""
        record = {"text": "I am happy today!"}
        raw, proc, status = preprocess_text_record(record, text_field="text")
        self.assertEqual(status, "ok")
        self.assertTrue(len(proc) > 0)
        self.assertEqual(raw, "I am happy today!")

    def test_missing_field_returns_null(self):
        """When the text field is absent, status is 'null'."""
        record = {}
        raw, proc, status = preprocess_text_record(record, text_field="text")
        self.assertEqual(status, "null")
        self.assertEqual(raw, "")
        self.assertEqual(proc, "")

    def test_none_value_returns_null(self):
        """None value returns null status."""
        record = {"text": None}
        raw, proc, status = preprocess_text_record(record, text_field="text")
        self.assertEqual(status, "null")

    def test_empty_string_returns_empty(self):
        """Empty string value returns 'empty' status."""
        record = {"text": ""}
        raw, proc, status = preprocess_text_record(record, text_field="text")
        self.assertEqual(status, "empty")

    def test_nan_string_returns_empty(self):
        """The string 'nan' returns 'empty' status."""
        record = {"text": "nan"}
        raw, proc, status = preprocess_text_record(record, text_field="text")
        self.assertEqual(status, "empty")

    def test_transcript_field_mosei(self):
        """CMU-MOSEI uses 'transcript' as the text field key."""
        record = {"transcript": "The model predicts emotion intensities."}
        raw, proc, status = preprocess_text_record(record, text_field="transcript")
        self.assertEqual(status, "ok")
        self.assertIn("model", proc)

    def test_normalize_false_returns_raw(self):
        """When normalize=False, processed text equals stripped raw text."""
        record = {"text": "I\u2019m NOT happy!!!"}
        raw, proc, status = preprocess_text_record(record, text_field="text", normalize=False)
        self.assertEqual(proc, raw)


# ---------------------------------------------------------------------------
# Audio preprocessing tests
# ---------------------------------------------------------------------------

class TestCheckWavHeader(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_wav(self, path: Path) -> None:
        """Write a minimal valid RIFF/WAVE file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"RIFF")
            fh.write(struct.pack("<I", 36 + 64))
            fh.write(b"WAVE")
            fh.write(b"fmt ")
            fh.write(struct.pack("<I", 16))
            fh.write(struct.pack("<H", 1))   # PCM
            fh.write(struct.pack("<H", 1))   # mono
            fh.write(struct.pack("<I", 16000))
            fh.write(struct.pack("<I", 32000))
            fh.write(struct.pack("<H", 2))
            fh.write(struct.pack("<H", 16))
            fh.write(b"data")
            fh.write(struct.pack("<I", 64))
            fh.write(b"\x00" * 64)

    def test_valid_wav_passes(self):
        path = self.tmpdir / "test.wav"
        self._make_wav(path)
        ok, status = check_wav_header(path)
        self.assertTrue(ok)
        self.assertEqual(status, AudioStatus.OK_WAV)

    def test_missing_file(self):
        ok, status = check_wav_header(self.tmpdir / "missing.wav")
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.MISSING_FILE)

    def test_too_small(self):
        path = self.tmpdir / "tiny.wav"
        path.write_bytes(b"\x00" * 10)
        ok, status = check_wav_header(path)
        self.assertFalse(ok)

    def test_wrong_riff_marker(self):
        path = self.tmpdir / "bad.wav"
        path.write_bytes(b"XXXX" + b"\x00" * 40)
        ok, status = check_wav_header(path)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.WAV_HEADER_INVALID)

    def test_string_path_accepted(self):
        """check_wav_header accepts a string path (auto-converted to Path)."""
        path = self.tmpdir / "str_path.wav"
        self._make_wav(path)
        ok, _ = check_wav_header(str(path))
        self.assertTrue(ok)


class TestValidateAudioArray(unittest.TestCase):

    def test_valid_2d_array(self):
        import numpy as np
        arr = np.zeros((50, 74))
        ok, status = validate_audio_array(arr)
        self.assertTrue(ok)
        self.assertEqual(status, AudioStatus.OK_NPY_FEATURE)

    def test_empty_array(self):
        import numpy as np
        arr = np.zeros((0, 74))
        ok, status = validate_audio_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.ARRAY_EMPTY_SHAPE)

    def test_1d_array_rejected(self):
        import numpy as np
        arr = np.zeros(74)
        ok, status = validate_audio_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.ARRAY_WRONG_DIM)

    def test_nan_values_rejected(self):
        import numpy as np
        arr = np.full((10, 74), float("nan"))
        ok, status = validate_audio_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.ARRAY_NAN_INF)

    def test_inf_values_rejected(self):
        import numpy as np
        arr = np.full((10, 74), float("inf"))
        ok, status = validate_audio_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.ARRAY_NAN_INF)

    def test_wrong_feature_dim(self):
        import numpy as np
        arr = np.zeros((10, 40))
        ok, status = validate_audio_array(arr, expected_feature_dim=74)
        self.assertFalse(ok)
        self.assertEqual(status, AudioStatus.ARRAY_WRONG_DIM)

    def test_correct_feature_dim(self):
        import numpy as np
        arr = np.zeros((10, 74))
        ok, status = validate_audio_array(arr, expected_feature_dim=74)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Video preprocessing tests
# ---------------------------------------------------------------------------

class TestCheckMp4Container(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mp4(self, path: Path) -> None:
        """Write a minimal valid MP4 container (ftyp + moov atoms).

        ftyp atom layout:
          4 bytes size (=32)
          4 bytes type 'ftyp'
          4 bytes major brand 'mp42'
          4 bytes minor version \x00*4
          4 bytes compat brand 'isom'
          4 bytes compat brand 'mp42'
          8 bytes padding = 32 total bytes
        moov atom:
          4 bytes size (=8)
          4 bytes type 'moov'
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            # ftyp: 8-byte header + 24-byte payload = 32 bytes total
            ftyp_payload = b"mp42" + b"\x00\x00\x00\x00" + b"isom" + b"mp42" + b"\x00" * 8
            fh.write(struct.pack(">I", 8 + len(ftyp_payload)) + b"ftyp" + ftyp_payload)
            # moov: minimal 8-byte atom
            fh.write(struct.pack(">I", 8) + b"moov")

    def test_valid_mp4_passes(self):
        path = self.tmpdir / "test.mp4"
        self._make_mp4(path)
        ok, status = check_mp4_container(path)
        self.assertTrue(ok)
        self.assertEqual(status, VisualStatus.OK_MP4)

    def test_missing_file(self):
        ok, status = check_mp4_container(self.tmpdir / "missing.mp4")
        self.assertFalse(ok)
        self.assertEqual(status, VisualStatus.MISSING_FILE)

    def test_too_small(self):
        path = self.tmpdir / "small.mp4"
        path.write_bytes(b"\x00" * 10)
        ok, status = check_mp4_container(path)
        self.assertFalse(ok)

    def test_no_moov_atom(self):
        path = self.tmpdir / "nomoov.mp4"
        with open(path, "wb") as fh:
            # ftyp only, no moov
            fh.write(struct.pack(">I", 32) + b"ftyp" + b"mp42" + b"\x00" * 20)
            fh.write(struct.pack(">I", 8) + b"free")
        ok, status = check_mp4_container(path)
        self.assertFalse(ok)
        self.assertEqual(status, VisualStatus.CORRUPTED_NO_MOOV)


class TestValidateVisualArray(unittest.TestCase):

    def test_valid_array(self):
        import numpy as np
        arr = np.zeros((30, 35))
        ok, status = validate_visual_array(arr, expected_feature_dim=35)
        self.assertTrue(ok)

    def test_empty_array(self):
        import numpy as np
        arr = np.zeros((0, 35))
        ok, status = validate_visual_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, VisualStatus.ARRAY_EMPTY_SHAPE)

    def test_nan_rejected(self):
        import numpy as np
        arr = np.full((10, 35), float("nan"))
        ok, status = validate_visual_array(arr)
        self.assertFalse(ok)
        self.assertEqual(status, VisualStatus.ARRAY_NAN_INF)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAlignmentConfig(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.manifest = self.tmpdir / "manifest.csv"
        self.manifest.write_text("sample_id,split\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_meld_config(self):
        cfg = AlignmentConfig(
            dataset="meld",
            manifest_path=self.manifest,
            output_dir=self.tmpdir / "out",
        )
        self.assertEqual(cfg.dataset, "meld")
        self.assertIsInstance(cfg.manifest_path, Path)

    def test_valid_mosei_config(self):
        cfg = AlignmentConfig(
            dataset="mosei",
            manifest_path=self.manifest,
            output_dir=self.tmpdir / "out",
        )
        self.assertEqual(cfg.dataset, "mosei")

    def test_invalid_dataset_raises(self):
        with self.assertRaises(ValueError):
            AlignmentConfig(
                dataset="iemocap",
                manifest_path=self.manifest,
                output_dir=self.tmpdir / "out",
            )

    def test_validate_missing_manifest_raises(self):
        cfg = AlignmentConfig(
            dataset="meld",
            manifest_path=self.tmpdir / "does_not_exist.csv",
            output_dir=self.tmpdir / "out",
        )
        with self.assertRaises(FileNotFoundError):
            cfg.validate()

    def test_paths_coerced_to_path_objects(self):
        """String paths are automatically converted to pathlib.Path."""
        cfg = AlignmentConfig(
            dataset="meld",
            manifest_path=str(self.manifest),
            output_dir=str(self.tmpdir / "out"),
            audio_root=str(self.tmpdir),
        )
        self.assertIsInstance(cfg.manifest_path, Path)
        self.assertIsInstance(cfg.output_dir, Path)
        self.assertIsInstance(cfg.audio_root, Path)


# ---------------------------------------------------------------------------
# Validation report tests
# ---------------------------------------------------------------------------

class TestRunValidation(unittest.TestCase):

    def _make_records(self):
        """Create a small batch of synthetic AlignmentRecords via the adapter."""
        from preprocessing.align_modalities import align_meld_record
        rows = [
            {
                "sample_id": "train_dia0_utt0", "split": "train",
                "dialogue_id": "0", "utterance_id": "0",
                "emotion": "neutral", "sentiment": "neutral",
                "video_path": "a.mp4", "audio_path": "a.wav",
                "has_text": "True", "has_video": "True", "has_audio": "True",
                "video_status": "ok", "audio_status": "ok",
                "is_fully_usable": "True", "exclusion_reason": "",
                "text": "Hello",
            },
            {
                "sample_id": "train_dia0_utt1", "split": "train",
                "dialogue_id": "0", "utterance_id": "1",
                "emotion": "joy", "sentiment": "positive",
                "video_path": "", "audio_path": "b.wav",
                "has_text": "True", "has_video": "False", "has_audio": "True",
                "video_status": "missing_file", "audio_status": "ok",
                "is_fully_usable": "False",
                "exclusion_reason": "missing_video_recording",
                "text": "Great",
            },
        ]
        return [
            align_meld_record(r, audio_root=None, video_root=None, verify_media=False)
            for r in rows
        ]

    def test_report_counts_correct(self):
        records = self._make_records()
        report = run_validation(records, expected_split="train")
        self.assertEqual(report.processed_count, 2)
        # First record: all flags True but no roots → is_fully_aligned may vary
        # At minimum, one record has exclusion_reason set
        self.assertGreaterEqual(report.excluded_count + report.partially_aligned_count, 1)

    def test_report_modality_counts(self):
        records = self._make_records()
        report = run_validation(records, expected_split="train")
        self.assertEqual(report.has_text_count, 2)

    def test_write_report_creates_files(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            records = self._make_records()
            report = run_validation(records, expected_split="train")
            write_report(report, records, output_dir=tmpdir)
            self.assertTrue((tmpdir / "alignment_report.json").exists())
            self.assertTrue((tmpdir / "alignment_report.md").exists())
            self.assertTrue((tmpdir / "alignment_exclusions.csv").exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_report_split_isolation_violations(self):
        """Cross-split records are counted in split_violation_count."""
        from preprocessing.align_modalities import align_meld_record
        row_train = {
            "sample_id": "train_dia0_utt0", "split": "train",
            "dialogue_id": "0", "utterance_id": "0",
            "emotion": "neutral", "sentiment": "neutral",
            "video_path": "", "audio_path": "",
            "has_text": "True", "has_video": "False", "has_audio": "False",
            "video_status": "ok", "audio_status": "ok",
            "is_fully_usable": "False",
            "exclusion_reason": "split_isolation_violation_test_in_train_manifest",
            "text": "Hi",
        }
        rec = align_meld_record(row_train, audio_root=None, video_root=None, verify_media=False)
        # Manually set split to test to simulate a violation
        rec2 = rec.__class__(
            **{**rec.__dict__, "split": "test"}
        )
        report = run_validation([rec, rec2], expected_split="train")
        self.assertEqual(report.split_violation_count, 1)


# ---------------------------------------------------------------------------
# Transcript source loader tests
# ---------------------------------------------------------------------------

class TestLoadTranscriptSource(unittest.TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_single_csv(self):
        """Standard MELD annotation CSV is parsed into (dia, utt) -> text."""
        csv_file = self.tmpdir / "annotations.csv"
        csv_file.write_text(
            "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode\n"
            "1,Hello there,Chandler,neutral,neutral,0,0,8,21\n"
            "2,How are you doing?,Joey,joy,positive,0,1,8,21\n",
            encoding="utf-8",
        )
        from preprocessing.preprocess_text import load_transcript_source

        mapping = load_transcript_source(csv_file)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[("0", "0")], "Hello there")
        self.assertEqual(mapping[("0", "1")], "How are you doing?")

    def test_load_directory_with_split(self):
        """Directory resolves {split}_sent_emo.csv correctly."""
        train_file = self.tmpdir / "train_sent_emo.csv"
        train_file.write_text(
            "Dialogue_ID,Utterance_ID,Utterance\n"
            "5,1,Train utterance text\n",
            encoding="utf-8",
        )
        dev_file = self.tmpdir / "dev_sent_emo.csv"
        dev_file.write_text(
            "Dialogue_ID,Utterance_ID,Utterance\n"
            "5,1,Dev utterance text\n",
            encoding="utf-8",
        )
        from preprocessing.preprocess_text import load_transcript_source

        train_mapping = load_transcript_source(self.tmpdir, expected_split="train")
        self.assertEqual(train_mapping[("5", "1")], "Train utterance text")

        dev_mapping = load_transcript_source(self.tmpdir, expected_split="dev")
        self.assertEqual(dev_mapping[("5", "1")], "Dev utterance text")

    def test_missing_source_raises(self):
        from preprocessing.preprocess_text import load_transcript_source

        with self.assertRaises(FileNotFoundError):
            load_transcript_source(self.tmpdir / "nonexistent.csv")


if __name__ == "__main__":
    unittest.main()
