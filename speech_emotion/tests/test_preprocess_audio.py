"""
speech_emotion/tests/test_preprocess_audio.py
=============================================
Automated tests for dataset integration, manifest resolution, duplicate handling,
missing audio accounting, split isolation, and audit report generation.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speech_emotion.audio_loader import write_wav
from speech_emotion.config import AudioConfig, TOTAL_MFCC_FEATURES
from speech_emotion.preprocess_audio import (
    load_manifest_records,
    process_split,
)


class TestPreprocessAudio(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_preprocess_audio_"))
        self.audio_dir = self.temp_dir / "audio"
        self.output_dir = self.temp_dir / "outputs"
        self.manifest_dir = self.temp_dir / "manifests"

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_wav(self, dia_id: int, utt_id: int, duration: float = 0.5, freq: float = 440.0) -> Path:
        sr = 16000
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        y = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        wav_path = self.audio_dir / f"dia{dia_id}_utt{utt_id}.wav"
        return write_wav(wav_path, y, sample_rate=sr)

    def _create_manifest_csv(self, filename: str, rows: list) -> Path:
        path = self.manifest_dir / filename
        fieldnames = ["Dialogue_ID", "Utterance_ID", "Emotion", "Split", "sample_id"]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return path

    def test_load_manifest_records_standard(self) -> None:
        """Manifest records are loaded and normalized into typed dictionaries."""
        rows = [
            {"Dialogue_ID": "0", "Utterance_ID": "0", "Emotion": "sadness", "Split": "dev", "sample_id": "dev_dia0_utt0"},
            {"Dialogue_ID": "0", "Utterance_ID": "1", "Emotion": "joy", "Split": "dev", "sample_id": "dev_dia0_utt1"},
        ]
        manifest_path = self._create_manifest_csv("dev_manifest.csv", rows)

        records = load_manifest_records(manifest_path, expected_split="dev")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["dialogue_id"], 0)
        self.assertEqual(records[0]["utterance_id"], 0)
        self.assertEqual(records[0]["emotion"], "sadness")
        self.assertEqual(records[0]["split"], "dev")

    def test_process_split_valid_records(self) -> None:
        """Valid audio files produce expected 40 MFCC feature columns and audit reports."""
        self._create_wav(dia_id=1, utt_id=0, freq=300.0)
        self._create_wav(dia_id=1, utt_id=1, freq=600.0)

        rows = [
            {"Dialogue_ID": "1", "Utterance_ID": "0", "Emotion": "neutral", "Split": "train", "sample_id": "train_dia1_utt0"},
            {"Dialogue_ID": "1", "Utterance_ID": "1", "Emotion": "joy", "Split": "train", "sample_id": "train_dia1_utt1"},
        ]
        manifest_path = self._create_manifest_csv("train_manifest.csv", rows)

        config = AudioConfig(sample_rate=16000, n_mfcc=20)
        feat_rows, fail_rows, report = process_split(
            split="train",
            audio_dir=self.audio_dir,
            manifest_path=manifest_path,
            output_dir=self.output_dir,
            config=config,
        )

        self.assertEqual(len(feat_rows), 2)
        self.assertEqual(len(fail_rows), 0)
        self.assertEqual(report.valid_count, 2)
        self.assertEqual(report.invalid_count, 0)
        self.assertEqual(report.total_records, 2)
        self.assertEqual(report.feature_shape, (2, 40))

        # Check output feature CSV
        feat_csv = self.output_dir / "train_mfcc.csv"
        self.assertTrue(feat_csv.is_file())
        with open(feat_csv, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out_rows = list(reader)
            self.assertEqual(len(out_rows), 2)
            self.assertIn("Dialogue_ID", reader.fieldnames)
            self.assertIn("Utterance_ID", reader.fieldnames)
            self.assertIn("emotion", reader.fieldnames)
            self.assertIn("mfcc_mean_01", reader.fieldnames)
            self.assertIn("mfcc_std_20", reader.fieldnames)
            self.assertEqual(len(reader.fieldnames), 43)  # 3 ID/emo cols + 40 features

        # Check JSON audit report
        audit_json = self.output_dir / "train_audio_audit.json"
        self.assertTrue(audit_json.is_file())
        with open(audit_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            self.assertEqual(data["valid_count"], 2)
            self.assertEqual(data["missing_audio_count"], 0)

        # Check Markdown report
        audit_md = self.output_dir / "train_audio_audit.md"
        self.assertTrue(audit_md.is_file())
        self.assertIn("Accounting Summary", audit_md.read_text(encoding="utf-8"))

    def test_missing_audio_recorded_in_failure_csv(self) -> None:
        """Missing audio files are logged to failures CSV and audit report without crashing."""
        # Create audio only for (dia2, utt0), leave (dia2, utt1) missing
        self._create_wav(dia_id=2, utt_id=0)

        rows = [
            {"Dialogue_ID": "2", "Utterance_ID": "0", "Emotion": "anger", "Split": "dev", "sample_id": "dev_dia2_utt0"},
            {"Dialogue_ID": "2", "Utterance_ID": "1", "Emotion": "fear", "Split": "dev", "sample_id": "dev_dia2_utt1"},
        ]
        manifest_path = self._create_manifest_csv("dev_manifest.csv", rows)

        feat_rows, fail_rows, report = process_split(
            split="dev",
            audio_dir=self.audio_dir,
            manifest_path=manifest_path,
            output_dir=self.output_dir,
        )

        self.assertEqual(len(feat_rows), 1)
        self.assertEqual(len(fail_rows), 1)
        self.assertEqual(report.valid_count, 1)
        self.assertEqual(report.missing_audio_count, 1)

        fail_entry = fail_rows[0]
        self.assertEqual(fail_entry["Dialogue_ID"], 2)
        self.assertEqual(fail_entry["Utterance_ID"], 1)
        self.assertEqual(fail_entry["reason"], "missing_audio")

    def test_duplicate_sample_id_detection(self) -> None:
        """Duplicate sample IDs in manifest are skipped and logged as failures."""
        self._create_wav(dia_id=3, utt_id=0)

        rows = [
            {"Dialogue_ID": "3", "Utterance_ID": "0", "Emotion": "joy", "Split": "test", "sample_id": "test_dia3_utt0"},
            {"Dialogue_ID": "3", "Utterance_ID": "0", "Emotion": "joy", "Split": "test", "sample_id": "test_dia3_utt0"},  # Duplicate
        ]
        manifest_path = self._create_manifest_csv("test_manifest.csv", rows)

        feat_rows, fail_rows, report = process_split(
            split="test",
            audio_dir=self.audio_dir,
            manifest_path=manifest_path,
            output_dir=self.output_dir,
        )

        self.assertEqual(len(feat_rows), 1)
        self.assertEqual(len(fail_rows), 1)
        self.assertEqual(report.duplicate_count, 1)
        self.assertEqual(fail_rows[0]["reason"], "duplicate_sample_id")

    def test_unexpected_emotion_label(self) -> None:
        """Records with non-standard emotion labels are excluded."""
        self._create_wav(dia_id=4, utt_id=0)

        rows = [
            {"Dialogue_ID": "4", "Utterance_ID": "0", "Emotion": "unknown_mood", "Split": "train", "sample_id": "train_dia4_utt0"},
        ]
        manifest_path = self._create_manifest_csv("train_bad_emo.csv", rows)

        feat_rows, fail_rows, report = process_split(
            split="train",
            audio_dir=self.audio_dir,
            manifest_path=manifest_path,
            output_dir=self.output_dir,
        )

        self.assertEqual(len(feat_rows), 0)
        self.assertEqual(len(fail_rows), 1)
        self.assertIn("unexpected_emotion", fail_rows[0]["reason"])

    def test_split_isolation_ignores_different_split(self) -> None:
        """Rows from unexpected splits are filtered during manifest loading."""
        rows = [
            {"Dialogue_ID": "5", "Utterance_ID": "0", "Emotion": "joy", "Split": "train", "sample_id": "train_dia5_utt0"},
            {"Dialogue_ID": "5", "Utterance_ID": "1", "Emotion": "joy", "Split": "test", "sample_id": "test_dia5_utt1"},
        ]
        manifest_path = self._create_manifest_csv("mixed_manifest.csv", rows)

        # Loading with expected_split="train" should retain only split="train"
        records = load_manifest_records(manifest_path, expected_split="train")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "train_dia5_utt0")


if __name__ == "__main__":
    unittest.main()
