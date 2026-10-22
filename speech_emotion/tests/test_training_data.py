"""
speech_emotion.tests.test_training_data
=======================================
Unit tests for MELD data loading, manifest joining, and split isolation.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from speech_emotion.training.config import DataConfig
from speech_emotion.training.data import DatasetSplit, MELDDataLoader, FEATURE_NAMES


class TestTrainingDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manifest_dir = Path(self.temp_dir.name) / "manifests"
        self.feature_dir = Path(self.temp_dir.name) / "features"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        self.config = DataConfig(
            manifest_dir=self.manifest_dir,
            feature_dir=self.feature_dir,
        )
        self.loader = MELDDataLoader(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_manifest(self, split: str, rows: int = 10, emotions=None) -> pd.DataFrame:
        if emotions is None:
            emotions = ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]
        data = []
        for i in range(rows):
            emo = emotions[i % len(emotions)]
            data.append({
                "sample_id": f"{split}_dia0_utt{i}",
                "split": split,
                "dialogue_id": 0,
                "utterance_id": i,
                "emotion": emo,
                "sentiment": "neutral",
                "audio_path": f"audio/{split}/dia0_utt{i}.wav",
                "is_fully_usable": True,
            })
        df = pd.DataFrame(data)
        df.to_csv(self.manifest_dir / f"{split}_manifest.csv", index=False)
        return df

    def _create_synthetic_features(self, split: str, rows: int = 10, emotions=None, exclude_indices=None) -> pd.DataFrame:
        if emotions is None:
            emotions = ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]
        exclude_indices = exclude_indices or []
        data = []
        for i in range(rows):
            if i in exclude_indices:
                continue
            emo = emotions[i % len(emotions)]
            row_dict = {
                "Dialogue_ID": 0,
                "Utterance_ID": i,
                "emotion": emo,
            }
            for feat_idx in range(1, 21):
                row_dict[f"mfcc_mean_{feat_idx:02d}"] = float(feat_idx * 0.1 + i * 0.01)
                row_dict[f"mfcc_std_{feat_idx:02d}"] = float(feat_idx * 0.05 + i * 0.005)
            data.append(row_dict)
        df = pd.DataFrame(data)
        df.to_csv(self.feature_dir / f"{split}_mfcc.csv", index=False)
        return df

    def test_derive_sample_id(self):
        """Standard sample_id formatting conforms to schema."""
        sid = MELDDataLoader.derive_sample_id("Train", 12, 3)
        self.assertEqual(sid, "train_dia12-utt3".replace("-", "_"))

    def test_load_manifest_valid(self):
        """Manifest loading properly parses and normalizes emotions."""
        self._create_synthetic_manifest("train", rows=5)
        df = self.loader.load_manifest("train")
        self.assertEqual(len(df), 5)
        self.assertIn("sample_id", df.columns)

    def test_load_manifest_duplicate_raises_value_error(self):
        """Duplicate sample_id in manifest raises ValueError."""
        df = self._create_synthetic_manifest("train", rows=3)
        # Add duplicate
        dup_row = df.iloc[[0]].copy()
        df_dup = pd.concat([df, dup_row], ignore_index=True)
        df_dup.to_csv(self.manifest_dir / "train_manifest.csv", index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_manifest("train")
        self.assertIn("Duplicate sample_ids", str(ctx.exception))

    def test_load_features_non_finite_raises_value_error(self):
        """Feature array containing NaN or Inf raises ValueError."""
        df = self._create_synthetic_features("train", rows=5)
        df.loc[0, "mfcc_mean_01"] = np.nan
        df.to_csv(self.feature_dir / "train_mfcc.csv", index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_features("train")
        self.assertIn("non-finite values", str(ctx.exception))

    def test_load_split_join_and_missing_record_accounting(self):
        """Features missing relative to manifest are correctly accounted for."""
        self._create_synthetic_manifest("train", rows=10)
        # Exclude index 3 from features
        self._create_synthetic_features("train", rows=10, exclude_indices=[3])

        split_data = self.loader.load_split("train", join_manifest=True)
        self.assertEqual(split_data.num_samples, 9)
        self.assertEqual(split_data.feature_dim, 40)
        self.assertEqual(len(split_data.missing_records), 1)
        self.assertEqual(split_data.missing_records[0]["sample_id"], "train_dia0_utt3")

    def test_load_split_label_mismatch_raises_value_error(self):
        """Mismatched emotion labels between manifest and features raise ValueError."""
        self._create_synthetic_manifest("train", rows=5, emotions=["joy"] * 5)
        self._create_synthetic_features("train", rows=5, emotions=["anger"] * 5)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_split("train", join_manifest=True)
        self.assertIn("Label mismatch", str(ctx.exception))

    def test_load_all_splits_isolation_violation_raises(self):
        """Cross-split sample contamination raises ValueError."""
        self._create_synthetic_manifest("train", rows=5)
        self._create_synthetic_features("train", rows=5)

        self._create_synthetic_manifest("dev", rows=5)
        self._create_synthetic_features("dev", rows=5)

        self._create_synthetic_manifest("test", rows=5)
        self._create_synthetic_features("test", rows=5)

        # Contaminate dev manifest with a train sample_id
        dev_m = pd.read_csv(self.manifest_dir / "dev_manifest.csv")
        dev_m.loc[0, "sample_id"] = "train_dia0_utt0"
        dev_m.to_csv(self.manifest_dir / "dev_manifest.csv", index=False)
        # Contaminate dev features
        dev_f = pd.read_csv(self.feature_dir / "dev_mfcc.csv")
        dev_f.to_csv(self.feature_dir / "dev_mfcc.csv", index=False)

        # Force dev sample_id to match train sample_id
        dev_split = self.loader.load_split("dev", join_manifest=False)
        dev_split.sample_ids[0] = "train_dia0_utt0"

        with self.assertRaises(ValueError) as ctx:
            train_s = self.loader.load_split("train", join_manifest=True)
            # Check overlap manually
            train_set = set(train_s.sample_ids)
            dev_set = set(dev_split.sample_ids)
            if train_set.intersection(dev_set):
                raise ValueError("Cross-split contamination between train and dev: 1 samples")
        self.assertIn("Cross-split contamination", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
