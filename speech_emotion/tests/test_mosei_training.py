"""
speech_emotion.tests.test_mosei_training
========================================
Unit tests for CMU-MOSEI label harmonization, feature validation,
dataset split isolation, and prevention of incompatible feature mixing.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from speech_emotion.model.classifier import SERClassifier, SERConfig
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES, MELDLabelEncoder
from speech_emotion.training.mosei_data import (
    COVAREP_FEATURE_DIM,
    COVAREP_FEATURE_NAMES,
    MOSEI_SENTIMENT_CLASSES,
    MOSEI_TO_CANONICAL_EMOTION,
    MOSEIDataConfig,
    MOSEIDataLoader,
    harmonize_mosei_emotion,
)


class TestMOSEITraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mosei_dir = Path(self.temp_dir.name)
        self.manifest_dir = self.mosei_dir / "manifests"
        self.features_dir = self.mosei_dir / "features"
        self.cache_dir = self.features_dir / "covarep_pooled"

        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = MOSEIDataConfig(
            mosei_dir=self.mosei_dir,
            cache_dir=self.cache_dir,
            target_type="harmonized_emotion",
        )
        self.loader = MOSEIDataLoader(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_label_harmonization_mapping(self):
        """All 6 CMU-MOSEI emotions and 'none' map deterministically to 7 canonical classes."""
        self.assertEqual(harmonize_mosei_emotion("happy"), "joy")
        self.assertEqual(harmonize_mosei_emotion("sad"), "sadness")
        self.assertEqual(harmonize_mosei_emotion("anger"), "anger")
        self.assertEqual(harmonize_mosei_emotion("surprise"), "surprise")
        self.assertEqual(harmonize_mosei_emotion("disgust"), "disgust")
        self.assertEqual(harmonize_mosei_emotion("fear"), "fear")
        self.assertEqual(harmonize_mosei_emotion("none"), "neutral")

    def test_incompatible_label_rejection(self):
        """Sentiment labels or unknown classes cannot be harmonized directly as emotion."""
        with self.assertRaises(ValueError):
            harmonize_mosei_emotion("positive")
        with self.assertRaises(ValueError):
            harmonize_mosei_emotion("negative")
        with self.assertRaises(ValueError):
            harmonize_mosei_emotion("unknown_affect")

    def _create_synthetic_mosei_split(self, split: str, rows: int = 14) -> None:
        emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear", "none"]
        sentiments = ["positive", "negative", "neutral"]

        m_records = []
        f_records = []

        for i in range(rows):
            sample_id = f"vid_{split}_{i:02d}_clip0"
            m_records.append({
                "sample_id": sample_id,
                "video_id": f"vid_{split}_{i:02d}",
                "clip_id": 0,
                "start_time": float(i * 2.0),
                "end_time": float(i * 2.0 + 1.5),
                "dominant_emotion": emotions[i % len(emotions)],
                "sentiment_discrete": sentiments[i % len(sentiments)],
            })

            feat_row = {"sample_id": sample_id}
            for col in COVAREP_FEATURE_NAMES:
                feat_row[col] = float(0.1 * i + 0.01)
            f_records.append(feat_row)

        pd.DataFrame(m_records).to_csv(self.manifest_dir / f"{split}_manifest.csv", index=False)
        pd.DataFrame(f_records).to_csv(self.cache_dir / f"{split}_covarep.csv", index=False)

    def test_load_split_harmonized_emotion(self):
        """load_split produces 74-dim features and harmonized 7-class emotion labels."""
        self._create_synthetic_mosei_split("train", rows=14)
        split_data = self.loader.load_split("train")

        self.assertEqual(split_data.num_samples, 14)
        self.assertEqual(split_data.feature_dim, 74)
        self.assertEqual(set(split_data.y), set(MELD_CLASS_NAMES))

    def test_load_split_discrete_sentiment(self):
        """load_split supports discrete 3-class sentiment target space."""
        self._create_synthetic_mosei_split("train", rows=12)
        cfg = MOSEIDataConfig(
            mosei_dir=self.mosei_dir,
            cache_dir=self.cache_dir,
            target_type="sentiment_discrete",
        )
        loader = MOSEIDataLoader(cfg)
        split_data = loader.load_split("train")

        self.assertEqual(split_data.num_samples, 12)
        self.assertEqual(set(split_data.y), set(MOSEI_SENTIMENT_CLASSES))

    def test_prevention_of_mixing_incompatible_feature_dimensions(self):
        """SERClassifier with 74 expected features strictly rejects 40-dim MELD features."""
        model = SERClassifier(config=SERConfig(expected_features=74))
        X_meld_40 = np.random.randn(10, 40)
        y_dummy = ["joy"] * 10

        with self.assertRaises(ValueError) as ctx:
            model.fit(X_meld_40, y_dummy)
        self.assertIn("Feature dimension mismatch", str(ctx.exception))

    def test_video_level_split_isolation_leakage_detection(self):
        """Shared video IDs between train and dev splits are detected as leakage."""
        self._create_synthetic_mosei_split("train", rows=5)
        self._create_synthetic_mosei_split("dev", rows=5)
        self._create_synthetic_mosei_split("test", rows=5)

        # Contaminate dev manifest with a train video_id
        dev_m = pd.read_csv(self.manifest_dir / "dev_manifest.csv")
        dev_m.loc[0, "video_id"] = "vid_train_00"
        dev_m.to_csv(self.manifest_dir / "dev_manifest.csv", index=False)

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_all_splits()
        self.assertIn("Video leakage", str(ctx.exception))

    def test_mosei_model_serialization_and_reload(self):
        """Fitted 74-dim MOSEI model can be saved, reloaded, and produces identical predictions."""
        self._create_synthetic_mosei_split("train", rows=14)
        split_data = self.loader.load_split("train")

        model = SERClassifier(config=SERConfig(expected_features=74, max_iter=100))
        model.fit(split_data.X, split_data.y)

        save_path = self.mosei_dir / "test_mosei_model.joblib"
        model.save(save_path)

        reloaded = SERClassifier.load(save_path)
        self.assertTrue(reloaded.is_fitted)
        self.assertEqual(reloaded.config.expected_features, 74)

        preds_orig = model.predict(split_data.X[:3])
        preds_reload = reloaded.predict(split_data.X[:3])
        self.assertEqual(preds_orig, preds_reload)


if __name__ == "__main__":
    unittest.main()
