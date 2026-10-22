"""
speech_emotion/tests/test_model.py
==================================
Unit tests for Speech Emotion Recognition (SER) classifier and prediction interface:
- Model construction and configuration defaults.
- Synthetic training on 40-dimensional feature vectors.
- Single-sample and batch prediction validity.
- Probability shape, bounds, and sum-to-one normalization.
- Strict input validation: dimension mismatch, NaN, Inf, non-finite values.
- Fit validation on label mismatches or unknown labels.
- Serialization and reload via joblib with prediction consistency.
- Handling of missing model files and safe overwrite controls.
- Convenience prediction from an audio file.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speech_emotion.audio_loader import write_wav
from speech_emotion.config import TOTAL_MFCC_FEATURES
from speech_emotion.model.classifier import SERClassifier, SERConfig
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES
from speech_emotion.model.predict import (
    SERPrediction,
    predict_emotion,
    predict_from_audio,
)


class TestSERModel(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_ser_model_"))

        # Generate a small, deterministic synthetic dataset across all 7 emotions
        np.random.seed(42)
        self.n_samples_per_class = 6
        self.classes = list(MELD_CLASS_NAMES)
        self.n_classes = len(self.classes)  # 7
        self.total_samples = self.n_samples_per_class * self.n_classes  # 42

        # Create 40-dimensional features with class-specific offsets to allow learning
        features_list = []
        labels_list = []
        for c_idx, class_name in enumerate(self.classes):
            # Class center offset
            center = np.zeros(TOTAL_MFCC_FEATURES)
            center[c_idx] = 3.0  # signal in c_idx
            noise = np.random.randn(self.n_samples_per_class, TOTAL_MFCC_FEATURES) * 0.2
            samples = center + noise
            features_list.append(samples)
            labels_list.extend([class_name] * self.n_samples_per_class)

        self.X_train = np.vstack(features_list)
        self.y_train = labels_list

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_construction_defaults(self) -> None:
        """Default SERClassifier uses expected config and 40 features."""
        model = SERClassifier()
        self.assertEqual(model.config.classifier_type, "logistic_regression")
        self.assertEqual(model.config.class_weight, "balanced")
        self.assertEqual(model.config.expected_features, 40)
        self.assertTrue(model.config.standardize)
        self.assertEqual(model.classes_, list(MELD_CLASS_NAMES))
        self.assertFalse(model.is_fitted)

    def test_unfitted_model_raises_runtime_error(self) -> None:
        """Calling predict or predict_proba on unfitted model raises RuntimeError."""
        model = SERClassifier()
        dummy_x = np.zeros((1, 40))
        with self.assertRaises(RuntimeError) as ctx:
            model.predict(dummy_x)
        self.assertIn("not fitted yet", str(ctx.exception))

        with self.assertRaises(RuntimeError):
            model.predict_proba(dummy_x)

    def test_training_and_single_sample_prediction(self) -> None:
        """Fitted model predicts valid string label on a single 1D feature vector."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_fitted)

        # Single 1D vector (40,)
        x_single = self.X_train[0]
        self.assertEqual(x_single.shape, (40,))

        pred = model.predict(x_single)
        self.assertIsInstance(pred, str)
        self.assertIn(pred, MELD_CLASS_NAMES)

    def test_batch_prediction_shape_and_types(self) -> None:
        """Batch input (N, 40) returns list of N valid emotion labels."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_train)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), len(self.X_train))
        for p in preds:
            self.assertIn(p, MELD_CLASS_NAMES)

    def test_predict_proba_shape_bounds_and_normalization(self) -> None:
        """predict_proba returns (N, 7) array where rows sum to 1.0 within [0.0, 1.0]."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        probs = model.predict_proba(self.X_train)
        self.assertEqual(probs.shape, (len(self.X_train), 7))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Each row must sum to 1.0
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_emotion_structured_output(self) -> None:
        """predict_emotion returns SERPrediction with class probabilities and confidence."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        # Single vector
        res = predict_emotion(model, self.X_train[0])
        self.assertIsInstance(res, SERPrediction)
        self.assertIn(res.predicted_label, MELD_CLASS_NAMES)
        self.assertEqual(res.predicted_index, list(MELD_CLASS_NAMES).index(res.predicted_label))
        self.assertGreater(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertEqual(len(res.probabilities), 7)
        self.assertEqual(res.probability_array.shape, (7,))

        # Dict conversion
        d = res.to_dict()
        self.assertEqual(d["predicted_label"], res.predicted_label)
        self.assertEqual(d["confidence"], res.confidence)
        self.assertIn("probabilities", d)

        # Batch
        batch_res = predict_emotion(model, self.X_train[:3])
        self.assertIsInstance(batch_res, list)
        self.assertEqual(len(batch_res), 3)
        for item in batch_res:
            self.assertIsInstance(item, SERPrediction)

    def test_dimension_mismatch_raises_value_error(self) -> None:
        """Features with wrong dimensionality (e.g. 39 or 41) raise ValueError."""
        model = SERClassifier()
        model.fit(self.X_train, self.y_train)

        # 39 features instead of 40
        bad_x_1d = np.zeros(39)
        with self.assertRaises(ValueError) as ctx:
            model.predict(bad_x_1d)
        self.assertIn("Expected 1D feature vector of length 40", str(ctx.exception))

        # (N, 41) instead of (N, 40)
        bad_x_2d = np.zeros((5, 41))
        with self.assertRaises(ValueError) as ctx:
            model.predict(bad_x_2d)
        self.assertIn("Expected 2D feature matrix with 40 columns", str(ctx.exception))

    def test_nan_or_inf_features_raise_value_error(self) -> None:
        """Features containing NaN or Inf raise ValueError."""
        model = SERClassifier()
        model.fit(self.X_train, self.y_train)

        nan_x = np.zeros(40)
        nan_x[5] = np.nan
        with self.assertRaises(ValueError) as ctx:
            model.predict(nan_x)
        self.assertIn("non-finite values", str(ctx.exception))

        inf_x = np.zeros((2, 40))
        inf_x[0, 10] = np.inf
        with self.assertRaises(ValueError):
            model.predict(inf_x)

    def test_fit_with_unknown_labels_raises_value_error(self) -> None:
        """fit() rejects unknown or malformed emotion labels."""
        model = SERClassifier()
        bad_labels = list(self.y_train)
        bad_labels[0] = "excited"  # not in MELD 7 emotions

        with self.assertRaises(ValueError) as ctx:
            model.fit(self.X_train, bad_labels)
        self.assertIn("Label validation failed", str(ctx.exception))

    def test_serialization_and_reload(self) -> None:
        """Model can be saved to disk and reloaded with identical predictions."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        save_path = self.temp_dir / "models" / "ser_model.joblib"
        model.save(save_path)
        self.assertTrue(save_path.is_file())

        # Reload model
        loaded_model = SERClassifier.load(save_path)
        self.assertTrue(loaded_model.is_fitted)
        self.assertEqual(loaded_model.classes_, model.classes_)

        # Verify exact predictions and probabilities match
        orig_preds = model.predict(self.X_train)
        loaded_preds = loaded_model.predict(self.X_train)
        self.assertEqual(orig_preds, loaded_preds)

        orig_probs = model.predict_proba(self.X_train)
        loaded_probs = loaded_model.predict_proba(self.X_train)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-7)

    def test_save_existing_file_without_overwrite_raises(self) -> None:
        """Saving to existing path without overwrite=True raises FileExistsError."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        save_path = self.temp_dir / "model.joblib"
        model.save(save_path)

        with self.assertRaises(FileExistsError):
            model.save(save_path, overwrite=False)

        # Overwrite=True succeeds
        model.save(save_path, overwrite=True)

    def test_load_non_existent_file_raises_file_not_found(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            SERClassifier.load(self.temp_dir / "does_not_exist.joblib")

    def test_predict_from_audio_file(self) -> None:
        """predict_from_audio extracts 40 MFCCs from a WAV file and predicts."""
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        # Generate a synthetic tone WAV file
        sr = 16000
        t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
        y = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        wav_path = self.temp_dir / "test_utterance.wav"
        write_wav(wav_path, y, sample_rate=sr)

        pred = predict_from_audio(model, wav_path)
        self.assertIsInstance(pred, SERPrediction)
        self.assertIn(pred.predicted_label, MELD_CLASS_NAMES)
        self.assertGreater(pred.confidence, 0.0)

    def test_predict_proba_with_subset_of_classes(self) -> None:
        """
        predict_proba always returns (N, 7) aligned to canonical MELD classes,
        even when one or more classes were absent during fitting.
        """
        # Train on only 3 classes: joy, neutral, sadness
        subset_classes = ["joy", "neutral", "sadness"]
        mask = [lbl in subset_classes for lbl in self.y_train]
        X_sub = self.X_train[mask]
        y_sub = [lbl for lbl in self.y_train if lbl in subset_classes]

        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(X_sub, y_sub)

        probs = model.predict_proba(X_sub)
        self.assertEqual(probs.shape, (len(X_sub), 7))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Check row sums equal 1.0
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

        # Absent classes must have exactly 0.0 probability
        # Canonical order: anger (0), disgust (1), fear (2), joy (3), neutral (4), sadness (5), surprise (6)
        absent_indices = [0, 1, 2, 6]  # anger, disgust, fear, surprise
        for idx in absent_indices:
            np.testing.assert_array_equal(probs[:, idx], 0.0)

        # Present classes must have positive probabilities
        present_indices = [3, 4, 5]  # joy, neutral, sadness
        for idx in present_indices:
            self.assertTrue(np.any(probs[:, idx] > 0.0))

        # Test predict_emotion structured output with absent classes
        pred = predict_emotion(model, X_sub[0])
        self.assertIsInstance(pred, SERPrediction)
        self.assertIn(pred.predicted_label, subset_classes)
        self.assertEqual(len(pred.probabilities), 7)
        for absent_name in ["anger", "disgust", "fear", "surprise"]:
            self.assertEqual(pred.probabilities[absent_name], 0.0)
        for present_name in subset_classes:
            self.assertIn(present_name, pred.probabilities)

    def test_load_malformed_artifact_fails_clearly(self) -> None:
        """Loading malformed, corrupted, or incompatible artifacts raises descriptive ValueError."""
        import joblib

        # 1. Non-dict, non-Pipeline artifact
        bad_obj_path = self.temp_dir / "bad_string.joblib"
        joblib.dump("just a string", bad_obj_path)
        with self.assertRaises(ValueError) as ctx:
            SERClassifier.load(bad_obj_path)
        self.assertIn("Unrecognized model artifact format", str(ctx.exception))

        # 2. Dict missing 'pipeline'
        missing_pipe_path = self.temp_dir / "no_pipeline.joblib"
        joblib.dump({"config": {}}, missing_pipe_path)
        with self.assertRaises(ValueError) as ctx:
            SERClassifier.load(missing_pipe_path)
        self.assertIn("missing or invalid 'pipeline'", str(ctx.exception))

        # 3. Dict with invalid 'pipeline' object
        invalid_pipe_path = self.temp_dir / "invalid_pipeline_type.joblib"
        joblib.dump({"pipeline": 12345}, invalid_pipe_path)
        with self.assertRaises(ValueError) as ctx:
            SERClassifier.load(invalid_pipe_path)
        self.assertIn("expected sklearn.pipeline.Pipeline", str(ctx.exception))

        # 4. Dict with incompatible config
        bad_cfg_path = self.temp_dir / "bad_config.joblib"
        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)
        joblib.dump({
            "pipeline": model._pipeline,
            "config": {"classifier_type": "unsupported_dnn"},
        }, bad_cfg_path)
        with self.assertRaises(ValueError) as ctx:
            SERClassifier.load(bad_cfg_path)
        self.assertIn("Incompatible configuration", str(ctx.exception))

        # 5. Corrupted binary file
        corrupt_path = self.temp_dir / "corrupted.joblib"
        corrupt_path.write_bytes(b"\x00\x01\x02\x03\x04INVALID_DATA")
        with self.assertRaises(ValueError) as ctx:
            SERClassifier.load(corrupt_path)
        self.assertIn("Failed to deserialize", str(ctx.exception))

    def test_predict_from_audio_uses_task4_config(self) -> None:
        """predict_from_audio defaults to standard Task 4 AudioConfig and detects mismatches."""
        from speech_emotion.config import AudioConfig

        model = SERClassifier(SERConfig(max_iter=500, random_state=42))
        model.fit(self.X_train, self.y_train)

        # Generate synthetic WAV
        sr = 16000
        t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
        y = (0.5 * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)
        wav_path = self.temp_dir / "test_task4_audio.wav"
        write_wav(wav_path, y, sample_rate=sr)

        # Default call uses standard Task 4 config (40 features, pad_mode='constant')
        pred = predict_from_audio(model, wav_path)
        self.assertIsInstance(pred, SERPrediction)
        self.assertEqual(len(pred.probability_array), 7)

        # Incompatible config (e.g. 26 features instead of 40) raises ValueError
        incompatible_cfg = AudioConfig(sample_rate=16000, n_mfcc=13)
        with self.assertRaises(ValueError) as ctx:
            predict_from_audio(model, wav_path, config=incompatible_cfg)
        self.assertIn("Expected 1D feature vector of length 40", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
