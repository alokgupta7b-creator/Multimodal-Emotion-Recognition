"""
speech_emotion.tests.test_training_pipeline
===========================================
Unit tests for SER training, tuning benchmark, and model evaluation pipeline.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from speech_emotion.model.classifier import SERClassifier, SERConfig
from speech_emotion.training.data import DatasetSplit
from speech_emotion.training.evaluate import evaluate_model_on_split
from speech_emotion.training.train import SERModelTrainer, train_single_model, evaluate_split


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic dataset split
        np.random.seed(42)
        n_samples = 35
        self.X_train = np.random.randn(n_samples, 40)
        self.y_train = (["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] * 5)
        self.train_split = DatasetSplit(
            split="train",
            sample_ids=[f"train_dia0_utt{i}" for i in range(n_samples)],
            X=self.X_train,
            y=self.y_train,
            dialogue_ids=np.zeros(n_samples, dtype=int),
            utterance_ids=np.arange(n_samples, dtype=int),
            missing_records=[],
            class_distribution={c: 5 for c in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]},
            metadata=pd.DataFrame(),
        )

        n_dev = 14
        self.X_dev = np.random.randn(n_dev, 40)
        self.y_dev = (["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] * 2)
        self.dev_split = DatasetSplit(
            split="dev",
            sample_ids=[f"dev_dia0_utt{i}" for i in range(n_dev)],
            X=self.X_dev,
            y=self.y_dev,
            dialogue_ids=np.zeros(n_dev, dtype=int),
            utterance_ids=np.arange(n_dev, dtype=int),
            missing_records=[],
            class_distribution={c: 2 for c in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]},
            metadata=pd.DataFrame(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_single_model_duration(self):
        """Single model fits properly and records duration."""
        cfg = SERConfig(classifier_type="logistic_regression", max_iter=100)
        model, duration = train_single_model(cfg, self.train_split)
        self.assertTrue(model.is_fitted)
        self.assertGreater(duration, 0.0)

    def test_evaluate_split_generates_valid_metrics(self):
        """evaluate_split computes latency and all evaluation metrics."""
        cfg = SERConfig(classifier_type="logistic_regression", max_iter=100)
        model, dur = train_single_model(cfg, self.train_split)
        res = evaluate_split(model, self.dev_split, model_name="LogRegTest", training_duration_sec=dur)

        self.assertEqual(res.num_samples, 14)
        self.assertTrue(0.0 <= res.accuracy <= 1.0)
        self.assertIsNotNone(res.inference_latency_ms)

    def test_run_benchmark_and_model_selection(self):
        """Trainer runs candidate models, saves artifacts, and selects top performer on dev."""
        trainer = SERModelTrainer(output_dir=self.output_dir, random_state=42)

        # Lightweight configs for fast test execution
        candidates = {
            "fast_logreg": SERConfig(classifier_type="logistic_regression", max_iter=50),
            "fast_svm": SERConfig(classifier_type="svm", C=1.0, svm_kernel="linear", max_iter=50),
            "fast_mlp": SERConfig(classifier_type="mlp", mlp_hidden_layers=(16,), max_iter=20),
        }

        best_name, best_model, results = trainer.run_benchmark(
            train_data=self.train_split,
            dev_data=self.dev_split,
            candidate_configs=candidates,
            selection_metric="macro_f1",
        )

        self.assertIn(best_name, candidates)
        self.assertTrue(best_model.is_fitted)
        self.assertEqual(len(results), 3)

        # Verify saved artifacts
        self.assertTrue((self.output_dir / "ser_best_model.joblib").exists())
        self.assertTrue((self.output_dir / "dev_benchmark_report.json").exists())
        self.assertTrue((self.output_dir / "dev_benchmark_report.md").exists())

    def test_evaluate_model_on_split_standalone(self):
        """evaluate_model_on_split loads saved artifact and produces file reports."""
        cfg = SERConfig(classifier_type="logistic_regression", max_iter=50)
        model, _ = train_single_model(cfg, self.train_split)
        saved_file = self.output_dir / "test_model.joblib"
        model.save(saved_file)

        eval_dir = self.output_dir / "reports"
        res = evaluate_model_on_split(
            model=saved_file,
            split_data=self.dev_split,
            output_dir=eval_dir,
            report_prefix="dev_eval",
        )

        self.assertEqual(res.num_samples, 14)
        self.assertTrue((eval_dir / "dev_eval_metrics.json").exists())
        self.assertTrue((eval_dir / "dev_eval_report.md").exists())


if __name__ == "__main__":
    unittest.main()
