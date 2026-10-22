"""
speech_emotion.tests.test_metrics
=================================
Unit tests for SER classification metrics computation and report formatting.
"""

import unittest
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES
from speech_emotion.training.metrics import compute_metrics, EvaluationResult


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions_metrics(self):
        """Perfect predictions yield 1.0 accuracy, macro F1, and balanced accuracy."""
        y_true = ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]
        y_pred = list(y_true)

        res = compute_metrics(y_true, y_pred, model_name="PerfectModel")
        self.assertAlmostEqual(res.accuracy, 1.0)
        self.assertAlmostEqual(res.balanced_accuracy, 1.0)
        self.assertAlmostEqual(res.macro_f1, 1.0)
        self.assertAlmostEqual(res.weighted_f1, 1.0)
        self.assertEqual(len(res.classes), 7)

    def test_unpredicted_classes_handled_safely(self):
        """When some classes are never predicted, metrics do not divide by zero and retain 7 classes."""
        y_true = ["neutral", "joy", "anger", "sadness", "surprise", "fear", "disgust"]
        # Only predict neutral
        y_pred = ["neutral"] * len(y_true)

        res = compute_metrics(y_true, y_pred, model_name="NeutralBiasModel")
        self.assertAlmostEqual(res.accuracy, 1.0 / 7.0)
        self.assertAlmostEqual(res.balanced_accuracy, 1.0 / 7.0)
        # Macro F1 should be non-zero for neutral and 0 for others without crashing
        self.assertTrue(0.0 <= res.macro_f1 <= 1.0)
        self.assertEqual(len(res.per_class_metrics), 7)
        self.assertEqual(res.per_class_metrics["disgust"]["precision"], 0.0)

    def test_confusion_matrix_shape_and_ordering(self):
        """Confusion matrix is 7x7 and rows/cols match canonical MELD ordering."""
        y_true = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        y_pred = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

        res = compute_metrics(y_true, y_pred)
        cm = res.confusion_matrix
        self.assertEqual(len(cm), 7)
        for row in cm:
            self.assertEqual(len(row), 7)

        # Diagonal entries should be 1, off-diagonal 0
        for i in range(7):
            for j in range(7):
                if i == j:
                    self.assertEqual(cm[i][j], 1)
                else:
                    self.assertEqual(cm[i][j], 0)

    def test_serialization_and_markdown(self):
        """EvaluationResult produces valid dict and Markdown with all required sections."""
        y_true = ["neutral", "joy", "anger"]
        y_pred = ["neutral", "joy", "sadness"]

        res = compute_metrics(
            y_true,
            y_pred,
            model_name="TestModel",
            training_duration_sec=1.23,
            inference_latency_ms=0.45,
            model_size_bytes=4096,
        )

        d = res.to_dict()
        self.assertEqual(d["model_name"], "TestModel")
        self.assertEqual(d["training_duration_sec"], 1.23)
        self.assertEqual(d["inference_latency_ms_per_sample"], 0.45)
        self.assertEqual(d["model_size_bytes"], 4096)

        md = res.to_markdown()
        self.assertIn("SER Evaluation Report: TestModel", md)
        self.assertIn("Confusion Matrix", md)
        self.assertIn("Training Duration", md)
        self.assertIn("Inference Latency", md)


if __name__ == "__main__":
    unittest.main()
