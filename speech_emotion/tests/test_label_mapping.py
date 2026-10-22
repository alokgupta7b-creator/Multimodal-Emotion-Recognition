"""
speech_emotion/tests/test_label_mapping.py
==========================================
Unit tests for MELD 7-class label encoder and mapping:
- Canonical 7 classes in alphabetical order.
- Bi-directional label encoding and decoding.
- Strict validation of invalid, empty, or unknown labels.
- Sequence encoding and decoding.
"""

from __future__ import annotations

import unittest
import numpy as np

from speech_emotion.model.label_mapping import (
    MELDLabelEncoder,
    MELD_CLASS_NAMES,
    NUM_MELD_CLASSES,
)


class TestLabelMapping(unittest.TestCase):

    def setUp(self) -> None:
        self.encoder = MELDLabelEncoder()

    def test_canonical_classes_order_and_count(self) -> None:
        """Encoder has exactly 7 classes in deterministic alphabetical order."""
        expected = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
        self.assertEqual(MELD_CLASS_NAMES, expected)
        self.assertEqual(self.encoder.classes, expected)
        self.assertEqual(self.encoder.num_classes, 7)
        self.assertEqual(NUM_MELD_CLASSES, 7)

    def test_encode_and_decode_single_label(self) -> None:
        """Single labels encode to integers [0..6] and decode back to original strings."""
        for expected_idx, label in enumerate(MELD_CLASS_NAMES):
            encoded = self.encoder.encode_label(label)
            self.assertEqual(encoded, expected_idx)

            decoded = self.encoder.decode_label(encoded)
            self.assertEqual(decoded, label)

    def test_case_and_whitespace_insensitivity(self) -> None:
        """Labels with irregular case and whitespace are normalized."""
        self.assertEqual(self.encoder.encode_label("  Joy  "), 3)
        self.assertEqual(self.encoder.encode_label("SADNESS"), 5)
        self.assertEqual(self.encoder.encode_label("NeUtRaL"), 4)

    def test_encode_sequence(self) -> None:
        """Encoding a list of labels produces an int64 numpy array."""
        labels = ["anger", "joy", "sadness", "joy", "neutral"]
        encoded = self.encoder.encode(labels)

        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.dtype, np.int64)
        np.testing.assert_array_equal(encoded, np.array([0, 3, 5, 3, 4], dtype=np.int64))

        decoded = self.encoder.decode(encoded)
        self.assertEqual(decoded, labels)

    def test_unknown_label_raises_value_error(self) -> None:
        """Labels from text datasets (e.g. 'love') or unknown emotions raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.encoder.encode_label("love")  # 6-class text emotion, not in MELD
        self.assertIn("Unknown emotion label 'love'", str(ctx.exception))

        with self.assertRaises(ValueError):
            self.encoder.encode_label("excited")

    def test_invalid_label_types_raise_value_error(self) -> None:
        """None, empty string, numbers, or malformed values raise ValueError."""
        with self.assertRaises(ValueError):
            self.encoder.encode_label(None)

        with self.assertRaises(ValueError):
            self.encoder.encode_label("")

        with self.assertRaises(ValueError):
            self.encoder.encode_label("   ")

        with self.assertRaises(ValueError):
            self.encoder.encode_label(123)

    def test_decode_out_of_bounds_index_raises_value_error(self) -> None:
        """Index outside [0..6] raises ValueError."""
        with self.assertRaises(ValueError):
            self.encoder.decode_label(-1)

        with self.assertRaises(ValueError):
            self.encoder.decode_label(7)

        with self.assertRaises(ValueError):
            self.encoder.decode_label(100)

        with self.assertRaises(ValueError):
            self.encoder.decode_label(2.5)  # non-integer

    def test_validate_labels(self) -> None:
        """validate_labels returns normalized lowercase strings or raises."""
        valid = ["Joy", "ANGER", "neutral"]
        normalized = self.encoder.validate_labels(valid)
        self.assertEqual(normalized, ["joy", "anger", "neutral"])

        invalid = ["joy", "invalid_emotion", "sadness"]
        with self.assertRaises(ValueError) as ctx:
            self.encoder.validate_labels(invalid)
        self.assertIn("Label validation failed at index 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
