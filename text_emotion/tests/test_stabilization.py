import unittest

from project_config import MELD_ROOT, meld_annotation_path
from text_emotion.preprocessing import preprocess_text


class TestTextStabilization(unittest.TestCase):
    def test_preprocessing_is_deterministic_and_preserves_negation(self):
        value = preprocess_text("I am NOT happy! Visit https://example.com")
        self.assertEqual(value, "i am not happy! visit")
        self.assertEqual(value, preprocess_text("I am NOT happy! Visit https://example.com"))

    def test_repository_local_meld_path(self):
        self.assertEqual(meld_annotation_path("train").parent.parent, MELD_ROOT)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            meld_annotation_path("validation")
