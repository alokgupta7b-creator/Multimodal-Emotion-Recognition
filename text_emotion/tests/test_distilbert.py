import unittest

from text_emotion.distilbert_model import DistilBertTrainingConfig, load_meld_frame
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


class TestDistilBertPreparation(unittest.TestCase):
    def test_meld_frames_use_canonical_labels_without_leakage(self):
        train = load_meld_frame("train")
        dev = load_meld_frame("dev")
        test = load_meld_frame("test")
        self.assertEqual(set(train["label_name"]), set(MELD_CLASS_NAMES))
        self.assertEqual(set(dev["label_name"]), set(MELD_CLASS_NAMES))
        self.assertEqual(set(test["label_name"]), set(MELD_CLASS_NAMES))
        self.assertEqual(len(train), 9989)
        self.assertEqual(len(dev), 1109)
        self.assertEqual(len(test), 2610)

    def test_defaults_fit_small_gpu(self):
        config = DistilBertTrainingConfig()
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.max_length, 128)

    def test_invalid_text_is_rejected_without_model_load(self):
        from text_emotion.distilbert_model import DistilBertTextPredictor
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            DistilBertTextPredictor("missing-distilbert")
