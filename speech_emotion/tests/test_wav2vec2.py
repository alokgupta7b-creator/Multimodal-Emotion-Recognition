import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from speech_emotion.wav2vec2_model import (
    WAV2VEC2_ROOT,
    Wav2Vec2Predictor,
    Wav2Vec2TrainingConfig,
    _AudioDataset,
    _collator,
    load_meld_audio_records,
    preprocess_speech_waveform,
    trim_leading_trailing_silence,
)
from calibration_utils import load_logit_calibration
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


class TestWav2Vec2Preparation(unittest.TestCase):
    def test_hardware_safe_defaults(self):
        config = Wav2Vec2TrainingConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.gradient_accumulation_steps, 4)

    def test_missing_audio_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_meld_audio_records("train", Path(directory))

    def test_invalid_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_meld_audio_records("validation", Path("."))

    def test_long_and_silent_audio_is_truncated_not_rejected(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            sf.write(path, np.zeros(16000 * 10, dtype=np.float32), 16000)
            dataset = _AudioDataset(
                [{"path": path, "label": 0, "sample_id": "sample"}],
                Wav2Vec2TrainingConfig(),
                torch,
            )
            item = dataset[0]
            self.assertEqual(item["input_values"].shape[0], 16000 * 8)

    def test_p0_silence_trimmer_is_scale_invariant_and_preserves_speech(self):
        sr = 16000
        t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False)
        speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        silence = np.zeros(sr * 2, dtype=np.float32)
        padded = np.concatenate([silence, speech, silence]).astype(np.float32)

        trimmed, info = trim_leading_trailing_silence(padded, sample_rate=sr)
        self.assertTrue(info["trimmed"])
        self.assertGreaterEqual(info["kept_sec"], 1.5)
        self.assertLess(info["kept_sec"], 2.3)

        quiet_trimmed, quiet_info = trim_leading_trailing_silence(padded * 0.05, sample_rate=sr)
        self.assertEqual(len(trimmed), len(quiet_trimmed))
        self.assertAlmostEqual(info["kept_sec"], quiet_info["kept_sec"], places=6)

    def test_p0_preprocess_validates_caps_8s_and_flags_low_volume(self):
        sr = 16000
        quiet = (0.002 * np.ones(sr * 10, dtype=np.float32))
        processed, meta = preprocess_speech_waveform(quiet, sample_rate=sr, max_seconds=8.0, trim_silence=True)
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(processed.shape[0], sr * 8)
        self.assertEqual(meta["low_volume_warning"], 1.0)

        with self.assertRaises(ValueError):
            preprocess_speech_waveform(np.array([np.nan, 0.1], dtype=np.float32))

    def test_p0_collator_emits_attention_mask(self):
        import torch
        from transformers import AutoFeatureExtractor

        extractor = AutoFeatureExtractor.from_pretrained("models/speech_emotion/wav2vec2/best", local_files_only=True)
        collate = _collator(extractor, torch, 16000)
        batch = collate(
            [
                {"input_values": np.ones(16000, dtype=np.float32) * 0.1, "labels": 0},
                {"input_values": np.ones(32000, dtype=np.float32) * 0.1, "labels": 4},
            ]
        )
        self.assertIn("attention_mask", batch)
        self.assertEqual(int(batch["attention_mask"][0].sum()), 16000)
        self.assertEqual(int(batch["attention_mask"][1].sum()), 32000)

    def test_p1_calibration_artifact_loads_and_toggles_cleanly(self):
        cal_file = WAV2VEC2_ROOT / "calibration_p1.json"
        self.assertTrue(cal_file.is_file())
        cal = load_logit_calibration(cal_file, MELD_CLASS_NAMES)
        self.assertGreater(cal.temperature, 0.0)
        self.assertEqual(cal.bias.shape, (7,))

        sr = 16000
        t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False)
        wave = (0.15 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, wave, sr)
            pred_cal = Wav2Vec2Predictor(device="cpu", use_calibration=True)
            pred_raw = Wav2Vec2Predictor(device="cpu", use_calibration=False)
            res_cal = pred_cal.predict_file(path)
            res_raw = pred_raw.predict_file(path)

        self.assertTrue(res_cal["calibrated"])
        self.assertFalse(res_raw["calibrated"])
        self.assertAlmostEqual(sum(res_cal["probabilities"].values()), 1.0, places=5)
        diffs = [abs(res_cal["probabilities"][k] - res_raw["probabilities"][k]) for k in MELD_CLASS_NAMES]
        self.assertGreater(max(diffs), 1e-3)

    def test_p1_calibration_rejects_mismatched_class_order(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad_cal.json"
            bad.write_text(
                json.dumps({"classes": list(reversed(MELD_CLASS_NAMES)), "temperature": 1.0, "bias": [0.0] * 7}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_logit_calibration(bad, MELD_CLASS_NAMES)

    def test_p2_speech_v2_training_helpers_bucket_and_augment(self):
        from speech_emotion.train_wav2vec2_v2 import (
            LengthBucketBatchSampler,
            LogitAdjustedCELoss,
            augment_speech_waveform,
        )

        wave = (0.12 * np.sin(np.linspace(0.0, 20.0 * np.pi, 16000))).astype(np.float32)
        aug = augment_speech_waveform(wave, sample_rate=16000)
        self.assertEqual(aug.dtype, np.float32)
        self.assertGreaterEqual(len(aug), int(0.4 * 16000))
        self.assertLessEqual(float(np.max(np.abs(aug))), 1.0)

        lengths = [16000, 64000, 20000, 58000, 18000, 61000]
        sampler = LengthBucketBatchSampler(lengths=lengths, batch_size=2, seed=42)
        batches = list(sampler)
        seen = sorted(i for b in batches for i in b)
        self.assertEqual(seen, list(range(len(lengths))))
        self.assertEqual(len(batches), 3)

        import torch

        priors = np.array([0.11, 0.03, 0.03, 0.17, 0.47, 0.07, 0.12], dtype=np.float64)
        loss_fn = LogitAdjustedCELoss(priors, tau=0.45, label_smoothing=0.03)
        loss = loss_fn(torch.randn(3, 7), torch.tensor([0, 2, 4], dtype=torch.long))
        self.assertTrue(bool(torch.isfinite(loss)))

