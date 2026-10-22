"""
speech_emotion/tests/test_feature_extraction.py
===============================================
Automated unit tests for MFCC feature extraction:
- Expected 40-feature convention (20 means + 20 stds).
- Finite outputs and shape consistency.
- Exact reproducibility.
- Parameter configurability (n_mfcc, hop_length, layouts).
- Signal sensitivity and frequency discrimination.
"""

from __future__ import annotations

import unittest
import numpy as np

from speech_emotion.config import AudioConfig, TOTAL_MFCC_FEATURES
from speech_emotion.feature_extraction import (
    MFCCExtractor,
    create_mel_filterbank,
    extract_mfcc_features,
)


class TestFeatureExtraction(unittest.TestCase):

    def _generate_tone(self, freq: float, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    def test_default_40_feature_convention(self) -> None:
        """Default configuration produces exactly 40 features (20 means + 20 stds)."""
        y = self._generate_tone(freq=440.0, duration=1.0)
        config = AudioConfig(sample_rate=16000, n_mfcc=20, feature_layout="mean_std")
        extractor = MFCCExtractor(config)

        feats = extractor.extract(y)

        self.assertEqual(feats.shape, (TOTAL_MFCC_FEATURES,))
        self.assertEqual(len(feats), 40)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_feature_names_match_convention(self) -> None:
        """AudioConfig generates standardized column names matching existing MELD artifacts."""
        config = AudioConfig(n_mfcc=20, feature_layout="mean_std")
        names = config.get_feature_names()

        self.assertEqual(len(names), 40)
        self.assertEqual(names[0], "mfcc_mean_01")
        self.assertEqual(names[19], "mfcc_mean_20")
        self.assertEqual(names[20], "mfcc_std_01")
        self.assertEqual(names[39], "mfcc_std_20")

    def test_reproducibility_identical_inputs(self) -> None:
        """Identical inputs and configuration produce identical feature vectors."""
        y = self._generate_tone(freq=500.0, duration=1.0)
        config = AudioConfig(sample_rate=16000, n_mfcc=20)
        extractor = MFCCExtractor(config)

        run1 = extractor.extract(y)
        run2 = extractor.extract(y)

        np.testing.assert_array_equal(run1, run2)

    def test_configurable_n_mfcc(self) -> None:
        """n_mfcc=13 yields 26 features (13 means + 13 stds)."""
        y = self._generate_tone(freq=600.0, duration=1.0)
        config = AudioConfig(n_mfcc=13, feature_layout="mean_std")
        extractor = MFCCExtractor(config)

        feats = extractor.extract(y)

        self.assertEqual(feats.shape, (26,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_configurable_layout_mean_only(self) -> None:
        """feature_layout='mean_only' yields n_mfcc features."""
        y = self._generate_tone(freq=440.0, duration=1.0)
        config = AudioConfig(n_mfcc=20, feature_layout="mean_only")
        extractor = MFCCExtractor(config)

        feats = extractor.extract(y)

        self.assertEqual(feats.shape, (20,))
        self.assertEqual(config.get_feature_names()[0], "mfcc_mean_01")
        self.assertEqual(len(config.get_feature_names()), 20)

    def test_configurable_layout_std_only(self) -> None:
        """feature_layout='std_only' yields n_mfcc standard deviation features."""
        y = self._generate_tone(freq=440.0, duration=1.0)
        config = AudioConfig(n_mfcc=20, feature_layout="std_only")
        extractor = MFCCExtractor(config)

        feats = extractor.extract(y)

        self.assertEqual(feats.shape, (20,))
        self.assertEqual(config.get_feature_names()[0], "mfcc_std_01")

    def test_frequency_sensitivity(self) -> None:
        """Signals of different frequencies (200 Hz vs 2000 Hz) yield distinct MFCCs."""
        y_low = self._generate_tone(freq=200.0, duration=1.0)
        y_high = self._generate_tone(freq=2000.0, duration=1.0)
        extractor = MFCCExtractor(AudioConfig())

        f_low = extractor.extract(y_low)
        f_high = extractor.extract(y_high)

        # Distance between feature vectors should be substantial
        l2_dist = np.linalg.norm(f_low - f_high)
        self.assertGreater(l2_dist, 5.0)

    def test_short_audio_signal_padded(self) -> None:
        """Short audio signal (< n_fft) is padded and extracts without crashing."""
        short_y = self._generate_tone(freq=440.0, duration=0.08)  # ~1280 samples < 2048 n_fft
        extractor = MFCCExtractor(AudioConfig())

        feats = extractor.extract(short_y)

        self.assertEqual(feats.shape, (40,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_empty_audio_raises_value_error(self) -> None:
        """Empty audio array raises ValueError."""
        extractor = MFCCExtractor(AudioConfig())
        with self.assertRaises(ValueError):
            extractor.extract(np.zeros(0, dtype=np.float32))

    def test_non_finite_audio_raises_value_error(self) -> None:
        """Waveform with NaN raises ValueError."""
        extractor = MFCCExtractor(AudioConfig())
        bad_y = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with self.assertRaises(ValueError):
            extractor.extract(bad_y)

    def test_mel_filterbank_matrix_properties(self) -> None:
        """Mel filterbank has expected dimensions and non-negative weights."""
        fb = create_mel_filterbank(sr=16000, n_fft=2048, n_mels=128, fmin=0.0, fmax=8000.0)

        self.assertEqual(fb.shape, (128, 1025))
        self.assertTrue(np.all(fb >= 0.0))
        # Each filter band should have non-zero total energy
        self.assertTrue(np.all(np.sum(fb, axis=1) > 0.0))

    def test_convenience_function(self) -> None:
        """extract_mfcc_features convenience function returns valid 40-element array."""
        y = self._generate_tone(freq=520.0, duration=0.5)
        feats = extract_mfcc_features(y, sr=16000)

        self.assertEqual(feats.shape, (40,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_invalid_config_raises_value_error(self) -> None:
        """Invalid configurations raise ValueError on init."""
        with self.assertRaises(ValueError):
            AudioConfig(sample_rate=-16000)
        with self.assertRaises(ValueError):
            AudioConfig(n_mfcc=0)
        with self.assertRaises(ValueError):
            AudioConfig(n_mels=-5)
        with self.assertRaises(ValueError):
            AudioConfig(fmax=100.0, fmin=200.0)
        with self.assertRaises(ValueError):
            AudioConfig(feature_layout="invalid_layout")

    def test_explicit_numpy_backend(self) -> None:
        """NumPy backend runs explicitly without librosa and produces 40 finite features."""
        y = self._generate_tone(freq=440.0, duration=1.0)
        config = AudioConfig(sample_rate=16000, n_mfcc=20, engine="numpy")
        extractor = MFCCExtractor(config)

        feats = extractor.extract(y)
        self.assertEqual(feats.shape, (40,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_pad_mode_configuration(self) -> None:
        """pad_mode='constant' vs 'reflect' produces expected valid outputs."""
        y = self._generate_tone(freq=440.0, duration=1.0)
        ext_const = MFCCExtractor(AudioConfig(engine="numpy", pad_mode="constant"))
        ext_reflect = MFCCExtractor(AudioConfig(engine="numpy", pad_mode="reflect"))

        f_const = ext_const.extract(y)
        f_reflect = ext_reflect.extract(y)

        self.assertEqual(f_const.shape, (40,))
        self.assertEqual(f_reflect.shape, (40,))
        self.assertTrue(np.all(np.isfinite(f_const)))
        self.assertTrue(np.all(np.isfinite(f_reflect)))

    def test_short_signal_constant_padding_regression(self) -> None:
        """
        Regression test: audio shorter than n_fft (e.g. 1872 samples < 2048)
        with pad_mode='constant' must pad with exact zeros (matching librosa)
        rather than symmetric pre-padding.
        """
        sr = 16000
        # 1872 samples is 0.117s, valid (> min_duration=0.1s) but < n_fft=2048
        y_short = self._generate_tone(freq=300.0, duration=1872 / sr, sr=sr)
        self.assertEqual(len(y_short), 1872)

        config = AudioConfig(sample_rate=sr, n_mfcc=20, engine="numpy", pad_mode="constant")
        extractor = MFCCExtractor(config)

        frames = extractor.extract_frames(y_short)
        # Expected frames: 1 + (1872 + 2048 - 2048) // 512 = 1 + 1872 // 512 = 4 frames
        self.assertEqual(frames.shape, (20, 4))

        feats = extractor.extract(y_short)
        self.assertEqual(feats.shape, (40,))
        self.assertTrue(np.all(np.isfinite(feats)))


if __name__ == "__main__":
    unittest.main()
