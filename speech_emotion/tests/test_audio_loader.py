"""
speech_emotion/tests/test_audio_loader.py
=========================================
Automated tests for audio loading, validation, channel downmix, and resampling.
All tests use synthetic audio fixtures; no external downloads required.
"""

from __future__ import annotations

import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from speech_emotion.audio_loader import (
    AudioLoadStatus,
    AudioMetadata,
    load_audio,
    resample_signal,
    validate_audio_signal,
    write_wav,
)


class TestAudioLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_audio_loader_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_synthetic_tone(
        self,
        duration: float = 1.0,
        sr: int = 16000,
        freq: float = 440.0,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    def test_load_valid_wav_mono(self) -> None:
        """Valid 16-bit PCM mono WAV loads correctly with accurate metadata."""
        y = self._generate_synthetic_tone(duration=1.5, sr=16000, freq=440.0)
        wav_path = self.temp_dir / "valid_mono.wav"
        write_wav(wav_path, y, sample_rate=16000, channels=1)

        loaded_y, meta = load_audio(wav_path, target_sr=16000, mono=True)

        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.OK)
        self.assertEqual(meta.sample_rate, 16000)
        self.assertEqual(meta.channels, 1)
        self.assertTrue(meta.is_mono)
        self.assertFalse(meta.resampled)
        self.assertAlmostEqual(meta.duration_seconds, 1.5, places=2)
        self.assertEqual(len(loaded_y), len(y))
        # Correlation with original signal should be close to 1.0
        corr = np.corrcoef(loaded_y, y)[0, 1]
        self.assertGreater(corr, 0.999)

    def test_load_stereo_converts_to_mono(self) -> None:
        """Stereo WAV is converted to mono by averaging channels (L + R) / 2."""
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        left = 0.6 * np.sin(2.0 * np.pi * 440.0 * t)
        right = 0.2 * np.sin(2.0 * np.pi * 880.0 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)

        wav_path = self.temp_dir / "stereo.wav"
        write_wav(wav_path, stereo, sample_rate=16000, channels=2)

        loaded_y, meta = load_audio(wav_path, target_sr=16000, mono=True)

        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.channels, 2)
        self.assertTrue(meta.is_mono)
        self.assertEqual(loaded_y.ndim, 1)

        # Expected mono is (L + R) / 2
        expected_mono = 0.5 * (left + right)
        max_diff = np.max(np.abs(loaded_y - expected_mono))
        self.assertLess(max_diff, 1e-3)

    def test_resampling_from_44100_to_16000(self) -> None:
        """Audio sampled at 44.1 kHz is resampled to 16 kHz when allow_resample=True."""
        y_44k = self._generate_synthetic_tone(duration=1.0, sr=44100, freq=300.0)
        wav_path = self.temp_dir / "sr44k.wav"
        write_wav(wav_path, y_44k, sample_rate=44100, channels=1)

        loaded_y, meta = load_audio(wav_path, target_sr=16000, allow_resample=True)

        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.sample_rate, 16000)
        self.assertEqual(meta.original_sample_rate, 44100)
        self.assertTrue(meta.resampled)
        self.assertEqual(len(loaded_y), 16000)
        self.assertAlmostEqual(meta.duration_seconds, 1.0, places=2)

    def test_resampling_from_8000_to_16000(self) -> None:
        """Audio sampled at 8 kHz is upsampled to 16 kHz."""
        y_8k = self._generate_synthetic_tone(duration=0.5, sr=8000, freq=400.0)
        wav_path = self.temp_dir / "sr8k.wav"
        write_wav(wav_path, y_8k, sample_rate=8000, channels=1)

        loaded_y, meta = load_audio(wav_path, target_sr=16000, allow_resample=True)

        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.sample_rate, 16000)
        self.assertEqual(len(loaded_y), 8000)

    def test_resampling_disabled_flags_unsupported_sample_rate(self) -> None:
        """When allow_resample=False, mismatched sample rate is flagged invalid."""
        y_44k = self._generate_synthetic_tone(duration=1.0, sr=44100)
        wav_path = self.temp_dir / "sr44k_no_resample.wav"
        write_wav(wav_path, y_44k, sample_rate=44100)

        loaded_y, meta = load_audio(wav_path, target_sr=16000, allow_resample=False)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.UNSUPPORTED_SAMPLE_RATE)
        self.assertIn("allow_resample is False", meta.reason)

    def test_missing_file_returns_file_not_found(self) -> None:
        """Non-existent audio file returns FILE_NOT_FOUND status."""
        missing_path = self.temp_dir / "does_not_exist.wav"
        loaded_y, meta = load_audio(missing_path)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.FILE_NOT_FOUND)
        self.assertEqual(len(loaded_y), 0)

    def test_empty_file_returns_empty_file_status(self) -> None:
        """0-byte file returns EMPTY_FILE status."""
        empty_path = self.temp_dir / "empty.wav"
        empty_path.write_bytes(b"")

        loaded_y, meta = load_audio(empty_path)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.EMPTY_FILE)
        self.assertEqual(len(loaded_y), 0)

    def test_truncated_header_returns_unreadable_header(self) -> None:
        """File with invalid or corrupt WAV header (>44 bytes) is caught gracefully."""
        corrupt_path = self.temp_dir / "corrupted_header.wav"
        # 64 bytes starting with RIFF but invalid chunks inside
        corrupt_path.write_bytes(b"RIFF" + struct.pack("<I", 56) + b"WAVEfmt " + b"\x00" * 48)

        loaded_y, meta = load_audio(corrupt_path)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.UNREADABLE_HEADER)

    def test_silent_audio_flagged(self) -> None:
        """Silent audio (all zeros) is flagged as SILENT_AUDIO."""
        silence = np.zeros(16000, dtype=np.float32)
        wav_path = self.temp_dir / "silence.wav"
        write_wav(wav_path, silence, sample_rate=16000)

        loaded_y, meta = load_audio(wav_path, silence_threshold_db=-60.0)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.SILENT_AUDIO)
        self.assertIn("RMS energy", meta.reason)

    def test_audio_too_short_flagged(self) -> None:
        """Audio shorter than min_duration is flagged as DURATION_TOO_SHORT."""
        short_y = self._generate_synthetic_tone(duration=0.05, sr=16000)
        wav_path = self.temp_dir / "too_short.wav"
        write_wav(wav_path, short_y, sample_rate=16000)

        loaded_y, meta = load_audio(wav_path, min_duration=0.1)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.DURATION_TOO_SHORT)

    def test_audio_too_long_flagged(self) -> None:
        """Audio exceeding max_duration is flagged as DURATION_TOO_LONG."""
        long_y = self._generate_synthetic_tone(duration=2.5, sr=16000)
        wav_path = self.temp_dir / "too_long.wav"
        write_wav(wav_path, long_y, sample_rate=16000)

        loaded_y, meta = load_audio(wav_path, max_duration=2.0)

        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.status, AudioLoadStatus.DURATION_TOO_LONG)

    def test_validate_audio_signal_non_finite_values(self) -> None:
        """Waveform with NaN or Inf is rejected."""
        y_nan = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        is_valid, status, reason, _ = validate_audio_signal(y_nan, sr=16000)

        self.assertFalse(is_valid)
        self.assertEqual(status, AudioLoadStatus.NON_FINITE_VALUES)

        y_inf = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        is_valid, status, reason, _ = validate_audio_signal(y_inf, sr=16000)

        self.assertFalse(is_valid)
        self.assertEqual(status, AudioLoadStatus.NON_FINITE_VALUES)

    def test_validate_audio_signal_zero_samples(self) -> None:
        """0-sample array is rejected as ZERO_FRAMES."""
        is_valid, status, reason, _ = validate_audio_signal(np.zeros(0, dtype=np.float32), sr=16000)
        self.assertFalse(is_valid)
        self.assertEqual(status, AudioLoadStatus.ZERO_FRAMES)


if __name__ == "__main__":
    unittest.main()
