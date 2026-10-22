# speech_emotion — Audio Preprocessing and Acoustic Feature Extraction (Task 4)

A dedicated, self-contained module for **speech audio validation, preprocessing, and acoustic feature extraction** for Multimodal Emotion Recognition research.

This module completes **Task 4: Audio Preprocessing and Feature Extraction**. It operates independently of dataset acquisition scripts and model training pipelines.

---

## 1. Scope and Purpose

The primary research objective of this repository is to develop a dynamic reliability and consistency estimation mechanism across text, speech, and facial modalities prior to fusion.

Before audio signals can be consumed by downstream acoustic emotion encoders or multimodal fusion networks, raw conversational speech recordings must be:
1. **Validated** for file existence, valid audio headers, finite numerical values (no NaN/Inf), non-zero length, and non-silent signal content.
2. **Standardized** to mono channel format and a consistent target sampling rate (16 kHz).
3. **Represented** using noise-robust acoustic feature descriptors (Mel-Frequency Cepstral Coefficients, MFCCs) with utterance-level temporal pooling.
4. **Aligned** strictly to source dialogue turns and ground-truth emotion labels without random repartitioning or cross-split leakage.
5. **Audited** with machine-readable JSON, Markdown logs, and per-sample exclusion records.

---

## 2. Architecture & File Structure

```text
speech_emotion/
├── README.md               # This documentation
├── __init__.py             # Public API exports
├── __main__.py             # CLI entry point (python -m speech_emotion)
├── config.py               # AudioConfig dataclass and MELD constants
├── audio_loader.py         # Native WAV decoder, channel downmix, resampling & validation
├── feature_extraction.py   # MFCC feature extractor (dual-engine: NumPy & librosa)
├── preprocess_audio.py     # Dataset integration, batch processor, deduplication & reports
└── tests/
    ├── __init__.py
    ├── test_audio_loader.py        # Audio loader, stereo downmix, and corruption tests
    ├── test_feature_extraction.py  # MFCC 40-feature shape, math, and consistency tests
    └── test_preprocess_audio.py    # Manifest parsing, duplicate & failure audit tests
```

---

## 3. Implementation Details

### 3.1 Audio Loading and Integrity Validation (`audio_loader.py`)
- **Native WAV Decoder**: Uses Python standard library `wave` and `numpy` to parse 16-bit, 24-bit, 32-bit float, and 8-bit PCM audio without requiring external C libraries.
- **Stereo-to-Mono Conversion**: Multi-channel inputs are downmixed by averaging channels across time:
  $$\text{mono}[t] = \frac{1}{C} \sum_{c=1}^C \text{channel}_c[t]$$
- **Resampling**: If the input sampling rate differs from the target (16,000 Hz), high-quality polyphase or linear interpolation resamples the signal.
- **Signal Validation**:
  - Rejects 0-sample or missing recordings (`zero_frames`, `file_not_found`).
  - Rejects truncated or corrupted headers (`unreadable_header`).
  - Detects non-finite values (`non_finite_values`).
  - Flags recordings with duration $< 0.10$ s (`duration_too_short`) or $> 60.0$ s (`duration_too_long`).
  - Computes RMS energy and flags recordings below the silence threshold (default $-60.0$ dB) as `silent_audio`.

### 3.2 Feature Extraction (`feature_extraction.py`)
- **MFCC Convention**: Extracts 20 Mel-Frequency Cepstral Coefficients per frame and applies utterance-level temporal pooling:
  - 20 coefficient means: $\mu_1, \mu_2, \dots, \mu_{20}$
  - 20 coefficient standard deviations: $\sigma_1, \sigma_2, \dots, \sigma_{20}$
  - Total feature dimension: **40 scalar features per utterance**.
- **Dual-Engine Design**:
  - `auto`: Automatically uses `librosa` if installed in the environment; otherwise seamlessly falls back to the native high-precision pure NumPy engine.
  - `numpy`: High-precision Slaney Mel filterbank (128 bands, 0 to 8000 Hz), STFT power spectrum with periodic Hann window and reflection padding, logarithmic compression, and orthogonal DCT-II.
  - `librosa`: Direct call to `librosa.feature.mfcc` for exact floating-point compatibility with previous baseline runs.

### 3.3 Dataset Integration & Auditing (`preprocess_audio.py`)
- Reads official MELD split manifests (`train_manifest.csv`, `dev_manifest.csv`, `test_manifest.csv`) or annotation CSVs.
- Preserves `Dialogue_ID`, `Utterance_ID`, `sample_id`, and `emotion` labels.
- Detects duplicate sample IDs and logs them as `duplicate_sample_id`.
- Never randomly repartitions data or fits dataset-wide transformations using validation/test splits.
- Writes structured outputs to `--output-dir` outside version control.

---

## 4. Configuration Parameters

The `AudioConfig` dataclass in `speech_emotion/config.py` provides full configurability:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `sample_rate` | `16000` | Target audio sampling rate in Hz |
| `mono` | `True` | Convert multi-channel audio to mono |
| `n_mfcc` | `20` | Number of MFCC coefficients per frame |
| `n_fft` | `2048` | FFT analysis window length in samples |
| `hop_length` | `512` | Frame hop step in samples (32 ms window, 16 ms hop at 16 kHz) |
| `n_mels` | `128` | Number of Mel filter bands |
| `fmin` | `0.0` | Lowest filter frequency in Hz |
| `fmax` | `8000.0` | Highest filter frequency in Hz (Nyquist) |
| `min_duration` | `0.1` | Minimum valid duration in seconds |
| `max_duration` | `60.0` | Maximum valid duration in seconds |
| `silence_threshold_db` | `-60.0` | RMS threshold for silence detection |
| `feature_layout` | `"mean_std"` | Pooling: `"mean_std"` (40 feats), `"mean_only"` (20), `"std_only"` (20) |
| `engine` | `"auto"` | `"auto"`, `"numpy"`, or `"librosa"` |

---

## 5. Output Schema

For each processed split, four files are written to `--output-dir`:

### 1. `{split}_mfcc.csv`
Tabular feature matrix with exactly 43 columns:
- `Dialogue_ID`: integer dialogue identifier.
- `Utterance_ID`: integer utterance turn identifier.
- `emotion`: categorical emotion label (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
- `mfcc_mean_01` .. `mfcc_mean_20`: mean of each cepstral coefficient across all frames.
- `mfcc_std_01` .. `mfcc_std_20`: standard deviation of each cepstral coefficient across all frames.

### 2. `{split}_mfcc_failures.csv`
Log of all excluded recordings with columns:
- `split`, `Dialogue_ID`, `Utterance_ID`, `reason`

### 3. `{split}_audio_audit.json`
Machine-readable audit report containing total counts, valid counts, missing audio counts, silence counts, emotion distributions, feature dimensions, and configuration details.

### 4. `{split}_audio_audit.md`
Human-readable Markdown audit report with formatted summary tables.

---

## 6. How to Run

### 6.1 Running Unit Tests

The test suite uses synthetic audio fixtures and requires no downloaded datasets:

```bash
# Run all speech_emotion tests (35 tests)
python -m unittest discover -s speech_emotion/tests -p "test_*.py" -v

# Run existing repository regression tests
python -m unittest discover -s preprocessing/tests -p "test_*.py" -v
python -m unittest discover -s datasets/CMU-MOSEI/tests -p "test_*.py" -v
```

### 6.2 Extracting Features on MELD Splits

```bash
# Process all splits (train, dev, test)
python -m speech_emotion \
    --split all \
    --audio-dir <path-to-meld-audio> \
    --manifest-dir datasets/MELD/manifests \
    --output-dir outputs/speech_features

# Process a single split with a sample limit for testing
python -m speech_emotion \
    --split dev \
    --audio-dir <path-to-meld-audio>/dev \
    --manifest-dir datasets/MELD/manifests \
    --output-dir outputs/speech_features \
    --limit 25
```

---

## 7. Verification & Accounting Summary

### 7.1 MELD Full-Dataset Verification Accounting

| Split | Total Annotations | Available Audio | Valid Features | Excluded | Exclusion Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 9,989 | 9,988 | **9,971** | 18 | 1 missing audio (`dia125_utt3`), 5 silent, 12 duration $<0.1$s |
| **Dev** | 1,109 | 1,108 | **1,105** | 4 | 1 missing audio (`dia110_utt7`), 1 silent, 2 duration $<0.1$s |
| **Test** | 2,610 | 2,610 | **2,606** | 4 | 0 missing, 2 silent, 2 duration $>60$s |
| **Total** | 13,708 | 13,706 | **13,682** | 26 | 2 missing audio, 8 silent, 14 duration $<0.1$s, 2 duration $>60$s |

### 7.2 Numerical Comparison with Existing MFCC Baseline

Comparison of features extracted by `speech_emotion` against existing MELD MFCC artifacts across all 1,105 aligned development utterances:
- **Columns & Schema**: 100% identical (43 columns: 3 metadata + 40 features).
- **Utterance & Emotion Alignment**: 100% matched across all records (`Dialogue_ID`, `Utterance_ID`).
- **Numerical Parity**:
  - Pearson correlation $r = 1.000000$ across all 40 coefficients.
  - Overall Mean Absolute Error (MAE): $8.14 \times 10^{-6}$.
  - Overall Median Absolute Error: $3.81 \times 10^{-6}$.
  - Overall Root Mean Squared Error (RMSE): $1.36 \times 10^{-5}$.
  - Overall Maximum Absolute Error: $1.22 \times 10^{-4}$ (within float32 precision limits).
