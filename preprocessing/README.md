# preprocessing — Multimodal Alignment and Preprocessing Module

A dedicated, self-contained module for **cross-modal text, audio, and video
alignment and preprocessing** in multimodal emotion recognition research.

This module is independent of dataset acquisition scripts, speech-emotion
recognition models, and fusion networks.  It provides reusable adapters for
both supported benchmarks — **MELD** and **CMU-MOSEI** — without duplicating
their dataset preparation pipelines.

---

## 1. Purpose

Before any multimodal model can be trained or evaluated, text, audio, and
visual representations for each utterance or segment must be:

1. **Normalised** consistently across datasets.
2. **Resolved** to their actual file paths or precomputed feature references.
3. **Aligned** per sample using stable identifiers — never by CSV row position.
4. **Validated** for missing modalities, invalid timestamps, duplicate IDs, and
   cross-split leakage.
5. **Reported** with full accounting of inclusions, exclusions, and modality
   availability.

This module performs exactly those tasks and writes reproducible, auditable
output reports.

---

## 2. Folder Structure

```text
preprocessing/
├── __init__.py             # Package init; version constant
├── __main__.py             # CLI entry point (python -m preprocessing)
├── config.py               # AlignmentConfig dataclass; dataset schema constants
├── csd_reader.py           # Safe reusable HDF5 archive reader and context manager
├── preprocess_text.py      # Text normalisation & external transcript loader
├── preprocess_audio.py     # Audio reference resolution and WAV/array validation
├── preprocess_video.py     # Visual reference resolution and MP4/array validation
├── align_modalities.py     # Cross-modal alignment; AlignmentRecord dataclass
├── validate_alignment.py   # Alignment validation report; JSON/Markdown/CSV output
├── README.md               # This document
└── tests/
    ├── __init__.py
    ├── test_alignment.py       # Alignment pipeline unit tests
    ├── test_csd_reader.py      # CSDArchiveReader hierarchy, intervals & cleanup tests
    └── test_preprocessing.py   # Text, audio, video sub-module unit tests
```

---

## 3. Module Responsibilities

| Module | Responsibility |
| :--- | :--- |
| `config.py` | Centralised schema constants (column names, label sets, ranges) and the `AlignmentConfig` dataclass. |
| `csd_reader.py` | Safe reusable HDF5 reader; handles `root/data/video_id` hierarchy, interval overlaps, shape checks, and deterministic cleanup. |
| `preprocess_text.py` | Unicode NFKC normalisation, URL/email removal, whitespace collapse, punctuation preservation, and external transcript joining. |
| `preprocess_audio.py` | WAV header validation (pure Python, no soundfile), COVAREP CSD reference resolution, numpy array shape/NaN validation. |
| `preprocess_video.py` | MP4 ISO atom scanner (no ffprobe), VisualFacet 4.2 CSD reference resolution, numpy array validation. |
| `align_modalities.py` | Identity-based cross-modal matching, duplicate detection, split isolation checks, `AlignmentRecord` production. |
| `validate_alignment.py` | Aggregated report generation (JSON + Markdown + exclusions CSV). |
| `__main__.py` | Command-line interface. |


---

## 4. Input Formats

### 4.1 MELD

Input: a manifest CSV produced by `datasets/MELD/prepare_dataset.py`.

Required columns:
```
sample_id, split, dialogue_id, utterance_id, emotion, sentiment,
video_path, audio_path, has_text, has_video, has_audio,
video_status, audio_status, is_fully_usable, exclusion_reason
```

Optional columns (present when `prepare_dataset.py` ran with
`include_transcript_text`):
```
text, speaker, season, episode, start_time, end_time
```

- **Sample identifier**: `{split}_dia{D}_utt{U}` (e.g. `train_dia0_utt0`).
- **Audio**: WAV files under a configurable `audio_root` directory.
- **Video**: MP4 clips under a configurable `video_root` directory.
- **Text**: Utterance string in the `text` column (or falls back to manifest flags).

### 4.2 CMU-MOSEI

Input: a manifest CSV produced by `datasets/CMU-MOSEI/prepare_dataset.py`.

Required columns:
```
sample_id, video_id, clip_id, start_time, end_time,
sentiment, sentiment_discrete, happy, sad, anger, surprise, disgust, fear,
dominant_emotion, transcript, split,
text_path, audio_path, visual_path,
has_text, has_audio, has_visual, is_fully_usable, exclusion_reason,
audio_status, video_status
```

- **Sample identifier**: `{video_id}_{clip_id}`.
- **Timestamps**: Continuous `[start_time, end_time]` in seconds.
- **Audio**: COVAREP precomputed features via `CMU_MOSEI_COVAREP.csd` or
  per-sample NPY arrays under a configurable `audio_root`.
- **Visual**: VisualFacet 4.2 features via `CMU_MOSEI_VisualFacet42.csd` or
  per-sample NPY arrays under a configurable `video_root`.
- **Text**: Utterance string in the `transcript` column.

---

## 5. How to Run

### 5.1 MELD — manifest flags only (no media files available)

```bash
python -m preprocessing \
    --dataset meld \
    --manifest datasets/MELD/manifests/train_manifest.csv \
    --output-dir outputs/alignment/meld/train \
    --skip-media-verify
```

### 5.2 MELD — with audio and video roots (and optional transcript source)

```bash
python -m preprocessing \
    --dataset meld \
    --manifest datasets/MELD/manifests/train_manifest.csv \
    --transcript-source datasets/MELD/annotations \
    --audio-root <path-to-MELD.Raw>/audio/train \
    --video-root <path-to-MELD.Raw>/train_splits \
    --output-dir outputs/alignment/meld/train \
    --expected-split train
```

If `--transcript-source` is omitted and the manifest lacks a `text` column,
the module marks text as `text_manifest_omitted_upstream_available` rather than
fabricating text from flags.


### 5.3 CMU-MOSEI — manifest flags only

```bash
python -m preprocessing \
    --dataset mosei \
    --manifest datasets/CMU-MOSEI/manifests/train_manifest.csv \
    --output-dir outputs/alignment/mosei/train \
    --skip-media-verify \
    --expected-split train
```

### 5.4 CMU-MOSEI — with CSD archives

```bash
python -m preprocessing \
    --dataset mosei \
    --manifest datasets/CMU-MOSEI/manifests/train_manifest.csv \
    --covarep-csd <path-to>/CMU_MOSEI_COVAREP.csd \
    --facet-csd  <path-to>/CMU_MOSEI_VisualFacet42.csd \
    --output-dir outputs/alignment/mosei/train \
    --expected-split train
```

All paths are provided at runtime.  No Windows-specific or user-specific
paths are hardcoded anywhere in this module.

---

## 6. Output Schema

Three files are written to `--output-dir` per run:

### `alignment_report.json`
A machine-readable JSON summary containing all count fields defined in
`AlignmentReport`:
- `input_count`, `processed_count`, `duplicate_count`
- `has_text_count`, `has_audio_count`, `has_visual_count`
- `missing_text_count`, `missing_audio_count`, `missing_visual_count`
- `fully_aligned_count`, `partially_aligned_count`, `excluded_count`
- `exclusion_breakdown` (dict: reason → count)
- `splits_found`, `cross_split_violations`, `warnings`

### `alignment_report.md`
Human-readable Markdown summary of the same counts.

### `alignment_exclusions.csv`
Per-sample exclusion log with columns:
```
sample_id, dataset, split, dialogue_or_video_id, segment_or_utterance_id,
has_text, has_audio, has_visual, alignment_status, exclusion_reason
```

Only records with `alignment_status != "ok"` appear here.

> [!IMPORTANT]
> **Fully aligned** means text + audio + visual are all present **and** no
> exclusion reason exists.  Records with one or more missing modalities are
> reported as **partially aligned** and are **not** counted in
> `fully_aligned_count`.

---

## 7. How to Run Tests

```bash
# Preprocessing module regression tests (90 tests, no real dataset required)
python -m unittest discover -s preprocessing/tests -p "test_*.py" -v

# Existing CMU-MOSEI dataset preparation tests (12 tests)
python -m unittest discover -s datasets/CMU-MOSEI/tests -p "test_*.py" -v
```

Expected results:
- preprocessing/tests: **90 tests, 0 failures**
- datasets/CMU-MOSEI/tests: **12 tests, 0 failures**

All tests use synthetic fixtures.  Real dataset files (CSD archives, WAV clips,
MP4 files) are not required.

---

## 8. Missing-Modality and Exclusion Policy

| Situation | Outcome |
| :--- | :--- |
| No `audio_root` or `covarep_csd` provided | `has_audio` is read from manifest flag; modality not re-verified |
| No `video_root` or `facet_csd` provided | `has_visual` is read from manifest flag; modality not re-verified |
| `--skip-media-verify` set | All modality flags are read from manifest; no file I/O |
| Manifest lacks `text` column but `has_text=True` | `exclusion_reason=text_manifest_omitted_upstream_available`; not fully aligned |
| Manifest has `--transcript-source` supplied | Transcripts joined on `(dialogue_id, utterance_id)` and split; `has_text=True` |
| Text field is absent, null, or empty | `has_text=False`; record excluded if fully-aligned count required |
| Inverted or negative MOSEI timestamps | Record excluded; `exclusion_reason=invalid_timestamps_*` |
| Duplicate `sample_id` within a manifest | Second occurrence is skipped; warning logged |
| Record from wrong split | `exclusion_reason=split_isolation_violation_*`; flagged in report |


Partially available samples (some modalities missing) are **retained** in
the output alignment records so that unimodal or bimodal training experiments
remain possible.  Only `is_fully_aligned = False` indicates that not all
modalities are available.

---

## 9. Relationship to Other Modules

| Module | Relationship |
| :--- | :--- |
| `datasets/MELD/` | **Input only**: this module reads MELD manifests. It does not modify MELD scripts or annotations. |
| `datasets/CMU-MOSEI/` | **Input only**: this module reads CMU-MOSEI manifests. It does not modify CMU-MOSEI scripts. |
| `text_emotion/` | **Independent**: the existing TF-IDF text baseline is separate. This module's text normalisation is compatible with downstream transformer-based extractors. |
| `speech_emotion/` (future) | **Consumer**: will consume `audio_ref` paths from `AlignmentRecord` objects. |
| `facial_emotion/` (future) | **Consumer**: will consume `visual_ref` paths from `AlignmentRecord` objects. |
| `fusion/` (future) | **Consumer**: will receive aligned record batches to compute reliability and consistency weights. |

---

## 10. Dependencies

This module requires only Python 3.8+ standard library for core functionality.

| Package | Purpose | Required? |
| :--- | :--- | :--- |
| `numpy` | Array shape/NaN validation in `validate_audio_array` / `validate_visual_array` | Optional (graceful degradation if absent) |
| `h5py` | CSD archive inspection in `resolve_mosei_audio` / `resolve_mosei_visual` | Optional (graceful degradation if absent) |

No model weights, GPU, or training infrastructure is required by this module.
