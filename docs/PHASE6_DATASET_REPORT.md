# Phase 6 MELD Dataset Acquisition and Preparation

## Acquisition

- Source URL: `https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`
- Source: official MELD download linked by [affective-meld.github.io](https://affective-meld.github.io/)
- Archive path: `datasets/MELD_raw_acquisition/MELD.Raw.tar.gz`
- Archive size: `10,878,146,150` bytes
- SHA-256: `A56B4407D574195CBCE470D86F9C9D72FCFEA59B0E34502ECD4BABEE4A5C613E`
- Official checksum: no checksum was published on the download page; the SHA-256 above was computed locally.
- Outer extraction: `datasets/MELD_raw_acquisition/extracted/MELD.Raw`
- Media extraction: `datasets/MELD_raw_acquisition/media`
- Prepared output: `datasets/MELD_raw_acquisition/prepared`
- Acquisition directory usage at validation: `34,074,544,652` bytes

The original archive remains preserved. Existing `datasets/MELD` annotations,
manifests, models and reports were not overwritten. The audiovisual material
must not be redistributed and remains subject to applicable MELD and source
media usage/copyright conditions.

## Per-split results

| Split | Manifest rows | Valid videos | Video failures | Valid WAVs | Audio failures |
|---|---:|---:|---:|---:|---:|
| Train | 9,989 | 9,988 | 1 | 9,988 | 1 |
| Dev | 1,109 | 1,108 | 1 | 1,108 | 1 |
| Test | 2,610 | 2,610 | 0 | 2,610 | 0 |
| **Total** | **13,708** | **13,706** | **2** | **13,706** | **2** |

The two missing WAVs are not unexplained:

1. `train_dia125_utt3`: source video exists but is corrupt; FFmpeg reports
   `moov atom not found`.
2. `dev_dia110_utt7`: source video is absent from the official extracted
   media tree.

These records are preserved in:

- `datasets/MELD_raw_acquisition/prepared/reports/train/video_failures.csv`
- `datasets/MELD_raw_acquisition/prepared/reports/dev/video_failures.csv`
- `datasets/MELD_raw_acquisition/prepared/reports/audio_conversion_failures.csv`

## Audio integrity

All 13,706 generated WAV files were checked with SoundFile:

- Zero-byte files: 0
- Unreadable files: 0
- Incorrect sample rate: 0
- Incorrect channel count: 0
- Non-positive durations: 0

Outputs are mono, 16,000 Hz PCM WAV files. The extraction script is
resumable: valid existing outputs are reported as `reused` and are not
overwritten; only missing or invalid outputs are attempted.

## Alignment and leakage

- All manifest rows have explicit audio/video processing status in the
  per-split reports.
- Sample IDs are unique within each split.
- No cross-split sample-ID overlap was reported.
- All seven expected emotion labels are present and valid.
- Audio paths are generated from the same dialogue/utterance IDs as the
  corresponding video and annotation rows.
- Basename repetition across split directories is expected because MELD
  dialogue IDs restart by split; full split-qualified sample IDs remain
  unique.
- Final validator status: **not valid**, solely because the two source-media
  failures above leave two manifest rows without both modalities.

Machine-readable reports:

- `datasets/MELD_raw_acquisition/prepared/reports/audio_integrity.json`
- `datasets/MELD_raw_acquisition/prepared/reports/multimodal_validation.json`
- Per-split `video_manifest.csv`, `video_failures.csv`, `audio_manifest.csv`
  and `audio_failures.csv`

No models were trained and no multimodal fusion was implemented.
