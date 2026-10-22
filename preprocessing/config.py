"""
preprocessing.config
====================
Centralised configuration for the multimodal alignment and preprocessing
pipeline.  All defaults are explicit; no Windows-specific paths are
hardcoded.

Usage
-----
    from preprocessing.config import AlignmentConfig, MELD_SCHEMA, MOSEI_SCHEMA

    cfg = AlignmentConfig(
        dataset="meld",
        manifest_path="datasets/MELD/manifests/train_manifest.csv",
        audio_root="<path-to-meld-audio>/train",
        video_root="<path-to-meld-raw>/train_splits",
        output_dir="outputs/meld/train",
    )
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset-specific schema constants
# ---------------------------------------------------------------------------

#: MELD manifest CSV column names produced by datasets/MELD/prepare_dataset.py
MELD_MANIFEST_COLUMNS = (
    "sample_id",
    "split",
    "dialogue_id",
    "utterance_id",
    "emotion",
    "sentiment",
    "video_path",
    "audio_path",
    "has_text",
    "has_video",
    "has_audio",
    "video_status",
    "audio_status",
    "is_fully_usable",
    "exclusion_reason",
)

#: MELD 7-class categorical emotion label space
MELD_EMOTIONS = frozenset(
    {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
)

#: MELD 3-class sentiment label space
MELD_SENTIMENTS = frozenset({"positive", "negative", "neutral"})

#: CMU-MOSEI manifest CSV column names produced by datasets/CMU-MOSEI/prepare_dataset.py
MOSEI_MANIFEST_COLUMNS = (
    "sample_id",
    "video_id",
    "clip_id",
    "start_time",
    "end_time",
    "sentiment",
    "sentiment_discrete",
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
    "dominant_emotion",
    "transcript",
    "split",
    "text_path",
    "audio_path",
    "visual_path",
    "has_text",
    "has_audio",
    "has_visual",
    "is_fully_usable",
    "exclusion_reason",
    "audio_status",
    "video_status",
)

#: CMU-MOSEI continuous emotion dimension names (intensity in [0.0, 3.0])
MOSEI_EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")

#: CMU-MOSEI continuous sentiment range
MOSEI_SENTIMENT_MIN: float = -3.0
MOSEI_SENTIMENT_MAX: float = 3.0

#: CMU-MOSEI continuous emotion intensity range
MOSEI_EMOTION_MIN: float = 0.0
MOSEI_EMOTION_MAX: float = 3.0

#: Valid split names shared across datasets
VALID_SPLITS = frozenset({"train", "dev", "test"})

#: Canonical output alignment record field names (dataset-agnostic)
ALIGNMENT_RECORD_FIELDS = (
    "dataset",
    "sample_id",
    "split",
    "dialogue_or_video_id",
    "segment_or_utterance_id",
    "start_time",
    "end_time",
    "text_raw",
    "text_normalized",
    "audio_ref",
    "visual_ref",
    "has_text",
    "has_audio",
    "has_visual",
    "is_fully_aligned",
    "target_label",
    "target_label_type",
    "alignment_status",
    "exclusion_reason",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AlignmentConfig:
    """
    Runtime configuration for the alignment and preprocessing pipeline.

    All path parameters accept ``str`` or ``pathlib.Path`` and are stored
    internally as ``pathlib.Path`` objects.  ``None`` means the corresponding
    resource is unavailable; the pipeline will record the modality as missing
    rather than raising an error.

    Parameters
    ----------
    dataset:
        One of ``"meld"`` or ``"mosei"``.
    manifest_path:
        Path to the pre-generated manifest CSV for one split (produced by
        ``datasets/MELD/prepare_dataset.py`` or
        ``datasets/CMU-MOSEI/prepare_dataset.py``).
    output_dir:
        Directory under which aligned records and validation reports are
        written.  Created on demand; never inside the source tree.
    audio_root:
        Root directory containing per-split audio files (WAV) for MELD, or
        precomputed COVAREP ``.npy`` feature arrays for CMU-MOSEI.  ``None``
        when audio is unavailable.
    video_root:
        Root directory containing raw MP4 video clips (MELD) or VisualFacet
        4.2 ``.npy`` arrays (CMU-MOSEI).  ``None`` when visual data is
        unavailable.
    text_root:
        Optional directory for per-sample text transcription files.  If
        ``None``, text is read from the manifest ``text`` / ``transcript``
        column or the ``Utterance`` field in the MELD annotation CSV.
    covarep_csd:
        Path to the CMU-MOSEI COVAREP CSD archive on the local filesystem.
        ``None`` for MELD or when the archive is unavailable.
    facet_csd:
        Path to the CMU-MOSEI VisualFacet 4.2 CSD archive.
        ``None`` for MELD or when the archive is unavailable.
    target_sample_rate:
        Target sampling rate (Hz) to report for audio resources.  No
        resampling is performed by this module; use this value only for
        downstream validation assertions.  Default: 16000.
    skip_media_verify:
        When ``True``, modality availability flags from the input manifest are
        accepted without re-checking the referenced files on disk.  Useful
        when media files live on a separate storage system.  Default: False.
    text_normalize:
        When ``True``, apply Unicode NFKC normalisation and whitespace
        collapsing to the text field.  Emotion-bearing punctuation is
        preserved.  Default: True.
    """

    dataset: str
    manifest_path: Path
    output_dir: Path

    # Optional modality roots (None = modality unavailable)
    audio_root: Optional[Path] = None
    video_root: Optional[Path] = None
    text_root: Optional[Path] = None
    transcript_source: Optional[Path] = None

    # CMU-MOSEI CSD archive paths (only relevant for mosei dataset)
    covarep_csd: Optional[Path] = None
    facet_csd: Optional[Path] = None

    # Processing options
    target_sample_rate: int = 16000
    skip_media_verify: bool = False
    text_normalize: bool = True

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower().strip()
        if self.dataset not in ("meld", "mosei"):
            raise ValueError(
                f"dataset must be 'meld' or 'mosei', got '{self.dataset}'"
            )

        self.manifest_path = Path(self.manifest_path)
        self.output_dir = Path(self.output_dir)

        if self.audio_root is not None:
            self.audio_root = Path(self.audio_root)
        if self.video_root is not None:
            self.video_root = Path(self.video_root)
        if self.text_root is not None:
            self.text_root = Path(self.text_root)
        if self.transcript_source is not None:
            self.transcript_source = Path(self.transcript_source)
        elif self.text_root is not None:
            self.transcript_source = Path(self.text_root)
        if self.covarep_csd is not None:
            self.covarep_csd = Path(self.covarep_csd)
        if self.facet_csd is not None:
            self.facet_csd = Path(self.facet_csd)

    def validate(self) -> None:
        """
        Raise ``FileNotFoundError`` if mandatory paths do not exist.

        Optional roots are checked only when they are set.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )
        if self.audio_root is not None and not self.audio_root.exists():
            raise FileNotFoundError(
                f"Audio root not found: {self.audio_root}"
            )
        if self.video_root is not None and not self.video_root.exists():
            raise FileNotFoundError(
                f"Video/visual root not found: {self.video_root}"
            )
        if self.transcript_source is not None and not self.transcript_source.exists():
            raise FileNotFoundError(
                f"Transcript source not found: {self.transcript_source}"
            )
        if self.covarep_csd is not None and not self.covarep_csd.exists():
            raise FileNotFoundError(
                f"COVAREP CSD archive not found: {self.covarep_csd}"
            )
        if self.facet_csd is not None and not self.facet_csd.exists():
            raise FileNotFoundError(
                f"VisualFacet CSD archive not found: {self.facet_csd}"
            )
