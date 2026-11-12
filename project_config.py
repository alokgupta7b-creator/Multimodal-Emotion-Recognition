"""Repository-wide paths and dataset configuration.

All paths are resolved from this file's location, so commands work from any
current working directory on Windows or Linux.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


REPOSITORY_ROOT = Path(__file__).resolve().parent
MELD_ROOT = REPOSITORY_ROOT / "datasets" / "MELD"
MELD_ANNOTATIONS = MELD_ROOT / "annotations"
MELD_MANIFESTS = MELD_ROOT / "manifests"
MELD_ACQUISITION_ROOT = MELD_ROOT.parent / "MELD_raw_acquisition"
MELD_ARCHIVE_PATH = MELD_ACQUISITION_ROOT / "MELD.Raw.tar.gz"
MELD_EXTRACTED_ROOT = MELD_ACQUISITION_ROOT / "extracted"
MELD_MEDIA_ROOT = MELD_ACQUISITION_ROOT / "media"
MELD_PREPARED_ROOT = MELD_ACQUISITION_ROOT / "prepared"
TEXT_MODEL_PATH = REPOSITORY_ROOT / "text_emotion_model.joblib"
TEXT_METADATA_PATH = REPOSITORY_ROOT / "text_emotion_model.metadata.json"
TEXT_LABELS_PATH = REPOSITORY_ROOT / "text_emotion_labels.json"

MELD_SPLITS = ("train", "dev", "test")


def meld_annotation_path(split: str) -> Path:
    """Return and validate the annotation path for a MELD split."""
    if split not in MELD_SPLITS:
        raise ValueError(f"Unknown MELD split {split!r}; expected one of {MELD_SPLITS}")
    path = MELD_ANNOTATIONS / f"{split}_sent_emo.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"MELD {split} annotations were not found at {path}. "
            "Place the official annotation CSVs in datasets/MELD/annotations."
        )
    return path


def meld_manifest_path(split: str) -> Path:
    """Return and validate the generated manifest path for a MELD split."""
    if split not in MELD_SPLITS:
        raise ValueError(f"Unknown MELD split {split!r}; expected one of {MELD_SPLITS}")
    path = MELD_MANIFESTS / f"{split}_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"MELD {split} manifest was not found at {path}.")
    return path


def validate_meld_dataset(require_manifests: bool = False) -> Dict[str, Path]:
    """Validate the repository-local MELD layout and return its key paths."""
    if not MELD_ROOT.is_dir():
        raise FileNotFoundError(f"MELD dataset directory was not found at {MELD_ROOT}.")
    paths: Dict[str, Path] = {}
    for split in MELD_SPLITS:
        paths[f"{split}_annotations"] = meld_annotation_path(split)
        if require_manifests:
            paths[f"{split}_manifest"] = meld_manifest_path(split)
    return paths
