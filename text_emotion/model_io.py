"""Trusted loading and persistence helpers for the text baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from project_config import REPOSITORY_ROOT, TEXT_LABELS_PATH, TEXT_METADATA_PATH


def _trusted_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to load model outside the repository: {resolved}") from exc
    if resolved.suffix != ".joblib":
        raise ValueError(f"Unexpected model artifact type: {resolved}")
    return resolved


def load_text_model(path: Path) -> Any:
    """Load a repository-local text artifact.

    Joblib is pickle-based; callers must not pass artifacts from untrusted
    sources. The repository-bound path check prevents accidental external
    loads but cannot make a malicious local pickle safe.
    """
    trusted = _trusted_path(Path(path))
    if not trusted.is_file():
        raise FileNotFoundError(f"Text model artifact not found at {trusted}")
    return joblib.load(trusted)


def write_model_metadata(model: Any, model_path: Path, sample_counts: dict[str, int]) -> None:
    """Persist class mapping and reproducibility metadata beside the model."""
    labels = [str(label) for label in model.classes_]
    TEXT_LABELS_PATH.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "format": "text-tfidf-logistic-regression",
        "classes": labels,
        "model_sha256": digest,
        "sample_counts": sample_counts,
        "preprocessing": "text_emotion.preprocessing.preprocess_text",
    }
    TEXT_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
