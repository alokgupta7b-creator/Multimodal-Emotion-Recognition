"""Facial dataset preparation with strict split and label validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from facial_emotion.config import CLASS_TO_INDEX, EMOTION_CLASSES


def load_fer_csv(path: Path) -> list[dict[str, Any]]:
    """Load a FER-style CSV with ``emotion`` and ``pixels`` columns.

    The loader accepts an optional ``Usage`` column but never creates splits;
    callers must use the source-provided train/validation/test assignment.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Facial dataset CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"emotion", "pixels"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Facial CSV must contain columns {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            raw_label = str(row["emotion"]).strip().lower()
            label = str(raw_label)
            if label.isdigit():
                index = int(label)
                if not 0 <= index < len(CLASS_TO_INDEX):
                    raise ValueError(f"Unknown numeric facial label at line {line_number}: {label}")
            elif label in CLASS_TO_INDEX:
                index = CLASS_TO_INDEX[label]
            else:
                raise ValueError(f"Unknown facial emotion at line {line_number}: {label}")
            pixels = [int(value) for value in str(row["pixels"]).split()]
            if not pixels or any(value < 0 or value > 255 for value in pixels):
                raise ValueError(f"Invalid pixel values at line {line_number}")
            rows.append({"label": index, "label_name": EMOTION_CLASSES[index], "pixels": pixels, "usage": row.get("Usage", "")})
    return rows


def find_local_facial_datasets(root: Path) -> list[Path]:
    """Find likely local FER-style CSVs without downloading anything."""
    root = Path(root)
    return sorted(path for path in root.rglob("*.csv") if path.name.lower() in {"fer2013.csv", "facial_emotions.csv"})
