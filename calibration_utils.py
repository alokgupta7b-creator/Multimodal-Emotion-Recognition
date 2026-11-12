"""Shared post-hoc probability calibration helpers for modality predictors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LogitCalibration:
    """Affine logit calibration transform: logits' = (logits / temperature) + bias."""

    temperature: float
    bias: np.ndarray
    method: str = "temp_plus_l2_bias"
    source_path: Path | None = None


def resolve_calibration_path(
    model_path: Path,
    explicit_path: Path | str | None = None,
    use_calibration: bool = True,
) -> Path | None:
    """Resolve explicit or default calibration artifact path without altering model weights."""
    if not use_calibration:
        return None
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Calibration file not found: {candidate}")
        return candidate

    candidates: list[Path] = []
    if model_path.is_dir():
        candidates.append(model_path / "calibration.json")
        if model_path.name == "best":
            candidates.append(model_path.parent / "calibration_p1.json")
    elif model_path.is_file():
        candidates.append(model_path.parent / "calibration.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_logit_calibration(
    calibration_path: Path | str,
    expected_classes: Sequence[str],
) -> LogitCalibration:
    """Load and validate a canonical-order logit calibration artifact."""
    path = Path(calibration_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected = [str(label).strip().lower() for label in expected_classes]
    raw_classes = payload.get("classes")
    if raw_classes is not None:
        actual = [str(label).strip().lower() for label in raw_classes]
        if actual != expected:
            raise ValueError(f"Calibration class order mismatch in {path}: {actual} != {expected}")

    temperature = float(payload.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Invalid calibration temperature {temperature!r} in {path}")

    raw_bias = payload.get("bias", [0.0] * len(expected))
    bias = np.asarray(raw_bias, dtype=np.float64)
    if bias.shape != (len(expected),) or not np.isfinite(bias).all():
        raise ValueError(f"Invalid calibration bias vector in {path}: shape={bias.shape}")

    return LogitCalibration(
        temperature=temperature,
        bias=bias,
        method=str(payload.get("method", "custom")),
        source_path=path,
    )


def apply_logit_calibration(
    logits: np.ndarray,
    calibration: LogitCalibration | None,
) -> np.ndarray:
    """Apply validated (logits / temperature) + bias transform when configured."""
    arr = np.asarray(logits, dtype=np.float64)
    if calibration is None:
        return arr
    return (arr / calibration.temperature) + calibration.bias

