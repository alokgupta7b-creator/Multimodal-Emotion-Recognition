"""Configuration for the MELD facial emotion recognition branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from preprocessing.config import MELD_EMOTIONS
MELD_SPLITS = ("train", "dev", "test")


@dataclass
class FacialConfig:
    """Settings for facial landmark extraction from MELD videos."""

    video_root: Path = Path(r"D:\Datasets\MELD\MELD.Raw")
    manifest_root: Path = Path("datasets/MELD/manifests")
    output_dir: Path = Path(r"D:\MELD_Facial_Output")

    frame_sample_count: int = 10
    face_detection_confidence: float = 0.3
    face_crop_padding: float = 1.4
    face_crop_size: int = 256

    def __post_init__(self) -> None:
        self.video_root = Path(self.video_root)
        self.manifest_root = Path(self.manifest_root)
        self.output_dir = Path(self.output_dir)

        if self.frame_sample_count <= 0:
            raise ValueError("frame_sample_count must be positive")
        if not 0.0 < self.face_detection_confidence <= 1.0:
            raise ValueError("face_detection_confidence must be in (0, 1]")
        if self.face_crop_padding < 1.0:
            raise ValueError("face_crop_padding must be >= 1.0")
        if self.face_crop_size <= 0:
            raise ValueError("face_crop_size must be positive")

    def manifest_path(self, split: str) -> Path:
        if split not in MELD_SPLITS:
            raise ValueError(f"Invalid MELD split: {split}")
        return self.manifest_root / f"{split}_manifest.csv"

    def validate(self, split: str | None = None) -> None:
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        if split is not None:
            path = self.manifest_path(split)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest not found: {path}")


EMOTION_LABELS = tuple(sorted(MELD_EMOTIONS))

