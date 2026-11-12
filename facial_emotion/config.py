from __future__ import annotations

from pathlib import Path

from speech_emotion.model.label_mapping import MELD_CLASS_NAMES

EMOTION_CLASSES = MELD_CLASS_NAMES
CLASS_TO_INDEX = {name: index for index, name in enumerate(EMOTION_CLASSES)}
FACIAL_MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "facial_emotion"
DEFAULT_IMAGE_SIZE = 96
