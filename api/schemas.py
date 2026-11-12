"""Pydantic response and request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class PredictionResponse(BaseModel):
    predicted_emotion: str
    probabilities: dict[str, float]
    confidence: float
    inference_duration_ms: float
    modality: str
    modality_status: str = "ok"
    quality: dict[str, Any] = Field(default_factory=dict)


class ModelStatus(BaseModel):
    loaded: bool
    model_identifier: str | None = None
    device: str | None = None
    error: str | None = None
    architecture: str | None = None
    checkpoint_sha256: str | None = None
    active_role: str | None = None
    preprocessing: str | None = None
    pooling_method: str | None = None
    calibrated: bool | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    classes: tuple[str, ...] = MELD_CLASS_NAMES


class ModelsStatusResponse(BaseModel):
    status: str
    models: dict[str, ModelStatus]


class MultimodalPredictionResponse(BaseModel):
    predicted_emotion: str
    probabilities: dict[str, float]
    confidence: float
    fusion_method: str
    modalities_available: dict[str, bool]
    exclusion_reasons: dict[str, str | None]
    modality_weights: dict[str, float]
    individual_predictions: dict[str, PredictionResponse | None]
    inference_duration_ms: float
    rule_applied: str = ""
    quality: dict[str, Any] = Field(default_factory=dict)

